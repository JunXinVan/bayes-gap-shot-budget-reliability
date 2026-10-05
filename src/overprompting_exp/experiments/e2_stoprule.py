from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from overprompting_exp.experiments.common import (
    build_model,
    build_prompt_for_k,
    build_task,
    iter_perturbed_prompts,
    parse_demo_selector,
    parse_grid,
    parse_perturbations,
    parse_template,
    prepare_run,
    run_output_dir,
)
from overprompting_exp.io_utils import write_json, write_jsonl
from overprompting_exp.plotting.curves import plot_e2_stoprule
from overprompting_exp.run_metadata import dump_run_metadata
from overprompting_exp.scoring.metrics import compute_example_metrics, kl_div


@dataclass(frozen=True)
class SelectionRuleConfig:
    lambdas: list[float]


@dataclass(frozen=True)
class DangerStopRuleConfig:
    tau_entropy_drop: float
    require_bg_non_decrease: bool = True


@dataclass(frozen=True)
class LegacyStopRuleConfig:
    lambda_entropy: float
    lambda_sensitivity: float
    eta_margin: float = 0.0  # reserved


def parse_selection_rule(cfg: dict[str, Any]) -> SelectionRuleConfig:
    s = cfg.get("selection_rule", {}) or {}
    lambdas = s.get("lambdas", [0.1, 0.3, 1.0, 3.0])
    return SelectionRuleConfig(lambdas=[float(x) for x in lambdas])


def parse_danger_stop_rule(cfg: dict[str, Any]) -> DangerStopRuleConfig | None:
    s = cfg.get("stop_rule", {}) or {}
    kind = str(s.get("kind", "danger_regime"))
    if kind not in {"danger_regime", "danger", "legacy_entropy_and_sensitivity"}:
        kind = "danger_regime"

    enabled = bool(s.get("enabled", kind != "legacy_entropy_and_sensitivity"))
    if not enabled or kind == "legacy_entropy_and_sensitivity":
        return None

    return DangerStopRuleConfig(
        tau_entropy_drop=float(s.get("tau_entropy_drop", s.get("lambda_entropy", 0.01))),
        require_bg_non_decrease=bool(s.get("require_bg_non_decrease", True)),
    )


def parse_legacy_stop_rule(cfg: dict[str, Any]) -> LegacyStopRuleConfig | None:
    s = cfg.get("stop_rule", {}) or {}
    kind = str(s.get("kind", "danger_regime"))
    enabled = bool(s.get("enabled", False))
    if kind != "legacy_entropy_and_sensitivity" and not enabled:
        return None
    if kind != "legacy_entropy_and_sensitivity":
        return None
    return LegacyStopRuleConfig(
        lambda_entropy=float(s.get("lambda_entropy", 0.01)),
        lambda_sensitivity=float(s.get("lambda_sensitivity", 0.02)),
        eta_margin=float(s.get("eta_margin", 0.0)),
    )


def _choose_k_selection(per_k: dict[int, dict[str, float]], lambda_bg: float) -> int:
    ks = sorted(per_k.keys())
    return min(ks, key=lambda k: per_k[k]["entropy"] + float(lambda_bg) * per_k[k]["bg_disagree"])


def _choose_k_bg_only(per_k: dict[int, dict[str, float]]) -> int:
    ks = sorted(per_k.keys())
    return min(ks, key=lambda k: per_k[k]["bg_disagree"])


def _choose_k_entropy_only(per_k: dict[int, dict[str, float]]) -> int:
    ks = sorted(per_k.keys())
    return min(ks, key=lambda k: per_k[k]["entropy"])


def _choose_k_fixed(per_k: dict[int, dict[str, float]], k_fixed: int) -> int:
    ks = sorted(per_k.keys())
    if not ks:
        raise ValueError("Empty k-grid")
    k_fixed = int(k_fixed)
    # snap to the nearest available k
    return min(ks, key=lambda k: (abs(int(k) - k_fixed), int(k)))


def _choose_k_stop_danger(per_k: dict[int, dict[str, float]], cfg: DangerStopRuleConfig) -> int:
    ks = sorted(per_k.keys())
    if not ks:
        raise ValueError("Empty k-grid")
    if len(ks) == 1:
        return ks[0]

    chosen = ks[-1]
    for i in range(len(ks) - 1):
        k = ks[i]
        k2 = ks[i + 1]
        entropy_drop = float(per_k[k]["entropy"] - per_k[k2]["entropy"])
        bg_delta = float(per_k[k2]["bg_disagree"] - per_k[k]["bg_disagree"])
        if entropy_drop <= cfg.tau_entropy_drop and (not cfg.require_bg_non_decrease or bg_delta >= 0.0):
            chosen = k
            break
    return chosen


def _choose_k_stop_legacy(per_k: dict[int, dict[str, float]], cfg: LegacyStopRuleConfig) -> int:
    ks = sorted(per_k.keys())
    if not ks:
        raise ValueError("Empty k-grid")
    if len(ks) == 1:
        return ks[0]

    chosen = ks[-1]
    for i in range(len(ks) - 1):
        k = ks[i]
        k2 = ks[i + 1]
        gain = max(float(per_k[k]["entropy"] - per_k[k2]["entropy"]), 0.0)
        bg = float(per_k[k]["bg_disagree"])
        if gain <= cfg.lambda_entropy and bg >= cfg.lambda_sensitivity:
            chosen = k
            break
    return chosen


def _choose_k_stop_gain_vs_sens(per_k: dict[int, dict[str, float]], *, lambda_bg: float) -> int:
    """v55 main-text sequential rule: stop at first k where Ghat_k <= lambda * Sens_hat_k.

    Here:
    - Ghat_k is the entropy drop between adjacent grid points.
    - Sens_hat_k is disagreement KL under Bayes-invariant perturbations ("bg_disagree").
    """
    ks = sorted(per_k.keys())
    if not ks:
        raise ValueError("Empty k-grid")
    if len(ks) == 1:
        return ks[0]

    chosen = ks[-1]
    lam = float(lambda_bg)
    for i in range(len(ks) - 1):
        k = ks[i]
        k2 = ks[i + 1]
        gain = max(float(per_k[k]["entropy"] - per_k[k2]["entropy"]), 0.0)
        sens = float(per_k[k]["bg_disagree"])
        if gain <= lam * sens:
            chosen = k
            break
    return chosen


def run_e2(config_path: Path) -> None:
    run, cfg = prepare_run(config_path)
    out_dir = run_output_dir(run)
    dump_run_metadata(out_dir, config_path=config_path)

    template = parse_template(cfg)
    grid = parse_grid(cfg)
    demo_selector = parse_demo_selector(cfg, default_seed=run.seed)
    perturbations = parse_perturbations(cfg)
    sel_cfg = parse_selection_rule(cfg)
    stop_cfg = parse_danger_stop_rule(cfg)
    legacy_stop_cfg = parse_legacy_stop_rule(cfg)
    position_lock_cfg = cfg.get("position_lock", {}) or {}
    position_lock_enabled = bool(position_lock_cfg.get("enabled", False))
    main_cfg = cfg.get("main_prompt_perturbation", {}) or {}
    main_enabled = bool(main_cfg.get("enabled", False))
    main_mode = str(main_cfg.get("mode", "mean")).lower()
    main_include_base = bool(main_cfg.get("include_base", False))
    main_max_samples = int(main_cfg.get("max_samples", 0))

    model = build_model(cfg, seed=run.seed)
    task = build_task(cfg, seed=run.seed)

    rows: list[dict[str, Any]] = []

    for batch in task.iter_batches():
        prompts_by_k = batch.user_prompts_by_k
        for idx in tqdm(range(len(batch.examples)), desc="examples"):
            ex = batch.examples[idx]
            pool = batch.demo_pools[idx]
            prompt_map = prompts_by_k[idx] if prompts_by_k is not None else None
            try:
                gold_idx = list(ex.candidates).index(ex.gold)
            except ValueError:
                gold_idx = 0

            k_entries: list[tuple[int, str, str, int, int | None]] = []
            for k in grid.k_values:
                if prompt_map is not None:
                    k_eff = int(k)
                    if k_eff not in prompt_map:
                        continue
                    user_prompt = str(prompt_map[k_eff])
                else:
                    k_eff = min(int(k), int(grid.max_demos), len(pool))
                    user_prompt = build_prompt_for_k(template, ex, pool, k_eff, selector=demo_selector)
                prompt = model.format_prompt(user_prompt=user_prompt, system_prompt=template.system)
                prompt_len_tokens_raw = int(model.prompt_token_count(prompt))
                query_start_token_raw = model.query_start_token(prompt, template.query_prefix)
                k_entries.append((k_eff, user_prompt, prompt, prompt_len_tokens_raw, query_start_token_raw))

            if not k_entries:
                continue

            pad_left_to_len = max(x[3] for x in k_entries) if position_lock_enabled else None

            def score_prompt_metrics(prompt_text: str):
                if pad_left_to_len is None:
                    dist = model.score_candidates(prompt_text, ex.candidates)
                else:
                    dist = model.score_candidates_padded(prompt_text, ex.candidates, pad_left_to_len=int(pad_left_to_len))
                p = dist.probs()
                m = compute_example_metrics(p, gold_idx)
                return p, m

            # precompute per-k metrics
            per_k: dict[int, dict[str, float]] = {}
            per_k_meta: dict[int, dict[str, Any]] = {}
            for k_eff, user_prompt, prompt, prompt_len_tokens_raw, query_start_token_raw in k_entries:
                pad_left = max(0, int(pad_left_to_len) - int(prompt_len_tokens_raw)) if pad_left_to_len is not None else 0
                prompt_len_tokens = int(prompt_len_tokens_raw) + int(pad_left)
                query_start_token = (
                    int(query_start_token_raw) + int(pad_left) if query_start_token_raw is not None else None
                )

                p_base, m_base = score_prompt_metrics(prompt)
                nll = m_base.nll
                acc = m_base.acc
                ent = m_base.ent

                perturbed_all = list(
                    iter_perturbed_prompts(
                        template=template,
                        example=ex,
                        demo_pool=None if prompt_map is not None else pool,
                        k=k_eff,
                        perturbations=perturbations,
                        base_prompt=user_prompt,
                        selector=demo_selector,
                    )
                )

                if main_enabled and perturbed_all:
                    variants = [p_prompt for _tag, p_prompt in perturbed_all]
                    if main_include_base:
                        variants = [user_prompt] + variants
                    if main_mode == "first":
                        variants = variants[:1]
                    if main_max_samples > 0:
                        variants = variants[:main_max_samples]
                    if variants:
                        nlls = []
                        accs = []
                        ents = []
                        for p_prompt in variants:
                            p_prompt_f = model.format_prompt(user_prompt=p_prompt, system_prompt=template.system)
                            _p_v, m_v = score_prompt_metrics(p_prompt_f)
                            nlls.append(float(m_v.nll))
                            accs.append(float(m_v.acc))
                            ents.append(float(m_v.ent))
                        nll = float(np.mean(nlls))
                        acc = float(np.mean(accs))
                        ent = float(np.mean(ents))

                sens_terms: list[float] = []
                loss_sens_terms: list[float] = []
                for _tag, p_prompt in perturbed_all:
                    p_prompt_f = model.format_prompt(user_prompt=p_prompt, system_prompt=template.system)
                    if pad_left_to_len is None:
                        q = model.score_candidates(p_prompt_f, ex.candidates).probs()
                    else:
                        q = model.score_candidates_padded(
                            p_prompt_f, ex.candidates, pad_left_to_len=int(pad_left_to_len)
                        ).probs()
                    sens_terms.append(kl_div(p_base, q))
                    nll_g = float(-np.log(max(float(q[gold_idx]), 1e-12)))
                    loss_sens_terms.append(abs(nll_g - float(m_base.nll)))

                per_k[k_eff] = {
                    "nll": float(nll),
                    "acc": float(acc),
                    "entropy": float(ent),
                    # v38 wording: disagreement-based BG proxy (label-free)
                    "bg_disagree": float(np.mean(sens_terms)) if sens_terms else 0.0,
                    # v55 Appendix heuristic: Sens^loss (label-dependent) / POSIX-style.
                    "sens_loss": float(np.mean(loss_sens_terms)) if loss_sens_terms else 0.0,
                }
                per_k_meta[k_eff] = {
                    "prompt_len_tokens": prompt_len_tokens,
                    "query_start_token": query_start_token,
                }

            ks = sorted(per_k.keys())
            oracle_k = min(ks, key=lambda k: per_k[k]["nll"])

            max_k = int(max(ks)) if ks else 0
            max_prompt_len_tokens = max(int(per_k_meta[k]["prompt_len_tokens"]) for k in ks)
            policies: list[tuple[str, int]] = []
            for lam in sel_cfg.lambdas:
                policies.append((f"selection_lambda_{lam:g}", _choose_k_selection(per_k, lambda_bg=lam)))
                policies.append(
                    (
                        f"stop_gain_vs_sens_lambda_{lam:g}",
                        _choose_k_stop_gain_vs_sens(per_k, lambda_bg=lam),
                    )
                )
            policies.append(("entropy_only", _choose_k_entropy_only(per_k)))
            policies.append(("bg_only", _choose_k_bg_only(per_k)))

            baselines = (cfg.get("baselines", {}) or {}).get("fixed_k", [8, 32])
            for k0 in baselines:
                policies.append((f"fixed_{int(k0)}", _choose_k_fixed(per_k, int(k0))))

            if stop_cfg is not None:
                policies.append(("stop_danger", _choose_k_stop_danger(per_k, stop_cfg)))
            if legacy_stop_cfg is not None:
                policies.append(("stop_legacy", _choose_k_stop_legacy(per_k, legacy_stop_cfg)))

            for policy_name, chosen_k in policies:
                rows.append(
                    {
                        "example_id": ex.example_id,
                        "policy": policy_name,
                        "oracle_k": oracle_k,
                        "chosen_k": chosen_k,
                        "k_max": max_k,
                        "prompt_len_tokens_max": max_prompt_len_tokens,
                        "oracle_nll": per_k[oracle_k]["nll"],
                        "chosen_nll": per_k[chosen_k]["nll"],
                        "oracle_acc": per_k[oracle_k]["acc"],
                        "chosen_acc": per_k[chosen_k]["acc"],
                        "oracle_entropy": per_k[oracle_k]["entropy"],
                        "chosen_entropy": per_k[chosen_k]["entropy"],
                        "oracle_bg_disagree": per_k[oracle_k]["bg_disagree"],
                        "chosen_bg_disagree": per_k[chosen_k]["bg_disagree"],
                        "oracle_sens_loss": per_k[oracle_k]["sens_loss"],
                        "chosen_sens_loss": per_k[chosen_k]["sens_loss"],
                        "oracle_prompt_len_tokens": int(per_k_meta[oracle_k]["prompt_len_tokens"]),
                        "chosen_prompt_len_tokens": int(per_k_meta[chosen_k]["prompt_len_tokens"]),
                        "oracle_query_start_token": per_k_meta[oracle_k]["query_start_token"],
                        "chosen_query_start_token": per_k_meta[chosen_k]["query_start_token"],
                        "chosen_prompt_len_over_max": float(
                            float(per_k_meta[chosen_k]["prompt_len_tokens"]) / float(max_prompt_len_tokens)
                            if max_prompt_len_tokens > 0
                            else 1.0
                        ),
                    }
                )

    if rows:
        agg: dict[str, Any] = {"n_examples": len({r["example_id"] for r in rows}), "policies": {}}
        for policy in sorted({str(r["policy"]) for r in rows}):
            sub = [r for r in rows if str(r["policy"]) == policy]
            regret = [float(r["chosen_nll"] - r["oracle_nll"]) for r in sub]
            saving = [
                (float(r["chosen_k"]) / float(r["k_max"]) if float(r["k_max"]) > 0 else 1.0) for r in sub
            ]
            token_saving = [float(r.get("chosen_prompt_len_over_max", 1.0)) for r in sub]
            agg["policies"][policy] = {
                "n": len(sub),
                "regret_nll_mean": float(np.mean(regret)),
                "regret_nll_median": float(np.median(regret)),
                "chosen_k_mean": float(np.mean([r["chosen_k"] for r in sub])),
                "oracle_k_mean": float(np.mean([r["oracle_k"] for r in sub])),
                "chosen_k_over_kmax_mean": float(np.mean(saving)),
                "chosen_prompt_len_over_max_mean": float(np.mean(token_saving)),
                "chosen_acc_mean": float(np.mean([r["chosen_acc"] for r in sub])),
            }
    else:
        agg = {}

    write_jsonl(out_dir / "decision.jsonl", rows)
    write_json(out_dir / "aggregate_decision.json", agg)

    if bool((cfg.get("plot", {}) or {}).get("enabled", False)):
        try:
            # plot one representative policy (prefer a selection lambda if present)
            plot_policy = str((cfg.get("plot", {}) or {}).get("policy", ""))
            if not plot_policy:
                plot_policy = f"selection_lambda_{sel_cfg.lambdas[0]:g}" if sel_cfg.lambdas else "entropy_only"
            plot_e2_stoprule(rows, out_dir / "figures" / "e2_oracle_vs_chosen.png", policy=plot_policy)
        except ImportError:
            pass
