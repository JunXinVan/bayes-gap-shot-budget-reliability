from __future__ import annotations

from collections import defaultdict
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
from overprompting_exp.io_utils import append_jsonl, write_json, write_jsonl
from overprompting_exp.plotting.curves import plot_e1_curves
from overprompting_exp.prompting.perturbations import apply_pad_shift_before_query
from overprompting_exp.run_metadata import dump_run_metadata
from overprompting_exp.scoring.metrics import brier_score, compute_example_metrics, kl_div, top1_confidence


def run_e1(config_path: Path) -> None:
    run, cfg = prepare_run(config_path)
    out_dir = run_output_dir(run)
    dump_run_metadata(out_dir, config_path=config_path)

    template = parse_template(cfg)
    grid = parse_grid(cfg)
    demo_selector = parse_demo_selector(cfg, default_seed=run.seed)
    perturbations = parse_perturbations(cfg)
    drift_cfg = cfg.get("drift", {}) or {}
    drift_enabled = bool(drift_cfg.get("enabled", False))
    position_lock_cfg = cfg.get("position_lock", {}) or {}
    position_lock_enabled = bool(position_lock_cfg.get("enabled", False))
    shift_cfg = cfg.get("shift_amplify", {}) or {}
    shift_enabled = bool(shift_cfg.get("enabled", False))
    shift_pad_token = str(shift_cfg.get("pad_token", "\n"))
    shift_pad_per_demo = int(shift_cfg.get("pad_per_demo", 0))
    main_cfg = cfg.get("main_prompt_perturbation", {}) or {}
    main_enabled = bool(main_cfg.get("enabled", False))
    main_mode = str(main_cfg.get("mode", "mean")).lower()
    main_include_base = bool(main_cfg.get("include_base", False))
    main_max_samples = int(main_cfg.get("max_samples", 0))
    if drift_enabled and position_lock_enabled:
        raise ValueError("drift.enabled is not compatible with position_lock.enabled yet. Disable drift for now.")
    if shift_enabled:
        if shift_pad_per_demo <= 0:
            raise ValueError("shift_amplify.pad_per_demo must be > 0 when shift_amplify.enabled is true.")
        if position_lock_enabled:
            raise ValueError("shift_amplify.enabled is not compatible with position_lock.enabled.")

    def apply_shift_knob(user_prompt: str, *, k_eff: int) -> str:
        if not shift_enabled:
            return user_prompt
        return apply_pad_shift_before_query(
            user_prompt,
            query_prefix=template.query_prefix,
            pad_token=shift_pad_token,
            pad_len=int(shift_pad_per_demo) * int(k_eff),
        )

    model = build_model(cfg, seed=run.seed)
    task = build_task(cfg, seed=run.seed)

    rows: list[dict[str, Any]] = []

    for batch in task.iter_batches():
        prompts_by_k = batch.user_prompts_by_k
        for idx in tqdm(range(len(batch.examples)), desc="examples"):
            ex = batch.examples[idx]
            pool = batch.demo_pools[idx]
            prompt_map = prompts_by_k[idx] if prompts_by_k is not None else None
            # map gold string -> index (only for classification tasks)
            is_open_ended = len(ex.candidates) == 0
            if is_open_ended:
                gold_idx = 0  # Not used for open-ended tasks
            else:
                try:
                    gold_idx = list(ex.candidates).index(ex.gold)
                except ValueError:
                    gold_idx = 0

            k_entries: list[tuple[int, str, str, str, int, int | None]] = []
            for k in grid.k_values:
                if prompt_map is not None:
                    k_eff = int(k)
                    if k_eff not in prompt_map:
                        continue
                    user_prompt_raw = str(prompt_map[k_eff])
                else:
                    k_eff = min(int(k), int(grid.max_demos), len(pool))
                    user_prompt_raw = build_prompt_for_k(template, ex, pool, k_eff, selector=demo_selector)
                user_prompt = apply_shift_knob(user_prompt_raw, k_eff=k_eff)
                prompt = model.format_prompt(user_prompt=user_prompt, system_prompt=template.system)
                prompt_len_tokens_raw = int(model.prompt_token_count(prompt))
                query_start_token_raw = model.query_start_token(prompt, template.query_prefix)
                k_entries.append((k_eff, user_prompt_raw, user_prompt, prompt, prompt_len_tokens_raw, query_start_token_raw))

            if not k_entries:
                continue

            pad_left_to_len = max(x[4] for x in k_entries) if position_lock_enabled else None

            for k_eff, user_prompt_raw, user_prompt, prompt, prompt_len_tokens_raw, query_start_token_raw in k_entries:
                pad_left = max(0, int(pad_left_to_len) - int(prompt_len_tokens_raw)) if pad_left_to_len is not None else 0
                prompt_len_tokens = int(prompt_len_tokens_raw) + int(pad_left)
                query_start_token = (
                    int(query_start_token_raw) + int(pad_left) if query_start_token_raw is not None else None
                )

                # Check if this is an open-ended task (no candidates)
                is_open_ended = len(ex.candidates) == 0

                if is_open_ended:
                    # Open-ended generation task (e.g., GSM8K)
                    # Use generate instead of score_candidates
                    generated_text = model.generate(prompt, max_new_tokens=256, temperature=0.0)

                    # Extract the last number from generated text
                    import re

                    numbers = re.findall(r"-?\d+\.?\d*", generated_text)
                    pred_answer = numbers[-1] if numbers else ""

                    # Compare with gold
                    acc = float(pred_answer.strip() == ex.gold.strip())

                    # For open-ended tasks, we don't have NLL/entropy/margin in the same way
                    # Use placeholders
                    nll = 0.0
                    ent = 0.0
                    margin = 0.0
                    conf = 0.0
                    brier = 0.0
                    bg_disagree = 0.0
                    sens_loss = 0.0
                else:
                    def score_prompt_stats_many(prompt_texts: list[str]):
                        if pad_left_to_len is None:
                            dists = model.score_candidates_many(prompt_texts, ex.candidates)
                        else:
                            dists = model.score_candidates_many_padded(
                                prompt_texts, ex.candidates, pad_left_to_len=int(pad_left_to_len)
                            )
                        stats = []
                        for dist in dists:
                            p = dist.probs()
                            m = compute_example_metrics(p, gold_idx)
                            stats.append(
                                (
                                    float(m.nll),
                                    float(m.acc),
                                    float(m.ent),
                                    float(m.margin),
                                    float(top1_confidence(p)),
                                    float(brier_score(p, gold_idx)),
                                    p,
                                )
                            )
                        return stats

                    perturbed_all = list(
                        iter_perturbed_prompts(
                            template=template,
                            example=ex,
                            demo_pool=None if prompt_map is not None else pool,
                            k=k_eff,
                            perturbations=perturbations,
                            base_prompt=user_prompt_raw,
                            selector=demo_selector,
                        )
                    )
                    # Apply the shift-amplification knob consistently to all perturbation variants.
                    perturbed_all = [(tag, apply_shift_knob(p_prompt, k_eff=k_eff)) for tag, p_prompt in perturbed_all]
                    perturbed_prompts_f = [
                        model.format_prompt(user_prompt=p_prompt, system_prompt=template.system)
                        for _tag, p_prompt in perturbed_all
                    ]

                    all_prompt_stats = score_prompt_stats_many([prompt] + perturbed_prompts_f)
                    nll_base, acc_base, ent_base, margin_base, conf_base, brier_base, p = all_prompt_stats[0]
                    perturbed_stats = all_prompt_stats[1:]

                    # Classification task with candidates
                    nll = nll_base
                    acc = acc_base
                    ent = ent_base
                    margin = margin_base
                    conf = conf_base
                    brier = brier_base

                    if main_enabled and perturbed_all:
                        variants = list(perturbed_stats)
                        if main_include_base:
                            variants = [all_prompt_stats[0]] + variants
                        if main_mode == "first":
                            variants = variants[:1]
                        if main_max_samples > 0:
                            variants = variants[:main_max_samples]

                        if variants:
                            nlls = [float(v[0]) for v in variants]
                            accs = [float(v[1]) for v in variants]
                            ents = [float(v[2]) for v in variants]
                            margins = [float(v[3]) for v in variants]
                            confs = [float(v[4]) for v in variants]
                            briers = [float(v[5]) for v in variants]
                            nll = float(np.mean(nlls))
                            acc = float(np.mean(accs))
                            ent = float(np.mean(ents))
                            margin = float(np.mean(margins))
                            conf = float(np.mean(confs))
                            brier = float(np.mean(briers))

                    # sensitivity proxy: average KL(q || q_g)
                    disagree_kl_terms: list[float] = []
                    loss_sens_terms: list[float] = []
                    for nll_g, _acc_g, _ent_g, _margin_g, _conf_g, _brier_g, q in perturbed_stats:
                        disagree_kl_terms.append(kl_div(p, q))
                        loss_sens_terms.append(abs(float(nll_g) - float(nll_base)))

                    bg_disagree = float(np.mean(disagree_kl_terms)) if disagree_kl_terms else 0.0
                    sens_loss = float(np.mean(loss_sens_terms)) if loss_sens_terms else 0.0

                hidden_drift = None
                if drift_enabled:
                    user_prompt_k1 = None
                    if prompt_map is not None:
                        user_prompt_k1 = prompt_map.get(int(k_eff + 1))
                    elif (k_eff + 1 <= int(grid.max_demos)) and (k_eff + 1 <= len(pool)):
                        user_prompt_k1 = build_prompt_for_k(
                            template, ex, pool, k_eff + 1, selector=demo_selector
                        )

                    if user_prompt_k1 is not None:
                        prompt_k1 = model.format_prompt(user_prompt=str(user_prompt_k1), system_prompt=template.system)
                        try:
                            h0 = model.get_query_hidden(prompt)
                            h1 = model.get_query_hidden(prompt_k1)
                            hidden_drift = float(np.linalg.norm(h1 - h0))
                        except NotImplementedError:
                            hidden_drift = None

                row = {
                    "example_id": ex.example_id,
                    "k": k_eff,
                    "prompt_len_tokens": prompt_len_tokens,
                    "query_start_token": query_start_token,
                    "nll": nll,
                    "acc": acc,
                    "entropy": ent,
                    "margin": margin,
                    "conf": conf,
                    "brier": brier,
                    # Backward-compatible key name: "sensitivity" (KL disagreement proxy).
                    "sensitivity": bg_disagree,
                    # v55 paper naming: BG proxy via disagreement KL, plus label-dependent loss sensitivity.
                    "bg_disagree": bg_disagree,
                    "sens_loss": sens_loss,
                    "hidden_drift": hidden_drift,
                    # entropy_drop_next will be filled in post-processing
                    "entropy_drop_next": None,
                }
                rows.append(row)
                # REAL-TIME WRITE: append to metrics.jsonl immediately
                append_jsonl(out_dir / "metrics.jsonl", row)

    # aggregate
    if rows:
        ks = sorted({int(r["k"]) for r in rows})
        agg = {}
        for k in ks:
            sub = [r for r in rows if int(r["k"]) == k]
            qpos_vals = [r["query_start_token"] for r in sub if r.get("query_start_token") is not None]
            drift_vals = [r["hidden_drift"] for r in sub if r.get("hidden_drift") is not None]
            drop_vals = [r["entropy_drop_next"] for r in sub if r.get("entropy_drop_next") is not None]
            agg[str(k)] = {
                "n": len(sub),
                "prompt_len_tokens_mean": float(np.mean([r["prompt_len_tokens"] for r in sub])),
                "query_start_token_mean": float(np.mean(qpos_vals)) if qpos_vals else None,
                "nll_mean": float(np.mean([r["nll"] for r in sub])),
                "acc_mean": float(np.mean([r["acc"] for r in sub])),
                "entropy_mean": float(np.mean([r["entropy"] for r in sub])),
                "entropy_drop_mean": float(np.mean(drop_vals)) if drop_vals else None,
                "conf_mean": float(np.mean([r["conf"] for r in sub])),
                "brier_mean": float(np.mean([r["brier"] for r in sub])),
                "sens_loss_mean": float(np.mean([r["sens_loss"] for r in sub])),
                "bg_disagree_mean": float(np.mean([r["bg_disagree"] for r in sub])),
                "sensitivity_mean": float(np.mean([r["sensitivity"] for r in sub])),
                "hidden_drift_mean": float(np.mean(drift_vals)) if drift_vals else None,
            }
    else:
        agg = {}

    write_jsonl(out_dir / "metrics.jsonl", rows)
    write_json(out_dir / "aggregate.json", agg)

    if bool((cfg.get("plot", {}) or {}).get("enabled", False)):
        try:
            plot_e1_curves(agg, out_dir / "figures" / "e1_curves.png")
        except ImportError:
            # plotting is optional
            pass
