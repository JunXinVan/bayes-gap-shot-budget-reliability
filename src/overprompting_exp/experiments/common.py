from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from overprompting_exp.config import RunConfig, load_yaml, parse_run
from overprompting_exp.io_utils import ensure_dir
from overprompting_exp.models.base import ModelAdapter
from overprompting_exp.models.dummy import DummyModelAdapter, DummyModelConfig
from overprompting_exp.models.hf import HFModelAdapter, HFModelConfig
from overprompting_exp.models.vllm import VLLMModelAdapter, VLLMModelConfig
from overprompting_exp.prompting.perturbations import (
    PerturbationSpec,
    apply_pad_shift_before_query,
    apply_reverse_order,
    apply_shuffle,
    apply_template_replacements,
    apply_template_replacements_before_query,
)
from overprompting_exp.prompting.template import PromptTemplate, build_user_prompt
from overprompting_exp.seed import set_global_seed
from overprompting_exp.scoring.cache import CacheConfig, CachedModelAdapter, ScoreCache
from overprompting_exp.tasks.local_jsonl import LocalJSONLConfig, LocalJSONLTask
from overprompting_exp.tasks.longiclbench import LongICLBenchConfig, LongICLBenchTask
from overprompting_exp.tasks.manyiclbench import ManyICLBenchConfig, ManyICLBenchTask
from overprompting_exp.tasks.hellaswag import HellaSwagConfig, HellaSwagTask
from overprompting_exp.tasks.gsm8k import GSM8KConfig, GSM8KTask
from overprompting_exp.tasks.hard_reasoning import HardReasoningConfig, HardReasoningTask
from overprompting_exp.tasks.mmlu_redux import MMLUReduxConfig, MMLUReduxTask
from overprompting_exp.types import Demo, Example


@dataclass(frozen=True)
class GridSpec:
    k_values: Sequence[int]
    max_demos: int


@dataclass(frozen=True)
class DemoSelectorSpec:
    mode: str
    seed: int


def parse_template(cfg: dict[str, Any]) -> PromptTemplate:
    p = cfg.get("prompt", {}) or {}
    return PromptTemplate(
        system=str(p.get("system", "")),
        demo_prefix=str(p.get("demo_prefix", "### Example {i}\n")),
        query_prefix=str(p.get("query_prefix", "### Query\n")),
        answer_prefix=str(p.get("answer_prefix", "### Answer\n")),
        query_preamble=str(p.get("query_preamble", "")),
        query_instruction=str(p.get("query_instruction", p.get("query_suffix", ""))),
    )


def parse_grid(cfg: dict[str, Any]) -> GridSpec:
    g = cfg.get("grid", {}) or {}
    return GridSpec(k_values=list(g.get("k_values", [0, 1, 2, 4, 8])), max_demos=int(g.get("max_demos", 8)))


def parse_demo_selector(cfg: dict[str, Any], *, default_seed: int = 0) -> DemoSelectorSpec:
    d = cfg.get("demo_selector", {}) or {}
    mode = str(d.get("mode", "benchmark_prefix")).strip().lower()
    if mode not in {"benchmark_prefix", "random_nested"}:
        raise ValueError(f"Unknown demo_selector.mode: {mode!r}")
    return DemoSelectorSpec(mode=mode, seed=int(d.get("seed", default_seed)))


def parse_perturbations(cfg: dict[str, Any]) -> list[PerturbationSpec]:
    def _parse_template_replacements(obj: Any) -> tuple[tuple[str, str], ...]:
        if obj is None:
            return ()
        if isinstance(obj, dict):
            return tuple((str(k), str(v)) for k, v in obj.items())
        if isinstance(obj, list):
            out_pairs: list[tuple[str, str]] = []
            for it in obj:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    out_pairs.append((str(it[0]), str(it[1])))
                    continue
                if isinstance(it, dict):
                    if "src" in it and "dst" in it:
                        out_pairs.append((str(it["src"]), str(it["dst"])))
                        continue
                    if len(it) == 1:
                        (k, v), *_rest = list(it.items())
                        out_pairs.append((str(k), str(v)))
                        continue
                raise ValueError(f"Invalid template replacement entry: {it!r}")
            return tuple(out_pairs)
        raise ValueError(f"Invalid template replacements spec (expected list/dict), got {type(obj)}")

    out: list[PerturbationSpec] = []
    for p in cfg.get("perturbations", []) or []:
        out.append(
            PerturbationSpec(
                kind=str(p["kind"]),
                num_samples=int(p.get("num_samples", 1)),
                seed=int(p.get("seed", 0)),
                pad_token=str(p.get("pad_token", "\n")),
                pad_lengths=tuple(int(x) for x in (p.get("pad_lengths", []) or [])),
                template_replacements=_parse_template_replacements(p.get("template_replacements")),
            )
        )
    return out


def build_model(cfg: dict[str, Any], seed: int) -> ModelAdapter:
    m = cfg.get("model", {}) or {}
    backend = str(m.get("backend", "dummy"))
    if backend == "dummy":
        d = m.get("dummy", {}) or {}
        base: ModelAdapter = DummyModelAdapter(
            DummyModelConfig(num_candidates=int(d.get("num_candidates", 4)), seed=seed)
        )
        return _maybe_wrap_cache(base, cfg, namespace="dummy")
    if backend == "hf":
        h = m.get("hf", {}) or {}
        scoring = cfg.get("scoring", {}) or {}
        batch_size = int(h.get("batch_size", scoring.get("batch_size", 4)))
        base = HFModelAdapter(
            HFModelConfig(
                model_name=str(h["model_name"]),
                dtype=str(h.get("dtype", "bfloat16")),
                device=str(h.get("device", "auto")),
                trust_remote_code=bool(h.get("trust_remote_code", False)),
                use_chat_template=bool(h.get("use_chat_template", True)),
                add_generation_prompt=bool(h.get("add_generation_prompt", True)),
                batch_size=batch_size,
                attn_implementation=(str(h.get("attn_implementation")) if h.get("attn_implementation") is not None else None),
            )
        )
        return _maybe_wrap_cache(base, cfg, namespace=str(h.get("model_name", "hf")))
    if backend == "vllm":
        v = m.get("vllm", {}) or {}
        base = VLLMModelAdapter(
            VLLMModelConfig(
                model_name=str(v["model_name"]),
                max_model_len=int(v["max_model_len"]) if v.get("max_model_len") is not None else None,
            )
        )
        return _maybe_wrap_cache(base, cfg, namespace=str(v.get("model_name", "vllm")))
    raise ValueError(f"Unknown model backend: {backend}")


def _parse_cache(cfg: dict[str, Any], namespace: str) -> CacheConfig:
    c = cfg.get("cache", {}) or {}
    enabled = bool(c.get("enabled", False))
    root = Path(c.get("dir", "data/cache_scores"))
    ns = str(c.get("namespace", namespace))
    return CacheConfig(enabled=enabled, dir=root, namespace=ns)


def _maybe_wrap_cache(base: ModelAdapter, cfg: dict[str, Any], namespace: str) -> ModelAdapter:
    cache_cfg = _parse_cache(cfg, namespace=namespace)
    if not cache_cfg.enabled:
        return base
    cache = ScoreCache(root=cache_cfg.dir, namespace=cache_cfg.namespace)
    return CachedModelAdapter(base=base, cache=cache)


def _effective_max_examples(c: dict[str, Any], default_max: int = 200) -> int:
    """Compute max_examples after applying optional data_fraction (e.g. 0.5 = 50%, 0.2 = 20%)."""
    max_examples = int(c.get("max_examples", default_max))
    fraction = float(c.get("data_fraction", 1.0))
    if fraction <= 0 or fraction > 1.0:
        raise ValueError("task.*.data_fraction must be in (0, 1], e.g. 0.5 for 50%")
    return max(1, int(max_examples * fraction))


def build_task(cfg: dict[str, Any], seed: int):
    t = cfg.get("task", {}) or {}
    backend = str(t.get("backend", "local_jsonl"))
    if backend == "local_jsonl":
        c = t.get("local_jsonl", {}) or {}
        raw_max = c.get("max_examples")
        fraction = float(c.get("data_fraction", 1.0))
        if fraction <= 0 or fraction > 1.0:
            raise ValueError("task.local_jsonl.data_fraction must be in (0, 1]")
        effective_max = max(1, int((raw_max or 999999) * fraction)) if raw_max is not None else None
        task = LocalJSONLTask(
            LocalJSONLConfig(
                path=Path(c["path"]),
                candidate_field=str(c.get("candidate_field", "candidates")),
                input_field=str(c.get("input_field", "input")),
                gold_field=str(c.get("gold_field", "gold")),
            ),
            seed=seed,
            max_examples=effective_max,
        )
        task.load()
        return task
    if backend == "manyiclbench":
        c = t.get("manyiclbench", {}) or {}
        task = ManyICLBenchTask(
            ManyICLBenchConfig(
                dataset_name=str(c.get("dataset_name", "launch/ManyICLBench")),
                task_name=str(c.get("task_name", "ARC-Challenge")),
                split=str(c.get("split", "seed0")),
                prefix_key=str(c.get("prefix_key", "8k")),
                max_examples=_effective_max_examples(c, 200),
                start_index=int(c.get("start_index", 0)),
                candidates=c.get("candidates"),
            ),
            seed=seed,
        )
        task.load()
        return task
    if backend == "longiclbench":
        c = t.get("longiclbench", {}) or {}
        task = LongICLBenchTask(
            LongICLBenchConfig(
                dataset_name=str(c.get("dataset_name", "TIGER-Lab/LongICLBench")),
                split=str(c.get("split", "Discovery")),
                max_examples=_effective_max_examples(c, 200),
                start_index=int(c.get("start_index", 0)),
                rounds=tuple(int(x) for x in (c.get("rounds", [1, 2, 3, 4, 5]) or [1, 2, 3, 4, 5])),
                batch_size=int(c.get("batch_size", 1)),
                candidates=c.get("candidates"),
            ),
            seed=seed,
        )
        task.load()
        return task
    if backend == "hellaswag":
        c = t.get("hellaswag", {}) or {}
        task = HellaSwagTask(
            HellaSwagConfig(
                data_dir=str(c.get("data_dir", "/root/autodl-tmp/data/hellaswag-master/data")),
                split=str(c.get("split", "val")),
                max_examples=_effective_max_examples(c, 200),
                start_index=int(c.get("start_index", 0)),
                candidates=c.get("candidates"),
                demo_pool_size=int(c.get("demo_pool_size", 100)),
            ),
            seed=seed,
        )
        task.load()
        return task
    if backend == "gsm8k":
        c = t.get("gsm8k", {}) or {}
        task = GSM8KTask(
            GSM8KConfig(
                data_dir=str(c.get("data_dir", "/root/autodl-tmp/data/gsm8k")),
                split=str(c.get("split", "test")),
                max_examples=_effective_max_examples(c, 200),
                start_index=int(c.get("start_index", 0)),
                demo_pool_size=int(c.get("demo_pool_size", 100)),
            ),
            seed=seed,
        )
        task.load()
        return task
    if backend == "mmlu_redux":
        c = t.get("mmlu_redux", {}) or {}
        task = MMLUReduxTask(
            MMLUReduxConfig(
                subset_path=str(c["subset_path"]),
                demo_pool_path=str(c["demo_pool_path"]),
                max_examples=int(c.get("max_examples", 228)),
                start_index=int(c.get("start_index", 0)),
                candidates=c.get("candidates"),
            ),
            seed=seed,
        )
        task.load()
        return task
    if backend == "hard_reasoning":
        c = t.get("hard_reasoning", {}) or {}
        task = HardReasoningTask(
            HardReasoningConfig(
                data_dir=str(c.get("data_dir", "/root/autodl-tmp/data/hard_reasoning")),
                split=str(c.get("split", "test")),
                max_examples=_effective_max_examples(c, 200),
                start_index=int(c.get("start_index", 0)),
                demo_pool_size=int(c.get("demo_pool_size", 100)),
            ),
            seed=seed,
        )
        task.load()
        return task
    raise ValueError(f"Unknown task backend: {backend}")


def _demo_pool_fingerprint(demo_pool: Sequence[Demo]) -> bytes:
    h = hashlib.sha256()
    for demo in demo_pool:
        h.update(demo.input_text.encode("utf-8"))
        h.update(b"\0")
        h.update(demo.output_text.encode("utf-8"))
        h.update(b"\0")
    return h.digest()


def _random_nested_prefix(demo_pool: Sequence[Demo], k: int, *, seed: int) -> list[Demo]:
    # Use one fixed permutation per run and per demo pool. Prefixing that
    # permutation keeps the path nested across k while degrading benchmark path
    # quality in a controlled way.
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    h.update(b"\0")
    h.update(_demo_pool_fingerprint(demo_pool))
    perm_seed = int.from_bytes(h.digest()[:8], byteorder="little", signed=False) % (2**32)
    rng = np.random.default_rng(perm_seed)
    perm = rng.permutation(len(demo_pool))
    return [demo_pool[int(i)] for i in perm[:k]]


def select_demos(
    demo_pool: Sequence[Demo], k: int, selector: DemoSelectorSpec | None = None
) -> list[Demo]:
    selector = selector or DemoSelectorSpec(mode="benchmark_prefix", seed=0)
    k_eff = min(int(k), len(demo_pool))
    if selector.mode == "benchmark_prefix":
        return list(demo_pool[:k_eff])
    if selector.mode == "random_nested":
        return _random_nested_prefix(demo_pool, k_eff, seed=int(selector.seed))
    raise ValueError(f"Unknown demo selector mode: {selector.mode!r}")


def build_prompt_for_k(
    template: PromptTemplate,
    example: Example,
    demo_pool: Sequence[Demo],
    k: int,
    selector: DemoSelectorSpec | None = None,
) -> str:
    demos = select_demos(demo_pool, k, selector=selector)
    return build_user_prompt(template, demos, example)


def iter_perturbed_prompts(
    template: PromptTemplate,
    example: Example,
    demo_pool: Sequence[Demo] | None,
    k: int,
    perturbations: Sequence[PerturbationSpec],
    base_prompt: str,
    selector: DemoSelectorSpec | None = None,
) -> Iterable[tuple[str, str]]:
    """Yield (tag, prompt_variant)."""
    demos = select_demos(demo_pool, k, selector=selector) if demo_pool is not None else None

    for spec in perturbations:
        if spec.kind == "reverse_order":
            if demos is None:
                continue
            demos_p = apply_reverse_order(demos)
            yield ("reverse_order", build_user_prompt(template, demos_p, example))
        elif spec.kind == "shuffle":
            if demos is None:
                continue
            # Deterministic but diverse shuffles: vary permutations by (example_id, k)
            # while keeping reproducibility under the global run/config seed.
            h = hashlib.sha256()
            h.update(str(spec.seed).encode("utf-8"))
            h.update(b"\0")
            h.update(str(example.example_id).encode("utf-8"))
            h.update(b"\0")
            h.update(str(k).encode("utf-8"))
            seed = int.from_bytes(h.digest()[:8], byteorder="little", signed=False) % (2**32)
            rng = np.random.default_rng(seed)
            for j in range(spec.num_samples):
                demos_p = apply_shuffle(demos, rng)
                yield (f"shuffle_{j}", build_user_prompt(template, demos_p, example))
        elif spec.kind == "pad_shift":
            for L in spec.pad_lengths:
                yield (
                    f"pad_{L}",
                    apply_pad_shift_before_query(
                        base_prompt, query_prefix=template.query_prefix, pad_token=spec.pad_token, pad_len=L
                    ),
                )
        elif spec.kind == "template_replace":
            if not spec.template_replacements:
                continue
            yield ("template_replace", apply_template_replacements(base_prompt, spec.template_replacements))
        elif spec.kind == "separator_replace":
            if not spec.template_replacements:
                continue
            # Separator edits should only touch demonstrations, not the final
            # query block, so the k=0 anchor remains identical to baseline.
            yield (
                "separator_replace",
                apply_template_replacements_before_query(
                    base_prompt, query_prefix=template.query_prefix, replacements=spec.template_replacements
                ),
            )
        else:
            raise ValueError(f"Unknown perturbation kind: {spec.kind}")


def prepare_run(config_path: Path) -> tuple[RunConfig, dict[str, Any]]:
    cfg = load_yaml(config_path)
    run = parse_run(cfg)
    set_global_seed(run.seed)
    return run, cfg


def run_output_dir(run: RunConfig) -> Path:
    out = run.output_dir / run.name
    ensure_dir(out)
    ensure_dir(out / "figures")
    return out
