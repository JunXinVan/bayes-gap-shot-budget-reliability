from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from overprompting_exp.tasks.base import TaskAdapter, TaskBatch
from overprompting_exp.types import Example


@dataclass(frozen=True)
class LongICLBenchConfig:
    dataset_name: str = "TIGER-Lab/LongICLBench"
    split: str = "Discovery"  # HF dataset uses dataset names as splits
    max_examples: int = 200
    start_index: int = 0  # deterministic slicing for lightweight multi-replicate runs
    rounds: Sequence[int] = (1, 2, 3, 4, 5)  # use the provided "N Round Prompt" fields
    batch_size: int = 1  # prompts are huge; keep memory bounded
    candidates: Sequence[str] | None = None  # optional override; otherwise inferred from gold labels


class LongICLBenchTask(TaskAdapter):
    """Adapter for the HF dataset `TIGER-Lab/LongICLBench`.

    LongICLBench ships pre-built prompts at multiple "rounds" (1..5). Each row contains:
    - `N Round Prompt`: the full prompt string (instruction + demonstrations + query stub)
    - `label`: a gold label (for candidate-set classification tasks)

    This adapter exposes those pre-built prompts via `TaskBatch.user_prompts_by_k` so the
    experiments can score candidate sets without reformatting the prompt text.
    """

    def __init__(self, cfg: LongICLBenchConfig, seed: int):
        self._cfg = cfg
        self._seed = seed
        self._ds = None
        self._indices: list[int] = []
        self._candidates: list[str] = []

    def _try_find_local_cache(self) -> Path | None:
        """Try to find cached LongICLBench dataset directory locally."""
        cache_root = Path("/root/autodl-tmp/hf_cache/datasets/long_icl_bench")
        if not cache_root.exists():
            return None
        # Map split name to local directory name
        task_name = self._cfg.split.lower()
        local_dir = cache_root / task_name
        if local_dir.exists() and local_dir.is_dir():
            return local_dir
        return None

    def load(self) -> None:
        try:
            from datasets import load_dataset, load_from_disk
        except Exception as e:  # pragma: no cover
            raise ImportError("LongICLBench requires extras: pip install -e '.[datasets]'") from e

        # Try local cache first
        local_cache_path = self._try_find_local_cache()
        if local_cache_path:
            print(f"[LongICLBench] Using local cache: {local_cache_path}")
            ds = load_from_disk(str(local_cache_path))
        else:
            print(f"[LongICLBench] Local cache not found, loading from HF Hub...")
            ds = load_dataset(self._cfg.dataset_name, split=str(self._cfg.split))
        if len(ds) < 1:
            raise RuntimeError(
                f"Empty LongICLBench split: dataset={self._cfg.dataset_name} split={self._cfg.split}"
            )

        rounds = tuple(int(r) for r in self._cfg.rounds)
        if not rounds:
            raise ValueError("LongICLBenchConfig.rounds must be non-empty.")
        for r in rounds:
            if r == 0:
                continue  # k=0 (zero-shot) will be handled separately
            col = f"{r} Round Prompt"
            if col not in ds.column_names:
                raise KeyError(f"LongICLBench missing column {col!r}. Available: {ds.column_names}")

        # Candidate set.
        if self._cfg.candidates is not None:
            candidates = [str(x).strip() for x in self._cfg.candidates if str(x).strip()]
        else:
            labels = [str(x).strip() for x in ds["label"]]
            # LongICLBench includes some structured-generation splits (e.g., FewNERD/DialogRE).
            # This codebase currently targets candidate-set classification tasks; fail fast if labels are not scalar.
            if any("\n" in l for l in labels):
                raise ValueError(
                    "This LongICLBench split appears to be structured / multi-line output (label contains newlines). "
                    "Choose a candidate-set classification split (e.g., GoEmotion/BANKING77/TacRED/Discovery)."
                )
            if any(", " in l for l in labels):
                raise ValueError(
                    "This LongICLBench split appears to be multi-label / structured output (label contains ', '). "
                    "Choose a candidate-set classification split (e.g., GoEmotion/BANKING77/TacRED/Discovery), "
                    "or set task.longiclbench.candidates explicitly."
                )
            candidates = sorted({l for l in labels if l})

        if not candidates:
            raise ValueError("Empty candidate set for LongICLBench; set task.longiclbench.candidates explicitly.")

        start = int(self._cfg.start_index)
        if start < 0 or start >= len(ds):
            raise ValueError(f"LongICLBench start_index out of range: {start} (len={len(ds)})")
        n = min(int(self._cfg.max_examples), len(ds) - start)
        self._ds = ds
        self._indices = list(range(start, start + n))
        self._candidates = list(candidates)

    def iter_batches(self) -> Iterable[TaskBatch]:
        if self._ds is None:
            raise RuntimeError("Call load() first")

        rounds = tuple(int(r) for r in self._cfg.rounds)
        bs = max(int(self._cfg.batch_size), 1)

        for start in range(0, len(self._indices), bs):
            batch_idx = self._indices[start : start + bs]

            examples: list[Example] = []
            pools = []
            prompts_by_k: list[Mapping[int, str]] = []

            for i in batch_idx:
                row = self._ds[int(i)]
                gold = str(row["label"]).strip()
                examples.append(
                    Example(example_id=str(i), input_text="", gold=gold, candidates=list(self._candidates))
                )
                pools.append([])  # demos not surfaced for this benchmark adapter

                m: dict[int, str] = {}
                for r in rounds:
                    if r == 0:
                        # Construct zero-shot prompt from 1 Round Prompt
                        # Extract instruction and query (remove demonstrations)
                        one_round = str(row["1 Round Prompt"])
                        # Split by the pattern to find demonstrations
                        parts = one_round.split('the most suitable conjunction word in the previous ( ) is')
                        if len(parts) >= 2:
                            # parts[0] = instruction + first demo sentence
                            # parts[-1] = last demo answer + query
                            instruction = parts[0].strip()
                            # Extract query from the last part (after the last answer)
                            last_part = parts[-1]
                            # Find the query (it ends without "the most suitable...")
                            query_lines = last_part.strip().split('\n')
                            # The query is the text before the final incomplete answer line
                            query = ''
                            for i, line in enumerate(query_lines):
                                if '( )' in line and i < len(query_lines) - 1:
                                    query = line.strip()
                                    break
                            if query:
                                # Reconstruct: instruction header + query
                                # Extract just the task description from instruction
                                header = instruction.split('The examples are as follows:')[0].strip()
                                m[0] = header + '\n' + query + '\nthe most suitable conjunction word in the previous ( ) is '
                            else:
                                m[0] = one_round  # fallback
                        else:
                            m[0] = one_round  # fallback
                    else:
                        m[int(r)] = str(row[f"{r} Round Prompt"])
                prompts_by_k.append(m)

            yield TaskBatch(examples=examples, demo_pools=pools, user_prompts_by_k=prompts_by_k)
