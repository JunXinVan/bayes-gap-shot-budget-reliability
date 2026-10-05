from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from overprompting_exp.tasks.base import TaskAdapter, TaskBatch
from overprompting_exp.types import Demo, Example


@dataclass(frozen=True)
class MMLUReduxConfig:
    subset_path: str
    demo_pool_path: str
    max_examples: int = 228
    start_index: int = 0
    candidates: Sequence[str] | None = None


class MMLUReduxTask(TaskAdapter):
    """Task adapter for locally materialized MMLU-Redux 2.0 support subsets.

    The subset JSONL stores normalized evaluation rows with fields:
      - id, input, gold, candidates, subject
    The demo JSONL stores normalized demo rows with fields:
      - id, input, output, subject

    The adapter keeps a single shared demo pool for all evaluation examples so the
    nested shot-scaling path is consistent across the benchmark.
    """

    def __init__(self, cfg: MMLUReduxConfig, seed: int):
        self._cfg = cfg
        self._seed = seed
        self._examples: list[Example] = []
        self._demo_pool: list[Demo] = []

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"MMLU-Redux JSONL not found: {path}")
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def load(self) -> None:
        subset_path = Path(self._cfg.subset_path)
        demo_pool_path = Path(self._cfg.demo_pool_path)
        subset_rows = self._load_jsonl(subset_path)
        demo_rows = self._load_jsonl(demo_pool_path)

        start = int(self._cfg.start_index)
        if start < 0:
            raise ValueError(f"start_index must be >= 0, got {start}")
        if start >= len(subset_rows):
            raise ValueError(f"start_index={start} out of range for subset length {len(subset_rows)}")
        n = min(int(self._cfg.max_examples), len(subset_rows) - start)

        candidates_override = [str(x) for x in self._cfg.candidates] if self._cfg.candidates is not None else None

        examples: list[Example] = []
        for row in subset_rows[start : start + n]:
            candidates = candidates_override or [str(c) for c in row.get("candidates", ["0", "1", "2", "3"])]
            gold = str(row["gold"])
            if gold not in candidates:
                raise ValueError(f"Gold label {gold!r} not found in candidates for row id={row.get('id')!r}")
            examples.append(
                Example(
                    example_id=str(row.get("id", len(examples))),
                    input_text=str(row["input"]),
                    gold=gold,
                    candidates=candidates,
                )
            )

        demo_pool: list[Demo] = []
        for row in demo_rows:
            demo_pool.append(Demo(input_text=str(row["input"]), output_text=str(row["output"])))

        if not examples:
            raise RuntimeError("No valid MMLU-Redux examples loaded")
        if not demo_pool:
            raise RuntimeError("No valid MMLU-Redux demos loaded")

        self._examples = examples
        self._demo_pool = demo_pool

    def iter_batches(self) -> Iterable[TaskBatch]:
        if not self._examples:
            raise RuntimeError("Call load() first")
        pools = [self._demo_pool for _ in self._examples]
        yield TaskBatch(examples=self._examples, demo_pools=pools)
