from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from overprompting_exp.tasks.base import TaskAdapter, TaskBatch
from overprompting_exp.types import Demo, Example


@dataclass(frozen=True)
class LocalJSONLConfig:
    path: Path
    candidate_field: str = "candidates"
    input_field: str = "input"
    gold_field: str = "gold"
    id_field: str = "id"
    demos_field: str = "demos"


class LocalJSONLTask(TaskAdapter):
    def __init__(self, cfg: LocalJSONLConfig, seed: int, max_examples: int | None = None):
        self._cfg = cfg
        self._seed = seed
        self._max_examples = max_examples
        self._rows: list[dict] = []

    def load(self) -> None:
        rows: list[dict] = []
        with self._cfg.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if self._max_examples is not None and len(rows) >= self._max_examples:
                    break
        self._rows = rows

    def iter_batches(self) -> Iterable[TaskBatch]:
        examples: list[Example] = []
        pools: list[list[Demo]] = []
        for idx, r in enumerate(self._rows):
            ex_id = str(r.get(self._cfg.id_field, idx))
            input_text = str(r[self._cfg.input_field])
            gold = str(r[self._cfg.gold_field])
            candidates = [str(c) for c in r[self._cfg.candidate_field]]

            demos_raw = r.get(self._cfg.demos_field, [])
            demo_pool = [Demo(input_text=str(d["input"]), output_text=str(d["output"])) for d in demos_raw]

            examples.append(Example(example_id=ex_id, input_text=input_text, gold=gold, candidates=candidates))
            pools.append(demo_pool)

        yield TaskBatch(examples=examples, demo_pools=pools)

