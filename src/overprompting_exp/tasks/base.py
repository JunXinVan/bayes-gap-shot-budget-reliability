from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from overprompting_exp.types import Demo, Example


@dataclass(frozen=True)
class TaskBatch:
    examples: Sequence[Example]
    demo_pools: Sequence[Sequence[Demo]]  # one pool per example (for nested-k coupling)
    # Optional fast path for benchmarks that ship pre-built prompts at each k/round.
    # When provided, experiments should use these prompts instead of reformatting demos.
    user_prompts_by_k: Sequence[Mapping[int, str]] | None = None


class TaskAdapter(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def iter_batches(self) -> Iterable[TaskBatch]:
        raise NotImplementedError
