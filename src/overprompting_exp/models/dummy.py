from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from overprompting_exp.models.base import ModelAdapter, ScoredDistribution


@dataclass
class DummyModelConfig:
    num_candidates: int
    seed: int = 0


class DummyModelAdapter(ModelAdapter):
    def __init__(self, cfg: DummyModelConfig):
        self._rng = np.random.default_rng(cfg.seed)

    def score_candidates(self, prompt: str, candidates: Sequence[str]) -> ScoredDistribution:
        # Deterministic-ish: prompt length affects the stream
        _ = len(prompt) % 997  # noqa: F841
        logp = self._rng.normal(size=(len(candidates),)).astype(np.float64)
        return ScoredDistribution(candidates=list(candidates), logp=logp)

    def get_query_hidden(self, prompt: str) -> np.ndarray:
        # Deterministic pseudo-hidden state for smoke tests.
        h = (hash(prompt) % 2**32) / 2**32
        rng = np.random.default_rng(int(h * 1_000_000_007))
        return rng.normal(size=(32,)).astype(np.float64)
