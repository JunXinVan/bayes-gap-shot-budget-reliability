from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from overprompting_exp.models.base import ModelAdapter, ScoredDistribution


@dataclass(frozen=True)
class VLLMModelConfig:
    model_name: str
    max_model_len: int | None = None


class VLLMModelAdapter(ModelAdapter):
    """vLLM backend (placeholder).

    This adapter is intentionally minimal for now. Candidate-scoring via vLLM can be made
    significantly faster than naive HF loops, but the exact API surface (prompt logprobs vs
    completion logprobs) depends on the vLLM version and serving mode.
    """

    def __init__(self, cfg: VLLMModelConfig):
        try:
            from vllm import LLM  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError("vLLM backend requires extras: pip install -e '.[vllm]'") from e

        self._cfg = cfg
        raise NotImplementedError(
            "VLLMModelAdapter is a scaffold. Use HF backend for now, or implement vLLM scoring for your setup."
        )

    def score_candidates(self, prompt: str, candidates: Sequence[str]) -> ScoredDistribution:  # pragma: no cover
        raise NotImplementedError

