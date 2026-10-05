from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ScoredDistribution:
    candidates: Sequence[str]
    logp: np.ndarray  # shape [C]

    def probs(self) -> np.ndarray:
        m = float(np.max(self.logp))
        p = np.exp(self.logp - m)
        return p / np.sum(p)


class ModelAdapter(ABC):
    @abstractmethod
    def score_candidates(self, prompt: str, candidates: Sequence[str]) -> ScoredDistribution:
        raise NotImplementedError

    def score_candidates_many(self, prompts: Sequence[str], candidates: Sequence[str]) -> list[ScoredDistribution]:
        """Score the same candidate set for multiple prompts.

        Default implementation falls back to per-prompt scoring. Backends with a
        real batched inference path should override this method.
        """
        return [self.score_candidates(prompt, candidates) for prompt in prompts]

    def score_candidates_padded(
        self,
        prompt: str,
        candidates: Sequence[str],
        *,
        pad_left_to_len: int | None = None,
    ) -> ScoredDistribution:
        """Score candidates while optionally applying a token-level left-padding intervention.

        This is used for E3-style "position-lock" treatments: pad the *prompt* to a target token
        length using masked pad tokens, so absolute position indices of downstream tokens are
        controlled across k.

        Backends that can implement masked padding + explicit position_ids should override this.
        Default implementation ignores padding and falls back to `score_candidates`.
        """
        _ = pad_left_to_len
        return self.score_candidates(prompt, candidates)

    def score_candidates_many_padded(
        self,
        prompts: Sequence[str],
        candidates: Sequence[str],
        *,
        pad_left_to_len: int | None = None,
    ) -> list[ScoredDistribution]:
        """Batched variant of ``score_candidates_padded``.

        Default implementation falls back to scoring prompts one by one.
        """
        return [
            self.score_candidates_padded(prompt, candidates, pad_left_to_len=pad_left_to_len)
            for prompt in prompts
        ]

    def format_prompt(self, user_prompt: str, system_prompt: str = "") -> str:
        system_prompt = system_prompt.strip()
        if not system_prompt:
            return user_prompt
        return system_prompt + "\n\n" + user_prompt

    def get_query_hidden(self, prompt: str) -> np.ndarray:
        raise NotImplementedError("Hidden-state extraction not implemented for this backend")

    def get_output_weight(self):  # type: ignore[no-untyped-def]
        """Return the model output projection weight W_out (if available).

        This is used for offline certificate computations (e.g., ||W_out||_op).
        Backends that do not expose weights should raise NotImplementedError.
        """
        raise NotImplementedError("Weight access not implemented for this backend")

    def prompt_token_count(self, prompt: str) -> int:
        """Return the token length of a (formatted) prompt.

        Default implementation is an approximation based on whitespace splitting.
        Backends with a real tokenizer (e.g., HF) should override this.
        """
        return len(str(prompt).split())

    def query_start_token(self, prompt: str, query_prefix: str) -> int | None:
        """Return the token index of the query block start (if found).

        We locate the last occurrence of `query_prefix` in the *formatted* prompt,
        then return the token count of the prefix substring. This is a key observable
        for position/shift analyses.

        Default implementation is an approximation based on whitespace splitting.
        """
        idx = str(prompt).rfind(str(query_prefix))
        if idx < 0:
            return None
        return len(str(prompt)[:idx].split())
