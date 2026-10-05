from __future__ import annotations

import errno
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from overprompting_exp.io_utils import ensure_dir
from overprompting_exp.models.base import ModelAdapter, ScoredDistribution


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = False
    dir: Path = Path("data/cache_scores")
    namespace: str = "default"


class ScoreCache:
    def __init__(self, root: Path, namespace: str):
        self._dir = root / namespace
        ensure_dir(self._dir)
        self._writes_disabled = False

    def _path_for_key(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    @staticmethod
    def make_key(prompt: str, candidates: Sequence[str], *, context: dict[str, object] | None = None) -> str:
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        h.update(b"\0")
        for c in candidates:
            h.update(c.encode("utf-8"))
            h.update(b"\0")
        if context:
            ctx = json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            h.update(b"\0ctx\0")
            h.update(ctx.encode("utf-8"))
        return h.hexdigest()

    def load_logp(self, key: str) -> list[float] | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return [float(x) for x in obj["logp"]]

    def save_logp(self, key: str, logp: Sequence[float]) -> None:
        if self._writes_disabled:
            return
        path = self._path_for_key(key)
        try:
            path.write_text(json.dumps({"logp": list(map(float, logp))}), encoding="utf-8")
        except OSError as e:
            # For long-running full-evaluation jobs, cache writes are auxiliary. If the
            # filesystem hits a quota or no-space condition, continue the run and simply
            # stop writing new cache entries rather than killing the experiment.
            if e.errno in {errno.EDQUOT, errno.ENOSPC}:
                self._writes_disabled = True
                print(
                    f"[score-cache] disabling further cache writes for {self._dir} after {e.strerror or e!s}",
                    file=sys.stderr,
                    flush=True,
                )
                return
            raise


class CachedModelAdapter(ModelAdapter):
    def __init__(self, base: ModelAdapter, cache: ScoreCache):
        self._base = base
        self._cache = cache

    def format_prompt(self, user_prompt: str, system_prompt: str = "") -> str:
        return self._base.format_prompt(user_prompt=user_prompt, system_prompt=system_prompt)

    def score_candidates(self, prompt: str, candidates: Sequence[str]) -> ScoredDistribution:
        key = self._cache.make_key(prompt, candidates)
        cached = self._cache.load_logp(key)
        if cached is not None:
            return ScoredDistribution(candidates=list(candidates), logp=np.asarray(cached, dtype=np.float64))

        dist = self._base.score_candidates(prompt, candidates)
        self._cache.save_logp(key, dist.logp.tolist())
        return dist

    def score_candidates_many(self, prompts: Sequence[str], candidates: Sequence[str]) -> list[ScoredDistribution]:
        results: list[ScoredDistribution | None] = [None] * len(prompts)
        missing_prompts: list[str] = []
        missing_indices: list[int] = []

        for i, prompt in enumerate(prompts):
            key = self._cache.make_key(prompt, candidates)
            cached = self._cache.load_logp(key)
            if cached is not None:
                results[i] = ScoredDistribution(candidates=list(candidates), logp=np.asarray(cached, dtype=np.float64))
            else:
                missing_prompts.append(prompt)
                missing_indices.append(i)

        if missing_prompts:
            dists = self._base.score_candidates_many(missing_prompts, candidates)
            for idx, prompt, dist in zip(missing_indices, missing_prompts, dists, strict=True):
                key = self._cache.make_key(prompt, candidates)
                self._cache.save_logp(key, dist.logp.tolist())
                results[idx] = dist

        if any(dist is None for dist in results):
            raise RuntimeError("Internal cache batching error: missing score_candidates_many result")
        return [dist for dist in results if dist is not None]

    def score_candidates_padded(
        self,
        prompt: str,
        candidates: Sequence[str],
        *,
        pad_left_to_len: int | None = None,
    ) -> ScoredDistribution:
        context = {"pad_left_to_len": int(pad_left_to_len)} if pad_left_to_len is not None else None
        key = self._cache.make_key(prompt, candidates, context=context)
        cached = self._cache.load_logp(key)
        if cached is not None:
            return ScoredDistribution(candidates=list(candidates), logp=np.asarray(cached, dtype=np.float64))

        dist = self._base.score_candidates_padded(prompt, candidates, pad_left_to_len=pad_left_to_len)
        self._cache.save_logp(key, dist.logp.tolist())
        return dist

    def score_candidates_many_padded(
        self,
        prompts: Sequence[str],
        candidates: Sequence[str],
        *,
        pad_left_to_len: int | None = None,
    ) -> list[ScoredDistribution]:
        context = {"pad_left_to_len": int(pad_left_to_len)} if pad_left_to_len is not None else None
        results: list[ScoredDistribution | None] = [None] * len(prompts)
        missing_prompts: list[str] = []
        missing_indices: list[int] = []

        for i, prompt in enumerate(prompts):
            key = self._cache.make_key(prompt, candidates, context=context)
            cached = self._cache.load_logp(key)
            if cached is not None:
                results[i] = ScoredDistribution(candidates=list(candidates), logp=np.asarray(cached, dtype=np.float64))
            else:
                missing_prompts.append(prompt)
                missing_indices.append(i)

        if missing_prompts:
            dists = self._base.score_candidates_many_padded(
                missing_prompts, candidates, pad_left_to_len=pad_left_to_len
            )
            for idx, prompt, dist in zip(missing_indices, missing_prompts, dists, strict=True):
                key = self._cache.make_key(prompt, candidates, context=context)
                self._cache.save_logp(key, dist.logp.tolist())
                results[idx] = dist

        if any(dist is None for dist in results):
            raise RuntimeError("Internal cache batching error: missing score_candidates_many_padded result")
        return [dist for dist in results if dist is not None]

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        return self._base.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def get_query_hidden(self, prompt: str) -> np.ndarray:  # pragma: no cover
        return self._base.get_query_hidden(prompt)

    def get_output_weight(self):  # type: ignore[no-untyped-def]  # pragma: no cover
        return self._base.get_output_weight()

    def prompt_token_count(self, prompt: str) -> int:
        return self._base.prompt_token_count(prompt)

    def query_start_token(self, prompt: str, query_prefix: str) -> int | None:
        return self._base.query_start_token(prompt, query_prefix)
