from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from overprompting_exp.types import Demo


@dataclass(frozen=True)
class PerturbationSpec:
    kind: str
    num_samples: int = 1
    seed: int = 0
    pad_token: str = "\n"
    pad_lengths: Sequence[int] = ()
    template_replacements: Sequence[tuple[str, str]] = ()


def apply_reverse_order(demos: Sequence[Demo]) -> list[Demo]:
    return list(reversed(demos))


def apply_shuffle(demos: Sequence[Demo], rng: np.random.Generator) -> list[Demo]:
    idx = np.arange(len(demos))
    rng.shuffle(idx)
    return [demos[i] for i in idx.tolist()]


def apply_pad_shift(prefix: str, pad_token: str, pad_len: int) -> str:
    return (pad_token * pad_len) + prefix


def apply_pad_shift_before_query(user_prompt: str, query_prefix: str, pad_token: str, pad_len: int) -> str:
    pad = pad_token * pad_len
    # Use the *last* occurrence so we pad immediately before the query block even
    # when the same prefix appears in demonstrations (e.g., ManyICLBench "Question:").
    idx = user_prompt.rfind(query_prefix)
    if idx < 0:
        return pad + user_prompt
    return user_prompt[:idx] + pad + user_prompt[idx:]


def apply_template_replacements(user_prompt: str, replacements: Sequence[tuple[str, str]]) -> str:
    out = str(user_prompt)
    for src, dst in replacements:
        src_s = str(src)
        if not src_s:
            continue
        out = out.replace(src_s, str(dst))
    return out


def apply_template_replacements_before_query(
    user_prompt: str, query_prefix: str, replacements: Sequence[tuple[str, str]]
) -> str:
    # Use the last query prefix so edits only touch the demonstration block and
    # preserve the final query format. This keeps k=0 identical to baseline.
    idx = user_prompt.rfind(query_prefix)
    if idx < 0:
        return apply_template_replacements(user_prompt, replacements)
    head = apply_template_replacements(user_prompt[:idx], replacements)
    return head + user_prompt[idx:]
