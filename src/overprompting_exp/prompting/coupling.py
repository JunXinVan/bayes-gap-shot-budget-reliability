from __future__ import annotations

from typing import Sequence

from overprompting_exp.prompting.template import PromptTemplate, build_user_prompt
from overprompting_exp.types import Demo, Example


def coupled_user_prompts(
    template: PromptTemplate,
    example: Example,
    demo_pool: Sequence[Demo],
    k: int,
) -> tuple[str, str]:
    """Return a nested prompt pair (k, k+1) built from a fixed demo pool order."""
    k0 = max(int(k), 0)
    demos_k = list(demo_pool[:k0])
    demos_k1 = list(demo_pool[: k0 + 1])
    return (
        build_user_prompt(template, demos_k, example),
        build_user_prompt(template, demos_k1, example),
    )

