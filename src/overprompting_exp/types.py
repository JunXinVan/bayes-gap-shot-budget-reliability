from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Example:
    example_id: str
    input_text: str
    gold: str
    candidates: Sequence[str]


@dataclass(frozen=True)
class Demo:
    input_text: str
    output_text: str

