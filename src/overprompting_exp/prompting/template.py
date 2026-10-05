from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from overprompting_exp.types import Demo, Example


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    demo_prefix: str  # may include {i}
    query_prefix: str
    answer_prefix: str
    query_preamble: str = ""
    query_instruction: str = ""

    def format_demo(self, demo: Demo, i: int) -> str:
        prefix = self.demo_prefix.format(i=i)
        return f"{prefix}{demo.input_text}\n{self.answer_prefix}{demo.output_text}\n\n"

    def format_query(self, example: Example) -> str:
        question_block = f"{self.query_prefix}{example.input_text}"
        query_preamble = str(self.query_preamble).strip()
        query_instruction = str(self.query_instruction).strip()
        parts: list[str] = []
        if query_preamble:
            parts.append(query_preamble)
        parts.append(question_block)
        if query_instruction:
            parts.append(query_instruction)
        parts.append(self.answer_prefix)
        return "\n".join(parts)

def build_user_prompt(template: PromptTemplate, demos: Sequence[Demo], example: Example) -> str:
    parts: list[str] = []
    for i, d in enumerate(demos, start=1):
        parts.append(template.format_demo(d, i))
    parts.append(template.format_query(example))
    return "".join(parts)

def build_prompt(template: PromptTemplate, demos: Sequence[Demo], example: Example) -> str:
    parts: list[str] = []
    if template.system.strip():
        parts.append(template.system.strip() + "\n\n")
    parts.append(build_user_prompt(template, demos, example))
    return "".join(parts)
