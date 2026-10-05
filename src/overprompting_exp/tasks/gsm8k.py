"""Task adapter for GSM8K (Grade School Math) dataset.

GSM8K contains grade school math word problems with:
- question: The math word problem
- answer: Reasoning process + final answer (after ####)

Example:
{
  "question": "Janet's ducks lay 16 eggs per day...",
  "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs...\n#### 18"
}
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from overprompting_exp.tasks.base import TaskAdapter, TaskBatch
from overprompting_exp.types import Demo, Example


@dataclass(frozen=True)
class GSM8KConfig:
    """Configuration for GSM8K task."""
    data_dir: str = "/root/autodl-tmp/data/gsm8k"
    split: str = "test"  # "train" or "test"
    max_examples: int = 200
    start_index: int = 0
    # For demo pool construction
    demo_pool_size: int = 100


def _extract_final_answer(answer_text: str) -> str:
    """Extract the final answer after ####."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    # Fallback: try to find a number at the end
    lines = answer_text.strip().split("\n")
    for line in reversed(lines):
        numbers = re.findall(r'-?\d+\.?\d*', line)
        if numbers:
            return numbers[-1]
    return ""


def _format_question(question: str) -> str:
    """Format the question."""
    return f"Question: {question}\nAnswer:"


class GSM8KTask(TaskAdapter):
    """Adapter for GSM8K math reasoning dataset.
    
    Task format: Given a math word problem, solve it step by step
    and provide the final numerical answer.
    """
    
    def __init__(self, cfg: GSM8KConfig, seed: int):
        self._cfg = cfg
        self._seed = seed
        self._examples: list[Example] = []
        self._demo_pool: list[Demo] = []
    
    def _load_jsonl(self, filename: str) -> list[dict]:
        """Load JSONL file."""
        path = Path(self._cfg.data_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"GSM8K data file not found: {path}")
        
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _build_demo_pool(self) -> list[Demo]:
        """Build demo pool from training set."""
        train_data = self._load_jsonl("train.jsonl")
        
        # Shuffle with seed for reproducibility
        rng = random.Random(self._seed)
        rng.shuffle(train_data)
        
        demos = []
        for item in train_data[:self._cfg.demo_pool_size]:
            question = item.get("question", "").strip()
            answer_full = item.get("answer", "").strip()
            final_answer = _extract_final_answer(answer_full)
            
            if not question or not final_answer:
                continue
            
            # Format as question + reasoning + final answer
            formatted_question = _format_question(question)
            # Include the reasoning process in the demo output
            formatted_answer = f"{answer_full}"
            
            demos.append(Demo(input_text=formatted_question, output_text=formatted_answer))
        
        return demos
    
    def load(self) -> None:
        """Load GSM8K data."""
        # Load demo pool from training set
        self._demo_pool = self._build_demo_pool()
        
        if not self._demo_pool:
            raise RuntimeError("Failed to build demo pool from training set")
        
        # Load test examples
        if self._cfg.split == "train":
            test_data = self._load_jsonl("train.jsonl")
        else:
            test_data = self._load_jsonl("test.jsonl")
        
        start = self._cfg.start_index
        if start < 0 or start >= len(test_data):
            raise ValueError(f"start_index={start} out of range for split length {len(test_data)}")
        
        n = min(self._cfg.max_examples, len(test_data) - start)
        
        # For GSM8K, candidates are open-ended (we'll use the gold answer for validation)
        # In practice, we extract the final number from model's output
        examples = []
        for i in range(n):
            item = test_data[start + i]
            question = item.get("question", "").strip()
            answer_full = item.get("answer", "").strip()
            gold_answer = _extract_final_answer(answer_full)
            
            if not question or not gold_answer:
                continue
            
            formatted_question = _format_question(question)
            
            examples.append(
                Example(
                    example_id=str(start + i),
                    input_text=formatted_question,
                    gold=gold_answer,
                    candidates=[]  # Open-ended, no fixed candidates
                )
            )
        
        self._examples = examples
        
        if not self._examples:
            raise RuntimeError("No valid examples loaded")
    
    def iter_batches(self) -> Iterable[TaskBatch]:
        """Yield batches of examples with demo pool."""
        if not self._examples:
            raise RuntimeError("Call load() first")
        
        # All examples share the same demo pool
        pools = [self._demo_pool for _ in self._examples]
        yield TaskBatch(examples=self._examples, demo_pools=pools)
