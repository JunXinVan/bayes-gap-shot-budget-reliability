"""Task adapter for Hard Reasoning (logic + multi-step math) dataset."""

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
class HardReasoningConfig:
    """Configuration for Hard Reasoning task."""
    data_dir: str = "/root/autodl-tmp/data/hard_reasoning"
    split: str = "test"  # "train" or "test"
    max_examples: int = 200
    start_index: int = 0
    demo_pool_size: int = 100


def _extract_answer(text: str) -> str:
    """Extract the final answer from generated text."""
    # Look for patterns like "The answer is X" or just the last number/word
    patterns = [
        r'(?:the answer is|answer:)\s*([\w\d]+)',
        r'(?:is|equals?)\s*([\w\d]+)\.?$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()
    
    # Fallback: return last word/number
    words = text.strip().split()
    if words:
        return words[-1].strip('.!?')
    return ""


class HardReasoningTask(TaskAdapter):
    """Adapter for Hard Reasoning (logic + multi-step math) dataset."""
    
    def __init__(self, cfg: HardReasoningConfig, seed: int):
        self._cfg = cfg
        self._seed = seed
        self._examples: list[Example] = []
        self._demo_pool: list[Demo] = []
    
    def _load_jsonl(self, filename: str) -> list[dict]:
        """Load JSONL file."""
        path = Path(self._cfg.data_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
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
            answer = item.get("answer", "").strip()
            
            if not question or not answer:
                continue
            
            # Format as question + reasoning + answer
            formatted_question = f"Problem: {question}\nSolution:"
            formatted_answer = f"Let me solve this step by step. The answer is {answer}."
            
            demos.append(Demo(input_text=formatted_question, output_text=formatted_answer))
        
        return demos
    
    def load(self) -> None:
        """Load Hard Reasoning data."""
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
        
        examples = []
        for i in range(n):
            item = test_data[start + i]
            question = item.get("question", "").strip()
            gold_answer = item.get("answer", "").strip()
            
            if not question or not gold_answer:
                continue
            
            formatted_question = f"Problem: {question}\nSolution:"
            
            examples.append(
                Example(
                    example_id=str(start + i),
                    input_text=formatted_question,
                    gold=gold_answer.lower(),
                    candidates=[]  # Open-ended
                )
            )
        
        self._examples = examples
        
        if not self._examples:
            raise RuntimeError("No valid examples loaded")
    
    def iter_batches(self) -> Iterable[TaskBatch]:
        """Yield batches of examples with demo pool."""
        if not self._examples:
            raise RuntimeError("Call load() first")
        
        pools = [self._demo_pool for _ in self._examples]
        yield TaskBatch(examples=self._examples, demo_pools=pools)
