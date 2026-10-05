"""Task adapter for HellaSwag dataset.

HellaSwag is a commonsense reasoning dataset where each example contains:
- ctx_a: First part of the context (sentence)
- ctx_b: Second part to be completed
- ctx: Combined context (ctx_a + " " + ctx_b)
- endings: List of 4 possible completions
- label: Index (0-3) of the correct ending
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from overprompting_exp.tasks.base import TaskAdapter, TaskBatch
from overprompting_exp.types import Demo, Example


@dataclass(frozen=True)
class HellaSwagConfig:
    """Configuration for HellaSwag task."""
    data_dir: str = "/root/autodl-tmp/data/hellaswag-master/data"
    split: str = "val"  # "train" or "val"
    max_examples: int = 200
    start_index: int = 0
    candidates: Sequence[str] | None = None  # ["0", "1", "2", "3"] by default
    # For demo pool construction
    demo_pool_size: int = 100  # Number of demos to load from training set


class HellaSwagTask(TaskAdapter):
    """Adapter for HellaSwag commonsense reasoning dataset.
    
    Task format: Given a context (ctx), select the most plausible
    continuation from 4 options (endings).
    """
    
    def __init__(self, cfg: HellaSwagConfig, seed: int):
        self._cfg = cfg
        self._seed = seed
        self._examples: list[Example] = []
        self._demo_pool: list[Demo] = []
    
    def _load_jsonl(self, filename: str) -> list[dict]:
        """Load JSONL file."""
        path = Path(self._cfg.data_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"HellaSwag data file not found: {path}")
        
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _format_question(self, ctx: str, endings: list[str]) -> str:
        """Format the question with context and options."""
        lines = [f"Context: {ctx}", ""]
        for i, ending in enumerate(endings):
            lines.append(f"{i}. {ending}")
        return "\n".join(lines)
    
    def _build_demo_pool(self) -> list[Demo]:
        """Build demo pool from training set."""
        train_data = self._load_jsonl("hellaswag_train.jsonl")
        
        # Shuffle with seed for reproducibility
        rng = random.Random(self._seed)
        rng.shuffle(train_data)
        
        demos = []
        for item in train_data[:self._cfg.demo_pool_size]:
            ctx = item.get("ctx", "").strip()
            endings = item.get("endings", [])
            label = item.get("label", 0)
            
            if not ctx or not endings or len(endings) != 4:
                continue
            
            # Format as question + answer
            question = self._format_question(ctx, endings)
            answer = str(label)  # "0", "1", "2", or "3"
            
            demos.append(Demo(input_text=question, output_text=answer))
        
        return demos
    
    def load(self) -> None:
        """Load HellaSwag data."""
        # Load demo pool from training set
        self._demo_pool = self._build_demo_pool()
        
        if not self._demo_pool:
            raise RuntimeError("Failed to build demo pool from training set")
        
        # Load test examples from validation set
        if self._cfg.split == "train":
            val_data = self._load_jsonl("hellaswag_train.jsonl")
        else:
            val_data = self._load_jsonl("hellaswag_val.jsonl")
        
        start = self._cfg.start_index
        if start < 0 or start >= len(val_data):
            raise ValueError(f"start_index={start} out of range for split length {len(val_data)}")
        
        n = min(self._cfg.max_examples, len(val_data) - start)
        
        # Candidates are always ["0", "1", "2", "3"]
        candidates = list(self._cfg.candidates) if self._cfg.candidates else ["0", "1", "2", "3"]
        
        examples = []
        for i in range(n):
            item = val_data[start + i]
            ctx = item.get("ctx", "").strip()
            endings = item.get("endings", [])
            label = item.get("label", 0)
            
            if not ctx or not endings or len(endings) != 4:
                continue
            
            question = self._format_question(ctx, endings)
            gold = str(label)
            
            examples.append(
                Example(
                    example_id=str(start + i),
                    input_text=question,
                    gold=gold,
                    candidates=list(candidates)
                )
            )
        
        self._examples = examples
        
        if not self._examples:
            raise RuntimeError("No valid examples loaded from validation set")
    
    def iter_batches(self) -> Iterable[TaskBatch]:
        """Yield batches of examples with demo pool."""
        if not self._examples:
            raise RuntimeError("Call load() first")
        
        # All examples share the same demo pool
        pools = [self._demo_pool for _ in self._examples]
        yield TaskBatch(examples=self._examples, demo_pools=pools)
