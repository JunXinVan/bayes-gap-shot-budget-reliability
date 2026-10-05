from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from overprompting_exp.tasks.base import TaskAdapter, TaskBatch
from overprompting_exp.types import Demo, Example


@dataclass(frozen=True)
class ManyICLBenchConfig:
    dataset_name: str = "launch/ManyICLBench"
    task_name: str = "ARC-Challenge"
    split: str = "seed0"
    prefix_key: str = "8k"  # choose which provided prompt prefix to parse into demos
    max_examples: int = 200
    start_index: int = 0  # for chunked runs / resuming
    candidates: Sequence[str] | None = None  # optional override; otherwise inferred


_KEYVAL_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z _-]{0,40}):\s*(?P<val>.*)$")


def _resolve_task_name(dataset_name: str, task_name: str) -> str:
    # Always skip online config lookup to avoid network requests - trust the task_name
    # Local cache loading will handle validation
    return task_name


def _resolve_split(split: str) -> str:
    # ManyICLBench uses seed splits (seed0/seed1/seed2). Keep "test" as a friendly alias.
    s = str(split)
    if s in {"test", "default"}:
        return "seed0"
    return s


def _split_prefix_and_answer_marker(lines: list[str]) -> tuple[str, str, str]:
    """Infer (query_marker, answer_marker, answer_value) from the last labeled line."""
    # Find the last "Key: Value" line (typically "Answer: X" or "Label: foo").
    for i in range(len(lines) - 1, -1, -1):
        m = _KEYVAL_RE.match(lines[i].strip())
        if m is None:
            continue
        key = str(m.group("key")).strip()
        val = str(m.group("val")).strip()
        answer_marker = f"{key}: "
        # Also infer the query marker from the first line if possible.
        m0 = _KEYVAL_RE.match(lines[0])
        query_marker = f"{m0.group('key').strip()}: " if m0 is not None else ""
        return (query_marker, answer_marker, val)
    raise ValueError("Could not find an answer/label line in a demo block.")


def _strip_query_marker(text: str, query_marker: str) -> str:
    if query_marker and text.startswith(query_marker):
        return text[len(query_marker) :]
    return text


def _parse_demo_blocks(prefix: str) -> tuple[list[Demo], str, str]:
    """Parse a ManyICLBench prompt prefix into (demos, query_marker, answer_marker)."""
    blocks = [b for b in prefix.split("\n\n") if b.strip()]
    if not blocks:
        raise ValueError("Empty ManyICLBench prefix.")

    # Infer markers from the first block.
    first_lines = blocks[0].splitlines()
    query_marker, answer_marker, _val = _split_prefix_and_answer_marker(first_lines)

    demos: list[Demo] = []
    for b in blocks:
        lines = b.splitlines()
        q_marker_b, a_marker_b, ans = _split_prefix_and_answer_marker(lines)
        # Require consistency within a task/prefix.
        if query_marker and q_marker_b and q_marker_b != query_marker:
            raise ValueError(f"Inconsistent query marker: {q_marker_b!r} vs {query_marker!r}")
        if answer_marker and a_marker_b and a_marker_b != answer_marker:
            raise ValueError(f"Inconsistent answer marker: {a_marker_b!r} vs {answer_marker!r}")

        # Remove the last labeled line from the input.
        # (Everything up to but excluding that line is the demo input.)
        demo_in = "\n".join(lines[:-1]).rstrip()
        demo_in = _strip_query_marker(demo_in, query_marker)
        demos.append(Demo(input_text=demo_in, output_text=str(ans).strip()))

    return (demos, query_marker, answer_marker)


def _strip_query_answer_stub(query: str, answer_marker: str) -> str:
    # Queries typically end with "Answer:" (no value). Remove that line if present.
    lines = query.splitlines()
    if not lines:
        return query
    last = lines[-1].strip()
    if last.lower().startswith(answer_marker.strip().lower().rstrip(":") + ":"):
        return "\n".join(lines[:-1]).rstrip()
    return query.rstrip()


def _infer_candidates(outputs: Sequence[str]) -> list[str]:
    labels = {str(x).strip() for x in outputs if str(x).strip()}
    if not labels:
        return []
    if labels.issubset({"A", "B", "C", "D"}):
        return ["A", "B", "C", "D"]
    # Stable ordering for multi-class labels.
    return sorted(labels)


class ManyICLBenchTask(TaskAdapter):
    """Adapter for the HF dataset `launch/ManyICLBench`.

    ManyICLBench stores, for each (task, seed split), a *single row* containing:
    - multiple prompt prefixes at different budgets (e.g., "1k", "2k", ..., "128k")
    - `Test Data`: list[str] of queries with an answer stub
    - `Test Target`: list[str] of gold labels/answers

    This adapter parses one chosen prefix (cfg.prefix_key) into a demo pool, and emits
    one (Example, demo_pool) pair per test query.
    """

    def __init__(self, cfg: ManyICLBenchConfig, seed: int):
        self._cfg = cfg
        self._seed = seed
        self._examples: list[Example] = []
        self._demo_pool: list[Demo] = []

    def _try_find_local_cache(self, task_name: str, split: str) -> str | None:
        """Try to find local cached dataset path to avoid network requests."""
        import os
        from pathlib import Path
        
        # Map dataset name to cache directory name
        dataset_to_cache = {
            "launch/ManyICLBench": "launch___many_icl_bench",
            "ShawnMenz/Translation_Bench": "ShawnMenz___Translation_Bench",
        }
        
        cache_name = dataset_to_cache.get(self._cfg.dataset_name)
        if not cache_name:
            return None
        
        # Common HF cache locations
        hf_cache_dirs = [
            Path(os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache")) / "datasets",
            Path.home() / ".cache" / "huggingface" / "hub" / "datasets",
            Path("/root/autodl-tmp/hf_cache") / "datasets",
        ]
        
        for cache_dir in hf_cache_dirs:
            if not cache_dir.exists():
                continue
            task_path = cache_dir / cache_name / task_name
            if task_path.exists():
                # Find the version directory and its subdirectories
                for version_dir in task_path.iterdir():
                    if version_dir.is_dir():
                        # Recursively search for arrow files matching the split
                        arrow_files = list(version_dir.rglob(f"*{split}.arrow"))
                        if arrow_files:
                            # Return the arrow file path
                            return str(arrow_files[0])
        return None

    def load(self) -> None:
        try:
            from datasets import Dataset
        except Exception as e:  # pragma: no cover
            raise ImportError("ManyICLBench requires extras: pip install -e '.[datasets]'") from e

        task_name = _resolve_task_name(self._cfg.dataset_name, self._cfg.task_name)
        split = _resolve_split(self._cfg.split)

        # Load from local cache only - no network requests
        arrow_file_path = self._try_find_local_cache(task_name, split)
        if not arrow_file_path:
            raise FileNotFoundError(f"Local cache not found for {self._cfg.dataset_name}/{task_name}/{split}. "
                                   f"Please download the dataset first.")
        
        ds = Dataset.from_file(arrow_file_path)
        if len(ds) < 1:
            raise RuntimeError(f"Empty ManyICLBench split: dataset={self._cfg.dataset_name} task={task_name} split={split}")
        row = ds[0]

        prefix_key = str(self._cfg.prefix_key)
        if prefix_key not in row:
            # case-insensitive fallback ("8K" -> "8k")
            for k in row.keys():
                if str(k).lower() == prefix_key.lower():
                    prefix_key = str(k)
                    break
        if prefix_key not in row:
            raise KeyError(f"prefix_key={self._cfg.prefix_key!r} not in dataset row keys: {sorted(row.keys())}")

        demo_pool, query_marker, answer_marker = _parse_demo_blocks(str(row[prefix_key]))

        test_data = row.get("Test Data")
        test_target = row.get("Test Target")
        if not isinstance(test_data, list) or not isinstance(test_target, list):
            raise KeyError("ManyICLBench row missing 'Test Data'/'Test Target' lists.")

        start = int(self._cfg.start_index)
        if start < 0:
            raise ValueError(f"start_index must be >= 0, got {start}")
        if start >= min(len(test_data), len(test_target)):
            raise ValueError(
                f"start_index={start} is out of range for split length "
                f"{min(len(test_data), len(test_target))}"
            )

        n = min(int(self._cfg.max_examples), len(test_data) - start, len(test_target) - start)

        golds = [str(x).strip() for x in test_target[start : start + n]]
        if self._cfg.candidates is not None:
            candidates = [str(x) for x in self._cfg.candidates]
        else:
            # Infer from demo outputs + test targets. This is exact for multiple-choice
            # tasks like ARC (A/B/C/D), and a best-effort fallback for multi-class tasks.
            candidates = _infer_candidates([d.output_text for d in demo_pool] + golds)
        if not candidates:
            raise ValueError(
                "Could not infer a non-empty candidate set. "
                "Set task.manyiclbench.candidates explicitly for this task."
            )

        examples: list[Example] = []
        for i in range(n):
            q_raw = str(test_data[start + i])
            q_body = _strip_query_answer_stub(q_raw, answer_marker=answer_marker)
            q_body = _strip_query_marker(q_body, query_marker=query_marker)
            examples.append(
                Example(example_id=str(start + i), input_text=q_body, gold=golds[i], candidates=list(candidates))
            )

        self._examples = examples
        self._demo_pool = demo_pool

    def iter_batches(self) -> Iterable[TaskBatch]:
        if not self._examples:
            raise RuntimeError("Call load() first")
        pools = [self._demo_pool for _ in self._examples]
        yield TaskBatch(examples=self._examples, demo_pools=pools)
