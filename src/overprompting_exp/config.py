from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data)} at {path}")
    return data


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    output_dir: Path


def parse_run(cfg: dict[str, Any]) -> RunConfig:
    run = cfg.get("run", {}) or {}
    return RunConfig(
        name=str(run.get("name", "run")),
        seed=int(run.get("seed", 0)),
        output_dir=Path(run.get("output_dir", "results")),
    )

