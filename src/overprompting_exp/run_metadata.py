from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from overprompting_exp.io_utils import ensure_dir, write_json


@dataclass(frozen=True)
class RunMetadata:
    config_text: str
    env: dict[str, Any]


def _safe_run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def _git_commit() -> str | None:
    return _safe_run(["git", "rev-parse", "HEAD"])


def _pip_freeze() -> str | None:
    return _safe_run([sys.executable, "-m", "pip", "freeze"])


def _torch_info() -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None

    info: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
    }
    if info["cuda_available"]:
        try:
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name_0"] = str(torch.cuda.get_device_name(0))
        except Exception:
            pass
    return info


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def collect_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git_commit": _git_commit(),
        "pip_freeze": _pip_freeze(),
        "packages": {
            "numpy": _pkg_version("numpy"),
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "datasets": _pkg_version("datasets"),
        },
        "torch": _torch_info(),
    }
    # Make sure JSON-serializable even if something odd slipped in.
    json.dumps(env, ensure_ascii=False)
    return env


def dump_run_metadata(out_dir: Path, *, config_path: Path) -> RunMetadata:
    ensure_dir(out_dir)
    config_text = config_path.read_text(encoding="utf-8")
    (out_dir / "config.yaml").write_text(config_text, encoding="utf-8")

    env = collect_env()
    write_json(out_dir / "env.json", env)
    return RunMetadata(config_text=config_text, env=env)

