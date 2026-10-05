from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from overprompting_exp.certificates.opnorm import estimate_output_opnorm_hf
from overprompting_exp.experiments.common import build_model, prepare_run
from overprompting_exp.io_utils import ensure_dir, write_json
from overprompting_exp.run_metadata import dump_run_metadata


def _sanitize_filename(s: str) -> str:
    return (
        s.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("@", "_")
        .replace(".", "_")
    )


@dataclass(frozen=True)
class OpNormRunConfig:
    num_iter: int
    seed: int
    out_dir: Path
    device: str | None
    dtype: str | None


def parse_opnorm_run(cfg: dict[str, Any], *, default_seed: int, default_out_dir: Path) -> OpNormRunConfig:
    c = (cfg.get("certificates", {}) or {}).get("opnorm", {}) or {}
    enabled = bool(c.get("enabled", True))
    if not enabled:
        raise RuntimeError("certificates.opnorm.enabled is false")
    return OpNormRunConfig(
        num_iter=int(c.get("num_iter", 50)),
        seed=int(c.get("seed", default_seed)),
        out_dir=Path(c.get("out_dir", default_out_dir)),
        device=str(c["device"]) if c.get("device") is not None else None,
        dtype=str(c["dtype"]) if c.get("dtype") is not None else None,
    )


def run_e5(config_path: Path) -> None:
    run, cfg = prepare_run(config_path)
    op_cfg = parse_opnorm_run(cfg, default_seed=run.seed, default_out_dir=Path("results") / "certificates")
    dump_run_metadata(op_cfg.out_dir, config_path=config_path)

    model = build_model(cfg, seed=run.seed)

    model_name = str(((cfg.get("model", {}) or {}).get("hf", {}) or {}).get("model_name", "model"))
    cert = estimate_output_opnorm_hf(
        model_adapter=model,
        model_name=model_name,
        num_iter=op_cfg.num_iter,
        seed=op_cfg.seed,
        device=op_cfg.device,
        dtype=op_cfg.dtype,
    )

    ensure_dir(op_cfg.out_dir)
    out_path = op_cfg.out_dir / f"{_sanitize_filename(model_name)}.json"
    write_json(out_path, cert.to_json())
