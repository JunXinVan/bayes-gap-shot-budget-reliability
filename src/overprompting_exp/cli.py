from __future__ import annotations

import argparse
from pathlib import Path

from overprompting_exp.experiments.e1_curves import run_e1
from overprompting_exp.experiments.e2_stoprule import run_e2
from overprompting_exp.experiments.e5_certificate import run_e5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="overprompting-exp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_e1 = sub.add_parser("e1", help="E1: curves (risk/acc/entropy/sensitivity vs k)")
    p_e1.add_argument("--config", type=Path, required=True)

    p_e2 = sub.add_parser("e2", help="E2: decision/selection rule evaluation")
    p_e2.add_argument("--config", type=Path, required=True)

    p_e5 = sub.add_parser("e5", help="E5: offline certificate (||W_out||_op)")
    p_e5.add_argument("--config", type=Path, required=True)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "e1":
        run_e1(args.config)
        return
    if args.cmd == "e2":
        run_e2(args.config)
        return
    if args.cmd == "e5":
        run_e5(args.config)
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
