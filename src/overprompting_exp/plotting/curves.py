from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_e1_curves(aggregate: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError("Plotting requires extras: pip install -e '.[plot]'") from e

    ks = sorted(int(k) for k in aggregate.keys())
    if not ks:
        return

    def series(key: str) -> list[float]:
        return [float(aggregate[str(k)][key]) for k in ks]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(ks, series("nll_mean"), marker="o")
    ax.set_title("Risk / NLL")
    ax.set_xlabel("k")

    ax = axes[0, 1]
    ax.plot(ks, series("acc_mean"), marker="o")
    ax.set_title("Accuracy")
    ax.set_xlabel("k")

    ax = axes[1, 0]
    ax.plot(ks, series("entropy_mean"), marker="o")
    ax.set_title("Entropy (PV proxy)")
    ax.set_xlabel("k")

    ax = axes[1, 1]
    ax.plot(ks, series("sensitivity_mean"), marker="o")
    ax.set_title("Sensitivity (KL proxy)")
    ax.set_xlabel("k")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_e2_stoprule(rows: list[dict[str, Any]], out_path: Path, *, policy: str | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError("Plotting requires extras: pip install -e '.[plot]'") from e

    if not rows:
        return

    if policy is not None:
        rows = [r for r in rows if str(r.get("policy", "")) == policy]
        if not rows:
            return

    chosen = [int(r["chosen_k"]) for r in rows]
    oracle = [int(r["oracle_k"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(oracle, chosen, s=10, alpha=0.6)
    ax.set_xlabel("oracle k*")
    ax.set_ylabel("chosen k-hat")
    title = "Decision rule: oracle vs chosen"
    if policy is not None:
        title += f" ({policy})"
    ax.set_title(title)
    lim = max(max(oracle), max(chosen))
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
