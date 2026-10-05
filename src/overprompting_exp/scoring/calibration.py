from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationSummary:
    ece: float
    brier_mean: float


def expected_calibration_error(conf: np.ndarray, acc: np.ndarray, num_bins: int = 15) -> float:
    """ECE over (confidence, accuracy) pairs.

    - conf: predicted top-1 probability
    - acc: 0/1 correctness
    """
    conf = np.asarray(conf, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < num_bins - 1 else conf <= hi)
        if not np.any(mask):
            continue
        w = float(np.mean(mask))
        gap = float(abs(np.mean(acc[mask]) - np.mean(conf[mask])))
        ece += w * gap
    return float(ece)

