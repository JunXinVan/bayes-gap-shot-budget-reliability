from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def clip_and_renorm(p: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"Expected a 1D probability vector, got shape={p.shape}")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be a positive finite float, got {alpha}")
    p = np.clip(p, alpha, 1.0)
    s = float(np.sum(p))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("Probability vector has non-finite or non-positive sum after clipping.")
    return p / s


def entropy(p: np.ndarray, alpha: float = 1e-6) -> float:
    p2 = clip_and_renorm(p, alpha=alpha)
    return float(-np.sum(p2 * np.log(p2)))


def kl_div(p: np.ndarray, q: np.ndarray, alpha: float = 1e-6) -> float:
    p2 = clip_and_renorm(p, alpha=alpha)
    q2 = clip_and_renorm(q, alpha=alpha)
    return float(np.sum(p2 * (np.log(p2) - np.log(q2))))


@dataclass(frozen=True)
class ExampleMetrics:
    nll: float
    acc: float
    ent: float
    margin: float


def compute_example_metrics(p: np.ndarray, gold_index: int, *, alpha: float = 1e-6) -> ExampleMetrics:
    nll = float(-np.log(max(p[gold_index], 1e-12)))
    pred = int(np.argmax(p))
    acc = float(pred == gold_index)

    sorted_p = np.sort(p)[::-1]
    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) >= 2 else float(sorted_p[0])
    return ExampleMetrics(nll=nll, acc=acc, ent=entropy(p, alpha=alpha), margin=margin)


def top1_confidence(p: np.ndarray) -> float:
    return float(np.max(p))


def brier_score(p: np.ndarray, gold_index: int) -> float:
    y = np.zeros_like(p, dtype=np.float64)
    y[gold_index] = 1.0
    return float(np.sum((p - y) ** 2))
