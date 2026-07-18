from __future__ import annotations

import numpy as np


def empirical_coverage(samples: np.ndarray, truth: np.ndarray, alpha: float = 0.1) -> float:
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    truth = np.asarray(truth, dtype=float)
    return float(np.mean((truth >= lower) & (truth <= upper)))
