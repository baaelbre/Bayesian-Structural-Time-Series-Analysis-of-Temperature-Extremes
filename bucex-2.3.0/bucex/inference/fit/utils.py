"""Small distribution helpers shared by the FS kernels."""
from __future__ import annotations

import numpy as np


Array = np.ndarray


def as_1d_observations(y: Array, *, family: str) -> Array:
    """Validate the univariate observation layout used by FS kernels."""

    values = np.asarray(y, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    raise ValueError(f"The FS {family} kernel supports univariate observations only.")


def sample_inverse_gamma(
    shape: float,
    scale: float,
    rng: np.random.Generator,
) -> float:
    if float(shape) <= 0.0 or float(scale) <= 0.0:
        raise ValueError("shape and scale must be positive.")
    precision = rng.gamma(shape=shape, scale=1.0 / scale)
    return float(1.0 / precision)


__all__ = ["as_1d_observations", "sample_inverse_gamma"]
