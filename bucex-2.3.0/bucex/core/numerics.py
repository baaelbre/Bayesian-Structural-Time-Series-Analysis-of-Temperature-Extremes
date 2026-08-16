"""Small numerical primitives shared by all inference engines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def symmetrize(matrix: Array) -> Array:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def psd_eigh(matrix: Array, *, tolerance: float = 1e-7) -> tuple[Array, Array]:
    """Eigendecompose a covariance, clipping only numerical PSD remnants.

    Singular structural smoothers can leave negative eigenvalues around
    ``1e-8`` after a Moore--Penrose Schur complement.  The tolerance is an
    absolute floor for sub-unit covariance matrices and relative above that;
    materially indefinite inputs still raise instead of being repaired.
    """
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    scale = max(float(np.max(np.abs(values))) if values.size else 0.0, 1.0)
    if np.any(values < -tolerance * scale):
        raise np.linalg.LinAlgError("Matrix is not positive semidefinite.")
    return np.clip(values, 0.0, None), vectors


def sample_mvn(mean: Array, covariance: Array, rng: np.random.Generator) -> Array:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    values, vectors = psd_eigh(covariance)
    if mean.size == 0:
        return mean.copy()
    return mean + vectors @ (np.sqrt(values) * rng.normal(size=mean.size))


def covariance_root(covariance: Array) -> Array:
    """Return ``L`` with ``L @ L.T == covariance`` for a PSD matrix."""

    values, vectors = psd_eigh(covariance)
    return vectors * np.sqrt(values)[None, :]


@dataclass(frozen=True)
class GaussianSupport:
    active_vectors: Array
    inactive_vectors: Array
    active_variances: Array
    log_constant: float
    tolerance: float


def gaussian_support(
    covariance: Array,
    *,
    tolerance: float = 1e-7,
    rank_tolerance: float | None = None,
) -> GaussianSupport:
    """Factor a possibly singular Gaussian covariance.

    ``tolerance`` controls the affine-support check only.  It must not also be
    used as a relative rank cutoff: structural time-series models routinely
    combine innovation variances that differ by many orders of magnitude (for
    example a monthly slope and a seasonal disturbance).  Treating a small but
    positive eigenvalue as zero changes the statistical model and makes valid
    paths appear off-support.

    The numerical rank cutoff is therefore close to machine precision unless
    callers explicitly request a larger value.  Exact structural zeros remain
    inactive, while genuinely stochastic small-variance coordinates remain in
    the Gaussian density.
    """

    covariance = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = psd_eigh(covariance, tolerance=tolerance)
    scale = max(
        float(np.max(eigenvalues)) if eigenvalues.size else 0.0,
        np.finfo(float).tiny,
    )
    if rank_tolerance is None:
        rank_tolerance = np.finfo(float).eps * max(1, covariance.shape[0])
    if float(rank_tolerance) < 0.0:
        raise ValueError("rank_tolerance must be non-negative.")
    active = eigenvalues > float(rank_tolerance) * scale
    variances = eigenvalues[active]
    log_constant = (
        0.0
        if variances.size == 0
        else float(-0.5 * (variances.size * np.log(2.0 * np.pi) + np.sum(np.log(variances))))
    )
    return GaussianSupport(
        active_vectors=eigenvectors[:, active],
        inactive_vectors=eigenvectors[:, ~active],
        active_variances=variances,
        log_constant=log_constant,
        tolerance=float(tolerance),
    )


def factored_singular_normal_logpdf(value: Array, mean: Array, factor: GaussianSupport):
    """Vectorized affine-support Gaussian density using a cached factorization."""

    value_arr, mean_arr = np.broadcast_arrays(np.asarray(value, dtype=float), np.asarray(mean, dtype=float))
    delta = value_arr - mean_arr
    scalar = delta.ndim == 1
    delta2 = delta[None, :] if scalar else delta.reshape(-1, delta.shape[-1])
    if factor.inactive_vectors.shape[1]:
        inactive = delta2 @ factor.inactive_vectors
        off_support = np.linalg.norm(inactive, axis=1) > factor.tolerance * (
            1.0 + np.linalg.norm(delta2, axis=1)
        )
    else:
        off_support = np.zeros(delta2.shape[0], dtype=bool)
    if factor.active_vectors.shape[1]:
        coordinates = delta2 @ factor.active_vectors
        quadratic = np.sum(coordinates**2 / factor.active_variances[None, :], axis=1)
        output = factor.log_constant - 0.5 * quadratic
    else:
        output = np.zeros(delta2.shape[0])
    output[off_support] = -np.inf
    return float(output[0]) if scalar else output.reshape(delta.shape[:-1])


def singular_normal_logpdf(
    value: Array,
    mean: Array,
    covariance: Array,
    *,
    tolerance: float = 1e-7,
) -> float:
    """Gaussian log density on the affine support of a PSD covariance."""

    value = np.asarray(value, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    delta = value - mean
    factor = gaussian_support(covariance, tolerance=tolerance)
    return float(factored_singular_normal_logpdf(value, mean, factor))


def logsumexp(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return -np.inf
    maximum = float(np.max(values[finite]))
    return float(maximum + np.log(np.sum(np.exp(values[finite] - maximum))))


def normalize_logweights(log_weights: Array) -> tuple[Array, float]:
    log_weights = np.asarray(log_weights, dtype=float)
    normalizer = logsumexp(log_weights)
    if not np.isfinite(normalizer):
        raise FloatingPointError("All particle weights are zero.")
    weights = np.exp(log_weights - normalizer)
    weights /= weights.sum()
    return weights, normalizer


def effective_sample_size(weights: Array) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(weights**2))


def systematic_resample(weights: Array, rng: np.random.Generator, size: int | None = None) -> Array:
    weights = np.asarray(weights, dtype=float)
    n = weights.size if size is None else int(size)
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions, side="right").astype(int)


def bounded_to_real(value: float, lower: float, upper: float) -> float:
    probability = (float(value) - float(lower)) / (float(upper) - float(lower))
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(np.log(probability) - np.log1p(-probability))


def real_to_bounded(value: float, lower: float, upper: float) -> float:
    if value >= 0.0:
        probability = 1.0 / (1.0 + np.exp(-value))
    else:
        exp_value = np.exp(value)
        probability = exp_value / (1.0 + exp_value)
    return float(lower + (upper - lower) * probability)


def bounded_log_jacobian(value: float, lower: float, upper: float) -> float:
    if value >= 0.0:
        log_p = -np.log1p(np.exp(-value))
        log_one_minus_p = -value - np.log1p(np.exp(-value))
    else:
        log_p = value - np.log1p(np.exp(value))
        log_one_minus_p = -np.log1p(np.exp(value))
    return float(np.log(upper - lower) + log_p + log_one_minus_p)
