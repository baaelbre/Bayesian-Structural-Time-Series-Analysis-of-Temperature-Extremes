"""Proper ensemble scores, including upper- and lower-tail weighting."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


Array = np.ndarray


def log_predictive_score(log_density_draws: Array) -> Array:
    """Negative log posterior-predictive density at every forecast time.

    ``log_density_draws`` contains the conditional log density of the held-out
    observation for every posterior draw and time.  The returned score is
    ``-log(mean(exp(log_density_draws)))``; lower values are better.  Working
    on the log scale avoids underflow for tail observations.
    """

    values = np.asarray(log_density_draws, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("log_density_draws must have shape (draws, time).")
    return -(logsumexp(values, axis=0) - np.log(values.shape[0]))


def _validate_ensemble(samples: Array, observed: Array) -> tuple[Array, Array]:
    samples = np.asarray(samples, dtype=float)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[1] != observed.size:
        raise ValueError("samples must have shape (draws, time) and match observed.")
    if samples.shape[0] < 2:
        raise ValueError("At least two ensemble members are required.")
    return samples, observed


def crps_ensemble(samples: Array, observed: Array) -> Array:
    """CRPS at every time using the empirical posterior predictive CDF."""

    samples, observed = _validate_ensemble(samples, observed)
    members = samples.shape[0]
    first = np.mean(np.abs(samples - observed[None, :]), axis=0)
    ordered = np.sort(samples, axis=0)
    coefficients = 2.0 * np.arange(1, members + 1) - members - 1.0
    pair_term = np.sum(coefficients[:, None] * ordered, axis=0) / members**2
    return first - pair_term


def threshold_weighted_crps(
    samples: Array,
    observed: Array,
    threshold: float,
    *,
    tail: str = "upper",
) -> Array:
    """Threshold-weighted CRPS via the corresponding censoring transform."""

    samples, observed = _validate_ensemble(samples, observed)
    tail = str(tail).lower()
    if tail == "upper":
        transformed_samples = np.maximum(samples, float(threshold))
        transformed_observed = np.maximum(observed, float(threshold))
    elif tail == "lower":
        transformed_samples = np.minimum(samples, float(threshold))
        transformed_observed = np.minimum(observed, float(threshold))
    else:
        raise ValueError("tail must be 'upper' or 'lower'.")
    return crps_ensemble(transformed_samples, transformed_observed)


def quantile_score(samples: Array, observed: Array, probability: float) -> Array:
    samples, observed = _validate_ensemble(samples, observed)
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1).")
    quantile = np.quantile(samples, probability, axis=0)
    error = observed - quantile
    return 2.0 * np.where(error >= 0.0, probability * error, (probability - 1.0) * error)


def exceedance_brier_score(
    samples: Array,
    observed: Array,
    threshold: float,
    *,
    tail: str = "upper",
) -> Array:
    samples, observed = _validate_ensemble(samples, observed)
    if tail == "upper":
        probability = np.mean(samples > threshold, axis=0)
        event = observed > threshold
    elif tail == "lower":
        probability = np.mean(samples < threshold, axis=0)
        event = observed < threshold
    else:
        raise ValueError("tail must be 'upper' or 'lower'.")
    return (probability - event.astype(float)) ** 2


def exceedance_log_score(
    samples: Array,
    observed: Array,
    threshold: float,
    *,
    tail: str = "upper",
    epsilon: float = 1e-12,
) -> Array:
    samples, observed = _validate_ensemble(samples, observed)
    if tail == "upper":
        probability = np.mean(samples > threshold, axis=0)
        event = observed > threshold
    elif tail == "lower":
        probability = np.mean(samples < threshold, axis=0)
        event = observed < threshold
    else:
        raise ValueError("tail must be 'upper' or 'lower'.")
    probability = np.clip(probability, epsilon, 1.0 - epsilon)
    return -(event * np.log(probability) + (~event) * np.log1p(-probability))


def evaluate_ensemble(
    samples: Array,
    observed: Array,
    *,
    thresholds: list[float] | tuple[float, ...] = (),
    quantiles: list[float] | tuple[float, ...] = (0.9, 0.95, 0.99),
    tail: str = "upper",
    log_density_draws: Array | None = None,
    aggregate: bool = True,
):
    score_values: list[tuple[str, float | None, Array]] = []
    score_values.append(("crps", None, crps_ensemble(samples, observed)))
    if log_density_draws is not None:
        log_values = np.asarray(log_density_draws, dtype=float)
        if log_values.ndim == 1:
            log_values = log_values[:, None]
        if log_values.shape[1] != np.asarray(observed).reshape(-1).size:
            raise ValueError("log_density_draws must match the observed time dimension.")
        score_values.append(("log", None, log_predictive_score(log_values)))
    for threshold in thresholds:
        for name, function in (
            ("twcrps", threshold_weighted_crps),
            ("exceedance_brier", exceedance_brier_score),
            ("exceedance_log", exceedance_log_score),
        ):
            values = function(samples, observed, float(threshold), tail=tail)
            score_values.append((name, float(threshold), values))
    for probability in quantiles:
        actual_probability = float(probability) if tail == "upper" else 1.0 - float(probability)
        values = quantile_score(samples, observed, actual_probability)
        score_values.append(("quantile", actual_probability, values))

    rows: list[dict[str, float | int | str | None]] = []
    if aggregate:
        for name, setting, values in score_values:
            rows.append(
                {
                    "score": name,
                    "setting": setting,
                    "mean": float(np.mean(values)),
                    "sum": float(np.sum(values)),
                }
            )
    else:
        for name, setting, values in score_values:
            rows.extend(
                {
                    "time_index": int(index),
                    "score": name,
                    "setting": setting,
                    "value": float(value),
                }
                for index, value in enumerate(np.asarray(values, dtype=float))
            )
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows
