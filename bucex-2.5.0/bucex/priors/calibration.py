"""Scientific calibration helpers for structural innovation priors.

The functions in this module work on the scale of observable temperature
changes.  They are intentionally independent of a sampler so they can be used
before a model is fitted and recorded in an analysis protocol.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm, t


def _horizon_periods(horizon_years: float, periods_per_year: int) -> int:
    years = float(horizon_years)
    frequency = int(periods_per_year)
    if not np.isfinite(years) or years <= 0.0:
        raise ValueError("horizon_years must be finite and positive.")
    if frequency < 1:
        raise ValueError("periods_per_year must be positive.")
    return max(1, int(round(years * frequency)))


def _central_z(probability: float) -> float:
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1).")
    return float(norm.ppf(0.5 * (1.0 + probability)))


def half_student_t_scale_for_median(
    median: float = 1.0,
    *,
    df: float = 4.0,
) -> float:
    """Scale a half-Student-t distribution to a requested median.

    This is useful for a hierarchical slab multiplier.  For example,
    ``half_student_t_scale_for_median(1, df=4)`` makes one the prior median
    multiplier, so the component ``coefficient_scale`` retains a direct
    scientific interpretation.
    """

    median = float(median)
    df = float(df)
    if not np.isfinite(median) or median <= 0.0:
        raise ValueError("median must be finite and positive.")
    if not np.isfinite(df) or df <= 0.0:
        raise ValueError("df must be finite and positive.")
    return median / float(t.ppf(0.75, df=df))


def calibrate_structural_scales(
    *,
    horizon_years: float,
    max_level_change: float,
    max_rate_change_per_decade: float,
    max_seasonal_innovation: float,
    periods_per_year: int = 12,
    probability: float = 0.90,
) -> dict[str, float]:
    """Translate plausible changes into FS normal-slab coefficient scales.

    Parameters are upper bounds that the analyst considers plausible with the
    supplied central probability. ``max_level_change`` refers to the
    accumulated random-walk-level contribution over ``horizon_years``.
    ``max_rate_change_per_decade`` refers to the change in the latent warming
    rate caused by slope innovations over the same horizon.
    ``max_seasonal_innovation`` is a one-step innovation in the dummy-seasonal
    state; seasonal propagation is model-specific and should subsequently be
    checked by simulation.

    The returned values calibrate the signed FS coefficients conditional on a
    hierarchical slab multiplier of one.  When the multiplier is learned, use
    :func:`half_student_t_scale_for_median` to keep that value at the prior
    median and inspect a full prior-predictive simulation as a final check.
    """

    periods = _horizon_periods(horizon_years, periods_per_year)
    z = _central_z(probability)
    bounds = np.asarray(
        (max_level_change, max_rate_change_per_decade, max_seasonal_innovation),
        dtype=float,
    )
    if np.any(~np.isfinite(bounds)) or np.any(bounds <= 0.0):
        raise ValueError("Every plausible-change bound must be finite and positive.")
    return {
        "level": float(max_level_change) / (z * np.sqrt(periods)),
        "trend": float(max_rate_change_per_decade)
        / (z * np.sqrt(periods) * int(periods_per_year) * 10.0),
        "season": float(max_seasonal_innovation) / z,
    }


def structural_scale_implications(
    *,
    level_sd: float,
    slope_sd: float,
    seasonal_sd: float,
    horizon_years: float,
    periods_per_year: int = 12,
    probability: float = 0.90,
) -> dict[str, Any]:
    """Report interpretable implications of proposed innovation scales.

    For a local-linear trend, a slope innovation persists.  Therefore this
    reports both the change in the latent rate and the cumulative level change
    induced by slope innovations.  The latter has variance
    ``slope_sd**2 * sum(k**2, k=1,...,h-1)``.
    """

    scales = np.asarray((level_sd, slope_sd, seasonal_sd), dtype=float)
    if np.any(~np.isfinite(scales)) or np.any(scales < 0.0):
        raise ValueError("Innovation SDs must be finite and non-negative.")
    periods = _horizon_periods(horizon_years, periods_per_year)
    z = _central_z(probability)
    level_path_sd = float(level_sd) * np.sqrt(periods)
    rate_decade_sd = (
        float(slope_sd) * np.sqrt(periods) * int(periods_per_year) * 10.0
    )
    slope_level_sd = float(slope_sd) * np.sqrt(
        periods * (periods - 1) * (2 * periods - 1) / 6.0
    )
    return {
        "horizon_years": float(horizon_years),
        "horizon_periods": periods,
        "periods_per_year": int(periods_per_year),
        "central_probability": float(probability),
        "level_random_walk_sd": level_path_sd,
        "level_random_walk_bound": z * level_path_sd,
        "rate_change_per_decade_sd": rate_decade_sd,
        "rate_change_per_decade_bound": z * rate_decade_sd,
        "slope_induced_level_change_sd": slope_level_sd,
        "slope_induced_level_change_bound": z * slope_level_sd,
        "one_step_seasonal_innovation_bound": z * float(seasonal_sd),
    }


__all__ = [
    "calibrate_structural_scales",
    "half_student_t_scale_for_median",
    "structural_scale_implications",
]
