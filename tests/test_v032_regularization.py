from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from bucex import (
    ComponentwiseBayesianLassoPrior,
    PosteriorBundle,
    fit_bayes,
    regularized_gaussian_priors,
    ssvs_gaussian_priors,
)
from bucex.inference.fit.noncentered_utils import _sample_gaussian


def test_regularized_profile_is_componentwise_and_scale_aware():
    priors = regularized_gaussian_priors()
    assert isinstance(priors.lasso, ComponentwiseBayesianLassoPrior)
    assert priors.beta0.sd == 0.005
    assert priors.lasso.coefficient_scale_for("level") == 0.03
    assert priors.lasso.coefficient_scale_for("trend") == 0.0002
    assert priors.lasso.coefficient_scale_for("season") == 0.03


def test_ssvs_profile_uses_monthly_slope_scale():
    priors = ssvs_gaussian_priors()
    assert priors.beta0.sd == 0.005
    assert priors.ssvs is not None
    assert priors.ssvs.innovation_slab_sd["trend"] == 0.0002


def test_gaussian_sampler_repairs_indefinite_covariance_without_warning():
    rng = np.random.default_rng(123)
    covariance = np.array([[1.0, 1.0000001], [1.0000001, 1.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        draw = _sample_gaussian(np.zeros(2), covariance, rng)
    assert np.all(np.isfinite(draw))
    assert not caught


def test_regularized_fit_saves_componentwise_lambdas():
    rng = np.random.default_rng(4)
    t = np.arange(48)
    y = 10.0 + 0.002 * t + np.sin(2.0 * np.pi * t / 12.0)
    y = y + rng.normal(0.0, 0.25, t.size)

    fit = fit_bayes(
        y,
        family="gaussian",
        period=12,
        priors="regularized",
        n_iter=18,
        burn=8,
        progress=False,
        seed=5,
    )

    assert fit.meta["prior_profile"] == "regularized"
    assert fit.meta["componentwise_lasso"] is True
    for block in ("level", "trend", "season"):
        assert f"lambda2_{block}" in fit.draws_static
        assert np.all(fit.draws_static[f"lambda2_{block}"] > 0.0)


def test_finite_level_rates_do_not_require_a_slope_state():
    dates = pd.date_range("2000-01-01", "2010-12-01", freq="MS")
    n = len(dates)
    draws = np.zeros((3, n + 1, 2))
    years_since_start = np.arange(n) / 12.0
    for m, offset in enumerate((0.0, 0.1, -0.1)):
        draws[m, 1:, 0] = 5.0 + offset + 0.2 * years_since_start
        draws[m, 1:, 1] = 0.0

    fit = PosteriorBundle(
        draws_states=draws,
        dates=dates.to_numpy(),
        state_names=("alpha", "beta"),
        transform_sign=1.0,
    )

    rate = fit.level_rate_draws(2000, 2010, scale="decade")
    assert np.allclose(rate, 2.0, atol=0.03)
    contrast = fit.rate_contrast_summary((2005, 2010), (2000, 2005))
    assert abs(contrast["median"]) < 0.05
