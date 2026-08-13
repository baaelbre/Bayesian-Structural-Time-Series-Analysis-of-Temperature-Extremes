from __future__ import annotations

import numpy as np

from bucex import (
    ComponentState,
    SSVSPrior,
    fit_bayes,
    fit_uccle_series,
    normal_gaussian_priors,
    ssvs_gaussian_priors,
)
from bucex.inference.fit.model_space import enumerate_structural_models
from bucex.inference.fit.noncentered_utils import infer_ncp_layout
from bucex.api.fit import _default_model


def test_v031_prior_profiles():
    normal = normal_gaussian_priors()
    ssvs = ssvs_gaussian_priors()
    assert normal.s_level is not None
    assert normal.lasso is None
    assert normal.ssvs is None
    assert isinstance(ssvs.ssvs, SSVSPrior)
    assert ssvs.s_level is None
    assert ssvs.lasso is None


def test_structural_model_space_has_18_models_and_no_zero_level():
    layout = infer_ncp_layout(_default_model("gaussian", 12))
    models = enumerate_structural_models(layout)
    assert len(models) == 18
    assert all(model.level != ComponentState.ZERO for model in models)


def test_gaussian_ssvs_draws_exact_structural_zeros():
    rng = np.random.default_rng(21)
    t = np.arange(72)
    y = 8.0 + 0.015 * t + 1.2 * np.sin(2.0 * np.pi * t / 12.0)
    y = y + rng.normal(0.0, 0.35, t.size)

    fit = fit_bayes(
        y,
        family="gaussian",
        period=12,
        priors="ssvs",
        n_iter=18,
        burn=6,
        progress=False,
        seed=22,
    )

    assert fit.meta["prior_profile"] == "ssvs"
    assert fit.meta["structural_ssvs"] is True
    assert fit.meta["model_selection_exact"] is True
    assert set(("state_level", "state_trend", "state_season")) <= set(fit.draws_static)
    assert np.all(fit.draws_static["state_level"] != int(ComponentState.ZERO))

    trend_fixed_or_zero = fit.draws_static["state_trend"] != int(ComponentState.DYNAMIC)
    season_fixed_or_zero = fit.draws_static["state_season"] != int(ComponentState.DYNAMIC)
    level_fixed = fit.draws_static["state_level"] == int(ComponentState.FIXED)
    assert np.all(fit.draws_static["s_trend"][trend_fixed_or_zero] == 0.0)
    assert np.all(fit.draws_static["s_season"][season_fixed_or_zero] == 0.0)
    assert np.all(fit.draws_static["s_level"][level_fixed] == 0.0)

    trend_zero = fit.draws_static["state_trend"] == int(ComponentState.ZERO)
    season_zero = fit.draws_static["state_season"] == int(ComponentState.ZERO)
    assert np.all(fit.draws_static["beta0"][trend_zero] == 0.0)
    assert np.all(fit.draws_static["gamma0_season"][season_zero] == 0.0)

    probabilities = fit.component_probabilities()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert fit.most_probable_structure()["level"] in {"fixed", "dynamic"}


def test_gev_ssvs_is_marked_as_laplace_model_selection():
    fit = fit_uccle_series(
        "TXx",
        data_dir="data",
        start="2000-01-01",
        end="2003-12-31",
        priors="ssvs",
        n_iter=8,
        burn=3,
        progress=False,
        seed=23,
    )
    assert fit.meta["structural_ssvs"] is True
    assert fit.meta["model_selection_exact"] is False
    assert fit.meta["model_selection_basis"] == "laplace_pseudo_observations"
