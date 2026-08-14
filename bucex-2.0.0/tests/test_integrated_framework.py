from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bucex as bx


QUICK = bx.MCMC(draws=2, warmup=1, chains=1, seed=20)


def _sample(family: str, seed: int = 1):
    rng = np.random.default_rng(seed)
    baseline = 8.0 + np.sin(2.0 * np.pi * np.arange(24) / 12.0)
    if family == "gaussian":
        return baseline + rng.normal(scale=0.3, size=24)
    return baseline + rng.gumbel(scale=0.5, size=24)


def test_plan_separates_parameterization_engine_and_asis():
    model = bx.structural_model("gev", period=12)
    plan = bx.plan(
        model,
        _sample("gev"),
        engine="pgas",
        parameterization="fs",
        asis=True,
    )
    assert plan.parameterization == "fruehwirth_schnatter"
    assert plan.engine == "pgas"
    assert plan.interweaves_with == "centered"
    assert plan.targets_exact_posterior
    with pytest.raises(ValueError, match="incompatible"):
        bx.plan(model, _sample("gev"), engine="ffbs")


@pytest.mark.parametrize(
    "parameterization,asis",
    [("centered", False), ("disturbance", True), ("fruehwirth_schnatter", True)],
)
def test_gaussian_parameterizations_share_one_result_contract(parameterization, asis):
    fit = bx.fit(
        _sample("gaussian"),
        family="gaussian",
        period=12,
        priors="normal",
        parameterization=parameterization,
        asis=asis,
        mcmc=QUICK,
    )
    assert type(fit) is bx.FitResult
    assert fit.state_names[:2] == ("level", "slope")
    assert fit.state_draws.shape == (1, 2, 25, 13)
    assert {"sd.level", "sd.slope", "sd.seasonal", "sigma"} <= set(
        fit.parameter_draws
    )
    assert fit.plan.parameterization == parameterization


@pytest.mark.parametrize("engine,exact", [("laplace", False), ("pgas", True)])
def test_gev_engines_share_one_result_contract(engine, exact):
    fit = bx.fit(
        _sample("gev"),
        family="gev",
        period=12,
        priors="normal",
        parameterization="fruehwirth_schnatter",
        engine=engine,
        mcmc=QUICK,
        particles=bx.Particles(n=16),
        laplace=bx.Laplace(max_iterations=6),
    )
    assert type(fit) is bx.FitResult
    assert fit.plan.targets_exact_posterior is exact
    assert {"sigma", "xi", "sd.level"} <= set(fit.parameter_draws)


@pytest.mark.parametrize(
    "profile,required",
    [
        ("manuscript_lasso", {"tau_level", "tau_trend", "tau_season", "lambda2"}),
        (
            "regularized_lasso",
            {"lambda2_level", "lambda2_trend", "lambda2_season"},
        ),
        (
            "regularized_horseshoe",
            {
                "horseshoe_global",
                "horseshoe_local_level",
                "horseshoe_local_trend",
                "horseshoe_local_season",
                "horseshoe_slab2",
            },
        ),
        ("pc", {"pc_tau_level", "pc_tau_trend", "pc_tau_season"}),
        ("normal", set()),
        ("ssvs", {"state_level", "state_trend", "state_season"}),
    ],
)
def test_fs_prior_profiles_resolve_through_fit(profile, required):
    fit = bx.fit(
        _sample("gaussian", 3),
        family="gaussian",
        period=12,
        priors=profile,
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=1, warmup=1, chains=1, seed=4),
    )
    assert fit.meta["prior_profile"] == profile
    assert required <= set(fit.parameter_draws)
    assert np.all(np.isfinite(fit.state_draws))


def test_incompatible_prior_and_model_combinations_fail_before_sampling():
    regression_model = bx.Model(
        bx.Gaussian(), [bx.LocalLevel(), bx.Regression(1, dynamic=True)]
    )
    with pytest.raises(ValueError, match="FS augmented"):
        bx.fit(
            np.arange(10.0),
            regression_model,
            exog=np.ones((10, 1)),
            parameterization="fruehwirth_schnatter",
            mcmc=QUICK,
        )
    with pytest.raises(ValueError, match="requires"):
        bx.fit(
            np.arange(12.0),
            family="gaussian",
            parameterization="centered",
            priors="regularized_horseshoe",
            mcmc=QUICK,
        )


def test_fs_ssvs_uses_semantic_component_summaries():
    fit = bx.fit(
        _sample("gaussian", 5),
        family="gaussian",
        period=12,
        priors="ssvs",
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=4, warmup=2, chains=1, seed=6),
    )
    table = fit.component_probabilities()
    assert list(table.columns) == ["zero", "fixed", "dynamic"]
    np.testing.assert_allclose(table.sum(axis=1), 1.0)
    assert fit.most_probable_structure()["level"] in {"fixed", "dynamic"}
    state_trend = np.asarray(fit.parameter("state_trend"), dtype=int)
    state_season = np.asarray(fit.parameter("state_season"), dtype=int)
    assert np.all(fit.parameter("signed_sd.slope")[state_trend != 2] == 0.0)
    assert np.all(fit.parameter("signed_sd.seasonal")[state_season != 2] == 0.0)


def test_structural_shrinkage_profiles_are_mathematically_distinct():
    manuscript = bx.manuscript_gaussian_priors()
    regularized = bx.regularized_gaussian_priors()
    horseshoe = bx.regularized_horseshoe_gaussian_priors()
    pc = bx.pc_gaussian_priors()
    assert isinstance(manuscript.lasso, bx.BayesianLassoPrior)
    assert isinstance(regularized.lasso, bx.ComponentwiseBayesianLassoPrior)
    assert horseshoe.horseshoe is not None and horseshoe.lasso is None
    assert pc.pc is not None
    assert np.exp(-pc.pc.standardized_rate_for("level")) == pytest.approx(
        pc.pc.alpha_for("level")
    )
    variance = horseshoe.horseshoe.conditional_variance(
        "level", local=100.0, global_scale=1.0, slab2=4.0
    )
    assert 0.0 < variance <= 4.0 * horseshoe.horseshoe.coefficient_scale_for("level") ** 2


def test_pandas_metadata_initialization_and_dynamic_regression():
    rng = np.random.default_rng(7)
    exog = rng.normal(size=(24, 1))
    values = pd.Series(
        1.5 * exog[:, 0] + rng.normal(scale=0.2, size=24),
        index=pd.date_range("2000-01-01", periods=24, freq="MS"),
        name="bulk_temperature",
    )
    model = bx.Model(
        bx.Gaussian(), [bx.LocalLevel(), bx.Regression(1, dynamic=True, name="x")]
    )
    fit = bx.fit(
        values,
        model,
        exog=exog,
        parameterization="disturbance",
        priors="normal",
        init={"sd.level": 0.02, "sd.x[1]": 0.01, "sigma": 0.3},
        mcmc=QUICK,
    )
    assert fit.series_name == "bulk_temperature"
    np.testing.assert_array_equal(fit.dates, values.index.to_numpy())
    assert "sd.x[1]" in fit.parameter_draws
    assert fit.initial_values["parameters_by_chain"][0]["sd.level"] == 0.02


def test_compatibility_wrapper_and_bulk_tail_delegate_to_fit():
    with pytest.deprecated_call():
        legacy = bx.fit_bayes(
            _sample("gaussian", 8),
            family="gaussian",
            period=12,
            priors="normal",
            n_iter=3,
            burn=1,
            seed=9,
        )
    assert type(legacy) is bx.FitResult
    pair = bx.fit_bulk_tail(
        _sample("gaussian", 10),
        _sample("gev", 11),
        period=12,
        parameterization="centered",
        bulk_priors="normal",
        tail_priors="normal",
        bulk_mcmc=QUICK,
        tail_mcmc=QUICK,
        tail_engine="laplace",
        laplace=bx.Laplace(max_iterations=6),
    )
    assert pair.bulk.family == "gaussian"
    assert pair.tail.family == "gev"
    assert not pair.metadata["joint_likelihood"]


def test_general_default_prior_and_legacy_initial_values_are_resolved():
    values = _sample("gaussian", 12)
    fit = bx.fit(
        values,
        bx.Model(bx.Gaussian(), [bx.LocalLevel()]),
        parameterization="centered",
        asis=True,
        init_params_state={"q_level": 0.04**2},
        init_params_obs={"sigma2": 0.3**2},
        mcmc=QUICK,
    )
    assert fit.priors.profile == "regularized"
    assert fit.initial_values["parameters_by_chain"][0]["sd.level"] == pytest.approx(0.04)
    assert fit.initial_values["parameters_by_chain"][0]["sigma"] == pytest.approx(0.3)
    assert {"sd.level", "asis.sd.level"} <= set(
        fit.sampler_diagnostics["acceptance"]
    )


def test_fit_bayes_is_only_a_monthly_translation_wrapper():
    with pytest.deprecated_call():
        fit = bx.fit_bayes(
            _sample("gaussian", 13),
            family="gaussian",
            priors="normal",
            n_iter=3,
            burn=1,
            seed=14,
        )
    assert fit.model.period == 12
    assert fit.plan.parameterization == "fruehwirth_schnatter"
    assert type(fit) is bx.FitResult
