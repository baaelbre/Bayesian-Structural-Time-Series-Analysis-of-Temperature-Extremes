from __future__ import annotations

import numpy as np
import pytest

import bucex as bx
from bucex.inference.fit.fs_utils import (
    canonicalize_ncp_params,
    infer_ncp_layout,
    mu_from_ncp,
    random_sign_switches,
)


def _multiseries(*, mixed: bool = False) -> bx.MultiSeriesModel:
    return bx.MultiSeriesModel(
        channels=(
            bx.Channel(
                "first",
                bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
            ),
            bx.Channel(
                "second",
                bx.GEV() if mixed else bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
            ),
        ),
        name="related summaries",
    )


def _dates(n_time: int) -> np.ndarray:
    return np.arange(
        np.datetime64("2000-01"),
        np.datetime64("2000-01") + np.timedelta64(n_time, "M"),
        dtype="datetime64[M]",
    )


def test_public_surface_and_hierarchical_inference_plan():
    model = _multiseries()
    values = np.zeros((12, 2))
    compiled = bx.compile_model(model, values)
    assert compiled.channel_names == ("first", "second")
    plan = bx.plan(model, values)
    assert plan.backend == "hierarchical_state_space"
    assert plan.engine == "ffbs"
    assert "separate latent paths" in plan.warnings[0]
    with pytest.raises(ValueError, match="requires parameterization"):
        bx.plan(model, values, parameterization="centered")
    with pytest.raises(ValueError, match="ASIS"):
        bx.plan(model, values, asis=True)


@pytest.mark.parametrize("pool", ("selection", "slab", "both"))
def test_hierarchical_prior_modes_are_explicit_and_seasonality_is_present(pool):
    prior = bx.HierarchicalPrior(pool=pool)
    assert prior.pools_selection is (pool in {"selection", "both"})
    assert prior.pools_slab is (pool in {"slab", "both"})
    season = prior.initial_probabilities("season")
    np.testing.assert_allclose(season.sum(), 1.0)
    assert season[0] == 0.0
    if pool == "slab":
        np.testing.assert_allclose(season, (0.0, 0.0, 1.0))


@pytest.mark.parametrize("pool", ("selection", "slab", "both"))
def test_hierarchical_gaussian_fit_is_joint_forecastable_and_serializable(
    tmp_path, pool
):
    rng = np.random.default_rng(240)
    n_time = 28
    season = 0.18 * np.sin(2.0 * np.pi * np.arange(n_time) / 4.0)
    values = np.column_stack(
        (
            0.02 * np.arange(n_time) + season + rng.normal(scale=0.15, size=n_time),
            0.01 * np.arange(n_time) + 0.7 * season + rng.normal(scale=0.2, size=n_time),
        )
    )
    fit = bx.fit(
        values,
        _multiseries(),
        priors=bx.HierarchicalPrior(pool=pool),
        mcmc=bx.MCMC(draws=4, warmup=3, chains=1, seed=241),
        dates=_dates(n_time),
    )
    assert fit.is_multiseries_model
    assert fit.plan.backend == "hierarchical_state_space"
    assert fit.meta["joint_model"]
    assert fit.meta["hierarchy_pool"] == pool
    assert fit.eta_draws().shape == (4, n_time, 2)
    probabilities = fit.component_probabilities()
    assert probabilities.shape == (6, 3)
    assert np.all(probabilities.xs("seasonal", level="process")["zero"] == 0.0)
    assert fit.hierarchical_probabilities().shape == (8, 4)
    assert fit.hierarchical_slab_summary().shape == (3, 5)
    for channel in fit.channel_names:
        for state in ("level", "slope"):
            key = f"initial.channel.{channel}.{state}"
            assert key in fit.parameter_draws
            assert np.all(np.isfinite(fit.parameter(key)))
    assert np.nanmax(fit.draws_aux["sign_invariance_error"]) <= 1e-12
    assert fit.channel_rate_summary("first", 2000, 2002)["channel"] == "first"
    assert fit.forecast(2, draws=3, seed=242).observations.shape == (3, 2, 2)

    archive = tmp_path / f"hierarchical-{pool}.bucex"
    fit.save(archive)
    restored = bx.FitResult.load(archive)
    assert isinstance(restored.model, bx.MultiSeriesModel)
    np.testing.assert_allclose(restored.eta_draws(), fit.eta_draws())

    pytest.importorskip("matplotlib")
    process_path = tmp_path / f"hierarchical-{pool}-process-sds.png"
    hierarchy_path = tmp_path / f"hierarchical-{pool}-population.png"
    fit.plot("process_sd", prior_draws=200, save=process_path)
    fit.plot("hierarchy", save=hierarchy_path)
    assert process_path.is_file() and process_path.stat().st_size > 0
    assert hierarchy_path.is_file() and hierarchy_path.stat().st_size > 0


def test_mixed_hierarchy_uses_guided_pgas_without_restoration():
    rng = np.random.default_rng(243)
    n_time = 16
    values = np.column_stack(
        (rng.normal(scale=0.2, size=n_time), rng.gumbel(scale=0.3, size=n_time))
    )
    fit = bx.fit(
        values,
        _multiseries(mixed=True),
        priors=bx.HierarchicalPrior(pool="selection"),
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=244),
        particles=bx.Particles(n=20, proposal="guided"),
        dates=_dates(n_time),
    )
    assert fit.plan.engine == "pgas"
    assert fit.plan.targets_exact_posterior
    assert fit.meta["model_selection_exact"]
    assert fit.meta["restored_iterations"] == 0
    assert "xi.second" in fit.parameter_draws
    assert np.all(np.isfinite(fit.state_draws))
    assert "initial.channel.second.slope" in fit.parameter_draws


def test_guided_disturbance_pgas_respects_the_singular_affine_support():
    rng = np.random.default_rng(245)
    n_time = 18
    values = rng.gumbel(size=n_time)
    fit = bx.fit(
        values,
        family="gev",
        period=4,
        priors="normal",
        engine="pgas",
        parameterization="disturbance",
        asis=False,
        particles=bx.Particles(n=20, proposal="guided"),
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=246),
    )
    assert fit.meta["restored_iterations"] == 0
    assert np.all(np.isfinite(fit.state_draws))
    assert np.all(np.isfinite(fit.parameter("initial.level")))
    assert np.all(np.isfinite(fit.parameter("initial.slope")))


class _AlwaysSwitch:
    def random(self):
        return 0.0


def test_random_fs_sign_switches_preserve_the_complete_predictor():
    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
    )
    layout = infer_ncp_layout(model)
    path = np.random.default_rng(247).normal(size=(21, layout.ncp_state_dim))
    params = canonicalize_ncp_params(
        {
            "alpha0": 0.4,
            "beta0": 0.01,
            "gamma0_season": (0.2, -0.1, 0.05),
            "s_level": 0.03,
            "s_trend": -0.0002,
            "s_season": 0.02,
        },
        layout,
    )
    before = mu_from_ncp(path, params, layout)
    switched_path, switched_params, decisions = random_sign_switches(
        path, params, layout, _AlwaysSwitch(), return_switches=True
    )
    after = mu_from_ncp(switched_path, switched_params, layout)
    assert all(decisions.values())
    np.testing.assert_allclose(after, before, atol=1e-12)
    assert switched_params["s_level"] == -params["s_level"]
    assert switched_params["s_trend"] == -params["s_trend"]
    assert switched_params["s_season"] == -params["s_season"]


def test_univariate_fs_reports_audited_sign_switches():
    fit = bx.fit(
        np.random.default_rng(248).normal(size=20),
        family="gaussian",
        period=4,
        priors="normal",
        parameterization="fs",
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=249),
    )
    assert fit.meta["sign_switching"]
    assert fit.meta["sign_switch_invariance_checked"]
    assert np.nanmax(fit.draws_aux["sign_invariance_error"]) <= 1e-12
    assert set(fit.sampler_diagnostics["sign_switch_counts_by_chain"][0]) == {
        "level",
        "trend",
        "season",
    }


def test_uccle_hierarchy_preserves_channel_order_and_minimum_orientation():
    model = bx.make_uccle_hierarchical_model(series=("TXm", "TXx", "TNn"))
    assert isinstance(model, bx.MultiSeriesModel)
    assert model.channel_names == ("TXm", "TXx", "TNn")
    np.testing.assert_array_equal(model.transform_signs, (1.0, 1.0, -1.0))


def test_level_slope_plot_uses_seasonally_adjusted_observations():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    rng = np.random.default_rng(250)
    n_time = 24
    values = (
        0.02 * np.arange(n_time)
        + 2.0 * np.sin(2.0 * np.pi * np.arange(n_time) / 4.0)
        + rng.normal(scale=0.1, size=n_time)
    )
    fit = bx.fit(
        values,
        family="gaussian",
        period=4,
        priors="normal",
        parameterization="fs",
        mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=251),
    )
    figure, axes = fit.plot("level_slope")
    plotted = np.asarray(axes[0].collections[0].get_offsets())[:, 1]
    seasonal = np.median(fit.state_original("seasonal[1]"), axis=0)
    np.testing.assert_allclose(plotted, fit.observed - seasonal)
    labels = axes[0].get_legend_handles_labels()[1]
    assert "seasonally adjusted observed" in labels
    import matplotlib.pyplot as plt

    plt.close(figure)
