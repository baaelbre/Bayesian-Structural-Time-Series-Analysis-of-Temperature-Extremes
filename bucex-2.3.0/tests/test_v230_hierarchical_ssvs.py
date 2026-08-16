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


def _panel(*, mixed: bool = False) -> bx.MultiSeriesModel:
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


def test_multiseries_and_factor_models_have_distinct_scientific_semantics():
    panel = _panel()
    y = np.zeros((12, 2))
    compiled = bx.compile_model(panel, y)
    assert panel.factor_names == ()
    assert compiled.factor_names == ()
    assert compiled.channel_names == ("first", "second")
    plan = bx.plan(panel, y)
    assert plan.backend == "multiseries_state_space"
    assert "no latent time path" in plan.warnings[0]

    factor = bx.FactorModel(
        channels=panel.channels,
        factors=(
            bx.Factor(
                "common",
                (
                    bx.LocalLinearTrend(
                        initial_level=0.0, initial_level_sd=0.0
                    ),
                ),
                {"first": 1.0, "second": bx.Loading.estimated(0.8)},
            ),
        ),
    )
    assert bx.plan(factor, y).backend == "factor_state_space"
    with pytest.raises(ValueError, match="requires parameterization"):
        bx.plan(panel, y, parameterization="centered")
    with pytest.raises(ValueError, match="ASIS"):
        bx.plan(panel, y, asis=True)


def test_hierarchical_gaussian_ssvs_is_joint_and_forecastable(tmp_path):
    rng = np.random.default_rng(230)
    y = np.column_stack(
        (
            np.linspace(0.0, 0.4, 28) + 0.2 * rng.normal(size=28),
            np.linspace(0.0, 0.2, 28) + 0.25 * rng.normal(size=28),
        )
    )
    fit = bx.fit(
        y,
        _panel(),
        priors="hierarchical_ssvs",
        mcmc=bx.MCMC(draws=4, warmup=3, chains=1, seed=231),
        dates=np.arange(
            np.datetime64("2000-01"), np.datetime64("2002-05"), dtype="datetime64[M]"
        ),
    )
    assert fit.is_multiseries_model
    assert not fit.is_factor_model
    assert fit.plan.backend == "hierarchical_ssvs"
    assert fit.meta["hierarchical_model_selection"]
    assert fit.meta["sign_switch_invariance_checked"]
    assert fit.eta_draws().shape == (4, 28, 2)
    assert fit.component_probabilities().shape == (6, 3)
    assert fit.hierarchical_probabilities().shape == (8, 4)
    assert fit.hierarchical_slab_summary().shape == (3, 5)
    assert fit.channel_rate_summary("first", 2000, 2002)["channel"] == "first"
    assert fit.forecast(2, draws=3, seed=232).observations.shape == (3, 2, 2)

    archive = tmp_path / "hierarchical.bucex"
    fit.save(archive)
    restored = bx.FitResult.load(archive)
    assert isinstance(restored.model, bx.MultiSeriesModel)
    np.testing.assert_allclose(restored.eta_draws(), fit.eta_draws())

    pytest.importorskip("matplotlib")
    figure_path = tmp_path / "hierarchical-process-sds.png"
    fit.plot("process_sd", prior_draws=300, save=figure_path)
    assert figure_path.is_file() and figure_path.stat().st_size > 0


def test_mixed_hierarchical_ssvs_uses_pgas_not_laplace():
    rng = np.random.default_rng(233)
    y = np.column_stack(
        (rng.normal(scale=0.2, size=16), rng.gumbel(scale=0.3, size=16))
    )
    fit = bx.fit(
        y,
        _panel(mixed=True),
        priors="hierarchical_ssvs",
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=234),
        particles=bx.Particles(n=16),
    )
    assert fit.plan.engine == "pgas"
    assert fit.plan.targets_exact_posterior
    assert fit.meta["model_selection_exact"]
    assert "xi.second" in fit.parameter_draws


class _AlwaysSwitch:
    def random(self):
        return 0.0


def test_random_fs_sign_switches_preserve_the_complete_predictor():
    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
    )
    layout = infer_ncp_layout(model)
    path = np.random.default_rng(235).normal(size=(21, layout.ncp_state_dim))
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
        np.random.default_rng(236).normal(size=20),
        family="gaussian",
        period=4,
        priors="normal",
        parameterization="fs",
        mcmc=bx.MCMC(draws=2, warmup=2, chains=1, seed=237),
    )
    assert fit.meta["sign_switching"]
    assert fit.meta["sign_switch_invariance_checked"]
    assert np.nanmax(fit.draws_aux["sign_invariance_error"]) <= 1e-12
    assert set(fit.sampler_diagnostics["sign_switch_counts_by_chain"][0]) == {
        "level",
        "trend",
        "season",
    }


def test_uccle_multiseries_convenience_graph_is_not_a_factor():
    model = bx.make_uccle_multiseries_model(series=("TXm", "TXx", "TNn"))
    assert isinstance(model, bx.MultiSeriesModel)
    assert model.factor_names == ()
    assert model.channel_names == ("TXm", "TXx", "TNn")
    np.testing.assert_array_equal(model.transform_signs, (1.0, 1.0, -1.0))
