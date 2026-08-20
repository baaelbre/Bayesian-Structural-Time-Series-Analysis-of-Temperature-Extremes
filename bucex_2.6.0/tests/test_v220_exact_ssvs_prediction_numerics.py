from __future__ import annotations

import numpy as np

import bucex as bx
from bucex.api.predict import Forecast
from bucex.inference.fit.fs_utils import (
    _backward_smoothing_moments,
    baseline_mu_path,
    canonicalize_ncp_params,
    design_matrix_ncp,
    ffbs_gaussian_1d,
    infer_ncp_layout,
    mu_from_ncp,
)
from bucex.inference.fit.model_space import (
    ComponentState,
    StructuralModelState,
    enforce_structural_state,
)


def _seasonal_model(period: int = 4):
    return bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(period)),
    )


def test_fs_design_and_reconstruction_are_identical_with_seasonality():
    model = _seasonal_model()
    layout = infer_ncp_layout(model)
    rng = np.random.default_rng(220)
    z_path = rng.normal(size=(31, layout.ncp_state_dim))
    params = canonicalize_ncp_params(
        {
            "alpha0": 1.2,
            "beta0": 0.03,
            "gamma0_season": (0.2, -0.1, 0.05),
            "s_level": 0.4,
            "s_trend": -0.02,
            "s_season": 0.1,
        },
        layout,
    )
    design, names, tbar = design_matrix_ncp(z_path, layout)
    theta = []
    for name in names:
        if name == "alpha_c":
            theta.append(params["alpha0"] + tbar * params["beta0"])
        elif name.startswith("gamma0_season_"):
            theta.append(params["gamma0_season"][int(name.rsplit("_", 1)[1]) - 1])
        else:
            theta.append(params[name])
    np.testing.assert_allclose(
        design @ np.asarray(theta),
        mu_from_ncp(z_path, params, layout),
        atol=1e-12,
    )


def test_fixed_fs_components_are_static_coefficients_not_small_shocks():
    model = _seasonal_model()
    layout = infer_ncp_layout(model)
    state = StructuralModelState(
        level=ComponentState.FIXED,
        trend=ComponentState.FIXED,
        season=ComponentState.FIXED,
    )
    params = enforce_structural_state(
        {
            "alpha0": 0.7,
            "beta0": 0.02,
            "gamma0_season": np.array((0.2, -0.1, 0.05)),
            "s_level": 2.0,
            "s_trend": 3.0,
            "s_season": 4.0,
        },
        state,
        layout,
    )
    assert params["s_level"] == params["s_trend"] == params["s_season"] == 0.0
    first = np.zeros((25, layout.ncp_state_dim))
    second = np.random.default_rng(2).normal(size=first.shape)
    expected = baseline_mu_path(24, params, layout)
    np.testing.assert_allclose(mu_from_ncp(first, params, layout), expected)
    np.testing.assert_allclose(mu_from_ncp(second, params, layout), expected)


def test_joseph_backward_covariance_is_psd_and_long_ffbs_is_stable():
    covariance = np.array(
        [[1.0, 1.0 - 1e-12], [1.0 - 1e-12, 1.0]], dtype=float
    )
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    process = np.diag([0.0, 1e-16])
    predicted = transition @ covariance @ transition.T + process
    _, conditional = _backward_smoothing_moments(
        np.zeros(2),
        covariance,
        np.zeros(2),
        transition,
        process,
        predicted,
    )
    assert np.min(np.linalg.eigvalsh(conditional)) >= -1e-12

    path = ffbs_gaussian_1d(
        np.zeros(1_000),
        transition,
        process,
        np.array((1.0, 0.0)),
        0.1,
        C0=np.zeros((2, 2)),
        rng=np.random.default_rng(12),
    )
    assert path.shape == (1_001, 2)
    assert np.all(np.isfinite(path))


def test_gev_pgas_ssvs_is_exact_invariant_and_records_model_moves():
    model = bx.Model(
        bx.GEV(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(4)),
    )
    simulation = bx.simulate(
        model,
        28,
        {
            "sd.level": 0.02,
            "sd.slope": 0.001,
            "sd.seasonal": 0.01,
            "sigma": 0.4,
            "xi": -0.05,
        },
        seed=221,
    )
    fitted = bx.fit(
        simulation.y,
        model,
        priors="ssvs",
        engine="pgas",
        mcmc=bx.MCMC(draws=4, warmup=4, chains=1, seed=222),
        particles=bx.Particles(n=20),
    )
    assert fitted.plan.targets_exact_posterior
    assert fitted.metadata["model_selection_exact"] is True
    assert "exact_gev_rjmh" in fitted.metadata["model_selection_basis"]
    assert "ssvs_model_move_accepted" in fitted.draws_aux
    assert "state_level" in fitted.parameter_draws


def test_forecast_log_score_pit_and_lfo_share_one_api():
    model = bx.Model(bx.Gaussian(), (bx.LocalLinearTrend(),))
    simulation = bx.simulate(
        model,
        28,
        {"sd.level": 0.02, "sd.slope": 0.001, "sigma": 0.3},
        seed=223,
    )
    fitted = bx.fit(
        simulation.y[:24],
        model,
        priors="normal",
        engine="ffbs",
        mcmc=bx.MCMC(draws=6, warmup=6, chains=1, seed=224),
    )
    forecast = fitted.forecast(2, draws=6, seed=225)
    observed = simulation.y[24:26]
    assert forecast.log_score(observed).shape == (2,)
    assert np.all((forecast.pit(observed) >= 0.0) & (forecast.pit(observed) <= 1.0))
    scores = forecast.score(observed)
    assert "log" in set(scores["score"])

    result = bx.leave_future_out(
        simulation.y,
        model,
        initial=24,
        horizon=2,
        step=2,
        fit_options={
            "priors": "normal",
            "engine": "ffbs",
            "mcmc": bx.MCMC(draws=3, warmup=3, chains=1, seed=226),
        },
        forecast_options={"draws": 3, "seed": 227},
        progress=False,
    )
    assert result.n_origins == 2
    assert {"crps", "log"}.issubset(set(result.scores["score"]))
    assert result.pit_diagnostics(bins=3).summary["n"] == 4


def test_lower_tail_pit_is_returned_on_the_original_orientation():
    forecast = Forecast(
        observations=np.zeros((2, 1)),
        eta=np.full((2, 1), -2.0),
        states=np.zeros((2, 1, 1)),
        parameters={"sigma": np.ones(2), "xi": np.zeros(2)},
        dates=np.array((1,)),
        family="gev",
        tail="lower",
        observation_model=bx.GEV(),
        transform_sign=-1.0,
    )
    expected = 1.0 - bx.GEV().cdf(1.0, 2.0, sigma=1.0, xi=0.0)
    np.testing.assert_allclose(forecast.pit(np.array((-1.0,))), expected)
