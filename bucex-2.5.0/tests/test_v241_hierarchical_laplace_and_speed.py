from __future__ import annotations

import numpy as np
import pytest

import bucex as bx
from bucex.inference.fit.fs_utils import (
    _ncp_exact_observation_loglik,
    _ncp_laplace_pseudo_data,
    _ncp_observation_logweights,
    canonicalize_ncp_params,
    infer_ncp_layout,
    measurement_vector,
)
from bucex.inference.fit._progress import compact_group, mcmc_progress_line


def _mixed_model() -> bx.MultiSeriesModel:
    components = (bx.LocalLinearTrend(), bx.DummySeasonal(4))
    return bx.MultiSeriesModel(
        channels=(
            bx.Channel("mean", bx.Gaussian(), components),
            bx.Channel("maximum", bx.GEV(), components),
        ),
        name="2.4.1 mixed hierarchy test",
    )


def _mixed_values(seed: int = 241) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_time = 16
    time = np.arange(n_time, dtype=float)
    season = 0.12 * np.sin(2.0 * np.pi * time / 4.0)
    return np.column_stack(
        (
            0.01 * time + season + rng.normal(scale=0.15, size=n_time),
            0.015 * time + 0.7 * season + rng.gumbel(scale=0.25, size=n_time),
        )
    )


def test_componentwise_selection_is_default_and_joint_space_is_explicit():
    prior = bx.HierarchicalPrior(pool="selection")
    assert prior.model_space == "componentwise"
    np.testing.assert_allclose(prior.initial_probabilities("trend"), 1.0 / 3.0)
    np.testing.assert_allclose(prior.initial_probabilities("season")[0], 0.0)

    joint = bx.HierarchicalPrior(
        pool="selection",
        model_space="joint_trend",
        trend_states=("fixed", "dynamic"),
        trend_concentration=(1.0, 1.0),
    )
    np.testing.assert_allclose(joint.initial_model_probabilities(), 0.25)
    np.testing.assert_allclose(joint.initial_probabilities("trend")[0], 0.0)

    with pytest.raises(ValueError, match="always estimates a slope"):
        bx.HierarchicalPrior(
            model_space="joint_trend",
            trend_states=("zero", "fixed", "dynamic"),
            trend_concentration=(1.0, 1.0, 1.0),
        )


def test_structural_scale_calibration_has_observable_implications():
    scales = bx.calibrate_structural_scales(
        horizon_years=20,
        max_level_change=1.5,
        max_rate_change_per_decade=0.4,
        max_seasonal_innovation=0.25,
        periods_per_year=12,
        probability=0.90,
    )
    implications = bx.structural_scale_implications(
        level_sd=scales["level"],
        slope_sd=scales["trend"],
        seasonal_sd=scales["season"],
        horizon_years=20,
        periods_per_year=12,
        probability=0.90,
    )
    assert implications["level_random_walk_bound"] == pytest.approx(1.5)
    assert implications["rate_change_per_decade_bound"] == pytest.approx(0.4)
    assert implications["one_step_seasonal_innovation_bound"] == pytest.approx(0.25)
    assert bx.half_student_t_scale_for_median(1.0, df=4.0) > 0.0


def test_mixed_hierarchical_laplace_plan_is_explicitly_approximate():
    values = _mixed_values()
    model = _mixed_model()
    laplace_plan = bx.plan(model, values, engine="laplace")
    assert laplace_plan.engine == "laplace"
    assert not laplace_plan.targets_exact_posterior
    assert laplace_plan.approximation == "iterated_laplace"
    assert any("exploratory" in warning.lower() for warning in laplace_plan.warnings)

    pgas_plan = bx.plan(model, values, engine="pgas")
    assert pgas_plan.targets_exact_posterior
    assert pgas_plan.approximation is None


def test_multiseries_progress_groups_shape_parameters_and_names_path_updates():
    line = mcmc_progress_line(
        label="hierarchical SSVS",
        engine="pgas",
        chain=2,
        chains=4,
        completed=25,
        total=100,
        warmup=50,
        saved=0,
        draws=50,
        elapsed=5.0,
        parameters={"xi": compact_group({"TXx": -0.12, "TNn": 0.04})},
        metrics={
            "particle_min_ess": 18.0,
            "particle_mean_unique_ancestors": 12.5,
            "particle_path_update_fraction": 0.37,
        },
        particles=24,
    )
    assert "chain 2/4" in line
    assert "xi=(TXx:-0.12,TNn:0.04)" in line
    assert "path_update=0.37" in line
    assert "changed_fraction" not in line


def test_laplace_screen_warm_starts_exact_pgas_and_parallel_updates():
    values = _mixed_values(seed=242)
    model = _mixed_model()
    prior = bx.HierarchicalPrior(
        pool="selection",
        model_space="joint_trend",
        trend_states=("fixed", "dynamic"),
        trend_concentration=(1.0, 1.0),
    )
    execution = bx.HierarchicalSampler(initializer="laplace", channel_workers=2)
    screen = bx.fit(
        values,
        model,
        priors=prior,
        engine="laplace",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=243),
        hierarchical_sampler=execution,
    )
    assert not screen.plan.targets_exact_posterior
    assert screen.meta["channel_workers"] == 2
    assert set(screen.hierarchical_trend_model_probabilities().index) == {
        "linear_trend",
        "rw1_drift",
        "rw2_smooth_trend",
        "local_linear_trend",
    }
    warm = screen.warm_start()
    assert "channel.mean.__centered_path" in warm
    assert "channel.maximum.__centered_path" in warm
    assert "hierarchy.trend_model_probabilities" in warm

    exact = bx.fit(
        values,
        model,
        priors=prior,
        engine="pgas",
        init=screen,
        particles=bx.Particles(n=16, proposal="guided"),
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=244),
        hierarchical_sampler=execution,
    )
    assert exact.plan.targets_exact_posterior
    assert exact.meta["external_warm_start"]
    assert exact.meta["warm_start_source"]["source_engine"] == "laplace"
    assert np.all(np.isfinite(exact.state_draws))
    assert np.all(
        np.isfinite(exact.draws_aux["particle_path_update_fraction"])
    )
    np.testing.assert_allclose(
        exact.draws_aux["particle_changed_fraction"],
        exact.draws_aux["particle_path_update_fraction"],
    )


def test_vectorized_gev_kernels_match_scalar_evaluation():
    observation = bx.GEV()
    model = bx.Model(observation, (bx.LocalLinearTrend(), bx.DummySeasonal(4)))
    layout = infer_ncp_layout(model)
    params_state = canonicalize_ncp_params(
        {
            "alpha0": 0.3,
            "beta0": 0.01,
            "gamma0_season": (0.1, -0.05, 0.02),
            "s_level": 0.03,
            "s_trend": 0.0002,
            "s_season": 0.02,
        },
        layout,
    )
    params_obs = {"sigma": 0.8, "sigma2": 0.64, "xi": -0.08}
    rng = np.random.default_rng(245)
    states = rng.normal(scale=0.2, size=(31, layout.ncp_state_dim))
    H = measurement_vector(params_state, layout)
    offset = 0.4
    y_t = 0.7

    vector = _ncp_observation_logweights(
        y_t, states, offset, H, model, params_obs
    )
    scalar = np.asarray(
        [
            _ncp_observation_logweights(
                y_t, state[None, :], offset, H, model, params_obs
            )[0]
            for state in states
        ]
    )
    np.testing.assert_allclose(vector, scalar, rtol=0.0, atol=1e-13)

    z_path = np.vstack((np.zeros((1, layout.ncp_state_dim)), states[:12]))
    # Use direct model reconstruction for the exact-likelihood comparison.
    from bucex.inference.fit.fs_utils import mu_from_ncp

    eta = mu_from_ncp(z_path, params_state, layout)
    y = eta + rng.normal(scale=0.2, size=eta.size)
    vector_loglik = _ncp_exact_observation_loglik(
        y, z_path, params_state, params_obs, model, layout
    )
    scalar_loglik = sum(
        float(observation.logpdf(y_i, eta_i, params=params_obs))
        for y_i, eta_i in zip(y, eta)
    )
    assert vector_loglik == pytest.approx(scalar_loglik, abs=1e-12)

    vector_y, vector_variance = _ncp_laplace_pseudo_data(
        y,
        eta,
        model,
        params_obs,
        curvature_floor=1e-6,
        maximum_variance=1e8,
        shift_limit=2.0,
    )
    scalar_y = []
    scalar_variance = []
    for y_i, eta_i in zip(y, eta):
        gradient = float(observation.grad_eta(y_i, eta_i, params=params_obs))
        hessian = float(observation.hess_eta(y_i, eta_i, params=params_obs))
        information = max(-hessian, 1e-6)
        scalar_variance.append(min(1.0 / information, 1e8))
        scalar_y.append(eta_i + np.clip(gradient / information, -2.0, 2.0))
    np.testing.assert_allclose(vector_y, scalar_y, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        vector_variance, scalar_variance, rtol=0.0, atol=1e-12
    )
