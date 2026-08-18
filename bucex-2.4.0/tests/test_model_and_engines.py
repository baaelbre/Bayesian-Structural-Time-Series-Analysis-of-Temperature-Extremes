from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

import bucex as bx
from bucex.core.numerics import gaussian_support, psd_eigh
from bucex.core.numerics import normalize_logweights
from bucex.inference.fit.fs_utils import _strict_normalize_logweights
from bucex.inference.state.laplace import observation_log_likelihood


def test_model_compiler_and_named_regression_design():
    model = bx.Model(
        bx.Gaussian(),
        [
            bx.LocalLinearTrend(),
            bx.DummySeasonal(4),
            bx.Regression(2, name="climate", feature_names=("nao", "enso")),
        ],
    )
    exog = pd.DataFrame(
        {
            "enso": np.linspace(1.0, 2.0, 12),
            "nao": np.linspace(-1.0, 1.0, 12),
            "unused": np.ones(12),
        }
    )
    compiled = bx.compile_model(model, np.arange(12.0), exog=exog)
    assert compiled.state_names == (
        "level",
        "slope",
        "seasonal[1]",
        "seasonal[2]",
        "seasonal[3]",
        "climate[nao]",
        "climate[enso]",
    )
    assert compiled.noise_names == ("level", "slope", "seasonal")
    np.testing.assert_allclose(compiled.design()[:, -2], exog["nao"])
    np.testing.assert_allclose(compiled.design()[:, -1], exog["enso"])


def test_general_disturbance_roundtrip_includes_dynamic_regression():
    model = bx.Model(
        bx.Gaussian(),
        [
            bx.LocalLinearTrend(),
            bx.DummySeasonal(4),
            bx.Regression(1, dynamic=True, name="x"),
        ],
    )
    exog = np.linspace(-1.0, 1.0, 20)[:, None]
    params = {
        "sd.level": 0.1,
        "sd.slope": 0.01,
        "sd.seasonal": 0.05,
        "sd.x[1]": 0.03,
        "sigma": 0.2,
    }
    simulation = bx.simulate(
        model,
        20,
        params,
        exog=exog,
        initial_state=np.zeros(6),
        seed=7,
    )
    compiled = bx.compile_model(model, simulation.y, exog=exog)
    reconstructed = compiled.from_disturbance(
        compiled.to_disturbance(simulation.states, params), params
    )
    np.testing.assert_allclose(reconstructed, simulation.states, atol=1e-11)


def test_gev_location_derivatives_match_finite_differences():
    observation = bx.GEV()
    sigma = 1.3
    eta = 0.4
    step = 1e-5
    for xi in (-0.3, 0.0, 0.25):
        y = 0.1
        objective = lambda value: float(
            observation.logpdf(y, value, sigma=sigma, xi=xi)
        )
        gradient = (objective(eta + step) - objective(eta - step)) / (2.0 * step)
        hessian = (
            objective(eta + step) - 2.0 * objective(eta) + objective(eta - step)
        ) / step**2
        assert observation.grad_eta(y, eta, sigma=sigma, xi=xi) == pytest.approx(
            gradient, abs=1e-6
        )
        assert observation.hess_eta(y, eta, sigma=sigma, xi=xi) == pytest.approx(
            hessian, abs=2e-3
        )


def test_kalman_likelihood_matches_joint_normal():
    y = np.asarray([0.2, -0.1, 0.4, 0.3])
    model = bx.Model(
        bx.Gaussian(), [bx.LocalLevel(initial_mean=0.0, initial_sd=1.2)]
    )
    compiled = bx.compile_model(model, y)
    params = {"sd.level": 0.3, "sigma": 0.5}
    result = bx.kalman_filter(y, compiled, params)
    covariance = np.fromfunction(
        lambda i, j: 1.2**2 + (np.minimum(i, j) + 1.0) * 0.3**2,
        (y.size, y.size),
    )
    covariance += 0.5**2 * np.eye(y.size)
    exact = multivariate_normal.logpdf(y, mean=np.zeros(y.size), cov=covariance)
    assert result.log_likelihood == pytest.approx(exact, abs=1e-9)


def test_iterated_laplace_and_singular_pgas_paths_are_valid():
    model = bx.Model(bx.GEV(), [bx.LocalLinearTrend(), bx.DummySeasonal(4)])
    params = {
        "sd.level": 0.05,
        "sd.slope": 0.004,
        "sd.seasonal": 0.03,
        "sigma": 0.8,
        "xi": -0.1,
    }
    simulation = bx.simulate(model, 16, params, initial_state=np.zeros(5), seed=10)
    compiled = bx.compile_model(model, simulation.y)
    laplace = bx.iterated_laplace(
        simulation.y,
        compiled,
        params,
        np.random.default_rng(11),
        max_iterations=20,
    )
    assert np.isfinite(
        observation_log_likelihood(
            simulation.y, compiled.eta(laplace.path), compiled, params
        )
    )
    result = bx.pgas(
        simulation.y,
        compiled,
        params,
        laplace.path,
        particles=bx.Particles(n=24),
        rng=np.random.default_rng(12),
    )
    assert result.exact_invariant
    compiled.to_disturbance(result.path, params)


def test_psd_and_tiny_positive_process_variance_are_not_conflated():
    values, _ = psd_eigh(np.diag([1.0, -5e-10]))
    np.testing.assert_allclose(values, [0.0, 1.0])
    with pytest.raises(np.linalg.LinAlgError):
        psd_eigh(np.diag([1.0, -1e-5]))
    model = bx.Model(bx.Gaussian(), [bx.LocalLinearTrend(), bx.DummySeasonal(4)])
    compiled = bx.compile_model(model, np.arange(20.0))
    support = gaussian_support(
        compiled.transition_cov(
            {
                "sd.level": 0.01,
                "sd.slope": 0.0002,
                "sd.seasonal": 0.02,
                "sigma": 1.0,
            }
        )
    )
    assert support.active_variances.size == 3


def test_particle_normalization_keeps_underflow_sized_log_weights():
    # exp(-1000) is zero in float64, but a log weight at -1000 is still a
    # mathematically valid ancestor when all alternatives are impossible.
    values = np.asarray([-1000.0, -np.inf, -np.inf])
    weights, normalizer = normalize_logweights(values)
    np.testing.assert_allclose(weights, [1.0, 0.0, 0.0])
    assert normalizer == pytest.approx(-1000.0)
    fs_weights, fs_log_weights, fs_normalizer = _strict_normalize_logweights(values)
    np.testing.assert_allclose(fs_weights, weights)
    assert fs_log_weights[0] == pytest.approx(0.0)
    assert fs_normalizer == pytest.approx(-1000.0)


def test_convenience_priors_are_local_scale_not_record_length_calibrated():
    short = np.tile([0.0, 1.0], 20)
    long = np.tile([0.0, 1.0], 200)
    model = bx.Model(bx.Gaussian(), [bx.LocalLevel()])
    short_prior = bx.default_priors(bx.compile_model(model, short))
    long_prior = bx.default_priors(bx.compile_model(model, long))
    ratio = long_prior.process["level"].upper / short_prior.process["level"].upper
    assert 0.95 < ratio < 1.05


def test_observation_reference_scale_removes_fixed_seasonal_cycle():
    rng = np.random.default_rng(46)
    seasonal = np.tile(np.linspace(-12.0, 12.0, 12), 20)
    values = seasonal + rng.normal(scale=0.4, size=seasonal.size)
    model = bx.Model(bx.Gaussian(), [bx.LocalLevel(), bx.DummySeasonal(12)])
    compiled = bx.compile_model(model, values)
    priors = bx.default_priors(compiled)
    assert compiled.observation_scale < compiled.y_scale / 4.0
    assert priors.observation_sd.scale == pytest.approx(compiled.observation_scale)
