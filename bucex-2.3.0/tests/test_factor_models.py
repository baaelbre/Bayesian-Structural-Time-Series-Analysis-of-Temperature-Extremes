from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import bucex as bx


def _gaussian_factor(estimated: bool = True):
    loading = bx.Loading.estimated(0.7, sd=1.0) if estimated else 0.7
    return bx.FactorModel(
        channels=(
            bx.Channel("bulk", bx.Gaussian()),
            bx.Channel("night", bx.Gaussian()),
        ),
        factors=(
            bx.Factor(
                "climate",
                (bx.LocalLevel(initial_mean=0.0, initial_sd=1.1),),
                {"bulk": 1.0, "night": loading},
            ),
        ),
        name="two-channel climate",
    )


def test_factor_identification_and_compiler_namespaces():
    with pytest.raises(ValueError, match="anchor"):
        bx.Factor(
            "unidentified",
            (bx.LocalLevel(),),
            {
                "a": bx.Loading.estimated(1.0),
                "b": bx.Loading.estimated(0.5),
            },
        )
    with pytest.raises(ValueError, match="full column rank"):
        bx.FactorModel(
            channels=(bx.Channel("a", bx.Gaussian()), bx.Channel("b", bx.Gaussian())),
            factors=(
                bx.Factor(
                    "first",
                    (bx.LocalLevel(),),
                    {"a": 1.0, "b": bx.Loading.estimated(0.5)},
                ),
                bx.Factor(
                    "second",
                    (bx.LocalLevel(),),
                    {"a": 1.0, "b": bx.Loading.estimated(-0.5)},
                ),
            ),
        )
    model = _gaussian_factor()
    y = np.column_stack((np.linspace(0.0, 1.0, 8), np.linspace(0.2, 0.8, 8)))
    compiled = bx.compile_model(model, y)
    params = {
        "sd.factor.climate.level": 0.1,
        "sigma.bulk": 0.2,
        "sigma.night": 0.3,
        "loading.climate.night": 0.65,
    }
    assert compiled.state_names == ("factor.climate.level",)
    assert compiled.noise_names == ("factor.climate.level",)
    assert compiled.design(params=params).shape == (8, 2, 1)
    np.testing.assert_allclose(
        compiled.loading_matrix(params),
        [[1.0], [0.65]],
    )


def test_multichannel_kalman_likelihood_matches_joint_normal():
    model = _gaussian_factor(estimated=False)
    y = np.asarray([[0.2, 0.0], [-0.1, 0.1], [0.4, 0.2], [0.3, 0.4]])
    compiled = bx.compile_model(model, y)
    params = {
        "sd.factor.climate.level": 0.25,
        "sigma.bulk": 0.4,
        "sigma.night": 0.6,
        "loading.climate.bulk": 1.0,
        "loading.climate.night": 0.7,
    }
    result = bx.kalman_filter(y, compiled, params)
    n_time = y.shape[0]
    covariance = np.zeros((2 * n_time, 2 * n_time))
    loadings = np.asarray([1.0, 0.7])
    sigmas = np.asarray([0.4, 0.6])
    for t in range(n_time):
        for s in range(n_time):
            latent = 1.1**2 + (min(t, s) + 1) * 0.25**2
            for i in range(2):
                for j in range(2):
                    covariance[2 * t + i, 2 * s + j] = loadings[i] * loadings[j] * latent
    covariance += np.diag(np.tile(sigmas**2, n_time))
    exact = multivariate_normal.logpdf(y.reshape(-1), mean=np.zeros(y.size), cov=covariance)
    assert result.log_likelihood == pytest.approx(exact, abs=1e-9)


def test_mixed_factor_simulation_and_pgas_are_exact_invariant():
    model = bx.FactorModel(
        channels=(
            bx.Channel("mean", bx.Gaussian()),
            bx.Channel("maximum", bx.GEV()),
        ),
        factors=(
            bx.Factor(
                "climate",
                (bx.LocalLevel(initial_mean=0.0, initial_sd=1.0),),
                {"mean": 1.0, "maximum": 0.8},
            ),
        ),
    )
    params = {
        "sd.factor.climate.level": 0.08,
        "sigma.mean": 0.25,
        "sigma.maximum": 0.35,
        "xi.maximum": -0.05,
    }
    simulation = bx.simulate(model, 12, params, seed=11)
    compiled = bx.compile_model(model, simulation.y)
    laplace = bx.iterated_laplace(
        simulation.y,
        compiled,
        params,
        np.random.default_rng(12),
        max_iterations=8,
    )
    result = bx.pgas(
        simulation.y,
        compiled,
        params,
        laplace.path,
        particles=bx.Particles(n=16),
        rng=np.random.default_rng(13),
    )
    assert result.exact_invariant
    assert result.path.shape == (13, 1)
    compiled.to_disturbance(result.path, params)


def test_factor_fit_forecast_risk_diagnostics_and_archive(tmp_path):
    model = bx.FactorModel(
        channels=(
            bx.Channel("mean", bx.Gaussian()),
            bx.Channel("minimum", bx.GEV(), tail="lower"),
        ),
        factors=(
            bx.Factor(
                "climate",
                (bx.LocalLevel(initial_mean=0.0, initial_sd=1.0),),
                {"mean": 1.0, "minimum": bx.Loading.estimated(0.7)},
            ),
        ),
        name="mixed climate",
    )
    parameters = {
        "sd.factor.climate.level": 0.08,
        "sigma.mean": 0.25,
        "sigma.minimum": 0.30,
        "xi.minimum": 0.0,
        "loading.climate.minimum": 0.7,
    }
    simulation = bx.simulate(model, 12, parameters, seed=21)
    fit = bx.fit(
        simulation.y,
        model,
        priors="normal",
        engine="pgas",
        parameterization="disturbance",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=22),
        particles=bx.Particles(n=16),
        laplace=bx.Laplace(max_iterations=8),
    )
    assert type(fit) is bx.FitResult
    assert fit.plan.backend == "factor_state_space"
    assert fit.plan.targets_exact_posterior
    assert fit.state_draws.shape == (1, 2, 13, 1)
    assert fit.eta_draws().shape == (2, 12, 2)
    assert fit.channel_eta_draws("minimum").shape == (2, 12)
    assert fit.factor_draws("climate").shape == (2, 12)
    assert fit.diagnostics()["pit"].shape == (12, 2)
    assert fit.return_level_draws(20, channel="minimum").shape == (2, 12)
    assert "<" in fit.event_label(0.0, channel="minimum")

    forecast = fit.forecast(3, draws=2, seed=23)
    assert forecast.observations.shape == (2, 3, 2)
    assert forecast.summary().shape[0] == 6
    assert forecast.return_level(20, channel="minimum").shape == (2, 3)
    held_out = np.zeros((3, 2), dtype=float)
    scores = forecast.score(held_out, aggregate=False)
    assert {"crps", "log"}.issubset(set(scores["score"]))
    assert set(scores["channel"]) == {"mean", "minimum"}
    for index, channel in enumerate(("mean", "minimum")):
        pit = forecast.pit(held_out[:, index], channel=channel)
        assert np.all((pit >= 0.0) & (pit <= 1.0))

    path = tmp_path / "factor.bucex"
    fit.save(path)
    restored = bx.FitResult.load(path)
    assert type(restored.model) is bx.FactorModel
    assert restored.channel_names == ("mean", "minimum")
    np.testing.assert_allclose(restored.state_draws, fit.state_draws)
    np.testing.assert_allclose(restored.eta_draws(), fit.eta_draws())


def test_univariate_model_remains_the_scalar_special_case():
    model = bx.Model(bx.Gaussian(), [bx.LocalLevel()])
    compiled = bx.compile_model(model, np.arange(6.0))
    assert isinstance(compiled, bx.CompiledModel)
    assert compiled.design().shape == (6, 1)
    assert compiled.eta(np.zeros((7, 1))).shape == (6,)


def test_uccle_factor_constructors_are_aligned_and_orientation_aware():
    data = bx.load_uccle_factor_data(
        "data", start="2000-01-01", end="2001-12-01"
    )
    model = bx.make_uccle_factor_model(structure="contrasts")
    assert tuple(data.columns) == model.channel_names
    assert model.factor_names == (
        "common",
        "day_night",
        "extremes_mean",
        "upper_lower",
    )
    # The original lower-tail contrast is -1; internal sign orientation turns
    # that into +1 for the model-scale TXn channel.
    assert model.factor("upper_lower").loading_for("TXn").value == pytest.approx(1.0)
    compiled = bx.compile_model(
        model,
        data.to_numpy() * model.transform_signs[None, :],
    )
    assert compiled.design().shape == (24, 6, compiled.state_dim)
