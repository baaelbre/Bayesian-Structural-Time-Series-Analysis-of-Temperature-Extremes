from __future__ import annotations

import numpy as np

import bucex as bx
from bucex.inference.fit.factor_fs import (
    CompiledFactorFS,
    _initial_fs_parameters,
)


def _two_channel_model(*, mixed: bool = False) -> bx.FactorModel:
    return bx.FactorModel(
        channels=(
            bx.Channel(
                "bulk",
                bx.Gaussian(),
                components=(bx.LocalLevel(), bx.DummySeasonal(period=4)),
            ),
            bx.Channel(
                "tail",
                bx.GEV() if mixed else bx.Gaussian(),
                components=(bx.LocalLevel(), bx.DummySeasonal(period=4)),
            ),
        ),
        factors=(
            bx.Factor(
                "common",
                (
                    bx.LocalLinearTrend(
                        initial_level=0.0,
                        initial_slope=0.0,
                        initial_level_sd=0.0,
                    ),
                ),
                {
                    "bulk": 1.0,
                    "tail": bx.Loading.estimated(0.8, sd=1.0),
                },
            ),
        ),
        name="v2.1 one-factor test",
    )


def test_uccle_v21_default_is_the_declared_one_factor_graph():
    model = bx.make_uccle_factor_model()
    assert model.factor_names == ("common",)
    assert model.supports_fs_parameterization
    assert model.factor("common").loading_for("TXm").fixed
    assert model.factor("common").loading_for("TXm").value == 1.0
    assert all(len(channel.components) == 2 for channel in model.channels)

    data = bx.load_uccle_factor_data(
        "data", start="2000-01-01", end="2001-12-01"
    )
    compiled = bx.compile_model(
        model, data.to_numpy() * model.transform_signs[None, :]
    )
    assert compiled.state_dim == 74
    assert compiled.noise_dim == 14
    plan = bx.plan(model, data)
    assert plan.parameterization == "fruehwirth_schnatter"
    assert plan.engine == "pgas"
    priors = bx.default_factor_priors(compiled)
    assert priors.profile == "regularized_horseshoe"
    assert set(priors.horseshoe_processes) == {
        f"channel.{channel}.level" for channel in model.channel_names
    }


def test_factor_fs_predictor_and_centered_state_are_algebraically_identical():
    model = _two_channel_model()
    y = np.random.default_rng(10).normal(size=(12, 2))
    compiled = bx.compile_model(model, y)
    priors = bx.default_factor_priors(compiled)
    fs = CompiledFactorFS(compiled)
    params = _initial_fs_parameters(
        compiled, fs, priors, np.random.default_rng(11), None
    )

    path = np.zeros((13, fs.state_dim))
    path[0] = fs.initial_mean
    rng = np.random.default_rng(12)
    for time in range(12):
        path[time + 1] = (
            fs.transition @ path[time]
            + fs.loading @ rng.normal(size=fs.noise_dim)
        )
    centered = fs.to_centered(path, params)
    np.testing.assert_allclose(
        fs.eta(path, params=params),
        compiled.eta(centered, params=params),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fs.from_centered(centered, params), path, atol=1e-12
    )


def test_factor_fs_and_disturbance_keep_one_fit_contract(tmp_path):
    model = _two_channel_model()
    rng = np.random.default_rng(20)
    dates = np.arange("2020-01", "2022-01", dtype="datetime64[M]")
    y = rng.normal(size=(dates.size, 2))
    fs_fit = bx.fit(
        y,
        model,
        dates=dates,
        parameterization="fs",
        asis=True,
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=21),
    )
    assert fs_fit.methods == {
        "engine": "ffbs",
        "parameterization": "fruehwirth_schnatter",
        "asis": True,
    }
    assert fs_fit.factor().shape == (2, dates.size)
    assert fs_fit.reconstructed_state("tail").shape == (2, dates.size)
    assert fs_fit.factor_rate_summary()["factor"] == "common"
    probabilities = fs_fit.factor_probabilities()
    assert set(probabilities["loading_above_threshold"]) == {"bulk", "tail"}
    assert fs_fit.auxiliary_draws["fs_state"].shape[:3] == (1, 2, 25)
    assert all(
        f"horseshoe.local.channel.{name}.level" in fs_fit.parameter_draws
        for name in ("bulk", "tail")
    )

    archive = tmp_path / "factor-fs.bucex"
    fs_fit.save(archive)
    restored = bx.FitResult.load(archive)
    assert restored.schema_version == "2.1"
    np.testing.assert_allclose(restored.factor(), fs_fit.factor())

    disturbance_fit = bx.fit(
        y,
        model,
        parameterization="disturbance",
        mcmc=bx.MCMC(draws=1, warmup=0, chains=1, seed=22),
    )
    assert disturbance_fit.methods["parameterization"] == "disturbance"
    assert disturbance_fit.meta["regularized_horseshoe"]
    assert set(disturbance_fit.meta["horseshoe_processes"]) == {
        "channel.bulk.level",
        "channel.tail.level",
    }


def test_mixed_factor_fs_uses_exact_pgas_joint_likelihood():
    model = _two_channel_model(mixed=True)
    rng = np.random.default_rng(30)
    y = np.column_stack(
        (rng.normal(10.0, 1.0, 10), 12.0 + rng.gumbel(size=10))
    )
    fit = bx.fit(
        y,
        model,
        parameterization="fruehwirth_schnatter",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=31),
        particles=bx.Particles(n=16),
    )
    assert fit.plan.engine == "pgas"
    assert fit.plan.targets_exact_posterior
    assert np.all(np.isfinite(fit.log_posterior))
    assert np.all(
        np.isfinite(
            fit.sampler_diagnostics["draw_metrics"]["particle_min_ess"]
        )
    )
