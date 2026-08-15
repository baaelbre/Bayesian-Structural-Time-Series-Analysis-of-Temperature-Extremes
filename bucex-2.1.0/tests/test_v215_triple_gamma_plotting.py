from __future__ import annotations

import numpy as np
import pytest

import bucex as bx


def _univariate_data(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(28)
    return 0.02 * time + 0.15 * np.sin(2.0 * np.pi * time / 4.0) + rng.normal(
        scale=0.20, size=time.size
    )


def _factor_model() -> bx.FactorModel:
    return bx.FactorModel(
        channels=(
            bx.Channel("reference", bx.Gaussian(), (bx.LocalLevel(),)),
            bx.Channel("response", bx.Gaussian(), (bx.LocalLevel(),)),
        ),
        factors=(
            bx.Factor(
                "common",
                (
                    bx.LocalLinearTrend(
                        initial_level=0.0,
                        initial_level_sd=0.0,
                        initial_slope=0.0,
                        initial_slope_sd=0.0,
                    ),
                ),
                {
                    "reference": 1.0,
                    "response": bx.Loading.estimated(0.8),
                },
            ),
        ),
    )


def test_triple_gamma_profiles_variance_and_shrinkage_factor():
    plain = bx.triple_gamma_gaussian_priors(
        spike_shape=0.5,
        tail_shape=0.5,
        learn_global=False,
    )
    regularized = bx.regularized_triple_gamma_gaussian_priors()
    assert plain.profile == "triple_gamma"
    assert regularized.profile == "regularized_triple_gamma"
    assert regularized.triple_gamma.regularized

    prior = plain.triple_gamma
    variance = prior.conditional_variance(
        "level", numerator=2.0, denominator=4.0, global_scale=3.0
    )
    assert variance == pytest.approx(0.03**2 * 1.5)
    assert prior.shrinkage_factor(
        numerator=2.0, denominator=4.0, global_scale=3.0
    ) == pytest.approx(1.0 / 2.5)


@pytest.mark.parametrize("profile", ("triple_gamma", "regularized_triple_gamma"))
def test_univariate_triple_gamma_profiles_fit(profile):
    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(period=4)),
    )
    fit = bx.fit(
        _univariate_data(2),
        model,
        parameterization="fruehwirth_schnatter",
        engine="ffbs",
        priors=profile,
        mcmc=bx.MCMC(draws=3, warmup=2, chains=1, seed=3),
    )
    required = {
        "triple_gamma_global",
        "triple_gamma_a",
        "triple_gamma_c",
        "triple_gamma_numerator_level",
        "triple_gamma_denominator_level",
        "triple_gamma_rho_level",
    }
    assert required <= set(fit.parameter_draws)
    if profile.startswith("regularized_"):
        assert "triple_gamma_slab2" in fit.parameter_draws
    rho = fit.parameter("triple_gamma_rho_level")
    assert np.all(np.isfinite(rho))
    assert np.all((0.0 < rho) & (rho < 1.0))
    assert fit.meta["shrinkage_update"] == "slice"


def test_factor_triple_gamma_customization_identification_and_roundtrip():
    model = _factor_model()
    compiled = bx.compile_model(model, np.zeros((18, 2)))
    priors = bx.identified_factor_priors(
        compiled,
        profile="regularized_triple_gamma",
        triple_gamma_options={
            "spike_shape": 0.20,
            "tail_shape": 0.30,
            "learn_global": False,
            "global_scale": 0.75,
        },
        reference_channel="reference",
    )
    assert priors.profile == "identified_regularized_triple_gamma"
    assert priors.triple_gamma.spike_shape == pytest.approx(0.20)
    assert priors.triple_gamma.tail_shape == pytest.approx(0.30)
    assert priors.triple_gamma.global_scale == pytest.approx(0.75)
    assert priors.triple_gamma_processes == ("channel.response.level",)
    assert "channel.reference.level" not in priors.triple_gamma.coefficient_scale

    restored = bx.FactorPriors.from_dict(priors.to_dict())
    assert restored.to_dict() == priors.to_dict()


@pytest.mark.parametrize(
    "parameterization", ("centered", "disturbance", "fruehwirth_schnatter")
)
def test_factor_triple_gamma_runs_in_every_factor_parameterization(parameterization):
    model = _factor_model()
    rng = np.random.default_rng(5)
    data = rng.normal(size=(18, 2))
    compiled = bx.compile_model(model, data)
    priors = bx.identified_factor_priors(
        compiled,
        profile="triple_gamma",
        reference_channel="reference",
    )
    fit = bx.fit(
        data,
        model,
        parameterization=parameterization,
        engine="ffbs",
        priors=priors,
        asis=parameterization != "centered",
        mcmc=bx.MCMC(draws=2, warmup=1, chains=1, seed=6),
    )
    name = "triple_gamma.rho.channel.response.level"
    assert name in fit.parameter_draws
    assert np.all(np.isfinite(fit.parameter(name)))
    assert fit.meta["triple_gamma"]


def test_analytic_pc_plot_acf_and_save_api(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(period=4)),
    )
    fit = bx.fit(
        _univariate_data(7),
        model,
        parameterization="fruehwirth_schnatter",
        priors="pc",
        mcmc=bx.MCMC(draws=5, warmup=2, chains=2, seed=8),
    )
    process_path = tmp_path / "nested" / "pc_process.png"
    process_figure, process_axes = fit.plot("process_sd", save=process_path)
    labels = {
        line.get_label() for axis in process_axes for line in axis.lines
    }
    assert "prior (analytic PC)" in labels
    assert process_path.is_file() and process_path.stat().st_size > 0

    acf_path = tmp_path / "acf.png"
    acf_figure, acf_axes = fit.plot(
        "acf",
        parameters=("sd.level", "sigma"),
        max_lag=3,
        save={"path": acf_path, "dpi": 90},
    )
    assert len(acf_axes) == 2
    assert acf_path.is_file() and acf_path.stat().st_size > 0

    forecast_path = tmp_path / "forecast.png"
    forecast_axis = fit.forecast(2, seed=9).plot(save=forecast_path)
    assert forecast_axis.figure is not None
    assert forecast_path.is_file() and forecast_path.stat().st_size > 0
    plt.close(process_figure)
    plt.close(acf_figure)
    plt.close(forecast_axis.figure)


def test_constant_ssvs_allocation_has_undefined_rhat_and_ess():
    values = np.zeros((4, 100))
    assert np.isnan(bx.rhat(values))
    assert np.isnan(bx.ess_bulk(values))

    forced = bx.SSVSPrior(
        level_dynamic_probability=0.0,
        trend_probabilities=(1.0, 0.0, 0.0),
        season_probabilities=(1.0, 0.0, 0.0),
    )
    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(period=4)),
    )
    fit = bx.fit(
        _univariate_data(10),
        model,
        parameterization="fruehwirth_schnatter",
        priors=bx.ssvs_gaussian_priors(period=4, ssvs=forced),
        asis=False,
        mcmc=bx.MCMC(draws=3, warmup=1, chains=2, seed=11),
    )
    row = fit.diagnostics()["parameters"].loc["state_season"]
    assert bool(row["constant"])
    assert np.isnan(row["rhat"])
    assert np.isnan(row["ess_bulk"])
    assert row["diagnostic"] == "constant draw; R-hat and ESS undefined"
    transitions = fit.component_transition_summary()
    assert transitions.loc["seasonal", "status"] == "constant posterior allocation"
