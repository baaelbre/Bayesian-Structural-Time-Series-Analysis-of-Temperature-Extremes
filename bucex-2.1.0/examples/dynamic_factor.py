"""Play 05: one common factor shared by Gaussian and GEV channels."""
from __future__ import annotations

import numpy as np
import pandas as pd

import bucex as bx
from play_config import describe, mcmc, particles, show_figures


N_TIME = 72
BASELINE = slice(0, 24)


def make_model() -> bx.FactorModel:
    return bx.FactorModel(
        channels=(
            bx.Channel(
                "mean",
                bx.Gaussian(),
                (bx.LocalLevel(), bx.DummySeasonal(12)),
            ),
            bx.Channel(
                "maximum",
                bx.GEV(),
                (bx.LocalLevel(), bx.DummySeasonal(12)),
            ),
        ),
        factors=(
            bx.Factor(
                "climate",
                (
                    bx.LocalLinearTrend(
                        initial_level=0.0,
                        initial_level_sd=0.0,
                        initial_slope=0.0,
                        initial_slope_sd=0.0,
                    ),
                ),
                {
                    "mean": 1.0,
                    "maximum": bx.Loading.estimated(0.9),
                },
            ),
        ),
        name="mixed Gaussian/GEV climate factor",
    )


def main() -> None:
    print(describe())
    model = make_model()
    truth = {
        "sd.factor.climate.level": 0.0,
        "sd.factor.climate.slope": 0.0015,
        "sd.channel.mean.level": 0.0,
        "sd.channel.mean.seasonal": 0.015,
        "sd.channel.maximum.level": 0.010,
        "sd.channel.maximum.seasonal": 0.020,
        "sigma.mean": 0.25,
        "sigma.maximum": 0.45,
        "xi.maximum": -0.08,
        "loading.climate.maximum": 0.90,
    }
    simulation = bx.simulate(model, N_TIME, truth, seed=50)
    dates = pd.date_range("2018-01-01", periods=N_TIME, freq="MS")
    data = pd.DataFrame(
        simulation.y, index=dates, columns=simulation.channel_names
    )
    compiled = bx.compile_model(model, data)
    priors = bx.identified_factor_priors(
        compiled,
        profile="regularized_horseshoe",
        smooth_factor=True,
        reference_channel="mean",
    )

    fit = bx.fit(
        data,
        model,
        engine="pgas",
        parameterization="fruehwirth_schnatter",
        priors=priors,
        asis=True,
        mcmc=mcmc(51, quick_draws=80, quick_warmup=100),
        particles=particles(),
    )
    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nSTATIC PARAMETERS\n", pd.DataFrame(fit.static_summary()).T.round(4))
    print(
        "\nIDENTIFICATION DIAGNOSTICS\n",
        fit.factor_identification_diagnostics(baseline=BASELINE).round(3),
    )
    print("\nPGAS DIAGNOSTICS\n", fit.diagnostics()["engine"])

    index = {name: i for i, name in enumerate(compiled.state_names)}
    factor = simulation.states[1:, index["factor.climate.level"]]
    factor = factor - np.mean(factor[BASELINE])
    deviations = {
        channel: simulation.states[
            1:, index[f"channel.{channel}.level"]
        ]
        for channel in model.channel_names
    }
    fit.plot(
        "factor_decomposition",
        baseline=BASELINE,
        truth={
            "predictor": {
                channel: simulation.eta[:, j]
                for j, channel in enumerate(model.channel_names)
            },
            "shared": {
                "mean": factor,
                "maximum": truth["loading.climate.maximum"] * factor,
            },
            "deviation": deviations,
        },
    )
    fit.plot(
        "parameter_density",
        parameters=(
            "sd.factor.climate.slope",
            "sd.channel.maximum.level",
            "loading.climate.maximum",
            "sigma.mean",
            "sigma.maximum",
            "xi.maximum",
        ),
        truths=truth,
    )
    fit.plot("traces", truths=truth)
    fit.plot(
        "loading_deviation",
        channel="maximum",
        summary="factor_projection",
        baseline=BASELINE,
    )
    fit.plot("identification", baseline=BASELINE)
    fit.plot(
        "idiosyncratic_innovations",
        channel="maximum",
        truth=deviations["maximum"],
    )
    show_figures()


if __name__ == "__main__":
    main()
