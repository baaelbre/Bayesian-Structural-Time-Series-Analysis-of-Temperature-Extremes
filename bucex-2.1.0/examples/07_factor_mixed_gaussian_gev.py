"""Example 7: a Gaussian mean and GEV maximum sharing one factor.

This demonstrates the proposed bulk/extreme dynamic-factor analysis.  The
Gaussian and GEV likelihoods are conditionally independent given the common
factor and their channel-specific states.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 120
DRAWS = 500
WARMUP = 750
CHAINS = 4
N_PARTICLES = 256
SEED = 701
BASELINE = slice(0, 36)


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
    model = make_model()
    truth = {
        "sd.factor.climate.level": 0.0,
        "sd.factor.climate.slope": 0.00015,
        "sd.channel.mean.level": 0.0,
        "sd.channel.mean.seasonal": 0.012,
        "sd.channel.maximum.level": 0.008,
        "sd.channel.maximum.seasonal": 0.016,
        "sigma.mean": 0.25,
        "sigma.maximum": 0.45,
        "xi.maximum": -0.08,
        "loading.climate.maximum": 0.90,
    }

    simulation_compiler = bx.compile_model(
        model,
        np.zeros((N_TIME, len(model.channel_names))),
    )
    initial_state = np.zeros(simulation_compiler.state_dim)
    initial_state[
        simulation_compiler.state_names.index("factor.climate.slope")
    ] = 0.006
    simulation = bx.simulate(
        model,
        N_TIME,
        truth,
        initial_state=initial_state,
        seed=SEED,
    )
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    data = pd.DataFrame(
        simulation.y,
        index=dates,
        columns=simulation.channel_names,
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
        parameterization="fruehwirth_schnatter",
        engine="pgas",
        priors=priors,
        asis=True,
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED + 1,
            progress=True,
        ),
        particles=bx.Particles(n=N_PARTICLES, proposal="guided"),
    )

    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nSCALAR DIAGNOSTICS\n", fit.diagnostics()["parameters"].round(4))
    print("\nPARTICLE DIAGNOSTICS\n", fit.diagnostics()["engine"])
    print(
        "\nIDENTIFICATION DIAGNOSTICS\n",
        fit.factor_identification_diagnostics(baseline=BASELINE).round(3),
    )

    state_index = {
        name: index for index, name in enumerate(compiled.state_names)
    }
    factor = simulation.states[1:, state_index["factor.climate.level"]]
    factor = factor - np.mean(factor[BASELINE])
    deviations = {
        channel: simulation.states[
            1:, state_index[f"channel.{channel}.level"]
        ]
        for channel in model.channel_names
    }
    fit.plot(
        "factor_decomposition",
        baseline=BASELINE,
        truth={
            "predictor": {
                channel: simulation.eta[:, index]
                for index, channel in enumerate(model.channel_names)
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
    plt.show()


if __name__ == "__main__":
    main()
