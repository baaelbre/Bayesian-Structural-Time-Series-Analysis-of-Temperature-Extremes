"""Example 6: an identified Gaussian dynamic-factor model.

Three series share one smooth local-linear trend.  The reference channel fixes
the factor scale and has no idiosyncratic local-level shock.  The other two
channels may deviate persistently from the common factor.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 240
DRAWS = 1_000
WARMUP = 1_500
CHAINS = 4
SEED = 601
BASELINE = slice(0, 60)


def make_model() -> bx.FactorModel:
    channels = tuple(
        bx.Channel(name, bx.Gaussian(), (bx.LocalLevel(),))
        for name in ("reference", "amplified", "damped")
    )
    common = bx.Factor(
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
            "reference": 1.0,                       # sign and scale anchor
            "amplified": bx.Loading.estimated(1.2),
            "damped": bx.Loading.estimated(0.8),
        },
    )
    return bx.FactorModel(channels, (common,), name="Gaussian warming factor")


def main() -> None:
    model = make_model()
    truth = {
        "sd.factor.common.level": 0.0,
        "sd.factor.common.slope": 0.00025,
        "sd.channel.reference.level": 0.0,
        "sd.channel.amplified.level": 0.006,
        "sd.channel.damped.level": 0.002,
        "sigma.reference": 0.10,
        "sigma.amplified": 0.14,
        "sigma.damped": 0.12,
        "loading.common.amplified": 1.35,
        "loading.common.damped": 0.65,
    }

    compiled_for_simulation = bx.compile_model(
        model,
        np.zeros((N_TIME, len(model.channel_names))),
    )
    initial_state = np.zeros(compiled_for_simulation.state_dim)
    initial_state[
        compiled_for_simulation.state_names.index("factor.common.slope")
    ] = 0.004
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
        smooth_factor=True,                 # no direct common-level shock
        reference_channel="reference",     # no reference idiosyncratic shock
    )

    fit = bx.fit(
        data,
        model,
        parameterization="fruehwirth_schnatter",
        engine="ffbs",
        priors=priors,
        asis=True,
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED + 1,
            progress=True,
        ),
    )

    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nIDENTIFICATION CONSTRAINTS\n", priors.metadata)
    print("\nSCALAR DIAGNOSTICS\n", fit.diagnostics()["parameters"].round(4))
    print(
        "\nLOADING--DEVIATION IDENTIFICATION\n",
        fit.factor_identification_diagnostics(baseline=BASELINE).round(3),
    )

    state_index = {
        name: index for index, name in enumerate(compiled.state_names)
    }
    true_factor = simulation.states[
        1:, state_index["factor.common.level"]
    ]
    true_factor = true_factor - np.mean(true_factor[BASELINE])
    true_deviation = {
        channel: simulation.states[
            1:, state_index[f"channel.{channel}.level"]
        ]
        for channel in model.channel_names
    }
    loadings = {
        "reference": 1.0,
        "amplified": truth["loading.common.amplified"],
        "damped": truth["loading.common.damped"],
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
                channel: loadings[channel] * true_factor
                for channel in model.channel_names
            },
            "deviation": true_deviation,
        },
    )
    fit.plot(
        "parameter_density",
        parameters=(
            "sd.factor.common.slope",
            "sd.channel.amplified.level",
            "sd.channel.damped.level",
            "loading.common.amplified",
            "loading.common.damped",
        ),
        truths=truth,
    )
    fit.plot("traces", truths=truth)

    for channel in ("amplified", "damped"):
        alpha = true_deviation[channel]
        fit.plot(
            "loading_deviation",
            channel=channel,
            summary="final_change",
            baseline=BASELINE,
            truth=(loadings[channel], alpha[-1] - alpha[0]),
        )
        fit.plot("idiosyncratic_innovations", channel=channel, truth=alpha)

    fit.plot(
        "identification",
        baseline=BASELINE,
        summaries=("factor_projection", "final_change"),
    )
    plt.show()


if __name__ == "__main__":
    main()
