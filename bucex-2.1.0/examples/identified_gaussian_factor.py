"""Play 04: shared warming versus channel-specific persistent change.

The complete predictor can be well identified even when its shared and
idiosyncratic pieces are not. Change BUCEX_FACTOR_IDENTIFICATION to compare
``pure_reference``, ``smooth_only``, ``unrestricted``, and ``loadings_only``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import bucex as bx
from play_config import describe, mcmc, show_figures


N_TIME = 240
BASELINE = slice(0, 60)
PRIOR = "regularized_horseshoe"
IDENTIFICATION = os.getenv(
    "BUCEX_FACTOR_IDENTIFICATION", "pure_reference"
).strip().lower()


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
            "reference": 1.0,
            "amplified": bx.Loading.estimated(1.20),
            "damped": bx.Loading.estimated(0.80),
        },
        description="smooth common warming trajectory",
    )
    return bx.FactorModel(channels, (common,), name="identified Gaussian factor")


def make_priors(compiled) -> bx.FactorPriors:
    options = {
        "pure_reference": dict(
            smooth_factor=True,
            reference_channel="reference",
            fixed_idiosyncratic=(),
        ),
        "smooth_only": dict(
            smooth_factor=True,
            reference_channel=None,
            fixed_idiosyncratic=(),
        ),
        "unrestricted": dict(
            smooth_factor=False,
            reference_channel=None,
            fixed_idiosyncratic=(),
        ),
        "loadings_only": dict(
            smooth_factor=True,
            reference_channel=None,
            fixed_idiosyncratic="all",
        ),
    }
    if IDENTIFICATION not in options:
        raise ValueError(
            f"Unknown identification strategy {IDENTIFICATION!r}; "
            f"choose {tuple(options)}."
        )
    return bx.identified_factor_priors(
        compiled,
        profile=PRIOR,
        **options[IDENTIFICATION],
    )


def main() -> None:
    print(describe())
    print(f"factor identification strategy: {IDENTIFICATION}")
    model = make_model()
    truth = {
        "sd.factor.common.level": 0.0,
        "sd.factor.common.slope": 0.00035,
        "sd.channel.reference.level": 0.0,
        "sd.channel.amplified.level": 0.006,
        "sd.channel.damped.level": 0.002,
        "sigma.reference": 0.10,
        "sigma.amplified": 0.14,
        "sigma.damped": 0.12,
        "loading.common.amplified": 1.35,
        "loading.common.damped": 0.65,
    }
    simulation = bx.simulate(model, N_TIME, truth, seed=40)
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    data = pd.DataFrame(
        simulation.y, index=dates, columns=simulation.channel_names
    )
    compiled = bx.compile_model(model, data)
    priors = make_priors(compiled)

    fit = bx.fit(
        data,
        model,
        parameterization="fruehwirth_schnatter",
        engine="ffbs",
        priors=priors,
        asis=True,
        mcmc=mcmc(41, quick_draws=120, quick_warmup=180),
    )
    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nIDENTIFICATION CONSTRAINTS\n", priors.metadata)
    diagnostics = fit.factor_identification_diagnostics(
        baseline=BASELINE
    )
    print("\nLOADING--DEVIATION DIAGNOSTICS\n", diagnostics.round(3))
    print(
        "\nInterpretation: the fixed reference loading identifies factor sign and "
        "scale. Smooth-factor and pure-reference constraints help dynamically, "
        "but estimated loadings can still compensate for persistent alpha paths "
        "in other channels. A ridge_flag says to interpret lambda*f + alpha more "
        "confidently than either term alone."
    )

    state_index = {
        name: index for index, name in enumerate(compiled.state_names)
    }
    true_factor = simulation.states[
        1:, state_index["factor.common.level"]
    ]
    centered_factor = true_factor - np.mean(true_factor[BASELINE])
    loadings = {
        "reference": 1.0,
        "amplified": truth["loading.common.amplified"],
        "damped": truth["loading.common.damped"],
    }
    true_deviation = {
        channel: simulation.states[
            1:, state_index[f"channel.{channel}.level"]
        ]
        for channel in model.channel_names
    }
    decomposition_truth = {
        "predictor": {
            channel: simulation.eta[:, index]
            for index, channel in enumerate(model.channel_names)
        },
        "shared": {
            channel: loadings[channel] * centered_factor
            for channel in model.channel_names
        },
        "deviation": true_deviation,
    }

    fit.plot(
        "factor_decomposition",
        baseline=BASELINE,
        truth=decomposition_truth,
    )
    density_parameters = [
        "sd.factor.common.slope",
        "sd.channel.amplified.level",
        "sd.channel.damped.level",
        "loading.common.amplified",
        "loading.common.damped",
    ]
    fit.plot(
        "parameter_density",
        parameters=density_parameters,
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
        fit.plot(
            "idiosyncratic_innovations",
            channel=channel,
            truth=alpha,
        )
    fit.plot(
        "identification",
        baseline=BASELINE,
        summaries=("factor_projection", "final_change"),
    )
    show_figures()


if __name__ == "__main__":
    main()
