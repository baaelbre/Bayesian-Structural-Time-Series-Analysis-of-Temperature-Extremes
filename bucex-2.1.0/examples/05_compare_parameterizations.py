"""Example 5: compare centered, disturbance, and FS parameterizations.

The scientific prior is matched across parameterizations: every innovation
SD has the same half-Normal prior and sigma**2 has the same inverse-gamma
prior.  Therefore converged posteriors should agree; only sampling efficiency
and runtime should differ.
"""
from __future__ import annotations

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 180
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
SEED = 501
PARAMETERIZATIONS = ("centered", "disturbance", "fruehwirth_schnatter")

PROCESS_SCALES = {
    "level": 0.03,
    "slope": 0.0002,
    "seasonal": 0.03,
}


def main() -> None:
    # Match the general state-space initial priors to the FS prior profile.
    model = bx.Model(
        bx.Gaussian(),
        (
            bx.LocalLinearTrend(
                initial_level_sd=np.sqrt(10.0),
                initial_slope_sd=0.005,
            ),
            bx.DummySeasonal(12, initial_sd=np.sqrt(5.0)),
        ),
        name="parameterization comparison",
    )
    truth = {
        "sd.level": 0.018,
        "sd.slope": 0.00015,
        "sd.seasonal": 0.012,
        "sigma": 0.25,
    }
    compiled = bx.compile_model(model, np.zeros(N_TIME))
    initial_state = np.zeros(compiled.state_dim)
    initial_state[compiled.state_names.index("slope")] = 0.006
    simulation = bx.simulate(
        model,
        N_TIME,
        truth,
        initial_state=initial_state,
        seed=SEED,
    )

    fs_prior = bx.normal_gaussian_priors(
        period=12,
        level_sd=PROCESS_SCALES["level"],
        trend_sd=PROCESS_SCALES["slope"],
        season_sd=PROCESS_SCALES["seasonal"],
    )
    general_prior = bx.Priors(
        process={
            name: bx.HalfNormalSD(scale)
            for name, scale in PROCESS_SCALES.items()
        },
        observation_sd=bx.InverseGammaVariance(shape=2.0, scale=1.0),
        profile="matched_half_normal",
    )

    parameters = ["sd.level", "sd.slope", "sd.seasonal", "sigma"]
    rows = []
    fits = {}
    for parameterization in PARAMETERIZATIONS:
        print(f"\n{'=' * 72}\n{parameterization}\n{'=' * 72}")
        prior = fs_prior if parameterization == "fruehwirth_schnatter" else general_prior
        started = perf_counter()
        fit = bx.fit(
            simulation.y,
            model,
            parameterization=parameterization,
            engine="ffbs",
            priors=prior,
            asis=True,
            mcmc=bx.MCMC(
                draws=DRAWS,
                warmup=WARMUP,
                chains=CHAINS,
                seed=SEED + 1,
                progress=True,
            ),
        )
        elapsed = perf_counter() - started
        fits[parameterization] = fit
        diagnostics = fit.diagnostics()["parameters"]
        summaries = fit.static_summary()
        for parameter in parameters:
            posterior = summaries[parameter]
            rows.append(
                {
                    "parameterization": parameterization,
                    "parameter": parameter,
                    "truth": truth[parameter],
                    "median": posterior["median"],
                    "lower90": posterior["lower"],
                    "upper90": posterior["upper"],
                    "rhat": diagnostics.loc[parameter, "rhat"],
                    "ess_bulk": diagnostics.loc[parameter, "ess_bulk"],
                    "seconds": elapsed,
                    "ess_per_second": diagnostics.loc[parameter, "ess_bulk"] / elapsed,
                }
            )
        print(fit.plan)

    comparison = pd.DataFrame(rows).set_index(
        ["parameterization", "parameter"]
    )
    print("\nPOSTERIOR AND EFFICIENCY COMPARISON\n", comparison.round(5).to_string())

    figure, axes = plt.subplots(
        len(PARAMETERIZATIONS), 1, figsize=(10, 9), sharex=True
    )
    for axis, parameterization in zip(axes, PARAMETERIZATIONS):
        fits[parameterization].plot("predictor", ax=axis)
        axis.plot(simulation.eta, color="black", linewidth=1.0, label="truth")
        axis.set_title(parameterization)
        axis.legend()
    figure.tight_layout()

    for parameterization, fit in fits.items():
        trace_figure, _ = fit.plot(
            "traces",
            parameters=parameters,
            truths=truth,
        )
        trace_figure.suptitle(parameterization, y=0.995)
        trace_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    plt.show()


if __name__ == "__main__":
    main()
