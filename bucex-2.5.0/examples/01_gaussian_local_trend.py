"""Example 1: fit a Gaussian local-linear-trend model with seasonality.

This is the simplest complete bucex workflow: define a model, simulate data,
fit it, check convergence, inspect recovery, forecast, and plot.  All settings
are visible below; change them here when experimenting.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


# ---------------------------------------------------------------------------
# 1. Settings a user is expected to change
# ---------------------------------------------------------------------------
N_TIME = 120
PERIOD = 12
PRIOR = "normal"       # try "pc", "triple_gamma", or "ssvs" afterwards
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
SEED = 101
FIGURE_DIR = Path("figures/01_gaussian_local_trend")


def main() -> None:
    # -----------------------------------------------------------------------
    # 2. Define the structural model
    # -----------------------------------------------------------------------
    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
        name="simulated monthly Gaussian series",
    )

    # Innovation SDs are changes per observation interval.  In particular,
    # sd.slope is the SD of changes in the slope, not the slope itself.
    truth = {
        "sd.level": 0.020,
        "sd.slope": 0.00015,
        "sd.seasonal": 0.015,
        "sigma": 0.30,
    }

    compiled = bx.compile_model(model, np.zeros(N_TIME))
    initial_state = np.zeros(compiled.state_dim)
    initial_state[compiled.state_names.index("slope")] = 0.008
    simulation = bx.simulate(
        model,
        N_TIME,
        truth,
        initial_state=initial_state,
        seed=SEED,
    )
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    data = pd.Series(simulation.y, index=dates, name="temperature")

    # -----------------------------------------------------------------------
    # 3. Fit the model
    # -----------------------------------------------------------------------
    fit = bx.fit(
        data,
        model,
        parameterization="fruehwirth_schnatter",
        engine="ffbs",                    # exact for Gaussian observations
        priors=PRIOR,
        asis=PRIOR != "ssvs",
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED + 1,
            progress=True,
        ),
    )

    # -----------------------------------------------------------------------
    # 4. Check the sampler before interpreting the posterior
    # -----------------------------------------------------------------------
    parameters = ("sd.level", "sd.slope", "sd.seasonal", "sigma")
    diagnostics = fit.diagnostics()["parameters"].loc[list(parameters)]
    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nCONVERGENCE DIAGNOSTICS\n", diagnostics.round(4))

    rows = []
    summaries = fit.static_summary()
    for parameter in parameters:
        row = summaries[parameter]
        rows.append(
            {
                "parameter": parameter,
                "truth": truth[parameter],
                "median": row["median"],
                "lower90": row["lower"],
                "upper90": row["upper"],
            }
        )
    print(
        "\nPARAMETER RECOVERY\n",
        pd.DataFrame(rows).set_index("parameter").round(5),
    )

    forecast = fit.forecast(12, draws=1_000, seed=SEED + 2)
    print("\n12-STEP FORECAST\n", forecast.summary().round(3))

    # -----------------------------------------------------------------------
    # 5. Plot the fit and the MCMC diagnostics
    # -----------------------------------------------------------------------
    predictor_figure, predictor_axis = fit.plot("predictor")
    predictor_axis.plot(dates, simulation.eta, color="black", label="true predictor")
    predictor_axis.legend()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    predictor_figure.savefig(FIGURE_DIR / "predictor.png", bbox_inches="tight")
    fit.plot("level_slope", save=FIGURE_DIR / "level_slope.png")
    fit.plot(
        "process_sd",
        truths=truth,
        title=f"{PRIOR.replace('_', ' ')} prior: Gaussian model",
        save=FIGURE_DIR / "prior_posterior_sd.png",
    )
    fit.plot(
        "traces", parameters=parameters, truths=truth,
        save=FIGURE_DIR / "traces.png",
    )
    fit.plot(
        "acf", parameters=parameters, max_lag=50,
        save=FIGURE_DIR / "acf.png",
    )
    fit.plot(
        "parameter_density", parameters=parameters, truths=truth,
        save=FIGURE_DIR / "densities.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
