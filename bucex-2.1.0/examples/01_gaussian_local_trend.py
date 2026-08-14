"""Example 1: fit a Gaussian local-linear-trend model with seasonality.

This is the simplest complete bucex workflow: define a model, simulate data,
fit it, check convergence, inspect recovery, forecast, and plot.  All settings
are visible below; change them here when experimenting.

Main result: variance allocation (sigma2 was slightly overestimated whereas slope and seasonal sd were too small).
Good stuff:
All 2values are essentially 1.
ESS values of 980–2450 are excellent.
All four chains overlap without visible drift or sticking.
The posterior predictor closely follows the true predictor.
Its credible interval covers nearly the complete true path.
The positive latent slope is recovered, and the forecast behaves coherently.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import replace
import bucex as bx


# ---------------------------------------------------------------------------
# 1. Settings a user is expected to change
# ---------------------------------------------------------------------------
N_TIME = 120
PERIOD = 12
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
SEED = 101


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
    priors = bx.pc_gaussian_priors(period=PERIOD)
    priors = replace(
        priors,
        sigma2=bx.InverseGammaPrior(
            a=2.0,
            b=truth["sigma"] ** 2,
        ),
    )

    fit = bx.fit(
        data,
        model,
        parameterization="fruehwirth_schnatter",
        engine="ffbs",                    # exact for Gaussian observations
        priors=priors,                      # interpretable shrinkage toward zero
        asis=True,
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
    _, predictor_axis = fit.plot("predictor")
    predictor_axis.plot(dates, simulation.eta, color="black", label="true predictor")
    predictor_axis.legend()
    fit.plot("level_slope")
    fit.plot("process_sd", truths=truth, title="PC prior: Gaussian model")
    fit.plot("traces", parameters=parameters, truths=truth)
    fit.plot("parameter_density", parameters=parameters, truths=truth)
    plt.show()


if __name__ == "__main__":
    main()
