"""Play 02: compare parameterizations on exactly the same Gaussian data."""
from __future__ import annotations

from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd

import bucex as bx
from play_config import describe, mcmc, show_figures


PARAMETERIZATIONS = (
    "centered",
    "disturbance",
    "fruehwirth_schnatter",
)
PRIOR = "normal"
ASIS = True
N_TIME = 120


def main() -> None:
    print(describe())
    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(12)),
        name="parameterization comparison",
    )
    truth = {
        "sd.level": 0.02,
        "sd.slope": 0.0015,
        "sd.seasonal": 0.015,
        "sigma": 0.25,
    }
    data = bx.simulate(model, N_TIME, truth, seed=20).y

    fits = {}
    rows = []
    for index, parameterization in enumerate(PARAMETERIZATIONS):
        started = perf_counter()
        fit = bx.fit(
            data,
            model,
            parameterization=parameterization,
            engine="ffbs",
            priors=PRIOR,
            asis=ASIS,
            mcmc=mcmc(21 + index),
        )
        elapsed = perf_counter() - started
        fits[parameterization] = fit
        diagnostics = fit.diagnostics()["parameters"]
        for parameter in ("sd.level", "sd.slope", "sd.seasonal", "sigma"):
            posterior = fit.static_summary()[parameter]
            rows.append(
                {
                    "parameterization": parameterization,
                    "parameter": parameter,
                    "median": posterior["median"],
                    "lower": posterior["lower"],
                    "upper": posterior["upper"],
                    "rhat": diagnostics.loc[parameter, "rhat"],
                    "ess_bulk": diagnostics.loc[parameter, "ess_bulk"],
                    "seconds": elapsed,
                }
            )
        print(f"\n{parameterization}\n{fit.plan}")

    comparison = pd.DataFrame(rows).set_index(
        ["parameterization", "parameter"]
    )
    print("\nPOSTERIOR AND SAMPLING COMPARISON\n", comparison.round(4))
    print(
        "\nThe posterior should agree across parameterizations after convergence; "
        "R-hat, ESS, trace behavior, and runtime tell us which geometry is easier."
    )

    figure, axes = plt.subplots(
        len(PARAMETERIZATIONS), 1, figsize=(10, 8), sharex=True
    )
    for axis, parameterization in zip(axes, PARAMETERIZATIONS):
        fits[parameterization].plot("state", state="level", ax=axis)
        axis.set_title(parameterization)
    figure.tight_layout()
    show_figures()


if __name__ == "__main__":
    main()
