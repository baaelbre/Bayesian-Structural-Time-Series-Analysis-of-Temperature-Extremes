"""Example 7: pool structure across Gaussian and GEV summaries.

The channels retain different likelihoods and different latent paths. Because
at least one likelihood is non-Gaussian, the joint state update uses guided
disturbance PGAS. This script is the compact regression test for the v2.4
singular-support hotfix and estimated initial levels/slopes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 120
PERIOD = 12
POOL = "selection"
DRAWS = 300
WARMUP = 300
CHAINS = 2
N_PARTICLES = 256
SEED = 701
SHOW_PLOTS = False
FIGURE_DIR = Path("figures/07_hierarchical_mixed")


def main() -> None:
    rng = np.random.default_rng(SEED)
    time = np.arange(N_TIME, dtype=float)
    month = np.arange(N_TIME) % PERIOD
    seasonal = 0.18 * np.sin(2.0 * np.pi * month / PERIOD)
    mean_eta = 8.0 + 0.010 * time + seasonal
    maximum_eta = 12.0 + 0.014 * time + 0.7 * seasonal
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    data = pd.DataFrame(
        {
            "mean": mean_eta + rng.normal(0.0, 0.25, N_TIME),
            "maximum": maximum_eta + rng.gumbel(0.0, 0.45, N_TIME),
        },
        index=dates,
    )
    model = bx.MultiSeriesModel(
        channels=(
            bx.Channel(
                "mean",
                bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
            ),
            bx.Channel(
                "maximum",
                bx.GEV(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
                tail="upper",
            ),
        ),
        name="Gaussian mean and GEV maximum",
    )

    fit = bx.fit(
        data,
        model,
        priors=bx.HierarchicalPrior(pool=POOL),
        parameterization="fruehwirth_schnatter",
        engine="pgas",
        asis=False,
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED + 1,
            progress=True,
        ),
        particles=bx.Particles(n=N_PARTICLES, proposal="guided"),
    )

    scientific = [
        name
        for name in fit.parameter_draws
        if name.startswith(
            ("sd.channel.", "sigma.", "xi.", "initial.level.", "initial.slope.")
        )
    ]
    diagnostics = fit.diagnostics()
    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nPARAMETERS\n", diagnostics["parameters"].loc[scientific].round(4))
    print("\nPARTICLES\n", diagnostics["engine"])
    print("\nRESTORATIONS\n", fit.metadata.get("restored_iterations_by_chain"))
    print("\nALLOCATIONS\n", fit.component_probabilities().round(3))
    print("\nPOPULATION PROBABILITIES\n", fit.hierarchical_probabilities().round(3))
    summaries = fit.static_summary()
    initial_names = (
        "initial.channel.mean.level",
        "initial.channel.mean.slope",
        "initial.channel.maximum.level",
        "initial.channel.maximum.slope",
    )
    print(
        "\nINITIAL STATES ARE ESTIMATED\n",
        {name: summaries[name] for name in initial_names},
    )

    # The publication run should have no restored iterations and should be
    # stable when N_PARTICLES is increased.
    if fit.metadata.get("restored_iterations", 0):
        raise RuntimeError("Guided PGAS restored an iteration; inspect this run.")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for channel in model.channel_names:
        fit.plot("channel", channel=channel, save=FIGURE_DIR / f"{channel}.png")
    fit.plot("component_probabilities", save=FIGURE_DIR / "allocations.png")
    fit.plot("hierarchy", save=FIGURE_DIR / "hierarchy.png")
    fit.plot("traces", parameters=scientific, save=FIGURE_DIR / "traces.png")
    fit.plot("acf", parameters=scientific, max_lag=50, save=FIGURE_DIR / "acf.png")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
