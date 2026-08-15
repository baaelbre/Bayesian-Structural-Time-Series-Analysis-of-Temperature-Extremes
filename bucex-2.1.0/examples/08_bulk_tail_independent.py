"""Example 8: parallel Gaussian bulk and GEV tail analyses.

This convenience workflow aligns data, configuration, summaries, and plots,
but it does not create a joint likelihood.  Use Example 7 when the bulk and
tail should share a latent warming factor.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 120
DRAWS = 500
WARMUP = 750
CHAINS = 4
N_PARTICLES = 256
SEED = 801
FIGURE_DIR = Path("figures/08_bulk_tail")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    time = np.arange(N_TIME)
    common = 8.0 + 0.012 * time + 0.7 * np.sin(2.0 * np.pi * time / 12.0)
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    bulk = pd.Series(
        common + rng.normal(scale=0.30, size=N_TIME),
        index=dates,
        name="monthly mean",
    )
    tail = pd.Series(
        common + rng.gumbel(scale=0.50, size=N_TIME),
        index=dates,
        name="monthly maximum",
    )

    pair = bx.fit_bulk_tail(
        bulk,
        tail,
        period=12,
        bulk_priors="pc",
        tail_priors="pc",
        bulk_mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED + 1,
            progress=True,
        ),
        tail_mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED + 2,
            progress=True,
        ),
        tail_engine="pgas",
        parameterization="fruehwirth_schnatter",
        asis=True,
        particles=bx.Particles(n=N_PARTICLES, proposal="guided"),
    )

    print("\nGAUSSIAN PLAN\n", pair.bulk.plan)
    print("\nGEV PLAN\n", pair.tail.plan)
    print("\nGAUSSIAN DIAGNOSTICS\n", pair.bulk.diagnostics()["parameters"].round(4))
    print("\nGEV DIAGNOSTICS\n", pair.tail.diagnostics()["parameters"].round(4))
    print("\nGEV PARTICLE DIAGNOSTICS\n", pair.tail.diagnostics()["engine"])
    print("\nGEV RESTORED FRACTION\n", pair.tail.metadata.get("restored_fraction", 0.0))

    pair.plot(save=FIGURE_DIR / "bulk_tail_levels.png")
    pair.bulk.plot("predictor", save=FIGURE_DIR / "bulk_predictor.png")
    pair.tail.plot("predictor", save=FIGURE_DIR / "tail_predictor.png")
    pair.bulk.plot("traces", save=FIGURE_DIR / "bulk_traces.png")
    pair.bulk.plot("acf", save=FIGURE_DIR / "bulk_acf.png")
    pair.tail.plot(
        "traces",
        parameters=("sd.level", "sd.slope", "sd.seasonal", "sigma", "xi"),
        save=FIGURE_DIR / "tail_traces.png",
    )
    pair.tail.plot("acf", save=FIGURE_DIR / "tail_acf.png")
    pair.tail.plot("endpoint", save=FIGURE_DIR / "tail_endpoint.png")
    plt.show()


if __name__ == "__main__":
    main()
