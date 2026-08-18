"""Example 14: screen a mixed hierarchy with Laplace, then run exact PGAS.

The first fit is deliberately exploratory: its GEV channels use an iterated
Laplace approximation.  The second fit exports the best complete posterior
draw from that screen and uses it to initialise an exact-invariant PGAS run.
Initialisation changes wall time and burn-in behaviour, never the PGAS target.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 120
PERIOD = 12
SCREEN_DRAWS = 200
SCREEN_WARMUP = 200
EXACT_DRAWS = 500
EXACT_WARMUP = 500
PARTICLES = 256
CHANNEL_WORKERS = 2
SEED = 1_401
RESULT_DIR = Path("results/14_laplace_then_pgas")


def model() -> bx.MultiSeriesModel:
    components = (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD))
    return bx.MultiSeriesModel(
        channels=(
            bx.Channel("mean", bx.Gaussian(), components),
            bx.Channel("maximum", bx.GEV(), components, tail="upper"),
        ),
        name="Laplace to PGAS demonstration",
    )


def simulate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    time = np.arange(N_TIME, dtype=float)
    season = 0.25 * np.sin(2.0 * np.pi * time / PERIOD)
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    return pd.DataFrame(
        {
            "mean": 8.0 + 0.010 * time + season + rng.normal(0, 0.3, N_TIME),
            "maximum": (
                12.0 + 0.014 * time + 0.7 * season
                + rng.gumbel(0, 0.45, N_TIME)
            ),
        },
        index=dates,
    )


def main() -> None:
    data = simulate()
    prior = bx.HierarchicalPrior(pool="both", model_space="joint_trend")
    execution = bx.HierarchicalSampler(
        initializer="laplace",
        channel_workers=CHANNEL_WORKERS,
    )

    print("\n1. EXPLORATORY HIERARCHICAL LAPLACE SCREEN")
    screen = bx.fit(
        data,
        model(),
        priors=prior,
        engine="laplace",
        parameterization="fs",
        asis=False,
        mcmc=bx.MCMC(
            draws=SCREEN_DRAWS,
            warmup=SCREEN_WARMUP,
            chains=1,
            seed=SEED + 1,
            progress=True,
        ),
        hierarchical_sampler=execution,
    )
    print(screen.plan)
    print("Exact posterior target?", screen.plan.targets_exact_posterior)
    print(screen.hierarchical_trend_model_probabilities().round(3))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    screen.save(RESULT_DIR / "screen.bucex")

    print("\n2. EXACT PGAS, INITIALISED FROM THE SCREEN")
    exact = bx.fit(
        data,
        model(),
        priors=prior,
        engine="pgas",
        parameterization="fs",
        asis=False,
        init=screen,  # equivalent to init=screen.warm_start()
        mcmc=bx.MCMC(
            draws=EXACT_DRAWS,
            warmup=EXACT_WARMUP,
            chains=2,
            seed=SEED + 2,
            progress=True,
        ),
        particles=bx.Particles(n=PARTICLES, proposal="guided"),
        hierarchical_sampler=execution,
    )
    print(exact.plan)
    print("Exact posterior target?", exact.plan.targets_exact_posterior)
    print("External warm start:", exact.meta["warm_start_source"])
    print("Trend classes:\n", exact.hierarchical_trend_model_probabilities().round(3))
    print("Particle diagnostics:\n", exact.diagnostics()["engine"])
    exact.save(RESULT_DIR / "exact_pgas.bucex")


if __name__ == "__main__":
    main()
