"""Example 6: pool structural evidence across related Gaussian series.

Every channel has its own level, slope, seasonality, and observation noise.
The hierarchy borrows strength only through shared structural-selection
probabilities, normal-slab scales, or both. It never forces the series to have
the same path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 240
PERIOD = 12
POOL = "selection"       # try "slab" or "both"
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
SEED = 601
SHOW_PLOTS = False
FIGURE_DIR = Path("figures/06_hierarchical_gaussian")


def simulate_data() -> pd.DataFrame:
    """Create four related, but deliberately non-identical, series."""

    rng = np.random.default_rng(SEED)
    time = np.arange(N_TIME, dtype=float)
    month = np.arange(N_TIME) % PERIOD
    seasonal = np.array(
        [-0.22, -0.15, -0.04, 0.05, 0.12, 0.17,
         0.19, 0.12, 0.04, -0.05, -0.12, -0.11]
    )[month]

    dynamic_level = np.cumsum(rng.normal(0.0, 0.015, N_TIME))
    dynamic_slope = 0.0015 + np.cumsum(rng.normal(0.0, 0.00008, N_TIME))
    changing_rate = np.cumsum(dynamic_slope)
    changing_season = seasonal * (0.7 + 0.002 * time)

    eta = np.column_stack(
        (
            0.0025 * time + seasonal,
            0.0018 * time + dynamic_level + 0.8 * seasonal,
            changing_rate + 0.5 * seasonal,
            0.0020 * time + changing_season,
        )
    )
    observed = eta + rng.normal(
        0.0,
        np.array([0.16, 0.18, 0.17, 0.16])[None, :],
        eta.shape,
    )
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    return pd.DataFrame(
        observed,
        index=dates,
        columns=("stable", "level_changes", "rate_changes", "season_changes"),
    )


def make_model(names: tuple[str, ...]) -> bx.MultiSeriesModel:
    return bx.MultiSeriesModel(
        channels=tuple(
            bx.Channel(
                name,
                bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
            )
            for name in names
        ),
        name="related Gaussian summaries",
    )


def main() -> None:
    data = simulate_data()
    model = make_model(tuple(data.columns))

    # Selection uses shared Dirichlet probabilities. Slab pooling uses a
    # shared multiplier with a half-t(df=4) hyperprior. Monthly seasonality is
    # fixed or dynamic, never absent.
    prior = bx.HierarchicalPrior(
        pool=POOL,
        season_states=("fixed", "dynamic"),
        slab_df=4.0,
    )
    fit = bx.fit(
        data,
        model,
        priors=prior,
        parameterization="fruehwirth_schnatter",
        engine="auto",                 # exact FFBS: every channel is Gaussian
        asis=False,
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED + 1,
            progress=True,
        ),
    )

    scientific = [
        name
        for name in fit.parameter_draws
        if name.startswith(("sd.channel.", "sigma.", "hierarchy."))
    ]
    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nMCMC DIAGNOSTICS\n", fit.diagnostics()["parameters"].loc[scientific].round(4))
    print("\nCHANNEL ALLOCATIONS\n", fit.component_probabilities().round(3))
    print("\nPOPULATION PROBABILITIES\n", fit.hierarchical_probabilities().round(3))
    print("\nPOOLED SLAB MULTIPLIERS\n", fit.hierarchical_slab_summary().round(3))
    print("\nALLOCATION SWITCHING\n", fit.component_transition_summary().round(3))

    print("\nCOMPLETE-PREDICTOR RATES PER DECADE")
    for channel in model.channel_names:
        print(channel, fit.channel_rate_summary(channel))

    forecast = fit.forecast(12, draws=1_000, seed=SEED + 2)
    print("\n12-MONTH FORECAST\n", forecast.summary().head(12).round(3))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for channel in model.channel_names:
        fit.plot("channel", channel=channel, save=FIGURE_DIR / f"{channel}.png")
    fit.plot("component_probabilities", save=FIGURE_DIR / "allocations.png")
    fit.plot("hierarchy", save=FIGURE_DIR / "hierarchy.png")
    fit.plot("process_sd", save=FIGURE_DIR / "process_sds.png")
    fit.plot("traces", parameters=scientific, save=FIGURE_DIR / "traces.png")
    fit.plot("acf", parameters=scientific, max_lag=50, save=FIGURE_DIR / "acf.png")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
