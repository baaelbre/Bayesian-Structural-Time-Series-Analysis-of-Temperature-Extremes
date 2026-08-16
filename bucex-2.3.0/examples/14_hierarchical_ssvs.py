"""Example 14: hierarchical SSVS for several related Gaussian series.

This is the smallest complete multiseries workflow.  The series do not share a
latent trajectory.  Instead, they jointly learn how often a level, trend, or
seasonal component is zero/fixed/dynamic and how wide the dynamic slabs should
be.  Use a FactorModel (Examples 6, 10, and 16) when the scientific target is a
common warming path and heterogeneous loadings.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 180
PERIOD = 12
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
SEED = 1_401
SAVE_FIT = False
SHOW_PLOTS = False
FIGURE_DIR = Path("figures/14_hierarchical_ssvs")


def simulate_related_series() -> tuple[pd.DataFrame, dict[str, str]]:
    """Create four series with deliberately different structural dynamics."""

    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    t = np.arange(N_TIME, dtype=float)
    phase = np.arange(N_TIME) % PERIOD
    seasonal_pattern = np.array(
        [-0.20, -0.12, -0.04, 0.04, 0.10, 0.14,
         0.16, 0.11, 0.04, -0.04, -0.10, -0.09]
    )

    # Fixed level + fixed slope + fixed seasonality.
    stable = 0.10 + 0.0025 * t + seasonal_pattern[phase]

    # Dynamic level, fixed slope, and fixed seasonality.
    level_change = np.cumsum(rng.normal(0.0, 0.018, N_TIME))
    adaptive = -0.15 + 0.0015 * t + level_change + 0.7 * seasonal_pattern[phase]

    # Dynamic slope with no level shock; the rate itself evolves.
    changing_slope = 0.001 + np.cumsum(rng.normal(0.0, 0.00010, N_TIME))
    changing_rate = np.cumsum(changing_slope) + 0.4 * seasonal_pattern[phase]

    # Dynamic level with no seasonal component.
    unseasonal = 0.20 + np.cumsum(rng.normal(0.0, 0.012, N_TIME))

    latent = np.column_stack((stable, adaptive, changing_rate, unseasonal))
    observed = latent + rng.normal(0.0, (0.16, 0.18, 0.17, 0.15), latent.shape)
    names = ("stable", "adaptive", "changing_rate", "unseasonal")
    truth = {
        "stable": "fixed level, fixed trend, fixed season",
        "adaptive": "dynamic level, fixed trend, fixed season",
        "changing_rate": "fixed level, dynamic trend, fixed season",
        "unseasonal": "dynamic level, fixed trend, zero season",
    }
    return pd.DataFrame(observed, index=dates, columns=names), truth


def make_model(names: tuple[str, ...]) -> bx.MultiSeriesModel:
    """Give every series the same selectable structural grammar."""

    return bx.MultiSeriesModel(
        channels=tuple(
            bx.Channel(
                name,
                bx.Gaussian(),
                components=(bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
            )
            for name in names
        ),
        name="four related Gaussian summaries",
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data, truth = simulate_related_series()
    model = make_model(tuple(data.columns))

    print("\nSIMULATED STRUCTURES")
    for name, structure in truth.items():
        print(f"  {name:14s}: {structure}")
    print("\nINFERENCE PLAN\n", bx.plan(model, data))

    # The string uses uniform Dirichlet allocation priors and learns one
    # half-t dynamic-slab multiplier per component from all four series.
    # Replace the string with bx.HierarchicalSSVSPrior(...) to calibrate those
    # hyperpriors explicitly.
    fit = bx.fit(
        data,
        model,
        priors="hierarchical_ssvs",
        parameterization="fruehwirth_schnatter",
        engine="auto",          # exact FFBS because every channel is Gaussian
        asis=False,              # SSVS itself changes parameter dimension
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
        for name, values in fit.parameter_draws.items()
        if np.asarray(values).ndim == 2
        and name.startswith(("sd.channel.", "sigma.", "hierarchical_ssvs."))
    ]
    print("\nRESOLVED INFERENCE PLAN\n", fit.plan)
    print(
        "\nSCIENTIFIC-PARAMETER DIAGNOSTICS\n",
        fit.diagnostics()["parameters"].loc[scientific].round(4),
    )
    print("\nSERIES-SPECIFIC STRUCTURAL PROBABILITIES\n")
    print(fit.component_probabilities().round(3).to_string())
    print("\nPOPULATION ALLOCATION PROBABILITIES\n")
    print(fit.hierarchical_probabilities().round(3).to_string())
    print("\nSHARED DYNAMIC-SLAB MULTIPLIERS\n")
    print(fit.hierarchical_slab_summary().round(3).to_string())
    print("\nMODEL-SWITCHING DIAGNOSTICS\n")
    print(fit.component_transition_summary().round(3).to_string())

    start_year = int(data.index[0].year)
    end_year = int(data.index[-1].year)
    print("\nCOMPLETE-PREDICTOR RATES PER DECADE")
    for name in data.columns:
        print(f"  {name:14s}: {fit.channel_rate_summary(name, start_year, end_year)}")

    forecast = fit.forecast(12, draws=min(fit.n_draws, 2_000), seed=SEED + 2)
    print("\n12-MONTH FORECAST (FIRST EIGHT ROWS)\n")
    print(forecast.summary().head(8).round(3).to_string(index=False))

    for name in data.columns:
        fit.plot(
            "channel",
            channel=name,
            save=FIGURE_DIR / f"{name}_predictor.png",
        )
    fit.plot(
        "component_probabilities",
        save=FIGURE_DIR / "component_probabilities.png",
    )
    fit.plot("hierarchy", save=FIGURE_DIR / "hierarchy.png")
    fit.plot(
        "process_sd",
        title="Hierarchical SSVS: prior and posterior process SDs",
        save=FIGURE_DIR / "process_sds.png",
    )
    fit.plot("traces", parameters=scientific, save=FIGURE_DIR / "traces.png")
    fit.plot(
        "acf", parameters=scientific, max_lag=50,
        save=FIGURE_DIR / "acf.png",
    )

    if SAVE_FIT:
        Path("results").mkdir(exist_ok=True)
        fit.save("results/hierarchical_ssvs.bucex")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
