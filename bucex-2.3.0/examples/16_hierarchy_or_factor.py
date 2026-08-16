"""Example 16: compare hierarchical SSVS with a common factor.

Both calls use ``bx.fit`` and return the same ``FitResult`` type, but the
scientific estimands differ:

* hierarchical SSVS asks which structural components recur across series;
* the factor model estimates one shared path, loadings, and deviations.

This script fits both to the same small Gaussian data set so that the API and
interpretation can be compared line by line.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 144
PERIOD = 12
DRAWS = 750
WARMUP = 750
CHAINS = 4
SEED = 1_601
SHOW_PLOTS = False
FIGURE_DIR = Path("figures/16_hierarchy_or_factor")


def simulate_common_signal() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    t = np.arange(N_TIME)
    common_slope = 0.002 + np.cumsum(rng.normal(0.0, 0.00008, N_TIME))
    common = np.cumsum(common_slope)
    season = 0.12 * np.sin(2.0 * np.pi * t / PERIOD)
    loadings = np.array((1.0, 1.35, 0.65))
    deviations = np.column_stack(
        (
            np.zeros(N_TIME),
            np.cumsum(rng.normal(0.0, 0.008, N_TIME)),
            np.cumsum(rng.normal(0.0, 0.004, N_TIME)),
        )
    )
    eta = common[:, None] * loadings + deviations + season[:, None]
    y = eta + rng.normal(0.0, (0.16, 0.20, 0.18), eta.shape)
    return pd.DataFrame(y, index=dates, columns=("reference", "amplified", "damped"))


def hierarchical_model(names: tuple[str, ...]) -> bx.MultiSeriesModel:
    return bx.MultiSeriesModel(
        tuple(
            bx.Channel(
                name,
                bx.Gaussian(),
                (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
            )
            for name in names
        ),
        name="hierarchical structural selection",
    )


def factor_model(names: tuple[str, ...]) -> bx.FactorModel:
    channels = tuple(
        bx.Channel(
            name,
            bx.Gaussian(),
            (bx.LocalLevel(), bx.DummySeasonal(PERIOD)),
        )
        for name in names
    )
    return bx.FactorModel(
        channels=channels,
        factors=(
            bx.Factor(
                "common",
                (
                    bx.LocalLinearTrend(
                        initial_level=0.0,
                        initial_level_sd=0.0,
                    ),
                ),
                {
                    "reference": 1.0,
                    "amplified": bx.Loading.estimated(1.0),
                    "damped": bx.Loading.estimated(0.8),
                },
            ),
        ),
        name="one common warming factor",
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = simulate_common_signal()
    names = tuple(data.columns)

    hierarchy = hierarchical_model(names)
    hierarchy_fit = bx.fit(
        data,
        hierarchy,
        priors="hierarchical_ssvs",
        parameterization="fs",
        engine="ffbs",
        asis=False,
        mcmc=bx.MCMC(
            draws=DRAWS, warmup=WARMUP, chains=CHAINS,
            seed=SEED + 1, progress=True,
        ),
    )

    factor = factor_model(names)
    compiled_factor = bx.compile_model(factor, data)
    factor_priors = bx.identified_factor_priors(
        compiled_factor,
        profile="regularized",
        smooth_factor=True,
        reference_channel="reference",
    )
    factor_fit = bx.fit(
        data,
        factor,
        priors=factor_priors,
        parameterization="fs",
        engine="ffbs",
        asis=True,
        mcmc=bx.MCMC(
            draws=DRAWS, warmup=WARMUP, chains=CHAINS,
            seed=SEED + 2, progress=True,
        ),
    )

    print("\nHIERARCHICAL SSVS: recurring structural complexity")
    print(hierarchy_fit.component_probabilities().round(3).to_string())
    print("\nLEARNED POPULATION PROBABILITIES")
    print(hierarchy_fit.hierarchical_probabilities().round(3).to_string())

    print("\nFACTOR MODEL: shared path and heterogeneous responses")
    print(factor_fit.factor_rate_summary("common"))
    loading_names = [
        name for name in factor_fit.parameter_draws if name.startswith("loading.")
    ]
    print(factor_fit.diagnostics()["parameters"].loc[loading_names].round(3))
    print("\nFACTOR IDENTIFICATION CHECK")
    print(factor_fit.factor_identification_diagnostics().round(3).to_string())

    print("\nCOMPLETE-PREDICTOR RATES: BOTH MODELS")
    for name in names:
        print(f"  {name}")
        print("    hierarchy:", hierarchy_fit.channel_rate_summary(name))
        print("    factor:   ", factor_fit.channel_rate_summary(name))

    hierarchy_fit.plot(
        "component_probabilities",
        save=FIGURE_DIR / "hierarchy_component_probabilities.png",
    )
    hierarchy_fit.plot("hierarchy", save=FIGURE_DIR / "hierarchy_population.png")
    factor_fit.plot("factor", factor="common", save=FIGURE_DIR / "common_factor.png")
    factor_fit.plot(
        "factor_decomposition",
        baseline=slice(0, PERIOD * 2),
        save=FIGURE_DIR / "factor_decomposition.png",
    )
    factor_fit.plot(
        "identification",
        baseline=slice(0, PERIOD * 2),
        save=FIGURE_DIR / "factor_identification.png",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
