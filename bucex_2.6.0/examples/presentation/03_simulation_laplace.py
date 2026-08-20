"""Simulate each structural design and fit it with Laplace state updates.

This file is deliberately self-contained. It uses an existing deterministic
simulation when one is present and otherwise creates it with
``bx.simulate_scenario`` before calling ``bx.fit`` directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bucex as bx


OUTPUT_DIR = Path("results/presentation")
N_MONTHS = 720
SCENARIOS = bx.STRUCTURAL_SCENARIOS

# Pilot settings. For the final analysis use 2_000, 2_000, and 4 chains.
DRAWS = 250
WARMUP = 250
CHAINS = 2
SEED = 26_000
PROGRESS = True

FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
DIAGNOSTIC_FIGURES = False
OVERWRITE = False


def load_or_simulate(base_scenario) -> tuple[pd.DataFrame, dict]:
    """Return this script's data, creating it through the simulation API."""

    scenario = base_scenario.resized(N_MONTHS)
    data_path = OUTPUT_DIR / "simulations" / "structure" / f"{scenario.name}.csv"
    truth_path = data_path.with_suffix(".json")
    if data_path.is_file() and truth_path.is_file() and not OVERWRITE:
        table = pd.read_csv(data_path, parse_dates=["date"])
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        if len(table) != N_MONTHS:
            raise ValueError(
                f"{data_path} has {len(table)} rows, expected {N_MONTHS}; "
                "set OVERWRITE=True to regenerate it."
            )
        return table, truth
    if not OVERWRITE and (data_path.exists() or truth_path.exists()):
        raise FileExistsError(
            f"Only one of {data_path.name} and {truth_path.name} exists; "
            "set OVERWRITE=True to regenerate both."
        )

    _, table = bx.simulate_scenario(scenario)
    truth = scenario.to_dict()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(data_path, index=False)
    truth_path.write_text(
        json.dumps(truth, indent=2, sort_keys=True), encoding="utf-8"
    )
    return table, truth


def main() -> None:
    for number, scenario in enumerate(SCENARIOS):
        table, truth = load_or_simulate(scenario)
        y = table["y"].to_numpy(float)

        # Componentwise SSVS: level is fixed/dynamic; slope and seasonality
        # are absent/fixed/dynamic. Innovation scales have Gaussian slabs.
        priors = bx.ssvs_gev_priors(
            period=12,
            alpha_mean=float(np.median(y)),
            beta_mean=0.0,
            beta_sd=0.01,
            innovation_slab_sd={
                "level": 0.12,
                "trend": 0.0015,
                "season": 0.10,
            },
            level_dynamic_probability=0.5,
            trend_probabilities=(1.0 / 3.0,) * 3,
            season_probabilities=(1.0 / 3.0,) * 3,
        )

        fit_path = (
            OUTPUT_DIR
            / "fits"
            / "simulations"
            / "laplace"
            / scenario.name
            / "combined.bucex"
        )
        if fit_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(fit_path)
            print(f"Reusing {fit_path}")
        else:
            laplace_fit = bx.fit(
                y,
                family="gev",
                period=12,
                priors=priors,
                engine="laplace",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(
                    draws=DRAWS,
                    warmup=WARMUP,
                    chains=CHAINS,
                    seed=SEED + 100 * number,
                    progress=PROGRESS,
                ),
                dates=table["date"].to_numpy(),
                name=scenario.name,
            )
            laplace_fit.metadata.update(
                {"example": "simulation_laplace", "truth": truth}
            )
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(fit_path)

        table_dir = (
            OUTPUT_DIR / "tables" / "simulations" / "laplace" / scenario.name
        )
        figure_dir = (
            OUTPUT_DIR / "figures" / "simulations" / "laplace" / scenario.name
        )
        bx.export_fit_results(
            laplace_fit, table_dir, source=fit_path, truth=truth
        )
        bx.plot_fit_results(
            laplace_fit,
            figure_dir,
            truth_table=table,
            truth=truth,
            formats=FIGURE_FORMATS,
            dpi=FIGURE_DPI,
            diagnostics=DIAGNOSTIC_FIGURES,
        )
        print(f"Laplace fit complete: {scenario.name}")


if __name__ == "__main__":
    main()
