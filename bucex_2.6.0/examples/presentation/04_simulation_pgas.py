"""Fit the structural simulations with Laplace-initialized PGAS.

The script is standalone: it simulates missing data and fits a missing Laplace
initializer itself. The important handoff is the public call
``bx.fit(..., engine="pgas", init=laplace_fit)`` below.
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

# Pilot settings. Final runs should use 2_000 draws, 2_000 warmup iterations,
# 4 chains, and 512 particles.
DRAWS = 250
WARMUP = 250
CHAINS = 2
PARTICLES = 128
SEED = 36_000
PROGRESS = True

FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
DIAGNOSTIC_FIGURES = False
OVERWRITE = False


def load_or_simulate(base_scenario) -> tuple[pd.DataFrame, dict]:
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
    fitted: dict[tuple[str, str], bx.FitResult] = {}
    truths: dict[str, dict] = {}

    for number, scenario in enumerate(SCENARIOS):
        table, truth = load_or_simulate(scenario)
        truths[scenario.name] = truth
        y = table["y"].to_numpy(float)

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

        laplace_path = (
            OUTPUT_DIR
            / "fits"
            / "simulations"
            / "laplace"
            / scenario.name
            / "combined.bucex"
        )
        if laplace_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(laplace_path)
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
            laplace_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(laplace_path)
        fitted[("laplace", scenario.name)] = laplace_fit

        pgas_path = (
            OUTPUT_DIR
            / "fits"
            / "simulations"
            / "pgas"
            / scenario.name
            / "combined.bucex"
        )
        if pgas_path.is_file() and not OVERWRITE:
            pgas_fit = bx.FitResult.load(pgas_path)
            print(f"Reusing {pgas_path}")
        else:
            # FitResult supplies static parameters and the complete centred
            # Laplace state path. PGAS still targets the exact posterior.
            pgas_fit = bx.fit(
                y,
                family="gev",
                period=12,
                priors=laplace_fit.priors,
                engine="pgas",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(
                    draws=DRAWS,
                    warmup=WARMUP,
                    chains=CHAINS,
                    seed=SEED + 10_000 + 100 * number,
                    progress=PROGRESS,
                ),
                particles=bx.Particles(n=PARTICLES, proposal="guided"),
                dates=table["date"].to_numpy(),
                name=scenario.name,
                init=laplace_fit,
            )
            pgas_fit.metadata.update(
                {
                    "example": "simulation_pgas",
                    "truth": truth,
                    "warm_start_source": str(laplace_path),
                }
            )
            pgas_path.parent.mkdir(parents=True, exist_ok=True)
            pgas_fit.save(pgas_path)
        fitted[("pgas", scenario.name)] = pgas_fit

        table_dir = OUTPUT_DIR / "tables" / "simulations" / "pgas" / scenario.name
        figure_dir = (
            OUTPUT_DIR / "figures" / "simulations" / "pgas" / scenario.name
        )
        bx.export_fit_results(pgas_fit, table_dir, source=pgas_path, truth=truth)
        bx.plot_fit_results(
            pgas_fit,
            figure_dir,
            truth_table=table,
            truth=truth,
            formats=FIGURE_FORMATS,
            dpi=FIGURE_DPI,
            diagnostics=DIAGNOSTIC_FIGURES,
        )
        print(f"PGAS fit complete: {scenario.name}")

    # One direct comparison table/figure for the results section.
    selection = bx.collect_selection_probabilities(fitted, truths)
    selection_path = (
        OUTPUT_DIR / "tables" / "30_results" / "simulation_selection_recovery.csv"
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(selection_path, index=False)
    bx.plot_selection_recovery(
        selection,
        OUTPUT_DIR / "figures" / "30_results",
        formats=FIGURE_FORMATS,
        dpi=FIGURE_DPI,
    )


if __name__ == "__main__":
    main()
