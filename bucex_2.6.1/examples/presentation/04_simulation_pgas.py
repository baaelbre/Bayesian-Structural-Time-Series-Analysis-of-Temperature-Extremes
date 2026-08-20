"""Fit the structural simulations with Laplace-initialized PGAS.

The script is standalone: it simulates missing data and fits a missing Laplace
initializer itself. The important handoff is the public call
``bx.fit(..., engine="pgas", init=laplace_fit)`` below.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if (SOURCE_ROOT / "bucex").is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


OUTPUT_DIR = Path("results/presentation")

# Simulation design. Keep these values aligned with scripts 02 and 03 when
# reusing their deterministic simulations and Laplace fits.
N_TIME = 800
PERIOD = 4
SIGMA = 1.50
XI = -0.30
INITIAL_LEVEL = 25.0
LINEAR_SLOPE = 0.006
RANDOM_WALK_SD = 0.060
LOCAL_LEVEL_SD = 0.035
LOCAL_SLOPE_SD = 0.00050
LOCAL_INITIAL_SLOPE = 0.003
DYNAMIC_SEASON_AMPLITUDE = 1.25
FIXED_SEASON_AMPLITUDE = 1.75
SEASONAL_SD = 0.040
SIMULATION_SEED = 2_610

# Fully exposed prior hyperparameters.
ALPHA_PRIOR_SD = 3.2
BETA_PRIOR_MEAN = 0.0
BETA_PRIOR_SD = 0.010
INITIAL_SEASON_PRIOR_SD = 2.5
SIGMA2_PRIOR_A = 2.0
SIGMA2_PRIOR_B = 2.25
XI_PRIOR_BOUNDS = (-0.50, 0.50)
XI_MAX_ABS = 0.50
INNOVATION_SLAB_SD = {
    "level": 0.10,
    "trend": 0.0008,
    "season": 0.07,
}
LEVEL_DYNAMIC_PROBABILITY = 0.50
TREND_PROBABILITIES = (1.0 / 3.0,) * 3
SEASON_PROBABILITIES = (1.0 / 3.0,) * 3

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


SCENARIOS = bx.make_structural_scenarios(
    n_time=N_TIME,
    period=PERIOD,
    sigma=SIGMA,
    xi=XI,
    initial_level=INITIAL_LEVEL,
    linear_slope=LINEAR_SLOPE,
    random_walk_sd=RANDOM_WALK_SD,
    local_level_sd=LOCAL_LEVEL_SD,
    local_slope_sd=LOCAL_SLOPE_SD,
    local_initial_slope=LOCAL_INITIAL_SLOPE,
    dynamic_season_amplitude=DYNAMIC_SEASON_AMPLITUDE,
    fixed_season_amplitude=FIXED_SEASON_AMPLITUDE,
    seasonal_sd=SEASONAL_SD,
    seed=SIMULATION_SEED,
)


def prior_settings() -> dict:
    return {
        "alpha_sd": ALPHA_PRIOR_SD,
        "beta_mean": BETA_PRIOR_MEAN,
        "beta_sd": BETA_PRIOR_SD,
        "seasonal_initial_sd": INITIAL_SEASON_PRIOR_SD,
        "sigma2": {"a": SIGMA2_PRIOR_A, "b": SIGMA2_PRIOR_B},
        "xi_bounds": list(XI_PRIOR_BOUNDS),
        "xi_max_abs": XI_MAX_ABS,
        "innovation_slab_sd": dict(INNOVATION_SLAB_SD),
        "level_dynamic_probability": LEVEL_DYNAMIC_PROBABILITY,
        "trend_probabilities": list(TREND_PROBABILITIES),
        "season_probabilities": list(SEASON_PROBABILITIES),
    }


def make_priors(y: np.ndarray) -> bx.FSGEVPriors:
    return bx.ssvs_gev_priors(
        period=PERIOD,
        alpha_mean=float(np.median(y)),
        alpha_sd=ALPHA_PRIOR_SD,
        beta_mean=BETA_PRIOR_MEAN,
        beta_sd=BETA_PRIOR_SD,
        seasonal_initial_sd=INITIAL_SEASON_PRIOR_SD,
        sigma2_prior=bx.InverseGammaPrior(SIGMA2_PRIOR_A, SIGMA2_PRIOR_B),
        xi_prior=bx.UniformPrior(*XI_PRIOR_BOUNDS),
        xi_max_abs=XI_MAX_ABS,
        innovation_slab_sd=INNOVATION_SLAB_SD,
        level_dynamic_probability=LEVEL_DYNAMIC_PROBABILITY,
        trend_probabilities=TREND_PROBABILITIES,
        season_probabilities=SEASON_PROBABILITIES,
    )


def validate_truth(truth: dict, scenario: bx.GEVScenario, path: Path) -> None:
    expected = scenario.to_dict()
    keys = (
        "name",
        "n_time",
        "period",
        "sigma",
        "xi",
        "level",
        "trend",
        "season",
        "sd_level",
        "sd_slope",
        "sd_seasonal",
        "initial_level",
        "initial_slope",
        "season_amplitude",
        "seed",
    )
    mismatches = [key for key in keys if truth.get(key) != expected[key]]
    if mismatches:
        raise ValueError(
            f"{path} was generated with different {mismatches}; "
            "set OVERWRITE=True to regenerate the simulation and fits."
        )


def validate_fit(fit: bx.FitResult, scenario: bx.GEVScenario, path: Path) -> None:
    if (
        fit.n_time != scenario.n_time
        or fit.model.period != scenario.period
        or fit.n_chains != CHAINS
        or fit.draws_per_chain != DRAWS
        or fit.metadata.get("prior_settings") != prior_settings()
    ):
        raise ValueError(
            f"{path} does not match the current data, prior, or MCMC controls; "
            "set OVERWRITE=True to refit."
        )


def load_or_simulate(scenario: bx.GEVScenario) -> tuple[pd.DataFrame, dict]:
    data_path = OUTPUT_DIR / "simulations" / "structure" / f"{scenario.name}.csv"
    truth_path = data_path.with_suffix(".json")
    if data_path.is_file() and truth_path.is_file() and not OVERWRITE:
        table = pd.read_csv(data_path, parse_dates=["date"])
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        if len(table) != N_TIME:
            raise ValueError(
                f"{data_path} has {len(table)} rows, expected {N_TIME}; "
                "set OVERWRITE=True to regenerate it."
            )
        validate_truth(truth, scenario, truth_path)
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

        priors = make_priors(y)

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
            validate_fit(laplace_fit, scenario, laplace_path)
        else:
            laplace_fit = bx.fit(
                y,
                family="gev",
                period=PERIOD,
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
                name=scenario.name,
            )
            laplace_fit.metadata.update(
                {
                    "example": "simulation_laplace_initializer",
                    "truth": truth,
                    "prior_settings": prior_settings(),
                }
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
            validate_fit(pgas_fit, scenario, pgas_path)
            print(f"Reusing {pgas_path}")
        else:
            # FitResult supplies static parameters and the complete centred
            # Laplace state path. PGAS still targets the exact posterior.
            pgas_fit = bx.fit(
                y,
                family="gev",
                period=PERIOD,
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
                name=scenario.name,
                init=laplace_fit,
            )
            pgas_fit.metadata.update(
                {
                    "example": "simulation_pgas",
                    "truth": truth,
                    "prior_settings": prior_settings(),
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
