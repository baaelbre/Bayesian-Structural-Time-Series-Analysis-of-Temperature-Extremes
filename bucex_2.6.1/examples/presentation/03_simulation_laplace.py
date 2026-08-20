"""Simulate each structural design and fit it with Laplace state updates.

This file is deliberately self-contained. It uses an existing deterministic
simulation when one is present and otherwise creates it with
``bx.simulate_scenario`` before calling ``bx.fit`` directly.
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

# Simulation design. Keep these values aligned with 02_structural_simulations.py
# when you want both files to reuse the same generated CSV/JSON artifacts.
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
TREND_PROBABILITIES = (1.0 / 3.0,) * 3  # absent, fixed, dynamic
SEASON_PROBABILITIES = (1.0 / 3.0,) * 3  # absent, fixed, dynamic

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
    """Construct the complete, editable prior through the public API."""

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
    """Prevent silent reuse after changing a simulation constant."""

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
            "set OVERWRITE=True to regenerate the simulation and fit."
        )


def load_or_simulate(scenario: bx.GEVScenario) -> tuple[pd.DataFrame, dict]:
    """Return this script's data, creating it through the simulation API."""

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
    for number, scenario in enumerate(SCENARIOS):
        table, truth = load_or_simulate(scenario)
        y = table["y"].to_numpy(float)

        # Componentwise SSVS: level is fixed/dynamic; slope and seasonality
        # are absent/fixed/dynamic. Innovation scales have Gaussian slabs.
        priors = make_priors(y)

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
            if (
                laplace_fit.n_time != scenario.n_time
                or laplace_fit.model.period != scenario.period
                or laplace_fit.n_chains != CHAINS
                or laplace_fit.draws_per_chain != DRAWS
                or laplace_fit.metadata.get("prior_settings") != prior_settings()
            ):
                raise ValueError(
                    f"{fit_path} does not match the current data, prior, or "
                    "MCMC controls; set OVERWRITE=True to refit."
                )
            print(f"Reusing {fit_path}")
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
                    "example": "simulation_laplace",
                    "truth": truth,
                    "prior_settings": prior_settings(),
                }
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
