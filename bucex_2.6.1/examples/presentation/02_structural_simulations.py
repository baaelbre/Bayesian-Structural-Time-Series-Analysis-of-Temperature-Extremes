"""Simulate the absent, fixed, and dynamic structural-component designs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if (SOURCE_ROOT / "bucex").is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


OUTPUT_DIR = Path("results/presentation")

# Common observation model and length.
N_TIME = 1000
PERIOD = 4
SIGMA = 1.50
XI = -0.30
INITIAL_LEVEL = 25.0

# Structural signal sizes. (sigma >> Q)
LINEAR_SLOPE = 0.006
RANDOM_WALK_SD = 0.05
LOCAL_LEVEL_SD = 0.05
LOCAL_SLOPE_SD = 0.00050
LOCAL_INITIAL_SLOPE = 0.003
DYNAMIC_SEASON_AMPLITUDE = .25
FIXED_SEASON_AMPLITUDE = .25
SEASONAL_SD = 0.05
SIMULATION_SEED = 13081997


FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
OVERWRITE = True


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


def main() -> None:
    tables: dict[str, pd.DataFrame] = {}
    truths: dict[str, dict] = {}

    for scenario in SCENARIOS:
        _, table = bx.simulate_scenario(scenario)
        truth = scenario.to_dict()

        data_path = (
            OUTPUT_DIR / "simulations" / "structure" / f"{scenario.name}.csv"
        )
        truth_path = data_path.with_suffix(".json")
        if not OVERWRITE and (data_path.exists() or truth_path.exists()):
            raise FileExistsError(
                f"Refusing to overwrite {data_path}; set OVERWRITE=True to rerun."
            )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(data_path, index=False)
        truth_path.write_text(
            json.dumps(truth, indent=2, sort_keys=True), encoding="utf-8"
        )
        tables[scenario.name] = table
        truths[scenario.name] = truth
        print(
            f"{scenario.name}: {truth['structural_truth']} "
            f"(period={scenario.period}, sigma={scenario.sigma:.2f}, "
            f"xi={scenario.xi:+.2f})"
        )

    catalog_path = (
        OUTPUT_DIR / "tables" / "10_simulations" / "structural_scenarios.csv"
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    pd.json_normalize(list(truths.values()), sep=".").to_csv(
        catalog_path, index=False
    )
    bx.plot_structural_simulations(
        tables,
        truths,
        OUTPUT_DIR / "figures" / "10_simulations" / "structure",
        formats=FIGURE_FORMATS,
        dpi=FIGURE_DPI,
    )


if __name__ == "__main__":
    main()
