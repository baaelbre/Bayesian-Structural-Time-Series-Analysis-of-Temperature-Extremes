"""Simulate how GEV shape and scale change a common local-level signal."""
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

# Simulation design: edit these values directly and rerun this file.
N_TIME = 600
PERIOD = 4
INITIAL_LEVEL = 25.0
LEVEL_PROCESS_SD = 0.08

TAIL_SIGMA = 1.50
TAIL_XI_VALUES = (-0.30, 0.0, 0.30)
TAIL_SEED = 2_601  # Common seed keeps the latent path matched across xi.

SCALE_SIGMA_VALUES = (0.75, 1.50, 3.00)
SCALE_XI = -0.30
SCALE_SEED = 2_602  # Common seed keeps the latent path matched across sigma.

FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
OVERWRITE = False


TAIL_SCENARIOS = bx.make_tail_scenarios(
    n_time=N_TIME,
    period=PERIOD,
    sigma=TAIL_SIGMA,
    xi_values=TAIL_XI_VALUES,
    initial_level=INITIAL_LEVEL,
    level_sd=LEVEL_PROCESS_SD,
    seed=TAIL_SEED,
)
SCALE_SCENARIOS = bx.make_scale_scenarios(
    n_time=N_TIME,
    period=PERIOD,
    sigma_values=SCALE_SIGMA_VALUES,
    xi=SCALE_XI,
    initial_level=INITIAL_LEVEL,
    level_sd=LEVEL_PROCESS_SD,
    seed=SCALE_SEED,
)


def main() -> None:
    grouped_tables: dict[str, dict[str, pd.DataFrame]] = {
        "tail": {},
        "scale": {},
    }
    grouped_truths: dict[str, dict[str, dict]] = {
        "tail": {},
        "scale": {},
    }

    for group, scenarios in (
        ("tail", TAIL_SCENARIOS),
        ("scale", SCALE_SCENARIOS),
    ):
        for scenario in scenarios:
            # Within each group the seed and latent process are identical. Only
            # xi (tail group) or sigma (scale group) changes.
            simulation, table = bx.simulate_scenario(scenario)
            truth = scenario.to_dict()

            data_path = (
                OUTPUT_DIR / "simulations" / group / f"{scenario.name}.csv"
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
            grouped_tables[group][scenario.name] = table
            grouped_truths[group][scenario.name] = truth
            print(
                f"{scenario.name}: n={simulation.y.size}, "
                f"period={scenario.period}, sigma={scenario.sigma:.2f}, "
                f"xi={scenario.xi:+.2f}"
            )

    table_dir = OUTPUT_DIR / "tables" / "10_simulations"
    table_dir.mkdir(parents=True, exist_ok=True)
    for group in ("tail", "scale"):
        pd.json_normalize(list(grouped_truths[group].values()), sep=".").to_csv(
            table_dir / f"{group}_scenarios.csv", index=False
        )
    bx.plot_tail_simulations(
        grouped_tables["tail"],
        grouped_truths["tail"],
        OUTPUT_DIR / "figures" / "10_simulations" / "tail",
        formats=FIGURE_FORMATS,
        dpi=FIGURE_DPI,
    )
    bx.plot_scale_simulations(
        grouped_tables["scale"],
        grouped_truths["scale"],
        OUTPUT_DIR / "figures" / "10_simulations" / "scale",
        formats=FIGURE_FORMATS,
        dpi=FIGURE_DPI,
    )


if __name__ == "__main__":
    main()
