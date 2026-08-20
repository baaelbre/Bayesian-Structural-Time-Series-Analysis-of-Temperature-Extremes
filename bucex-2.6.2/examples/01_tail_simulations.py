"""Simulate matched GEV series while changing only shape or scale.

The models are deliberately constructed in this file with the public bucex
API. Run with ``python examples/01_tail_simulations.py``.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genextreme

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (SOURCE_ROOT / "bucex").is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


# Results. The timestamp can be shared across HPC jobs through BUCEX_RUN_ID;
# the identifying settings are always appended automatically.
RESULTS_ROOT = Path(os.environ.get("BUCEX_RESULTS_ROOT", "results"))
SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = os.environ.get("BUCEX_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OVERWRITE = os.environ.get("BUCEX_OVERWRITE", "0").lower() in {"1", "true", "yes"}

# Simulation design.
N_TIME = int(os.environ.get("BUCEX_N_TIME", "800"))
PERIOD = int(os.environ.get("BUCEX_PERIOD", "4"))
START_DATE = "1901-01-01"
INITIAL_LEVEL = 25.0
LEVEL_PROCESS_SD = 0.08

TAIL_SIGMA = 1.50
TAIL_XI_VALUES = (-0.30, 0.0, 0.30)
TAIL_SEED = 2_601  # Same seed gives all three series the same latent path.

SCALE_SIGMA_VALUES = (0.75, 1.50, 3.00)
SCALE_XI = -0.30
SCALE_SEED = 2_602  # Same seed gives all three series the same latent path.

# The theoretical density plots show this central probability range. Their
# horizontal axis is y - mu, so the changing local-level path plays no role.
DENSITY_PROBABILITY_RANGE = (0.001, 0.995)
DENSITY_GRID_POINTS = 900

FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
COLORS = {"navy": "#123B4A", "teal": "#1D7F7A", "grey": "#7A8589"}

RUN_SIGNATURE = (
    f"n{N_TIME}_p{PERIOD}"
    f"__tail-sigma{TAIL_SIGMA:g}-xi{'_'.join(f'{value:g}' for value in TAIL_XI_VALUES)}"
    f"__scale-sigma{'_'.join(f'{value:g}' for value in SCALE_SIGMA_VALUES)}-xi{SCALE_XI:g}"
    f"__qlevel{LEVEL_PROCESS_SD:g}_seed{TAIL_SEED}_{SCALE_SEED}"
)
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


# Each scenario is an ordinary bucex model plus its parameters and initial
# state. No presentation-specific scenario builder is involved.
TAIL_SCENARIOS = (
    {
        "name": "bounded_tail",
        "title": r"bounded tail ($\xi=-0.30$)",
        "model": bx.Model(bx.GEV(xi_bounds=(-0.5, 0.5)), (bx.LocalLevel(mode="dynamic"),), name="bounded tail"),
        "params": {"sigma": TAIL_SIGMA, "xi": TAIL_XI_VALUES[0], "sd.level": LEVEL_PROCESS_SD},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": TAIL_SEED,
    },
    {
        "name": "gumbel_tail",
        "title": r"exponential tail ($\xi=0$)",
        "model": bx.Model(bx.GEV(xi_bounds=(-0.5, 0.5)), (bx.LocalLevel(mode="dynamic"),), name="exponential tail"),
        "params": {"sigma": TAIL_SIGMA, "xi": TAIL_XI_VALUES[1], "sd.level": LEVEL_PROCESS_SD},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": TAIL_SEED,
    },
    {
        "name": "heavy_tail",
        "title": r"heavy tail ($\xi=+0.30$)",
        "model": bx.Model(bx.GEV(xi_bounds=(-0.5, 0.5)), (bx.LocalLevel(mode="dynamic"),), name="heavy tail"),
        "params": {"sigma": TAIL_SIGMA, "xi": TAIL_XI_VALUES[2], "sd.level": LEVEL_PROCESS_SD},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": TAIL_SEED,
    },
)

SCALE_SCENARIOS = (
    {
        "name": "low_scale",
        "title": r"low scale ($\sigma=0.75$)",
        "model": bx.Model(bx.GEV(xi_bounds=(-0.5, 0.5)), (bx.LocalLevel(mode="dynamic"),), name="low scale"),
        "params": {"sigma": SCALE_SIGMA_VALUES[0], "xi": SCALE_XI, "sd.level": LEVEL_PROCESS_SD},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SCALE_SEED,
    },
    {
        "name": "reference_scale",
        "title": r"reference scale ($\sigma=1.50$)",
        "model": bx.Model(bx.GEV(xi_bounds=(-0.5, 0.5)), (bx.LocalLevel(mode="dynamic"),), name="reference scale"),
        "params": {"sigma": SCALE_SIGMA_VALUES[1], "xi": SCALE_XI, "sd.level": LEVEL_PROCESS_SD},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SCALE_SEED,
    },
    {
        "name": "high_scale",
        "title": r"high scale ($\sigma=3.00$)",
        "model": bx.Model(bx.GEV(xi_bounds=(-0.5, 0.5)), (bx.LocalLevel(mode="dynamic"),), name="high scale"),
        "params": {"sigma": SCALE_SIGMA_VALUES[2], "xi": SCALE_XI, "sd.level": LEVEL_PROCESS_SD},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SCALE_SEED,
    },
)


def main() -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "legend.frameon": False})
    dates = pd.date_range(START_DATE, periods=N_TIME, freq="QS")
    grouped_tables: dict[str, dict[str, pd.DataFrame]] = {"shape": {}, "scale": {}}
    grouped_scenarios = {"shape": TAIL_SCENARIOS, "scale": SCALE_SCENARIOS}

    config_path = OUTPUT_DIR / "run_config.json"
    if config_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Refusing to overwrite {config_path}; set BUCEX_OVERWRITE=1 to rerun."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {
        "script": SCRIPT_NAME,
        "created_at": datetime.now().astimezone().isoformat(),
        "run_timestamp": RUN_TIMESTAMP,
        "run_signature": RUN_SIGNATURE,
        "output_directory": str(OUTPUT_DIR),
        "bucex_version": bx.__version__,
        "simulation": {
            "n_time": N_TIME,
            "period": PERIOD,
            "start_date": START_DATE,
            "initial_level": INITIAL_LEVEL,
            "level_process_sd": LEVEL_PROCESS_SD,
            "tail_sigma": TAIL_SIGMA,
            "tail_xi_values": list(TAIL_XI_VALUES),
            "tail_seed": TAIL_SEED,
            "scale_sigma_values": list(SCALE_SIGMA_VALUES),
            "scale_xi": SCALE_XI,
            "scale_seed": SCALE_SEED,
        },
        "density": {
            "probability_range": list(DENSITY_PROBABILITY_RANGE),
            "grid_points": DENSITY_GRID_POINTS,
            "relative_to_location": True,
        },
        "figures": {"formats": list(FIGURE_FORMATS), "dpi": FIGURE_DPI},
    }
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    for group, scenarios in grouped_scenarios.items():
        for scenario in scenarios:
            simulation = bx.simulate(
                scenario["model"],
                N_TIME,
                scenario["params"],
                initial_state=scenario["initial_state"],
                seed=scenario["seed"],
            )
            level_index = scenario["model"].state_names.index("level")
            table = pd.DataFrame(
                {
                    "date": dates,
                    "y": simulation.y,
                    "eta": simulation.eta,
                    "level": simulation.states[1:, level_index],
                }
            )
            grouped_tables[group][scenario["name"]] = table

            data_path = OUTPUT_DIR / "simulations" / group / f"{scenario['name']}.csv"
            truth_path = data_path.with_suffix(".json")
            if not OVERWRITE and (data_path.exists() or truth_path.exists()):
                raise FileExistsError(f"Refusing to overwrite {data_path}; set BUCEX_OVERWRITE=1.")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(data_path, index=False)
            truth = {
                "name": scenario["name"],
                "n_time": N_TIME,
                "period": PERIOD,
                "model": scenario["model"].to_dict(),
                "params": scenario["params"],
                "initial_state": scenario["initial_state"].tolist(),
                "seed": scenario["seed"],
            }
            truth_path.write_text(json.dumps(truth, indent=2, sort_keys=True), encoding="utf-8")

    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for group, scenarios in grouped_scenarios.items():
        all_y = np.concatenate([grouped_tables[group][scenario["name"]]["y"].to_numpy() for scenario in scenarios])
        padding = 0.04 * max(float(np.ptp(all_y)), 1.0)
        common_ylim = (float(np.min(all_y) - padding), float(np.max(all_y) + padding))

        # One time series per figure. Within a group every figure has exactly
        # the same y range, so xi or sigma can be compared without rescaling.
        for scenario in scenarios:
            table = grouped_tables[group][scenario["name"]]
            figure, axis = plt.subplots(figsize=(11, 4.2))
            axis.scatter(table["date"], table["y"], s=7, alpha=0.27, color=COLORS["grey"], label="observed")
            axis.plot(table["date"], table["eta"], color=COLORS["teal"], linewidth=1.8, label="latent location")
            axis.set_ylim(*common_ylim)
            axis.set_title(scenario["title"])
            axis.set_ylabel("GEV observation")
            axis.grid(axis="y", alpha=0.35)
            axis.legend(loc="upper left")
            figure.tight_layout()
            for extension in FIGURE_FORMATS:
                figure.savefig(figure_dir / f"{group}_{scenario['name']}.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(figure)

        # Compare the actual GEV densities, rather than kernel densities of the
        # simulated non-stationary observations.  SciPy uses c = -xi.  Setting
        # loc=0 makes the horizontal axis the value relative to mu_t.
        density_ranges = []
        finite_upper_endpoints = []
        for scenario in scenarios:
            sigma = float(scenario["params"]["sigma"])
            xi = float(scenario["params"]["xi"])
            density_ranges.append(
                genextreme.ppf(
                    DENSITY_PROBABILITY_RANGE,
                    c=-xi,
                    loc=0.0,
                    scale=sigma,
                )
            )
            if xi < 0.0:
                finite_upper_endpoints.append(-sigma / xi)

        lower = min(float(values[0]) for values in density_ranges)
        upper = max(float(values[1]) for values in density_ranges)
        if finite_upper_endpoints:
            upper = max(upper, max(finite_upper_endpoints))
        padding = 0.05 * max(upper - lower, 1.0)
        density_grid = np.linspace(lower - padding, upper + padding, DENSITY_GRID_POINTS)

        figure, axis = plt.subplots(figsize=(9, 4.6), layout="constrained")
        palette = ("#2778B4", COLORS["teal"], "#E76F51")
        for scenario, color in zip(scenarios, palette):
            sigma = float(scenario["params"]["sigma"])
            xi = float(scenario["params"]["xi"])
            density = genextreme.pdf(
                density_grid,
                c=-xi,
                loc=0.0,
                scale=sigma,
            )
            if group == "shape":
                label = rf"$\xi={xi:+.2f}$, $\sigma={sigma:.2f}$"
            else:
                label = rf"$\sigma={sigma:.2f}$, $\xi={xi:+.2f}$"
            if xi < 0.0:
                endpoint = -sigma / xi
                label += rf"; endpoint $={endpoint:.2f}$"
                axis.axvline(
                    endpoint,
                    color=color,
                    linestyle="--",
                    linewidth=1.25,
                    alpha=0.85,
                )
            axis.plot(
                density_grid,
                density,
                color=color,
                linewidth=2.2,
                label=label,
            )

        axis.set_xlim(lower - padding, upper + padding)
        axis.set_title(
            "The GEV shape changes the tail and its support"
            if group == "shape"
            else "The GEV scale changes dispersion and the bounded endpoint"
        )
        axis.set_xlabel(r"value relative to location, $y-\mu$")
        axis.set_ylabel("density")
        axis.grid(axis="y", alpha=0.35)
        axis.legend()
        for extension in FIGURE_FORMATS:
            figure.savefig(
                figure_dir / f"{group}_theoretical_density_comparison.{extension}",
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        plt.close(figure)

    catalog = []
    for group, scenarios in grouped_scenarios.items():
        for scenario in scenarios:
            catalog.append({"group": group, "name": scenario["name"], **scenario["params"], "seed": scenario["seed"]})
    catalog_path = OUTPUT_DIR / "tables" / "scenarios.csv"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(catalog).to_csv(catalog_path, index=False)
    print(f"Tail and scale outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
