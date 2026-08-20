"""Simulate the absent, fixed, and dynamic structural-component designs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import bucex as bx


OUTPUT_DIR = Path("results/presentation")
N_MONTHS = 720
SCENARIOS = bx.STRUCTURAL_SCENARIOS
FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
OVERWRITE = False


def main() -> None:
    tables: dict[str, pd.DataFrame] = {}
    truths: dict[str, dict] = {}

    for base_scenario in SCENARIOS:
        scenario = base_scenario.resized(N_MONTHS)
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
            f"(sigma={scenario.sigma:.2f}, xi={scenario.xi:+.2f})"
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
