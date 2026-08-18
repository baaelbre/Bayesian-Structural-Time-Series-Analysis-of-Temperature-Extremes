"""Example 16: combine and diagnose four independently run HPC chains."""
from __future__ import annotations

from pathlib import Path

import bucex as bx


CHAIN_DIR = Path("results/hpc_chains")
FIGURE_DIR = Path("figures/16_hpc_combined")


def main() -> None:
    paths = [CHAIN_DIR / f"chain_{index:02d}.bucex" for index in range(1, 5)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run Example 15 for every chain first: {missing}")
    fit = bx.combine_fits([bx.FitResult.load(path) for path in paths])
    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nDIAGNOSTICS\n", fit.diagnostics()["parameters"].round(4))
    print("\nTREND MODEL CLASSES\n", fit.hierarchical_trend_model_probabilities().round(3))
    print("\nCHANNEL ALLOCATIONS\n", fit.component_probabilities().round(3))

    combined = CHAIN_DIR / "combined_four_chains.bucex"
    fit.save(combined)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fit.plot("trend_models", save=FIGURE_DIR / "trend_models.png")
    fit.plot("traces", save=FIGURE_DIR / "traces.png")
    fit.plot("acf", max_lag=50, save=FIGURE_DIR / "acf.png")
    print(combined)


if __name__ == "__main__":
    main()
