"""Example 16: combine four independent componentwise-SSVS PGAS chains."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bucex as bx


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-dir", type=Path, default=Path("results/hpc_chains"))
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/16_hpc_combined"),
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    paths = [args.chain_dir / f"chain_{index:02d}.bucex" for index in range(1, 5)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing independent chains: {missing}")

    fit = bx.combine_fits([bx.FitResult.load(path) for path in paths])
    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nPARAMETER DIAGNOSTICS\n", fit.diagnostics()["parameters"].round(4))
    print("\nCHANNEL COMPONENT PROBABILITIES\n", fit.component_probabilities().round(3))
    print("\nCHANNEL JOINT STRUCTURES\n", fit.structural_model_probabilities().round(3))
    print("\nPOPULATION COMPONENT PROBABILITIES\n", fit.hierarchical_probabilities().round(3))
    print("\nPOOLED SLAB MULTIPLIERS\n", fit.hierarchical_slab_summary().round(3))
    print("\nALLOCATION SWITCHING\n", fit.component_transition_summary().round(3))

    combined = args.chain_dir / "combined_four_chains.bucex"
    fit.save(combined)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    fit.plot("component_probabilities", save=args.figure_dir / "allocations.png")
    fit.plot("hierarchy", save=args.figure_dir / "hierarchy.png")
    fit.plot("process_sd", save=args.figure_dir / "process_sds.png")
    fit.plot("traces", save=args.figure_dir / "traces.png")
    fit.plot("acf", max_lag=50, save=args.figure_dir / "acf.png")
    plt.close("all")
    print("\nCOMBINED FIT\n", combined)


if __name__ == "__main__":
    main()
