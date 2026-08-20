#!/usr/bin/env python3
"""End-to-end software smoke test for the v2.6 presentation workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


def run(work_dir: Path) -> dict[str, object]:
    started = time.perf_counter()
    config = bx.PresentationConfig.for_profile(
        "smoke",
        output_dir=work_dir,
        progress=False,
        draws=1,
        warmup=1,
        particles=16,
    )
    workflow = bx.PresentationWorkflow(config)
    data = workflow.run_data(figures=False)
    tail = workflow.run_simulations(kind="tail", figures=False)
    structure = workflow.run_simulations(
        kind="structure", scenario="stationary", figures=False
    )
    laplace_sim = workflow.fit_simulations(
        engine="laplace", scenario="stationary", figures=False
    )
    pgas_sim = workflow.fit_simulations(
        engine="pgas", scenario="stationary", figures=False
    )
    laplace_txx = workflow.fit_uccle(
        engine="laplace", series="TXx", figures=False
    )
    pgas_txx = workflow.fit_uccle(
        engine="pgas", series="TXx", figures=False
    )

    exact = bx.FitResult.load(workflow.paths.simulation_fit("pgas", "stationary"))
    screen = bx.FitResult.load(
        workflow.paths.simulation_fit("laplace", "stationary")
    )
    if screen.plan.targets_exact_posterior:
        raise RuntimeError("The Laplace fit was incorrectly marked exact.")
    if not exact.plan.targets_exact_posterior:
        raise RuntimeError("The PGAS fit was not marked exact-invariant.")
    if exact.meta.get("warm_start_source_engine") != "laplace":
        raise RuntimeError("PGAS did not retain Laplace warm-start provenance.")
    if "reference_ancestor_change_rate" not in exact.diagnostics()["engine"]:
        raise RuntimeError("The PGAS degeneracy diagnostic is missing.")

    manifest = json.loads(workflow.paths.manifest.read_text(encoding="utf-8"))
    required = {
        "data",
        "simulations:tail:all",
        "simulations:structure:stationary",
        "simulation-fit:laplace:stationary:combined",
        "simulation-fit:pgas:stationary:combined",
        "uccle-fit:laplace:TXx:combined",
        "uccle-fit:pgas:TXx:combined",
    }
    missing = sorted(required - set(manifest["tasks"]))
    if missing:
        raise RuntimeError(f"Manifest is missing workflow tasks: {missing}")

    return {
        "bucex_version": bx.__version__,
        "work_dir": str(work_dir),
        "data_artifacts": len(data),
        "tail_artifacts": len(tail),
        "structure_artifacts": len(structure),
        "laplace_simulation_artifacts": len(laplace_sim),
        "pgas_simulation_artifacts": len(pgas_sim),
        "laplace_txx_artifacts": len(laplace_txx),
        "pgas_txx_artifacts": len(pgas_txx),
        "warm_start_source": exact.meta["warm_start_source_engine"],
        "pgas_engine_diagnostics": exact.diagnostics()["engine"],
        "manifest_tasks": sorted(manifest["tasks"]),
        "total_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/presentation_smoke_2.6.0.json"),
    )
    args = parser.parse_args()
    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="bucex-2.6.0-") as directory:
            result = run(Path(directory))
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        result = run(args.work_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
