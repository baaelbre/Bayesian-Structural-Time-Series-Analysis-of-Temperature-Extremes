#!/usr/bin/env python3
"""End-to-end software smoke test for the v2.5 presentation workflow."""
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
    )
    workflow = bx.PresentationWorkflow(config)
    data_table = workflow.validate_data()
    benchmark = workflow.run_txx_benchmarks(
        benchmark="stationary", engine="pgas"
    )[0]
    txx = workflow.run_txx_ssvs(engine="pgas")
    validation = workflow.run_txx_validation(
        model_name="stationary", engine="pgas"
    )
    independent = workflow.run_six_univariate(series="TXm")
    screen = workflow.run_hierarchy_screen(pool="selection")
    exact = workflow.run_hierarchy_pgas(pool="selection")
    sensitivity = workflow.run_sensitivity(pool="both", engine="laplace")
    report = workflow.report(figures=False)

    exact_fit = bx.FitResult.load(exact)
    screen_fit = bx.FitResult.load(screen)
    combined_contract = bx.combine_fits(
        [bx.FitResult.load(txx), bx.FitResult.load(txx)]
    )
    if combined_contract.n_chains != 2:
        raise RuntimeError("Deserialized SSVS archives did not combine as chains.")
    manifest = json.loads(workflow.paths.manifest.read_text(encoding="utf-8"))
    required = {
        "data",
        "txx-benchmarks-stationary-chain-combined",
        "txx-ssvs",
        "txx-validation-stationary",
        "six-univariate-TXm-chain-combined",
        "hierarchy-screen",
        "hierarchy-pgas",
        "sensitivity-both-laplace",
        "report",
    }
    missing = sorted(required - set(manifest["stages"]))
    if missing:
        raise RuntimeError(f"Manifest is missing workflow stages: {missing}")
    if screen_fit.plan.targets_exact_posterior:
        raise RuntimeError("Laplace screen was incorrectly marked exact.")
    if not exact_fit.plan.targets_exact_posterior:
        raise RuntimeError("PGAS fit was not marked exact-invariant.")
    if exact_fit.meta.get("trend_model_space") != "componentwise":
        raise RuntimeError("Workflow did not use componentwise SSVS.")

    return {
        "bucex_version": bx.__version__,
        "work_dir": str(work_dir),
        "data_table": str(data_table),
        "benchmark": str(benchmark),
        "txx": str(txx),
        "validation_artifacts": len(validation),
        "independent_artifacts": len(independent),
        "screen": str(screen),
        "exact": str(exact),
        "sensitivity": str(sensitivity),
        "report_artifacts": len(report),
        "deserialized_combine_chains": combined_contract.n_chains,
        "manifest_stages": sorted(manifest["stages"]),
        "total_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/presentation_smoke_2.5.0.json"),
    )
    args = parser.parse_args()
    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="bucex-2.5.0-") as directory:
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
