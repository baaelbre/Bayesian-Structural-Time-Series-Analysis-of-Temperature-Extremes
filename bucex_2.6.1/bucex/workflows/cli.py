"""Command-line interface for the focused v2.6.1 presentation workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import PresentationConfig, WorkflowPaths
from .scenarios import ALL_SCENARIOS, STRUCTURAL_SCENARIOS
from .uccle import (
    EXTREME_SERIES,
    INFERENCE_ENGINES,
    PRESENTATION_STAGES,
    PresentationWorkflow,
)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot", "publication"),
        default="pilot",
        help="Named runtime profile; individual settings below may override it.",
    )
    parser.add_argument("--start", help="first Uccle month (YYYY-MM-DD)")
    parser.add_argument("--end", help="last Uccle month (YYYY-MM-DD)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/presentation")
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--seed", type=int, default=26_000)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument(
        "--simulation-length",
        "--simulation-months",
        dest="simulation_months",
        type=int,
        help="number of simulated blocks",
    )
    parser.add_argument("--simulation-period", type=int)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=("pdf", "png"),
    )
    parser.add_argument("--figure-dpi", type=int, default=180)


def _config(args: argparse.Namespace) -> PresentationConfig:
    return PresentationConfig.for_profile(
        args.profile,
        start=args.start,
        end=args.end,
        seed=args.seed,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        progress=args.progress,
        draws=args.draws,
        warmup=args.warmup,
        chains=args.chains,
        particles=args.particles,
        simulation_months=args.simulation_months,
        simulation_period=args.simulation_period,
        figure_formats=tuple(args.formats),
        figure_dpi=args.figure_dpi,
    )


def _add_execution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=INFERENCE_ENGINES, default="laplace")
    parser.add_argument("--scenario", choices=tuple(item.name for item in ALL_SCENARIOS))
    parser.add_argument("--series", choices=EXTREME_SERIES)
    parser.add_argument("--chain", type=int, help="one-based independent HPC chain")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="make stage figures (defaults on for data/simulation stages)",
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--strict", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bucex-presentation",
        description=(
            "Generate the Uccle record, structural simulations, Laplace fits, "
            "Laplace-initialized PGAS fits, and presentation results."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="print the resolved experimental plan")
    _add_config(plan)

    run = commands.add_parser("run", help="run one stage or the full local sequence")
    run.add_argument("stage", choices=(*PRESENTATION_STAGES, "all"))
    _add_config(run)
    _add_execution_options(run)

    combine = commands.add_parser(
        "combine", help="combine independent chain files produced on an HPC"
    )
    combine.add_argument("target", choices=("simulation-fit", "uccle-fit"))
    _add_config(combine)
    _add_execution_options(combine)

    report = commands.add_parser(
        "report", help="regenerate every table and figure from combined fits"
    )
    _add_config(report)
    report.add_argument("--strict", action="store_true")
    report.add_argument("--diagnostics", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(args)
    if args.command == "plan":
        paths = WorkflowPaths.from_config(config)
        print(
            json.dumps(
                {
                    "configuration": config.to_dict(),
                    "stages": list(PRESENTATION_STAGES),
                    "tail_scenarios": [
                        item.name for item in ALL_SCENARIOS if item.group == "tail"
                    ],
                    "scale_scenarios": [
                        item.name for item in ALL_SCENARIOS if item.group == "scale"
                    ],
                    "structural_scenarios": [
                        item.name for item in STRUCTURAL_SCENARIOS
                    ],
                    "uccle_series": list(EXTREME_SERIES),
                    "manifest": str(paths.manifest),
                },
                indent=2,
            )
        )
        return 0

    workflow = PresentationWorkflow(config)
    if args.command == "report":
        workflow.report(strict=args.strict, diagnostics=args.diagnostics)
        return 0

    if args.command == "combine":
        figures = True if args.figures is None else args.figures
        common = {
            "engine": args.engine,
            "overwrite": args.overwrite,
            "figures": figures,
            "diagnostics": args.diagnostics,
        }
        if args.target == "simulation-fit":
            workflow.combine_simulation_fits(
                scenario=args.scenario,
                **common,
            )
        else:
            workflow.combine_uccle_fits(series=args.series, **common)
        return 0

    stage = args.stage
    default_figures = stage in {
        "data",
        "tail-simulations",
        "structural-simulations",
    }
    figures = default_figures if args.figures is None else args.figures
    if stage == "all":
        workflow.run_all(
            overwrite=args.overwrite,
            diagnostics=args.diagnostics,
        )
    elif stage == "data":
        workflow.run_data(overwrite=args.overwrite, figures=figures)
    elif stage == "tail-simulations":
        workflow.run_simulations(
            kind="tail",
            scenario=args.scenario,
            overwrite=args.overwrite,
            figures=figures,
        )
    elif stage == "structural-simulations":
        workflow.run_simulations(
            kind="structure",
            scenario=args.scenario,
            overwrite=args.overwrite,
            figures=figures,
        )
    elif stage == "simulation-fit":
        workflow.fit_simulations(
            engine=args.engine,
            scenario=args.scenario,
            chain=args.chain,
            overwrite=args.overwrite,
            figures=figures,
            diagnostics=args.diagnostics,
        )
    elif stage == "uccle-fit":
        workflow.fit_uccle(
            engine=args.engine,
            series=args.series,
            chain=args.chain,
            overwrite=args.overwrite,
            figures=figures,
            diagnostics=args.diagnostics,
        )
    elif stage == "report":
        workflow.report(strict=args.strict, diagnostics=args.diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
