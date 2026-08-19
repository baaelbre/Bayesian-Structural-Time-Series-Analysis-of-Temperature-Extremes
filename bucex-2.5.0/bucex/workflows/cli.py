"""Command-line driver for the staged Uccle presentation workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..datasets import UCCLE_SERIES
from .config import PresentationConfig, WorkflowPaths
from .uccle import (
    BENCHMARK_MODELS,
    PRESENTATION_STAGES,
    VALIDATION_MODELS,
    PresentationWorkflow,
)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=("smoke", "pilot", "publication"), default="pilot")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output-dir", type=Path, default=Path("results/presentation"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--seed", type=int, default=25_000)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)


def _config(args: argparse.Namespace) -> PresentationConfig:
    return PresentationConfig.for_profile(
        args.profile,
        start=args.start,
        end=args.end,
        pool=getattr(args, "pool", None) or "selection",
        seed=args.seed,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        progress=args.progress,
        draws=args.draws,
        warmup=args.warmup,
        chains=args.chains,
        particles=args.particles,
        channel_workers=args.workers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bucex-presentation",
        description="Run the Uccle analysis in paper/presentation order.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="print stages and resolved settings")
    _add_config(plan)

    run = commands.add_parser("run", help="run one stage or the local sequence")
    run.add_argument(
        "stage",
        choices=(*PRESENTATION_STAGES, "all"),
    )
    _add_config(run)
    run.add_argument("--benchmark", choices=BENCHMARK_MODELS)
    run.add_argument("--validation-model", choices=VALIDATION_MODELS)
    run.add_argument("--series", choices=UCCLE_SERIES)
    run.add_argument("--pool", choices=("selection", "slab", "both"), default="selection")
    run.add_argument("--engine", choices=("laplace", "pgas"), default="pgas")
    run.add_argument("--chain", type=int)
    run.add_argument("--screen", type=Path)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--figures", action="store_true")
    run.add_argument("--diagnostic-plots", action="store_true")
    run.add_argument("--include-sensitivity", action="store_true")
    run.add_argument("--initial", type=int, help="initial TXx observations for LFO")
    run.add_argument("--horizon", type=int, default=12)
    run.add_argument("--step", type=int)

    combine = commands.add_parser("combine", help="combine independent HPC chains")
    combine.add_argument(
        "target",
        choices=("txx-benchmarks", "txx-ssvs", "six-univariate", "hierarchy-pgas", "sensitivity"),
    )
    _add_config(combine)
    combine.add_argument("--benchmark", choices=BENCHMARK_MODELS)
    combine.add_argument("--series", choices=UCCLE_SERIES)
    combine.add_argument("--pool", choices=("selection", "slab", "both"), default="selection")
    combine.add_argument("--engine", choices=("laplace", "pgas"), default="pgas")
    combine.add_argument("--overwrite", action="store_true")
    combine.add_argument("--figures", action="store_true")
    combine.add_argument("--diagnostic-plots", action=argparse.BooleanOptionalAction, default=True)

    report = commands.add_parser("report", help="regenerate outputs from saved fits")
    _add_config(report)
    report.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True)
    report.add_argument("--diagnostic-plots", action="store_true")
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
                    "txx_benchmarks": list(BENCHMARK_MODELS),
                    "manifest": str(paths.manifest),
                },
                indent=2,
            )
        )
        return 0

    workflow = PresentationWorkflow(config)
    if args.command == "report":
        workflow.report(
            figures=args.figures, diagnostic_plots=args.diagnostic_plots
        )
        return 0
    if args.command == "combine":
        workflow.combine(
            args.target,
            benchmark=args.benchmark,
            series=args.series,
            pool=args.pool,
            engine=args.engine,
            overwrite=args.overwrite,
            figures=args.figures,
            diagnostic_plots=args.diagnostic_plots,
        )
        return 0

    stage = args.stage
    common = {"overwrite": args.overwrite, "figures": args.figures}
    if stage == "data":
        workflow.validate_data()
    elif stage == "txx-benchmarks":
        workflow.run_txx_benchmarks(
            benchmark=args.benchmark,
            chain=args.chain,
            engine=args.engine,
            **common,
        )
    elif stage == "txx-ssvs":
        workflow.run_txx_ssvs(
            chain=args.chain,
            engine=args.engine,
            diagnostic_plots=args.diagnostic_plots,
            **common,
        )
    elif stage == "txx-validation":
        workflow.run_txx_validation(
            model_name=args.validation_model,
            engine=args.engine,
            initial=args.initial,
            horizon=args.horizon,
            step=args.step,
            overwrite=args.overwrite,
        )
    elif stage == "six-univariate":
        workflow.run_six_univariate(
            series=args.series,
            chain=args.chain,
            gev_engine=args.engine,
            **common,
        )
    elif stage == "hierarchy-screen":
        workflow.run_hierarchy_screen(pool=args.pool, **common)
    elif stage == "hierarchy-pgas":
        workflow.run_hierarchy_pgas(
            pool=args.pool,
            chain=args.chain,
            screen=args.screen,
            diagnostic_plots=args.diagnostic_plots,
            **common,
        )
    elif stage == "sensitivity":
        workflow.run_sensitivity(
            pool=args.pool,
            engine=args.engine,
            chain=args.chain,
            **common,
        )
    elif stage == "report":
        workflow.report(
            figures=args.figures, diagnostic_plots=args.diagnostic_plots
        )
    elif stage == "all":
        workflow.run_all(
            overwrite=args.overwrite,
            figures=args.figures,
            include_sensitivity=args.include_sensitivity,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
