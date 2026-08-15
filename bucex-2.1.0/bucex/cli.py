"""Command-line access to the canonical Uccle workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import FitResult, combine_fits
from .datasets import UCCLE_SERIES, fit_uccle_series, validate_uccle_data
from .inference import Laplace, MCMC, Particles


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _fit_summary(fit: FitResult, path: Path) -> dict[str, object]:
    return {
        "output": str(path),
        "series": fit.series_name,
        "plan": fit.plan.to_dict(),
        "state_shape": list(fit.state_draws.shape),
        "parameters": sorted(fit.parameter_draws),
        "prior_profile": fit.meta["prior_profile"],
        "engine_diagnostics": fit.diagnostics()["engine"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bucex-uccle",
        description="Fit and inspect the six bundled Uccle temperature series.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fit_parser = commands.add_parser("fit", help="fit one Uccle series")
    fit_parser.add_argument("series", choices=UCCLE_SERIES)
    fit_parser.add_argument("--data-dir", type=Path)
    fit_parser.add_argument("--output", type=Path)
    fit_parser.add_argument("--start")
    fit_parser.add_argument("--end")
    fit_parser.add_argument("--engine", choices=("auto", "ffbs", "laplace", "pgas"), default="auto")
    fit_parser.add_argument(
        "--parameterization",
        choices=("auto", "centered", "disturbance", "fruehwirth_schnatter"),
        default="auto",
    )
    fit_parser.add_argument(
        "--priors",
        choices=(
            "manuscript_lasso",
            "regularized_lasso",
            "regularized_horseshoe",
            "triple_gamma",
            "regularized_triple_gamma",
            "pc",
            "normal",
            "ssvs",
        ),
        default="manuscript_lasso",
    )
    fit_parser.add_argument("--asis", action=argparse.BooleanOptionalAction, default=True)
    fit_parser.add_argument("--draws", type=int, default=2_000)
    fit_parser.add_argument("--warmup", type=int, default=1_000)
    fit_parser.add_argument("--thin", type=int, default=1)
    fit_parser.add_argument("--chains", type=int, default=4)
    fit_parser.add_argument("--particles", type=int, default=256)
    fit_parser.add_argument(
        "--particle-proposal", choices=("bootstrap", "guided"), default="guided"
    )
    fit_parser.add_argument("--laplace-iterations", type=int, default=30)
    fit_parser.add_argument("--laplace-tolerance", type=float, default=1e-5)
    fit_parser.add_argument("--seed", type=int, default=40)
    fit_parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    validate_parser = commands.add_parser(
        "validate-data", help="validate bundled or supplied Uccle data"
    )
    validate_parser.add_argument("--data-dir", type=Path)
    validate_parser.add_argument("--check-daily", action="store_true")

    inspect_parser = commands.add_parser("inspect", help="inspect a saved fit")
    inspect_parser.add_argument("fit", type=Path)

    combine_parser = commands.add_parser(
        "combine", help="combine compatible independently saved fits as chains"
    )
    combine_parser.add_argument("fits", nargs="+", type=Path)
    combine_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        table = validate_uccle_data(args.data_dir, check_daily=args.check_daily)
        print(
            json.dumps(
                table.reset_index().to_dict(orient="records"),
                indent=2,
                default=_json_default,
            )
        )
        return 0

    if args.command == "inspect":
        fit = FitResult.load(args.fit)
        print(json.dumps(_fit_summary(fit, args.fit), indent=2, default=_json_default))
        return 0

    if args.command == "combine":
        combined = combine_fits(FitResult.load(path) for path in args.fits)
        combined.save(args.output)
        print(
            json.dumps(
                _fit_summary(combined, args.output),
                indent=2,
                default=_json_default,
            )
        )
        return 0

    output = args.output or Path("results") / f"{args.series}.bucex"
    output.parent.mkdir(parents=True, exist_ok=True)
    fit = fit_uccle_series(
        args.series,
        args.data_dir,
        start=args.start,
        end=args.end,
        engine=args.engine,
        parameterization=args.parameterization,
        priors=args.priors,
        asis=args.asis,
        mcmc=MCMC(
            draws=args.draws,
            warmup=args.warmup,
            thin=args.thin,
            chains=args.chains,
            seed=args.seed,
            progress=args.progress,
        ),
        particles=Particles(
            n=args.particles,
            proposal=args.particle_proposal,
        ),
        laplace=Laplace(
            max_iterations=args.laplace_iterations,
            tolerance=args.laplace_tolerance,
        ),
    )
    fit.save(output)
    print(json.dumps(_fit_summary(fit, output), indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
