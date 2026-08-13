#!/usr/bin/env python3
"""Fit one Uccle series with the bucex 1.1 FS engine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bucex as bx


parser = argparse.ArgumentParser()
parser.add_argument("--series", required=True, choices=bx.UCCLE_SERIES)
parser.add_argument("--data-dir", default=None)
parser.add_argument("--out-dir", type=Path, default=None)
parser.add_argument("--engine", choices=("auto", "laplace", "pgas"), default="auto")
parser.add_argument(
    "--priors",
    choices=(
        "manuscript",
        "regularized_lasso",
        "horseshoe",
        "pc",
        "normal",
        "ssvs",
    ),
    default="regularized_lasso",
)
parser.add_argument("--draws", type=int, default=2000)
parser.add_argument("--warmup", type=int, default=2000)
parser.add_argument("--thin", type=int, default=1)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--seed", type=int, default=40)
parser.add_argument("--particles", type=int, default=256)
parser.add_argument("--proposal", choices=("guided", "bootstrap"), default="guided")
parser.add_argument("--asis", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--progress", action="store_true")
parser.add_argument(
    "--quick",
    action="store_true",
    help="Run a short smoke fit in a separate default output directory.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Explicitly replace an existing fit archive.",
)
args = parser.parse_args()

if args.quick:
    args.draws = min(args.draws, 20)
    args.warmup = min(args.warmup, 20)
    args.chains = 1
    args.particles = min(args.particles, 64)

default_root = "results/uccle_v1_1_quick" if args.quick else "results/uccle_v1_1"
out_dir = Path(default_root) if args.out_dir is None else args.out_dir
fit_dir = out_dir / "fits"
summary_dir = out_dir / "summaries"
diagnostic_dir = out_dir / "diagnostics"
figure_dir = out_dir / "figures"
for directory in (fit_dir, summary_dir, diagnostic_dir, figure_dir):
    directory.mkdir(parents=True, exist_ok=True)

fit_path = fit_dir / f"{args.series}.bucex"
if fit_path.exists() and not args.overwrite:
    raise FileExistsError(
        f"{fit_path} already exists; choose another --out-dir or pass --overwrite."
    )

fit = bx.fit_uccle_series(
    args.series,
    args.data_dir,
    priors=args.priors,
    engine=args.engine,
    parameterization="fruehwirth_schnatter",
    asis=args.asis,
    mcmc=bx.MCMC(
        draws=args.draws,
        warmup=args.warmup,
        thin=args.thin,
        chains=args.chains,
        seed=args.seed,
        progress=args.progress,
    ),
    particles=bx.Particles(n=args.particles, proposal=args.proposal),
)
fit.save(fit_path)

pd.DataFrame(fit.static_summary()).T.to_csv(
    summary_dir / f"{args.series}_posterior.csv"
)
diagnostics = fit.diagnostics()
diagnostics["parameters"].to_csv(diagnostic_dir / f"{args.series}_mcmc.csv")
with (diagnostic_dir / f"{args.series}_engine.json").open("w", encoding="utf-8") as handle:
    json.dump(
        {
            "engine": diagnostics["engine"],
            "warnings": diagnostics["warnings"],
            "inference": fit.methods,
            "prior_profile": fit.meta["prior_profile"],
            "chain_seeds": fit.meta["chain_seeds"],
        },
        handle,
        indent=2,
    )

years = pd.to_datetime(fit.dates).year
if years.min() <= 1892 and years.max() >= 2022:
    periods = {
        "early_1892_1949": (1892, 1949),
        "mid_1950_1979": (1950, 1979),
        "recent_1980_2022": (1980, 2022),
    }
    fit.period_rate_summary(periods).to_csv(
        summary_dir / f"{args.series}_period_rates.csv"
    )
    pd.DataFrame(
        [fit.rate_contrast_summary((1980, 2022), (1950, 1979))]
    ).to_csv(summary_dir / f"{args.series}_acceleration.csv", index=False)

if args.priors == "ssvs":
    fit.component_probabilities().to_csv(
        summary_dir / f"{args.series}_component_probabilities.csv"
    )
    fit.structural_model_probabilities().to_csv(
        summary_dir / f"{args.series}_structural_models.csv", index=False
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for kind in ("level_slope", "process_sd"):
    figure, _ = fit.plot(type=kind)
    figure.savefig(
        figure_dir / f"{args.series}_{kind}.png", dpi=250, bbox_inches="tight"
    )
    plt.close(figure)
if fit.family == "gev" and np.any(fit.parameter("xi") < 0.0):
    figure, _ = fit.plot(type="endpoint")
    figure.savefig(
        figure_dir / f"{args.series}_endpoint.png", dpi=250, bbox_inches="tight"
    )
    plt.close(figure)

print(f"Saved {fit.family}/{fit.plan.engine} FS fit to {fit_path}")
