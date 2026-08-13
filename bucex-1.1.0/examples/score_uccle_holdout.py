#!/usr/bin/env python3
"""Fit a training window and calculate bulk and tail-weighted holdout scores."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import bucex as bx


parser = argparse.ArgumentParser()
parser.add_argument("--series", required=True, choices=bx.UCCLE_SERIES)
parser.add_argument("--data-dir", default=None)
parser.add_argument("--holdout", type=int, default=120, help="Number of final months held out.")
parser.add_argument("--out-dir", type=Path, default=Path("results/uccle_v1_1_holdout"))
parser.add_argument("--engine", choices=("auto", "laplace", "pgas"), default="auto")
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--warmup", type=int, default=1000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--particles", type=int, default=256)
parser.add_argument("--forecast-draws", type=int, default=4000)
parser.add_argument("--seed", type=int, default=80)
args = parser.parse_args()

values = bx.load_uccle_series(args.series, args.data_dir)
if not 12 <= args.holdout < len(values) - 24:
    raise ValueError("holdout must leave at least 24 training observations.")
training = values.iloc[: -args.holdout]
observed = values.iloc[-args.holdout :]
info = bx.UCCLE_INFO[args.series]

fit = bx.fit(
    training,
    family=info["family"],
    period=12,
    tail=info["tail"],
    engine=args.engine,
    parameterization="auto",
    asis=True,
    mcmc=bx.MCMC(
        draws=args.draws,
        warmup=args.warmup,
        chains=args.chains,
        seed=args.seed,
    ),
    particles=bx.Particles(n=args.particles),
)
forecast = fit.forecast(
    args.holdout,
    draws=args.forecast_draws,
    seed=args.seed + 1,
    dates=observed.index.to_numpy(),
)

probabilities = np.asarray([0.90, 0.95, 0.99])
threshold_probabilities = probabilities if info["tail"] == "max" else 1.0 - probabilities
thresholds = np.quantile(training.to_numpy(), threshold_probabilities)
scores = forecast.score(
    observed.to_numpy(),
    thresholds=[float(value) for value in thresholds],
    quantiles=probabilities.tolist(),
)

args.out_dir.mkdir(parents=True, exist_ok=True)
fit.save(args.out_dir / f"{args.series}_training.bucex")
forecast.summary().to_csv(args.out_dir / f"{args.series}_forecast.csv", index=False)
scores.to_csv(args.out_dir / f"{args.series}_scores.csv", index=False)
print(scores.to_string(index=False))
