"""Fit one Uccle temperature series with bucex.

Example:
    python examples/fit_uccle_series.py \
        --series TXx \
        --data-dir data \
        --out-dir results/uccle_v02 \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from bucex import fit_uccle_series


parser = argparse.ArgumentParser()

parser.add_argument(
    "--series",
    required=True,
    choices=["TXm", "TNm", "TXx", "TXn", "TNx", "TNn"],
)

parser.add_argument("--data-dir", default="data")
parser.add_argument("--out-dir", default="results/uccle_v02")

parser.add_argument("--n-iter", type=int, default=20_000)
parser.add_argument("--burn", type=int, default=5_000)
parser.add_argument("--thin", type=int, default=1)
parser.add_argument("--seed", type=int, default=40)

parser.add_argument(
    "--priors",
    default="manuscript",
    help="Prior profile passed to fit_uccle_series.",
)

args = parser.parse_args()

series = args.series

out_dir = Path(args.out_dir)
fit_dir = out_dir / "fits"
figure_dir = out_dir / "figures"
summary_dir = out_dir / "summaries"

fit_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)
summary_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"Fitting {series}")
print(f"Iterations: {args.n_iter}")
print(f"Burn-in:    {args.burn}")
print(f"Thin:       {args.thin}")
print(f"Seed:       {args.seed}")
print(f"Priors:     {args.priors}")
print("=" * 70)

fit = fit_uccle_series(
    series=series,
    data_dir=args.data_dir,
    priors=args.priors,
    n_iter=args.n_iter,
    burn=args.burn,
    thin=args.thin,
    seed=args.seed,
    progress=True,
)

# Save the complete posterior object immediately.
fit_file = fit_dir / f"{series}_bucex_v0.2.pkl"
fit.save(fit_file)

# Save static posterior summaries.
summary = pd.DataFrame(fit.static_summary()).T
summary.index.name = "parameter"
summary.to_csv(summary_dir / f"{series}_posterior_summary.csv")

# Level and slope.
fig, _ = fit.plot(
    type="level_slope",
    credible_interval=0.90,
)

fig.savefig(
    figure_dir / f"{series}_level_slope.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

# GEV-specific figures.
if series in {"TXx", "TXn", "TNx", "TNn"}:
    fig, _ = fit.plot(
        type="endpoint",
        credible_interval=0.90,
    )

    fig.savefig(
        figure_dir / f"{series}_endpoint.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

print(f"Finished {series}")
print(f"Saved fit to: {fit_file}")