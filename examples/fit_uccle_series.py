"""Fit one Uccle temperature series with bucex.

Example:
    python examples/fit_uccle_series.py \
        --series TXx \
        --data-dir data \
        --out-dir results/uccle_v033_regularized \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from bucex import __version__, fit_uccle_series


parser = argparse.ArgumentParser()

parser.add_argument(
    "--series",
    required=True,
    choices=["TXm", "TNm", "TXx", "TXn", "TNx", "TNn"],
)

parser.add_argument("--data-dir", default="data")
parser.add_argument(
    "--out-dir",
    default=None,
    help="Output directory. Default: results/uccle_v033_<priors>.",
)

parser.add_argument("--n-iter", type=int, default=8_000)
parser.add_argument("--burn", type=int, default=1_500)
parser.add_argument("--thin", type=int, default=1)
parser.add_argument("--seed", type=int, default=40)
parser.add_argument(
    "--max-state-tries",
    type=int,
    default=25,
    help="Maximum Laplace retry attempts before restoring an iteration.",
)

parser.add_argument(
    "--priors",
    default="regularized",
    choices=["manuscript", "normal", "regularized", "ssvs"],
    help="Prior profile: manuscript, scale-aware normal, component-wise regularized lasso, or structural SSVS.",
)

args = parser.parse_args()

series = args.series

out_dir = Path(args.out_dir or f"results/uccle_v033_{args.priors}")
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
    state_method="laplace" if series in {"TXx", "TXn", "TNx", "TNn"} else "auto",
    state_kwargs={"max_state_tries": args.max_state_tries},
)

# Save the complete posterior object immediately.
fit_file = fit_dir / f"{series}_bucex_v{__version__}.pkl"
fit.save(fit_file)

# Save static posterior summaries.
summary = pd.DataFrame(fit.static_summary()).T
summary.index.name = "parameter"
summary.to_csv(summary_dir / f"{series}_posterior_summary.csv")

# Finite-period rates based on the fitted level, not the instantaneous slope.
periods = {
    "early_1892_1949": (1892, 1949),
    "mid_1950_1979": (1950, 1979),
    "recent_1980_2022": (1980, 2022),
}
fit.period_rate_summary(periods).to_csv(
    summary_dir / f"{series}_period_rates.csv"
)
pd.DataFrame(
    [
        fit.rate_contrast_summary(
            recent=(1980, 2022),
            reference=(1950, 1979),
        )
    ]
).to_csv(summary_dir / f"{series}_rate_acceleration.csv", index=False)

if args.priors == "ssvs":
    fit.component_probabilities().to_csv(
        summary_dir / f"{series}_component_probabilities.csv"
    )
    fit.structural_model_probabilities().to_csv(
        summary_dir / f"{series}_structural_models.csv", index=False
    )
    fit.component_transition_summary().to_csv(
        summary_dir / f"{series}_component_transitions.csv"
    )

    fig, _ = fit.plot(type="component_probabilities")
    fig.savefig(
        figure_dir / f"{series}_component_probabilities.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

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