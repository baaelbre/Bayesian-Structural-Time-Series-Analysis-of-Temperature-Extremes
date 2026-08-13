"""Re-fit the six Uccle series and create the central manuscript-style figures.

A full run uses the manuscript setting of 20,000 iterations and 5,000 burn-in.
Use --quick to verify the complete workflow without waiting for convergence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from bucex import fit_uccle_all

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
parser.add_argument("--out-dir", default="results/bucex_v0.2")
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

n_iter = 150 if args.quick else 20_000
burn = 50 if args.quick else 5_000

fits = fit_uccle_all(
    args.data_dir,
    n_iter=n_iter,
    burn=burn,
    thin=1,
    seed=40,
    progress=True,
)
fits.save(out / "fits")
fits.summary().to_csv(out / "posterior_summary.csv")

fig, _ = fits.plot(type="level_slope", credible_interval=0.90)
fig.savefig(out / "levels_and_slopes.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, _ = fits["TXx"].plot(
    type="return_period",
    threshold=[36.8, 39.7],
    annual=True,
    credible_interval=0.90,
    max_return_period=10_000,
)
fig.savefig(out / "TXx_return_periods.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, _ = fits["TXx"].plot(
    type="endpoint",
    threshold=39.7,
    credible_interval=0.90,
)
fig.savefig(out / "TXx_endpoint.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, _ = fits["TNx"].plot(
    type="exceedance",
    threshold=20.0,
    annual=True,
    credible_interval=0.90,
)
fig.savefig(out / "TNx_tropical_night_probability.png", dpi=300, bbox_inches="tight")
plt.close(fig)
