# uccle_dgev_plotter.py
from __future__ import annotations
"""
Uccle DGEV Laplace Plotter (TXx, TXn, TNx, TNn, Precx; Seasonal / Monthly)
===========================================================================

This is the *Uccle* wrapper around the generic DGEV Laplace harmonic plotter:

• Same figures, same internals (from DGEVPlotter):
  - overview.png
  - states.png
  - traces_acf__*.png
  - posteriors__*.png
  - quick_report.png
  - return_levels_N*.png
  - return_periods_u*.png

• Uccle-specific default roots (where the Laplace runs live):
  TXx, Seasonal  → results/uccle/TX/TXx/Seasonal/Laplace
  TXx, Monthly   → results/uccle/TX/TXx/Monthly/Laplace
  TXn, Seasonal  → results/uccle/TX/TXn/Seasonal/Laplace
  TXn, Monthly   → results/uccle/TX/TXn/Monthly/Laplace
  TNx, Seasonal  → results/uccle/TN/TNx/Seasonal/Laplace
  TNx, Monthly   → results/uccle/TN/TNx/Monthly/Laplace
  TNn, Seasonal  → results/uccle/TN/TNn/Seasonal/Laplace
  TNn, Monthly   → results/uccle/TN/TNn/Monthly/Laplace
  Precx, Seasonal→ results/uccle/Prec/Precx/Seasonal/Laplace
  Precx, Monthly → results/uccle/Prec/Precx/Monthly/Laplace

Usage
-----
# Example: latest Seasonal TXx Laplace run
python -u uccle_dgev_plotter.py --series TXx --agg Seasonal --show

# Example: Monthly TNn, only quick report
python -u uccle_dgev_plotter.py --series TNn --agg Monthly \
  --skip-overview --skip-states --skip-grouped-traces --skip-grouped-post --show

# Or point directly at a posterior.npz or its run directory:
python -u uccle_dgev_plotter.py \
  --target results/uccle/TX/TXx/Seasonal/Laplace/TXx_dynamic_dynamic_dynamic_20251116_151500

# Example with return levels / periods:
python -u uccle_dgev_plotter.py --series TXx --agg Seasonal \
  --rl-N 50 --rl-season 2 --rp-u 35 --rp-yearly
"""

import os
import sys
import argparse
from typing import Optional

from simulator.dgev_laplace_plotter import (  # type: ignore
    DGEVPlotter,
    load_posterior,
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p: Optional[str]) -> None:
    """Create directory `p` if not empty and it does not yet exist."""
    if p:
        os.makedirs(p, exist_ok=True)


def _uccle_root(series: str, agg: str) -> str:
    """
    Map (series, aggregation) to the root directory that contains Laplace runs.

    series ∈ {TXx, TXn, TNx, TNn, Precx}
    agg    ∈ {"Seasonal", "Monthly"}  (case-insensitive, first letter enough)

    Returns a path like:
      results/uccle/TX/TXx/Seasonal/Laplace
    """
    base = "results/uccle"

    agg_norm = agg.strip().lower()
    if agg_norm.startswith("s"):
        agg_dir = "Seasonal"
    elif agg_norm.startswith("m"):
        agg_dir = "Monthly"
    else:
        raise ValueError(f"Unknown aggregation '{agg}' (use Seasonal or Monthly).")

    mapping = {
        "TXx": os.path.join(base, "TX", "TXx", agg_dir, "Laplace"),
        "TXn": os.path.join(base, "TX", "TXn", agg_dir, "Laplace"),
        "TNx": os.path.join(base, "TN", "TNx", agg_dir, "Laplace"),
        "TNn": os.path.join(base, "TN", "TNn", agg_dir, "Laplace"),
        "Precx": os.path.join(base, "Prec", "Precx", agg_dir, "Laplace"),
    }
    if series not in mapping:
        raise ValueError(
            f"Unknown series '{series}'. Expected one of {list(mapping.keys())}."
        )
    return mapping[series]


def _find_latest_posterior_npz(root: str) -> str:
    """
    Recursively search under `root` for posterior_*.npz and return the path
    to the *most recent* one (by modification time).

    This matches the behavior where each Laplace run lives in a subdirectory like:
      results/uccle/TX/TXx/Seasonal/Laplace/TXx_dynamic_dynamic_dynamic_20251116_151500
    and that subdir contains a single posterior_*.npz + meta.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root directory '{root}' does not exist.")

    best_path: Optional[str] = None
    best_mtime: float = -1.0

    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".npz"):
                continue
            if not fn.startswith("posterior_"):
                # If you want to allow any .npz, drop this check.
                continue
            full = os.path.join(dirpath, fn)
            mtime = os.path.getmtime(full)
            if mtime > best_mtime:
                best_mtime = mtime
                best_path = full

    if best_path is None:
        raise FileNotFoundError(f"No posterior_*.npz found under '{root}'.")

    return best_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Uccle DGEV Laplace Plotter (TXx/TXn/TNx/TNn/Precx; Seasonal/Monthly)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Either point directly at a run or let the wrapper find the latest under the Uccle root
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Run dir or posterior.npz. "
            "If omitted, we search in the Uccle Laplace root derived from --series and --agg, "
            "and take the latest posterior_*.npz."
        ),
    )

    p.add_argument(
        "--series",
        type=str,
        choices=["TXx", "TXn", "TNx", "TNn", "Precx"],
        default="TXx",
        help="Which Uccle DGEV series to plot.",
    )

    p.add_argument(
        "--agg",
        type=str,
        choices=["Seasonal", "Monthly"],
        default="Monthly",
        help="Seasonal (DJF/MAM/JJA/SON) or Monthly aggregation.",
    )

    p.add_argument(
        "--level",
        type=float,
        default=0.90,
        help="Credible band level for μ_t and states.",
    )

    p.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show figures interactively.",
    )

    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for figures (default: <run-dir>/figures).",
    )

    # toggles for figure families
    p.add_argument("--skip-overview", action="store_true", default=False)
    p.add_argument("--skip-states", action="store_true", default=False)
    p.add_argument("--skip-grouped-traces", action="store_true", default=False)
    p.add_argument("--skip-grouped-post", action="store_true", default=False)
    p.add_argument("--skip-quick", action="store_true", default=False)

    p.add_argument(
        "--max-lag",
        type=int,
        default=200,
        help="ACF / ESS max lag for trace plots.",
    )

    # -----------------------------------------------------------------------
    # Return levels / return periods
    # -----------------------------------------------------------------------
    p.add_argument(
        "--rl-N",
        type=int,
        default=1000,
        help=(
            "If set, plot block-wise N-block return level time series "
            "(N is in block units: years if each block is yearly)."
        ),
    )
    p.add_argument(
        "--rl-season",
        type=int,
        default=None,
        help=(
            "Season index within the period for return levels (0..period-1). "
            "Default: all blocks."
        ),
    )
    p.add_argument(
        "--rp-u",
        type=float,
        default=None,
        help=(
            "If set, plot time-varying return periods for threshold u "
            "in data units."
        ),
    )
    p.add_argument(
        "--rp-yearly",
        action="store_true",
        default=False,
        help=(
            "If set, aggregate within each calendar year when computing "
            "return periods (at least one exceedance in that year). "
            "Otherwise, block-wise return periods."
        ),
    )
    p.add_argument(
        "--rp-season",
        type=int,
        default=None,
        help=(
            "Season index within the period for block-wise return periods "
            "(0..period-1). Ignored if --rp-yearly is set. Default: all blocks."
        ),
    )

    args = p.parse_args()

    # -----------------------------------------------------------------------
    # Resolve which posterior to load
    # -----------------------------------------------------------------------
    if args.target is not None:
        # Use the generic loader; it will handle both a directory and a .npz path.
        draws, meta, npz_path = load_posterior(args.target)
    else:
        root = _uccle_root(args.series, args.agg)
        npz_latest = _find_latest_posterior_npz(root)
        draws, meta, npz_path = load_posterior(npz_latest)

    run_dir = os.path.dirname(npz_path)
    out_dir = args.out or os.path.join(run_dir, "figures")
    _ensure_dir(out_dir)
    print(f"[info] Uccle DGEV ({args.series}, {args.agg})")
    print(f"[info] Posterior source: {npz_path}")
    print(f"[info] Saving figures to: {out_dir}")

    # -----------------------------------------------------------------------
    # Build plotter and generate figures
    # -----------------------------------------------------------------------
    pl = DGEVPlotter(draws=draws, meta=meta, level=float(args.level))

    if not args.skip_overview:
        pl.figure_overview(out_dir, args.show)

    if not args.skip_states:
        pl.figure_states(out_dir, args.show)

    if not args.skip_grouped_traces:
        pl.figure_traces_grouped_all(out_dir, args.show, int(args.max_lag))

    #if not args.skip_grouped_post:
    #    pl.figure_posteriors_grouped_all(out_dir, args.show)

    if not args.skip_quick:
        pl.quick_report(out_dir, args.show)

    # -----------------------------------------------------------------------
    # Return levels / return periods
    # -----------------------------------------------------------------------
    if args.rl_N is not None:
        pl.figure_return_levels(
            N=float(args.rl_N),
            season=args.rl_season,
            save_dir=out_dir,
            show=args.show,
        )

    if args.rp_u is not None:
        pl.figure_return_periods(
            u=float(args.rp_u),
            yearly=bool(args.rp_yearly),
            season=None if args.rp_yearly else args.rp_season,
            save_dir=out_dir,
            show=args.show,
        )

    print("[done] Uccle DGEV Laplace plots written.")
