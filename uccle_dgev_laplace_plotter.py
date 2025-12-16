# uccle_dgev_laplace_plotter.py
from __future__ import annotations
"""
Uccle DGEV Laplace Plotter (TXx, TXn, TNx, TNn, Precx; Seasonal / Monthly)
===========================================================================

Uccle wrapper around the generic Laplace DGEV plotter:

    simulator.dgev_laplace_plotter.DGEVPlotter
    simulator.dgev_laplace_plotter.load_posterior

Key features
------------
- Same figures as the generic DGEV plotter (overview, trace/hist/ACF, states, quick report).
- Uccle-specific default roots for Laplace runs.
- Robust "latest run" discovery even if files are named posterior_*.npz (not necessarily posterior.npz).
- Works with load_posterior returning a PosteriorBundle object (NOT a tuple).
- Optional post-hoc burn-in and thinning on the stored draws.

Default Uccle Laplace roots
---------------------------
  TXx, Seasonal   → results/uccle/TX/TXx/Seasonal/Laplace
  TXx, Monthly    → results/uccle/TX/TXx/Monthly/Laplace
  TXn, Seasonal   → results/uccle/TX/TXn/Seasonal/Laplace
  TXn, Monthly    → results/uccle/TX/TXn/Monthly/Laplace
  TNx, Seasonal   → results/uccle/TN/TNx/Seasonal/Laplace
  TNx, Monthly    → results/uccle/TN/TNx/Monthly/Laplace
  TNn, Seasonal   → results/uccle/TN/TNn/Seasonal/Laplace
  TNn, Monthly    → results/uccle/TN/TNn/Monthly/Laplace
  Precx, Seasonal → results/uccle/Prec/Precx/Seasonal/Laplace
  Precx, Monthly  → results/uccle/Prec/Precx/Monthly/Laplace

Examples
--------
# latest Seasonal TXx Laplace run
python -u uccle_dgev_laplace_plotter.py --series TXx --agg Seasonal --show

# monthly TNn, only quick report
python -u uccle_dgev_laplace_plotter.py --series TNn --agg Monthly \
  --skip-overview --skip-states --skip-traces --show

# explicit run directory or posterior .npz
python -u uccle_dgev_laplace_plotter.py --target path/to/run_or_posterior.npz

# with post-hoc burn/thin
python -u uccle_dgev_laplace_plotter.py --series TXx --agg Seasonal --burn 1000 --thin 5
"""

import os
import re
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simulator.dgev_laplace_plotter import DGEVPlotter, load_posterior  # type: ignore


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_dir(p: Optional[str]) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def uccle_root(series: str, agg: str) -> str:
    """
    Map (series, aggregation) to root containing Laplace runs.

    series ∈ {TXx, TXn, TNx, TNn, Precx}
    agg    ∈ {Seasonal, Monthly} (case-insensitive; "s"/"m" ok)
    """
    base = "results/uccle"
    s = str(series).strip()
    a = str(agg).strip().lower()

    if a.startswith("s"):
        agg_dir = "Seasonal"
    elif a.startswith("m"):
        agg_dir = "Monthly"
    else:
        raise ValueError(f"Unknown aggregation {agg!r} (use Seasonal or Monthly).")

    mapping = {
        "TXx": os.path.join(base, "TX", "TXx", agg_dir, "Laplace"),
        "TXn": os.path.join(base, "TX", "TXn", agg_dir, "Laplace"),
        "TNx": os.path.join(base, "TN", "TNx", agg_dir, "Laplace"),
        "TNn": os.path.join(base, "TN", "TNn", agg_dir, "Laplace"),
        "Precx": os.path.join(base, "Prec", "Precx", agg_dir, "Laplace"),
    }
    if s not in mapping:
        raise ValueError(f"Unknown series {series!r}. Expected one of {list(mapping.keys())}.")
    return mapping[s]


def _extract_ts(path_str: str) -> Optional[float]:
    """
    Extract YYYYMMDD_HHMMSS from path (common in your run directory names),
    return epoch seconds. None if not found/parsable.
    """
    m = re.search(r"(\d{8})_(\d{6})", path_str)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.timestamp()
    except Exception:
        return None


def find_latest_posterior_npz(root: str) -> Optional[str]:
    """
    Recursively search for posterior*.npz under `root` and return the latest.

    Preference:
      1) largest YYYYMMDD_HHMMSS found in the *path*
      2) fallback: largest modification time
    """
    rp = Path(root)
    if not rp.exists():
        return None

    cands = list(rp.rglob("posterior*.npz"))
    if not cands:
        # also allow any .npz containing "posterior" (more permissive)
        cands = [p for p in rp.rglob("*.npz") if "posterior" in p.name.lower()]
    if not cands:
        return None

    def key(p: Path) -> Tuple[int, float]:
        ts = _extract_ts(str(p))
        if ts is not None:
            return (1, ts)
        return (0, p.stat().st_mtime)

    best = max(cands, key=key)
    return str(best)


def resolve_bundle(
    target: Optional[str],
    series: str,
    agg: str,
    root: Optional[str],
) -> Any:
    """
    Returns PosteriorBundle from load_posterior() (NOT a tuple).
    """
    if target:
        return load_posterior(target)

    search_root = root or uccle_root(series, agg)
    print(f"[info] searching latest posterior under: {search_root!r}")

    npz = find_latest_posterior_npz(search_root)
    if npz is None:
        print(
            f"[error] No posterior .npz found under {search_root!r}.\n"
            f"  → Provide --target or check that your Laplace run wrote posterior*.npz."
        )
        raise SystemExit(1)

    print(f"[info] using latest posterior: {npz}")
    return load_posterior(npz)


def apply_burn_thin(
    draws: Dict[str, Any],
    meta: Dict[str, Any],
    burn: int = 0,
    thin: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Post-hoc burn-in + thinning for arrays whose first dim matches n_samp
    inferred from draws['mu'].
    """
    if "mu" not in draws:
        print("[warn] 'mu' not in draws; skipping post-hoc burn/thin.")
        return draws, meta

    burn = int(burn or 0)
    thin = int(thin or 1)
    if burn < 0:
        raise ValueError(f"--burn must be >= 0, got {burn}")
    if thin < 1:
        raise ValueError(f"--thin must be >= 1, got {thin}")

    mu = np.asarray(draws["mu"])
    if mu.ndim < 2:
        print("[warn] 'mu' does not look like (n_samp, T); skipping post-hoc burn/thin.")
        return draws, meta

    n_samp = int(mu.shape[0])
    if burn >= n_samp:
        raise ValueError(f"--burn={burn} ≥ number of draws ({n_samp}).")

    idx = slice(burn, None, thin)
    n_used = math.ceil((n_samp - burn) / thin)
    print(f"[info] post-hoc burn/thin: raw n={n_samp}, burn={burn}, thin={thin} → used n={n_used}")

    for k, v in list(draws.items()):
        if not isinstance(v, np.ndarray):
            continue
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n_samp:
            draws[k] = arr[idx, ...]

    postproc = meta.get("postproc", {})
    postproc.update(
        {
            "extra_burn": int(burn),
            "thin": int(thin),
            "n_samples_raw": int(n_samp),
            "n_samples_used": int(np.asarray(draws["mu"]).shape[0]),
        }
    )
    meta["postproc"] = postproc
    return draws, meta


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Uccle DGEV Laplace Plotter (TXx/TXn/TNx/TNn/Precx; Seasonal/Monthly)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--target",
        type=str,
        default=None,
        help=(
            "Run directory or posterior .npz. If omitted, search under the Uccle Laplace root "
            "derived from --series and --agg."
        ),
    )
    p.add_argument("--series", type=str, choices=["TXx", "TXn", "TNx", "TNn", "Precx"], default="TXn")
    p.add_argument("--agg", type=str, choices=["Seasonal", "Monthly"], default="Monthly")
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override the search root used when --target is omitted.",
    )

    p.add_argument("--level", type=float, default=0.90, help="Credible band level for μ_t and states.")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    p.add_argument("--out", type=str, default=None, help="Output directory (default: <run-dir>/figures).")

    # Figure toggles
    p.add_argument("--skip-overview", action="store_true", default=False)
    p.add_argument("--skip-states", action="store_true", default=False)
    p.add_argument("--skip-traces", action="store_true", default=False)
    p.add_argument("--skip-quick", action="store_true", default=False)

    p.add_argument("--max-lag", type=int, default=200, help="ACF / ESS max lag for trace plots.")

    # Post-hoc chain processing
    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Resolve posterior bundle (object)
    bundle = resolve_bundle(args.target, args.series, args.agg, args.root)

    # load_posterior returns PosteriorBundle, not a tuple
    draws: Dict[str, Any] = bundle.draws
    meta: Dict[str, Any] = bundle.meta
    npz_path: str = bundle.npz_path

    # Optional post-hoc burn/thin
    if args.burn > 0 or args.thin > 1:
        draws, meta = apply_burn_thin(draws, meta, burn=args.burn, thin=args.thin)

    run_dir = os.path.dirname(npz_path)
    out_dir = args.out or os.path.join(run_dir, "figures")
    ensure_dir(out_dir)

    print(f"[info] Uccle DGEV Laplace ({args.series}, {args.agg})")
    print(f"[info] Posterior source: {npz_path}")
    print(f"[info] Saving figures to: {out_dir}")

    pl = DGEVPlotter(draws=draws, meta=meta, level=float(args.level))

    # Overview
    if not args.skip_overview:
        pl.figure_overview(save_dir=out_dir, show=args.show)

    # Trace + hist + ACF (σ, ξ, process scales, other scalars)
    if not args.skip_traces:
        pl.figure_trace_acf_core(
            save_dir=out_dir,
            show=args.show,
            max_lag=int(args.max_lag),
        )

    # Separate states (level, slope, seasonality)
    if not args.skip_states:
        # Keep same default slope scaling as in DLM plotter (120 months ~ 10 years)
        pl.figure_states_separate(
            save_dir=out_dir,
            show=args.show,
            slope_scale=120.0,
            center="mean",
        )

    # Quick report
    if not args.skip_quick:
        pl.quick_report(save_dir=out_dir, show=args.show)

    print("[done] Uccle DGEV Laplace plots written.")


if __name__ == "__main__":
    main()
