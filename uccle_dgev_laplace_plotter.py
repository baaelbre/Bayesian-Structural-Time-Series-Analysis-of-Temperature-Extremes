# uccle_dgev_laplace_plotter.py
from __future__ import annotations
"""
Uccle DGEV Laplace Plotter (TXx, TXn, TNx, TNn, Precx; Seasonal / Monthly)
===========================================================================

Uccle wrapper around the generic Laplace DGEV plotter:

    simulator.dgev_laplace_plotter.DGEVPlotter

Key features
------------
- Same figures + CLI interface style as simulator.dgev_laplace_plotter.py:
    --target / --root / --level / --interval / --show / --out
    --minima / --maxima
    --skip-overview / --skip-traceacf / --skip-states / --skip-quick
    --overview-kw / --traceacf-kw / --states-kw / --quick-kw (repeatable K=V)
- Uccle-specific default root selection via --series and --agg when --target is omitted.
- Robust "latest run" discovery even if files are named posterior_*.npz (not necessarily posterior.npz).
- Ensures TNn and TXn are treated as minima series (negated convention) by:
    (i) injecting meta['series']=<series> (if missing), and
    (ii) defaulting minima=True for series in {TXn, TNn} unless user forces --maxima.

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
# latest Seasonal TXx Laplace run (ETI ribbons)
python -u uccle_dgev_laplace_plotter.py --series TXx --agg Seasonal --show

# monthly TNn (minima series), HPD ribbons:
python -u uccle_dgev_laplace_plotter.py --series TNn --agg Monthly --interval hpd --show

# explicit run directory or posterior .npz
python -u uccle_dgev_laplace_plotter.py --target path/to/run_or_posterior.npz

# override kwargs
python -u uccle_dgev_laplace_plotter.py --series TXx --agg Seasonal \
  --states-kw center=mean --states-kw slope_scale=120 \
  --traceacf-kw max_lag=400
"""

import os
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from simulator.dgev_laplace_plotter import DGEVPlotter  # type: ignore

# ---------------------------------------------------------------------
# I/O helpers: posterior loader
# ---------------------------------------------------------------------
try:
    from optimization.posterior_bundle import load_posterior  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e


# ---------------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------------
def _ensure_dir(p: Optional[str]) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _parse_agg(agg: str) -> str:
    a = str(agg).strip().lower()
    if a.startswith("s"):
        return "Seasonal"
    if a.startswith("m"):
        return "Monthly"
    raise ValueError(f"Unknown aggregation {agg!r} (use Seasonal or Monthly).")


def uccle_root(series: str, agg: str) -> str:
    """
    Map (series, aggregation) to root containing Laplace runs.
    series ∈ {TXx, TXn, TNx, TNn, Precx}
    agg    ∈ {Seasonal, Monthly} (case-insensitive; 's'/'m' ok)
    """
    base = "results/uccle"
    s = str(series).strip()
    agg_dir = _parse_agg(agg)

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


def _extract_ts_from_path(path_str: str) -> Optional[float]:
    """
    Extract YYYYMMDD_HHMMSS from a path (common in your run directory names),
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
        cands = [p for p in rp.rglob("*.npz") if "posterior" in p.name.lower()]
    if not cands:
        return None

    def key(p: Path) -> Tuple[int, float]:
        ts = _extract_ts_from_path(str(p))
        if ts is not None:
            return (1, ts)
        return (0, p.stat().st_mtime)

    best = max(cands, key=key)
    return str(best)


# ---------------------------------------------------------------------------
# CLI kw override parsing (same as generic plotter)
# ---------------------------------------------------------------------------
def _parse_value(raw: str):
    import ast

    s = raw.strip()
    low = s.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _set_nested(d: dict, key: str, value):
    parts = [p for p in key.split(".") if p]
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _parse_kv_list(items: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Expected K=V, got: {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        val = _parse_value(v)
        if "." in k:
            _set_nested(out, k, val)
        else:
            out[k] = val
    return out


# ---------------------------------------------------------------------------
# Bundle resolution
# ---------------------------------------------------------------------------
def resolve_bundle(
    *,
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
    print(f"[info] --target not provided; searching latest posterior under: {search_root!r}")

    npz = find_latest_posterior_npz(search_root)
    if npz is None:
        print(
            f"[error] No posterior .npz found under {search_root!r}.\n"
            f"  → Provide --target or check that your Laplace run wrote posterior*.npz."
        )
        raise SystemExit(1)

    print(f"[info] using latest posterior: {npz}")
    return load_posterior(npz)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Uccle wrapper for the Laplace DGEV plotter.\n"
            "If --target is omitted, we search the default Uccle Laplace root derived from --series and --agg."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Uccle selectors
    p.add_argument("--series", type=str, choices=["TXx", "TXn", "TNx", "TNn", "Precx"], default="TNn")
    p.add_argument("--agg", type=str, choices=["Seasonal", "Monthly"], default="Monthly")

    # Same “core” args as the generic plotter
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to a run directory or directly to a posterior .npz. If omitted, searches under --root (or the Uccle default root).",
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Search root if --target is omitted. If not provided, uses the Uccle default root for (--series, --agg).",
    )
    p.add_argument("--level", type=float, default=0.90, help="Credible mass for ribbons.")
    p.add_argument("--interval", type=str, default="hpd", choices=["eti", "hpd"], help="Credible interval type for ribbons (ETI or HPD).")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    p.add_argument("--out", type=str, default=None, help="Directory to save figures. Default: <run>/figures")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--minima", action="store_true", help="Force minima=True (back-transform negated location-scale quantities).")
    g.add_argument("--maxima", action="store_true", help="Force minima=False (no back-transform).")

    p.add_argument("--skip-overview", action="store_true", help="Skip overview figure.")
    p.add_argument("--skip-traceacf", action="store_true", help="Skip trace+hist+ACF panels.")
    p.add_argument("--skip-states", action="store_true", help="Skip separate state plots.")
    p.add_argument("--skip-quick", action="store_true", help="Skip quick report.")

    # Kw overrides (same pattern as the generic plotter)
    p.add_argument(
        "--overview-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_overview(...). Repeatable. Supports nested keys via dots.",
    )
    p.add_argument(
        "--traceacf-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_trace_acf_core(...). Repeatable. Supports nested keys via dots.",
    )
    p.add_argument(
        "--states-kw",
        action="append",
        default=["slope_scale=120", "center=median"],
        metavar="K=V",
        help="Override kwargs for plotter.figure_states_separate(...). Repeatable. Supports nested keys via dots.",
    )
    p.add_argument(
        "--quick-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.quick_report(...). Repeatable. Supports nested keys via dots.",
    )

    # Backward-compatible alias (older uccle plotter used --skip-traces)
    p.add_argument("--skip-traces", action="store_true", default=False, help=argparse.SUPPRESS)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Harmonize alias
    if getattr(args, "skip_traces", False):
        args.skip_traceacf = True

    # Load bundle
    bundle = resolve_bundle(target=args.target, series=args.series, agg=args.agg, root=args.root)
    draws: Dict[str, Any] = bundle.draws
    meta: Dict[str, Any] = bundle.meta if isinstance(bundle.meta, dict) else {}
    npz_path: str = bundle.npz_path

    # Inject Uccle context for minima detection heuristics
    meta = dict(meta)
    meta.setdefault("series", args.series)
    meta.setdefault("agg", args.agg)

    # Choose minima override:
    # - explicit flags win
    # - else: enforce minima for TNn/TXn by default (user request)
    minima_override: Optional[bool]
    if args.minima:
        minima_override = True
    elif args.maxima:
        minima_override = False
    else:
        minima_override = True if args.series in {"TXn", "TNn"} else None

    # Output directory
    run_dir = os.path.dirname(npz_path)
    out_dir = args.out or os.path.join(run_dir, "figures")
    _ensure_dir(out_dir)

    print(f"[info] Uccle DGEV Laplace ({args.series}, {args.agg})")
    print(f"[info] Posterior source: {npz_path}")
    print(f"[info] Saving figures to: {out_dir}")

    # Instantiate plotter (interval controls ribbons; per-figure overrides can still be passed via *-kw)
    plotter = DGEVPlotter(
        draws=draws,
        meta=meta,
        level=float(args.level),
        minima=minima_override,
        interval=args.interval,
    )

    overview_kw = _parse_kv_list(args.overview_kw)
    traceacf_kw = _parse_kv_list(args.traceacf_kw)
    states_kw = _parse_kv_list(args.states_kw)
    quick_kw = _parse_kv_list(args.quick_kw)

    if not args.skip_overview:
        plotter.figure_overview(save_dir=out_dir, show=args.show, **overview_kw)

    if not args.skip_traceacf:
        plotter.figure_trace_acf_core(save_dir=out_dir, show=args.show, **traceacf_kw)

    if not args.skip_states:
        plotter.figure_states_separate(save_dir=out_dir, show=args.show, **states_kw)

    if not args.skip_quick:
        plotter.quick_report(save_dir=out_dir, show=args.show, **quick_kw)

    print("[done] Uccle DGEV Laplace plots written.")


if __name__ == "__main__":
    main()
