# %% simulator/uccle_dgev_laplace_plotter.py
from __future__ import annotations
"""
Uccle DGEV Laplace Plotter (TXx, TXn, TNx, TNn, Precx; Seasonal / Monthly)
===========================================================================

Uccle wrapper around the generic Laplace DGEV plotter:

    simulator.dgev_laplace_plotter.DGEVPlotter

Key features
------------
- Mirrors simulator/dlm_plotter.py CLI style as closely as possible:
    --target / --root / --level / --show / --out / --start-date
    --skip-overview / --skip-traceacf / --skip-states / --skip-quick / --skip-qhist
    --overview-kw / --traceacf-kw / --states-kw / --quick-kw / --qhist-kw (repeatable K=V; nested via dots)
    optional: --print-level-slope / --times, --print-static (if supported by DGEVPlotter)
- Uccle-specific default root selection via --series and --agg when --target is omitted.
- Robust "latest run" discovery:
    1) optimization.posterior_bundle.find_latest_run (expects posterior.npz)
    2) fallback recursive search for posterior*.npz (find_latest_posterior_npz)
- Ensures TNn and TXn are treated as minima series by default unless user forces --maxima.
- Styling to match Uccle DLM plotter conventions:
    * TX* series: red line + red band
    * TN* series: blue line + blue band
    * State component plots: NO titles by default + NO legend by default

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

# monthly TNn (minima series)
python -u uccle_dgev_laplace_plotter.py --series TNn --agg Monthly --show

# explicit run dir or posterior .npz
python -u uccle_dgev_laplace_plotter.py --target path/to/run_or_posterior.npz

# override kwargs
python -u uccle_dgev_laplace_plotter.py --series TXx --agg Seasonal \
  --states-kw center=mean --states-kw slope_scale=40 \
  --traceacf-kw max_lag=400
"""

import os
import sys
import argparse
import inspect
from typing import Any, Dict, Optional, List

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simulator.dgev_laplace_plotter import DGEVPlotter  # type: ignore

# ---------------------------------------------------------------------
# I/O helpers: posterior loader
# ---------------------------------------------------------------------
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e

# ---------------------------------------------------------------------
# Shared utils (same file as dlm_plotter.py uses)
# ---------------------------------------------------------------------
try:
    from simulator.utils import (  # type: ignore
        _ensure_dir,
        find_latest_posterior_npz,
        _parse_kv_list,
    )
except Exception as e:
    raise ImportError(
        "Could not import simulator.utils.\n"
        "Make sure simulator/utils.py is on PYTHONPATH."
    ) from e


# ---------------------------------------------------------------------------
# Uccle path + styling helpers
# ---------------------------------------------------------------------------
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


def uccle_color(series: str) -> str:
    s = str(series).strip()
    if s.startswith("TX"):
        return "tab:red"
    if s.startswith("TN"):
        return "tab:blue"
    if s.startswith("Prec"):
        return "tab:green"
    return "C0"


def default_slope_scale(series: str, agg: str) -> float:
    # convert per-step slope to per-decade:
    # monthly step: 120 months / decade; seasonal step (quarterly): 40 seasons / decade
    agg_dir = _parse_agg(agg)
    return 120.0 if agg_dir == "Monthly" else 40.0


def _filter_kwargs_for(fn, kw: Dict[str, Any], *, label: str) -> Dict[str, Any]:
    """
    Filter kw dict to only parameters accepted by fn (robust against plotter API changes).
    """
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
    except Exception:
        return kw

    out = {k: v for k, v in kw.items() if k in allowed}
    dropped = sorted(set(kw.keys()) - set(out.keys()))
    if dropped:
        print(f"[info] {label}: dropping unsupported kwargs: {dropped}")
    return out


# ---------------------------------------------------------------------------
# Bundle resolution
# ---------------------------------------------------------------------------
def resolve_bundle(*, target: Optional[str], series: str, agg: str, root: Optional[str]) -> Any:
    """
    Returns PosteriorBundle from load_posterior() (NOT a tuple).
    """
    if target:
        return load_posterior(target)

    search_root = root or uccle_root(series, agg)
    print(f"[info] --target not provided; searching for latest run under: {search_root!r}")

    # Prefer the "run directory" locator (posterior.npz), else fallback to newest posterior*.npz
    run_path = find_latest_run(root=search_root)
    if run_path is None:
        npz = find_latest_posterior_npz(search_root)
        if npz is None:
            print(
                f"[error] No posterior runs found under {search_root!r}.\n"
                f"  → Provide --target or check that your Laplace run wrote posterior*.npz."
            )
            raise SystemExit(1)
        print(f"[info] find_latest_run found nothing; using latest npz: {npz}")
        return load_posterior(npz)

    print(f"[info] Using latest run: {run_path}")
    return load_posterior(run_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Uccle wrapper for the Laplace DGEV plotter.\n"
            "If --target is omitted, we search the default Uccle Laplace root derived from --series and --agg.\n"
            "Use --<section>-kw K=V (repeatable) to override kwargs; nested dicts via dots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Uccle selectors
    p.add_argument("--series", type=str, choices=["TXx", "TXn", "TNx", "TNn", "Precx"], default="TNx")
    p.add_argument("--agg", type=str, choices=["Seasonal", "Monthly"], default="Monthly")

    # Core args (mirrors dlm_plotter style)
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
    p.add_argument("--level", type=float, default=0.90, help="Credible band mass/level.")
    p.add_argument("--interval", type=str, default="hpd", choices=["eti", "hpd"], help="Credible interval type (ETI or HPD).")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    p.add_argument("--out", type=str, default=None, help="Directory to save figures. Default: <run>/figures")
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Override meta start_date (YYYY-MM-DD) for building a calendar axis (if plotter supports it).",
    )

    # minima/maxima override
    g = p.add_mutually_exclusive_group()
    g.add_argument("--minima", action="store_true", help="Force minima=True (back-transform negated location-scale quantities).")
    g.add_argument("--maxima", action="store_true", help="Force minima=False (no back-transform).")

    # Skip toggles (include qhist to mirror dlm_plotter)
    p.add_argument("--skip-overview", action="store_true", help="Skip overview figure.")
    p.add_argument("--skip-traceacf", action="store_true", help="Skip trace+hist+ACF panels.")
    p.add_argument("--skip-states", action="store_true", help="Skip separate state plots.")
    p.add_argument("--skip-quick", action="store_true", help="Skip quick report.")
    p.add_argument("--skip-qhist", action="store_true", help="Skip separate log10(Q) histogram (if supported).")

    # Kw overrides
    p.add_argument("--overview-kw", action="append", default=[], metavar="K=V", help="Override kwargs for plotter.figure_overview(...). Repeatable.")
    p.add_argument("--traceacf-kw", action="append", default=[], metavar="K=V", help="Override kwargs for plotter.figure_trace_acf_core(...). Repeatable.")
    p.add_argument("--states-kw", action="append", default=[], metavar="K=V", help="Override kwargs for plotter.figure_states_separate(...). Repeatable.")
    p.add_argument("--quick-kw", action="append", default=[], metavar="K=V", help="Override kwargs for plotter.quick_report(...). Repeatable.")
    p.add_argument("--qhist-kw", action="append", default=[], metavar="K=V", help="Override kwargs for plotter.figure_process_variances_hist(...). Repeatable.")

    # Optional printing (only used if plotter implements it)
    gg = p.add_mutually_exclusive_group()
    gg.add_argument(
        "--print-level-slope", dest="print_level_slope", action="store_true", default=True,
        help="Print level/slope summaries at times given by --times (if supported)."
    )
    gg.add_argument(
        "--no-print-level-slope", dest="print_level_slope", action="store_false",
        help="Disable printing of level/slope summaries."
    )
    p.add_argument("--times", type=str, default="start,mid,end", help="Comma-separated times for level/slope printing.")

    p.add_argument("--print-static", action="store_true", default=True, help="Print summaries for static parameters (if supported).")
    p.add_argument("--static-level", type=float, default=None, help="Credible level for static params (defaults to --level).")
    p.add_argument("--static-center", type=str, default="median", help="Center for static summaries: median or mean.")
    p.add_argument("--static-digits", type=int, default=4, help="Digits for static summary printing.")
    p.add_argument("--static-max-cols", type=int, default=None, help="Max columns to print per vector parameter (None = all).")
    p.add_argument("--static-no-diag", action="store_true", default=False, help="Disable ESS/Geweke diagnostics in static summary (if supported).")

    # Backward-compatible alias
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

    # Inject Uccle context for minima detection and labeling
    meta = dict(meta)
    meta.setdefault("series", args.series)
    meta.setdefault("agg", args.agg)
    meta.setdefault("start_date", "1892-01-01")  # only used if missing

    if args.start_date:
        meta["start_date"] = str(args.start_date)

    # Choose minima override:
    # - explicit flags win
    # - else: enforce minima for TXn/TNn by default
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

    # Instantiate plotter
    plotter = DGEVPlotter(
        draws=draws,
        meta=meta,
        level=float(args.level),
        minima=minima_override,
        interval=str(args.interval),
    )

    # Parse kwargs
    overview_kw = _parse_kv_list(args.overview_kw)
    traceacf_kw = _parse_kv_list(args.traceacf_kw)
    states_kw = _parse_kv_list(args.states_kw)
    quick_kw = _parse_kv_list(args.quick_kw)
    qhist_kw = _parse_kv_list(args.qhist_kw)

    # Defaults to match DLM Uccle conventions
    col = uccle_color(args.series)
    slope_sc = default_slope_scale(args.series, args.agg)

    # Apply defaults only if user didn't override
    overview_kw.setdefault("color", col)
    quick_kw.setdefault("color", col)
    states_kw.setdefault("color", col)

    states_kw.setdefault("slope_scale", slope_sc)

    # state plots: NO titles + NO legend by default (matches Uccle DLM plotter)
    states_kw.setdefault("title_level", "")
    states_kw.setdefault("title_slope", "")
    states_kw.setdefault("title_seasonality", "")
    states_kw.setdefault("show_legend", False)

    # (Optional) printing, only if plotter supports the methods
    if args.print_level_slope and hasattr(plotter, "print_level_slope_at"):
        raw_times = [s.strip() for s in str(args.times).split(",") if s.strip() != ""]
        times: List[Any] = []
        for rt in raw_times:
            times.append(int(rt) if rt.isdigit() else rt)

        try:
            plotter.print_level_slope_at(times=times, slope_scale=slope_sc)  # type: ignore[attr-defined]
        except TypeError:
            # older API: no slope_scale
            plotter.print_level_slope_at(times=times)  # type: ignore[attr-defined]

    if args.print_static and hasattr(plotter, "print_static_params"):
        try:
            plotter.print_static_params(  # type: ignore[attr-defined]
                level=args.static_level,
                center=args.static_center,
                digits=int(args.static_digits),
                max_vector_cols=args.static_max_cols,
                include_diagnostics=(not args.static_no_diag),
            )
        except TypeError:
            # older API: accept fewer kwargs
            plotter.print_static_params()  # type: ignore[attr-defined]

    # Call plots (filter kwargs for robustness)
    if not args.skip_overview:
        kw = _filter_kwargs_for(plotter.figure_overview, overview_kw, label="overview")
        plotter.figure_overview(save_dir=out_dir, show=args.show, **kw)

    if not args.skip_traceacf:
        kw = _filter_kwargs_for(plotter.figure_trace_acf_core, traceacf_kw, label="traceacf")
        plotter.figure_trace_acf_core(save_dir=out_dir, show=args.show, **kw)

    if not args.skip_states:
        kw = _filter_kwargs_for(plotter.figure_states_separate, states_kw, label="states")
        plotter.figure_states_separate(save_dir=out_dir, show=args.show, **kw)

    if not args.skip_quick:
        kw = _filter_kwargs_for(plotter.quick_report, quick_kw, label="quick")
        plotter.quick_report(save_dir=out_dir, show=args.show, **kw)

    if not args.skip_qhist and hasattr(plotter, "figure_process_variances_hist"):
        kw = _filter_kwargs_for(plotter.figure_process_variances_hist, qhist_kw, label="qhist")
        plotter.figure_process_variances_hist(save_dir=out_dir, show=args.show, **kw)  # type: ignore[attr-defined]

    print("[done] Uccle DGEV Laplace plots written.")


if __name__ == "__main__":
    main()
