# %% simulator/uccle_dgev_laplace_plotter.py
from __future__ import annotations
"""
Uccle DGEV Laplace Plotter (TXx, TXn, TNx, TNn, Precx; Seasonal / Monthly)
===========================================================================
"""

import os
import sys
import argparse
import inspect
import re
from typing import Any, Dict, Optional, List, Union
import matplotlib as mpl

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simulator.dgev_laplace_plotter import DGEVPlotter  # type: ignore
from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
from simulator.utils import (  # type: ignore
        _ensure_dir,
        find_latest_posterior_npz,
        _parse_kv_list,
    )

mpl.rcParams.update({
    # global base font
    "font.size": 18,

    # titles + axis labels
    "axes.titlesize": 16,
    "axes.labelsize": 16,

    # tick labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,

    # legends
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
})

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
    agg_dir = _parse_agg(agg)
    return 120.0 if agg_dir == "Monthly" else 40.0


def _filter_kwargs_for(fn, kw: Dict[str, Any], *, label: str) -> Dict[str, Any]:
    """Drop unknown kwargs (robust to older plotter signatures)."""
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


def _parse_years_list(s: str) -> Union[str, List[int]]:
    """
    years="auto" or a CSV like "1950,1980,2020".
    """
    st = str(s).strip()
    if st == "" or st.lower() == "auto":
        return "auto"
    out: List[int] = []
    for tok in st.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        if not re.fullmatch(r"-?\d+", tok):
            raise ValueError(f"Invalid year token {tok!r} in --seasonal-years.")
        out.append(int(tok))
    return out


# ---------------------------------------------------------------------------
# Bundle resolution
# ---------------------------------------------------------------------------
def resolve_bundle(*, target: Optional[str], series: str, agg: str, root: Optional[str]) -> Any:
    if target:
        return load_posterior(target)

    search_root = root or uccle_root(series, agg)
    print(f"[info] --target not provided; searching for latest run under: {search_root!r}")

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
            "Use --<section>-kw K=V (repeatable) to override kwargs; nested dicts via dots.\n"
            "Seasonal diagnostics are DGEVPlotter-aligned (which=total|dynamic|baseline)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--series", type=str, choices=["TXx", "TXn", "TNx", "TNn", "Precx"], default="TNn")
    p.add_argument("--agg", type=str, choices=["Seasonal", "Monthly"], default="Monthly")

    p.add_argument("--target", type=str, default=None)
    p.add_argument("--root", type=str, default=None)

    p.add_argument("--level", type=float, default=0.90)
    p.add_argument("--interval", type=str, default="eti", choices=["eti", "hpd"])
    p.add_argument("--show", action="store_true", default=False)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--start-date", type=str, default=None)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--minima", action="store_true", help="Force minima=True (back-transform negated location series).")
    g.add_argument("--maxima", action="store_true", help="Force minima=False (no back-transform).")

    # main figure toggles
    p.add_argument("--skip-overview", action="store_true")
    p.add_argument("--skip-traceacf", action="store_true")
    p.add_argument("--skip-states", action="store_true")
    p.add_argument("--skip-quick", action="store_true")
    p.add_argument("--skip-qhist", action="store_true")

    # seasonal toggles (aligned with DGEVPlotter)
    p.add_argument("--skip-seasonal-patterns", action="store_true")
    p.add_argument("--skip-seasonal-variance", action="store_true")
    p.add_argument("--skip-seasonal-heatmap", action="store_true")
    p.add_argument("--skip-seasonal-bymonth", action="store_true")

    p.add_argument(
        "--seasonal-years",
        type=str,
        default="auto",
        help="Comma-separated calendar years for seasonal patterns, or 'auto'. Example: 1950,1980,2020",
    )
    p.add_argument(
        "--seasonal-which",
        type=str,
        default="total",
        choices=["total", "dynamic", "baseline"],
        help="Which seasonal contribution to plot (used as default for seasonal diagnostics).",
    )

    # kwargs overrides
    p.add_argument("--overview-kw", action="append", default=[], metavar="K=V")
    p.add_argument("--traceacf-kw", action="append", default=[], metavar="K=V")
    p.add_argument("--states-kw", action="append", default=[], metavar="K=V")
    p.add_argument("--quick-kw", action="append", default=[], metavar="K=V")
    p.add_argument("--qhist-kw", action="append", default=[], metavar="K=V")

    p.add_argument("--seasonal-patterns-kw", action="append", default=[], metavar="K=V")
    p.add_argument("--seasonal-variance-kw", action="append", default=[], metavar="K=V")
    p.add_argument("--seasonal-heatmap-kw", action="append", default=[], metavar="K=V")
    p.add_argument("--seasonal-bymonth-kw", action="append", default=[], metavar="K=V")

    # printing toggles (match plotter)
    gg = p.add_mutually_exclusive_group()
    gg.add_argument("--print-level-slope", dest="print_level_slope", action="store_true", default=True)
    gg.add_argument("--no-print-level-slope", dest="print_level_slope", action="store_false")
    p.add_argument("--times", type=str, default="start,mid,end")
    p.add_argument("--slope-scale", type=float, default=None, help="Override slope scale (default depends on agg).")

    p.add_argument("--print-static", action="store_true", default=True)
    p.add_argument("--static-level", type=float, default=None)
    p.add_argument("--static-center", type=str, default="median")
    p.add_argument("--static-digits", type=int, default=4)
    p.add_argument("--static-max-cols", type=int, default=None)
    p.add_argument("--static-no-diag", action="store_true", default=False)
    p.add_argument("--static-interval", type=str, default=None, choices=["eti", "hpd"])

    # backwards compat alias
    p.add_argument("--skip-traces", action="store_true", default=False, help=argparse.SUPPRESS)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    # backwards compat
    if getattr(args, "skip_traces", False):
        args.skip_traceacf = True

    bundle = resolve_bundle(target=args.target, series=args.series, agg=args.agg, root=args.root)
    draws: Dict[str, Any] = bundle.draws
    meta: Dict[str, Any] = bundle.meta if isinstance(bundle.meta, dict) else {}
    npz_path: str = bundle.npz_path

    # meta defaults
    meta = dict(meta)
    meta.setdefault("series", args.series)
    meta.setdefault("agg", args.agg)
    meta.setdefault("start_date", "1892-01-01")

    # ensure period consistent with agg if missing
    if "period" not in meta:
        meta["period"] = 12 if _parse_agg(args.agg) == "Monthly" else 4

    if args.start_date:
        meta["start_date"] = str(args.start_date)

    # minima policy:
    # - explicit flags win
    # - otherwise: TXn/TNn default minima (because stored as negated)
    # - else let plotter meta detection decide (None)
    minima_override: Optional[bool]
    if args.minima:
        minima_override = True
    elif args.maxima:
        minima_override = False
    else:
        minima_override = True if args.series in {"TXn", "TNn"} else None

    run_dir = os.path.dirname(npz_path)
    out_dir = args.out or os.path.join(run_dir, "figures")
    _ensure_dir(out_dir)

    print(f"[info] Uccle DGEV Laplace ({args.series}, {args.agg})")
    print(f"[info] Posterior source: {npz_path}")
    print(f"[info] Saving figures to: {out_dir}")

    plotter = DGEVPlotter(
        draws=draws,
        meta=meta,
        level=float(args.level),
        minima=minima_override,
        interval=str(args.interval),
    )

    # parse kw override sections
    overview_kw = _parse_kv_list(args.overview_kw)
    traceacf_kw = _parse_kv_list(args.traceacf_kw)
    states_kw = _parse_kv_list(args.states_kw)
    quick_kw = _parse_kv_list(args.quick_kw)
    qhist_kw = _parse_kv_list(args.qhist_kw)

    seasonal_patterns_kw = _parse_kv_list(args.seasonal_patterns_kw)
    seasonal_variance_kw = _parse_kv_list(args.seasonal_variance_kw)
    seasonal_heatmap_kw = _parse_kv_list(args.seasonal_heatmap_kw)
    seasonal_bymonth_kw = _parse_kv_list(args.seasonal_bymonth_kw)

    # uccle styling defaults
    col = uccle_color(args.series)
    slope_sc = float(args.slope_scale) if (args.slope_scale is not None) else default_slope_scale(args.series, args.agg)

    overview_kw.setdefault("color", col)
    quick_kw.setdefault("color", col)
    states_kw.setdefault("color", col)
    states_kw.setdefault("slope_scale", slope_sc)

    # Uccle state style policy
    states_kw.setdefault("title_level", "")
    states_kw.setdefault("title_slope", "")
    states_kw.setdefault("title_seasonality", "")
    states_kw.setdefault("show_legend", False)
    states_kw.setdefault("ylims", {"slope": (-0.5, 1.5)})

    # if user wants a specific seasonal contribution on the state plot
    # (default: dynamic is the cleanest for the state itself)
    states_kw.setdefault("seasonality_which", "dynamic")

    # printing
    if args.print_level_slope and hasattr(plotter, "print_level_slope_at"):
        raw_times = [s.strip() for s in str(args.times).split(",") if s.strip()]
        times: List[Any] = [int(rt) if rt.isdigit() else rt for rt in raw_times]
        try:
            plotter.print_level_slope_at(times=times, slope_scale=slope_sc)
        except TypeError:
            # older signature without slope_scale
            plotter.print_level_slope_at(times=times)

    if args.print_static and hasattr(plotter, "print_static_params"):
        plotter.print_static_params(
            level=args.static_level,
            center=args.static_center,
            digits=int(args.static_digits),
            max_vector_cols=args.static_max_cols,
            include_diagnostics=(not args.static_no_diag),
            interval=args.static_interval,
        )

    # main figures
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

    # seasonal diagnostics (delegated to DGEVPlotter)
    seasonal_which = str(args.seasonal_which).strip().lower()
    years = _parse_years_list(args.seasonal_years)

    if not args.skip_seasonal_patterns and hasattr(plotter, "figure_seasonal_patterns"):
        seasonal_patterns_kw.setdefault("which", seasonal_which)
        seasonal_patterns_kw.setdefault("years", years)
        kw = _filter_kwargs_for(plotter.figure_seasonal_patterns, seasonal_patterns_kw, label="seasonal_patterns")
        plotter.figure_seasonal_patterns(save_dir=out_dir, show=args.show, **kw)  # type: ignore[attr-defined]

    if not args.skip_seasonal_variance and hasattr(plotter, "figure_seasonal_variance"):
        seasonal_variance_kw.setdefault("which", seasonal_which)
        kw = _filter_kwargs_for(plotter.figure_seasonal_variance, seasonal_variance_kw, label="seasonal_variance")
        plotter.figure_seasonal_variance(save_dir=out_dir, show=args.show, **kw)  # type: ignore[attr-defined]

    if not args.skip_seasonal_heatmap and hasattr(plotter, "figure_seasonal_dummies_heatmap"):
        seasonal_heatmap_kw.setdefault("which", seasonal_which)
        kw = _filter_kwargs_for(plotter.figure_seasonal_dummies_heatmap, seasonal_heatmap_kw, label="seasonal_heatmap")
        plotter.figure_seasonal_dummies_heatmap(save_dir=out_dir, show=args.show, **kw)  # type: ignore[attr-defined]

    if not args.skip_seasonal_bymonth and hasattr(plotter, "figure_seasonal_dummies_by_month"):
        seasonal_bymonth_kw.setdefault("which", seasonal_which)
        kw = _filter_kwargs_for(plotter.figure_seasonal_dummies_by_month, seasonal_bymonth_kw, label="seasonal_bymonth")
        plotter.figure_seasonal_dummies_by_month(save_dir=out_dir, show=args.show, **kw)  # type: ignore[attr-defined]

    print("[done] Uccle DGEV Laplace plots written.")


if __name__ == "__main__":
    main()