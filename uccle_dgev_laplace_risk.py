# %% simulator/uccle_dgev_laplace_risk.py
from __future__ import annotations
"""
Uccle wrapper: Bayesian exceedance risk for DGEV Laplace posterior
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional, List

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update({
    # global base font
    "font.size": 14,

    # titles + axis labels
    "axes.titlesize": 16,
    "axes.labelsize": 16,

    # tick labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,

    # legends
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
})

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simulator.dgev_laplace_risk import (  # type: ignore
    DGEVLaplaceRisk,
    resolve_bundle as _resolve_bundle,
    apply_burn_thin as _apply_burn_thin,
)


# =============================================================================
# Uccle root selection
# =============================================================================
def _series_group(series: str) -> str:
    s = str(series).strip().upper()
    if s.startswith("TX"):
        return "TX"
    if s.startswith("TN"):
        return "TN"
    raise ValueError(f"Unknown Uccle temperature series: {series!r} (expected TX* or TN*).")


def default_root(series: str, freq: str) -> str:
    return os.path.join("results", "uccle", _series_group(series), str(series), str(freq), "Laplace")


# =============================================================================
# Parsing helpers
# =============================================================================
def _parse_date(s: Optional[str]) -> Optional[datetime]:
    """Accepts YYYY, YYYY-MM, YYYY-MM-DD."""
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    parts = [int(p) for p in ss.split("-")]
    if len(parts) == 1:
        return datetime(parts[0], 1, 1)
    if len(parts) == 2:
        return datetime(parts[0], parts[1], 1)
    if len(parts) == 3:
        return datetime(parts[0], parts[1], parts[2])
    raise ValueError("date must be YYYY, YYYY-MM, or YYYY-MM-DD")


def _parse_csv_floats(s: Optional[str]) -> List[float]:
    if s is None:
        return []
    out: List[float] = []
    for tok in str(s).split(","):
        t = tok.strip()
        if t:
            out.append(float(t))
    return out


def _parse_csv_strings(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    out: List[str] = []
    for tok in str(s).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


# =============================================================================
# Uccle color policy: single-color + colormap
# =============================================================================
def _is_tx(series: str) -> bool:
    return str(series).strip().upper().startswith("TX")


def _single_color(series: str) -> str:
    return "red" if _is_tx(series) else "blue"


def _cmap(series: str, user_cmap: Optional[str] = None) -> str:
    """
    Default colormaps:
      TX -> Reds
      TN -> Blues
    user_cmap can override.
    """
    if user_cmap and str(user_cmap).strip():
        return str(user_cmap).strip()
    return "Reds" if _is_tx(series) else "Blues"


def _set_single_color_cycle(color: str) -> None:
    try:
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[str(color)])
    except Exception:
        pass


# =============================================================================
# Run one series
# =============================================================================
def _run_one(series: str, args: argparse.Namespace) -> None:
    series = str(series)

    # Determine root for latest-run discovery (unless --target provided)
    root = args.root or default_root(series, args.freq)

    bundle = _resolve_bundle(target=args.target, root=root)
    draws, meta, npz_path = bundle.draws, dict(bundle.meta), bundle.npz_path

    # Uccle defaults
    meta.setdefault("start_date", "1892-01-01")
    meta.setdefault("period", 12)

    # Post-hoc burn/thin
    if int(args.burn) > 0 or int(args.thin) > 1:
        draws, meta = _apply_burn_thin(draws, meta, burn=int(args.burn), thin=int(args.thin))

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "risk")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[info] series={series} | using posterior: {npz_path}")
    print(f"[info] series={series} | saving outputs to: {out_dir}")

    # Thresholds are always on PLOT/original scale
    thresholds = _parse_csv_floats(args.thresholds)
    if not thresholds:
        raise ValueError("Provide --thresholds as comma-separated floats, e.g. --thresholds 35,37,39")

    # Event: auto -> let engine choose (maxima->gt, minima->lt)
    ev = str(args.event).strip().lower()
    if ev == "auto":
        ev_opt = None
    else:
        if ev not in ("gt", "lt"):
            raise ValueError("--event must be auto, gt, or lt")
        ev_opt = ev

    # Calendar axis
    sd = _parse_date(args.start_date)
    if sd is None:
        sd = _parse_date(meta.get("start_date")) or _parse_date("1892-01-01")

    risk = DGEVLaplaceRisk(draws, meta, npz_path=npz_path)

    rr = risk.compute(
        thresholds=thresholds,
        event=ev_opt,
        horizon=int(args.horizon),
        seed=int(args.seed),
        start_date=sd,
    )

    # Save draw-level probabilities
    risk.save(rr, out_path=os.path.join(out_dir, "exceedance_probs.npz"))

    # ---- plotting style policy ----
    # - If combine: use colormap shading (Reds/Blues)
    # - Else: single-color cycle (red/blue)
    combine = bool(args.combine)

    # defaults consistent with your risk engine:
    # when combine=True, bands are often off; when combine=False, bands on
    band = (not combine) if (args.band is None) else bool(args.band)
    legend = combine if (args.legend is None) else bool(args.legend)
    shade = combine if (args.shade is None) else bool(args.shade)

    if not combine:
        _set_single_color_cycle(_single_color(series))

    cmap_name = _cmap(series, args.cmap)

    # Fine (block-scale)
    risk.plot_fine(
        rr,
        out_dir=out_dir,
        level=float(args.level),
        window=int(args.window_months) if args.window_months is not None else None,
        show=bool(args.show),
        y_mode=str(args.y_mode),
        yscale=str(args.yscale),
        combine=combine,
        band=band,
        legend=legend,
        split_line=bool(args.split_line),
        rp_cap_years=float(args.rp_cap_years),
        small_prob_policy=str(args.small_prob_policy),
        shade=shade,
        cmap=str(cmap_name),
        cmap_min=float(args.cmap_min),
        cmap_max=float(args.cmap_max),
    )

    # Annual
    if not combine:
        _set_single_color_cycle(_single_color(series))

    risk.plot_annual(
        rr,
        out_dir=out_dir,
        level=float(args.level),
        window_years=int(args.window_years) if args.window_years is not None else None,
        show=bool(args.show),
        y_mode=str(args.y_mode),
        yscale=str(args.yscale),
        combine=combine,
        band=band,
        legend=legend,
        split_line=bool(args.split_line),
        rp_cap_years=float(args.rp_cap_years),
        small_prob_policy=str(args.small_prob_policy),
        shade=shade,
        cmap=str(cmap_name),
        cmap_min=float(args.cmap_min),
        cmap_max=float(args.cmap_max),
    )

    # Optional reporting at times
    times = _parse_csv_strings(args.times)
    if times:
        report_scale = str(args.report_scale).strip().lower()
        if report_scale not in ("fine", "annual"):
            raise ValueError("--report-scale must be fine or annual")

        report_csv = str(args.report_csv).strip()
        if not report_csv:
            report_csv = os.path.join(out_dir, f"risk_report_times_{report_scale}_{str(args.y_mode).lower()}.csv")

        risk.print_at_times(
            rr,
            times=times,
            scale=report_scale,
            y_mode=str(args.y_mode),
            level=float(args.level),
            rp_cap_years=float(args.rp_cap_years),
            small_prob_policy=str(args.small_prob_policy),
            csv_path=report_csv,
        )

    print(f"[done] series={series} | risk probabilities + plots written.\n")


# =============================================================================
# CLI
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Uccle wrapper around simulator.dgev_laplace_risk:\n"
            "- Loads latest DGEV Laplace posterior (per series) by default.\n"
            "- Computes Bayesian event probabilities for EVERY posterior draw.\n"
            "- Writes <run>/risk/exceedance_probs.npz and risk plots.\n"
            "- Can plot prob or return period, linear or log, and combine thresholds.\n"
            "- Uses Reds (TX*) and Blues (TN*) colormaps when combining thresholds.\n"
            "- Can print/save summaries at selected times.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Run dir or posterior.npz. If omitted, uses latest under the series root.",
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Search root when --target is omitted. If omitted, uses Uccle layout "
            "results/uccle/<GROUP>/<SERIES>/<FREQ>/Laplace."
        ),
    )

    p.add_argument("--series", type=str, default="TXx", choices=["TXx", "TXn", "TNx", "TNn"])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", dest="run_all", action="store_true", default=False, help="Run TXx, TXn, TNx, TNn (default).")
    g.add_argument("--one", dest="run_all", action="store_false", help="Run only --series.")

    p.add_argument("--freq", type=str, choices=["Monthly", "Seasonal"], default="Monthly")

    p.add_argument("--out", type=str, default=None, help="Output directory. Default: <run>/risk")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")

    # post-hoc chain trimming
    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    # risk inputs
    p.add_argument("--thresholds", type=str, default="36.8,39.7", help="Comma-separated thresholds on plot/original scale.")
    p.add_argument("--event", type=str, default="auto", help="auto, gt, or lt (event on plot scale).")

    # calendar axis
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY / YYYY-MM / YYYY-MM-DD). If omitted: meta['start_date'] or 1892-01-01.",
    )

    # optional forecast extension
    p.add_argument("--horizon", type=int, default=0, help="Extra future steps to extend risk (simulate latent states).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed used only if --horizon>0.")

    # credible band / windows
    p.add_argument("--level", type=float, default=0.90, help="Credible band level (over posterior draws).")
    p.add_argument("--window-months", type=int, default=240, help="Fine plot: last N points shown.")
    p.add_argument("--window-years", type=int, default=60, help="Annual plot: last N year groups shown.")

    # plotting controls
    p.add_argument("--y-mode", type=str, default="rp", choices=["prob", "rp"], help="Plot y as probability or return period (years).")
    p.add_argument("--yscale", type=str, default="log", choices=["linear", "log"], help="Plot y-axis scale.")
    p.add_argument("--combine", action=argparse.BooleanOptionalAction, default=True, help="Overlay multiple thresholds on one plot.")

    p.add_argument("--band", action=argparse.BooleanOptionalAction, default=True, help="Show credible bands (default: on if not combine).")
    p.add_argument("--legend", action=argparse.BooleanOptionalAction, default=None, help="Show legend (default: on if combine).")
    p.add_argument("--split-line", action=argparse.BooleanOptionalAction, default=False, help="Draw split line at last observed point.")

    # RP guardrails
    p.add_argument("--rp-cap-years", type=float, default=10_000.0, help="Return-period cap in years (guards against tiny probabilities).")
    p.add_argument("--small-prob-policy", type=str, default="clip", choices=["clip", "mask"], help="How to handle probs below the rp-cap floor.")

    # colormap controls (used when combine=True; can override defaults)
    p.add_argument("--shade", action=argparse.BooleanOptionalAction, default=None, help="Use colormap shading (default: on if combine).")
    p.add_argument("--cmap", type=str, default="", help="Override colormap name (default: Reds for TX, Blues for TN).")
    p.add_argument("--cmap-min", type=float, default=0.35, help="Lower end of colormap range [0,1] (lighter).")
    p.add_argument("--cmap-max", type=float, default=0.95, help="Upper end of colormap range [0,1] (darker).")

    # reporting controls
    p.add_argument("--times", type=str, default="1900,1970,2020", help="Comma-separated times for reporting (e.g. 1950-07,2020-07 or numeric x).")
    p.add_argument("--report-scale", type=str, default="annual", choices=["fine", "annual"], help="Whether to report on fine or annual scale.")
    p.add_argument("--report-csv", type=str, default="", help="Optional CSV path for the report (default: <out>/risk_report_times_*.csv).")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if bool(args.run_all):
        for s in ["TXx", "TXn", "TNx", "TNn"]:
            _run_one(s, args)
    else:
        _run_one(str(args.series), args)


if __name__ == "__main__":
    main()
