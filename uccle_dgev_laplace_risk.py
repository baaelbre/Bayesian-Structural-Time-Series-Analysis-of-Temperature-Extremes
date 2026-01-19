# %% simulator/uccle_dgev_laplace_risk.py
from __future__ import annotations
"""
Uccle wrapper: Bayesian exceedance risk for DGEV Laplace posterior
=================================================================

Thin Uccle wrapper around:
    simulator.dgev_laplace_risk.py   (NEW version: prob/RP, log/linear, combine, reporting)

What this wrapper does
----------------------
- Picks the correct Uccle root layout:
    results/uccle/<GROUP>/<SERIES>/<FREQ>/Laplace
  where GROUP is TX or TN inferred from SERIES.
- Loads the latest posterior run (unless --target is given).
- Applies optional post-hoc burn/thin.
- Calls DGEVLaplaceRisk.compute(...) for thresholds on the PLOT/original scale.
- Saves exceedance_probs.npz (draw-level probabilities) and writes plots.
- Optional: prints/saves probability or return-period summaries at selected times.

Color policy (Uccle-style)
--------------------------
- TX* series -> red
- TN* series -> blue
Applied via matplotlib color cycle before calling the plotters.

Examples
--------
# run all 4 series on latest runs:
python -u simulator/uccle_dgev_laplace_risk.py --thresholds 35,37,39 --combine

# one series only:
python -u simulator/uccle_dgev_laplace_risk.py --one --series TXx --thresholds 35,37,39 --y-mode rp --yscale log --combine

# specify a particular run dir / posterior.npz:
python -u simulator/uccle_dgev_laplace_risk.py --one --series TXn --target <path/to/run/or/posterior.npz> --thresholds 0,-5,-10

# post-hoc trimming:
python -u simulator/uccle_dgev_laplace_risk.py --thresholds 35,37 --burn 200 --thin 5

# report at times:
python -u simulator/uccle_dgev_laplace_risk.py --one --series TXx --thresholds 35,37,39 --times 1950-07,2020-07 --y-mode rp --report-scale fine
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Make project root importable (mirrors your other Uccle wrappers)
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
# Uccle plotting color policy
# =============================================================================
def _set_single_color_cycle(color: str) -> None:
    try:
        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=[str(color)])
    except Exception:
        pass


def _uccle_color(series: str) -> str:
    s = str(series).upper()
    if s.startswith("TX"):
        return "red"
    if s.startswith("TN"):
        return "blue"
    return "C0"


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
        sd = _parse_date(meta.get("start_date"))

    # Instantiate risk engine (NEW)
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

    # Plots (Uccle color policy)
    _set_single_color_cycle(_uccle_color(series))

    # Fine (block-scale)
    risk.plot_fine(
        rr,
        out_dir=out_dir,
        level=float(args.level),
        window=int(args.window_months) if args.window_months is not None else None,
        show=bool(args.show),
        y_mode=str(args.y_mode),
        yscale=str(args.yscale),
        combine=bool(args.combine),
        band=(not bool(args.no_band)),
        rp_cap_years=float(args.rp_cap_years),
        small_prob_policy=str(args.small_prob_policy),
    )

    # Annual
    _set_single_color_cycle(_uccle_color(series))
    risk.plot_annual(
        rr,
        out_dir=out_dir,
        level=float(args.level),
        window_years=int(args.window_years) if args.window_years is not None else None,
        show=bool(args.show),
        y_mode=str(args.y_mode),
        yscale=str(args.yscale),
        combine=bool(args.combine),
        band=(not bool(args.no_band)),
        rp_cap_years=float(args.rp_cap_years),
        small_prob_policy=str(args.small_prob_policy),
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
            "Uccle wrapper around simulator.dgev_laplace_risk (NEW):\n"
            "- Loads latest DGEV Laplace posterior (per series) by default.\n"
            "- Computes Bayesian event probabilities for EVERY posterior draw.\n"
            "- Writes <run>/risk/exceedance_probs.npz and risk plots.\n"
            "- Can plot prob or return period, linear or log, and combine thresholds.\n"
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

    p.add_argument("--series", type=str, default="TXn", choices=["TXx", "TXn", "TNx", "TNn"])
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
    p.add_argument("--thresholds", type=str, default="0,-1,-2,-3,-4,-5", help="Comma-separated thresholds on plot/original scale.")
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
    p.add_argument("--window-years", type=int, default=120, help="Annual plot: last N year groups shown.")

    # NEW plotting controls
    p.add_argument("--y-mode", type=str, default="prob", choices=["prob", "rp"], help="Plot y as probability or return period (years).")
    p.add_argument("--yscale", type=str, default="linear", choices=["linear", "log"], help="Plot y-axis scale.")
    p.add_argument("--combine", action="store_true", default=False, help="Overlay multiple thresholds on one plot.")
    p.add_argument("--no-band", action="store_true", default=False, help="Disable credible bands (useful when combining).")
    p.add_argument("--rp-cap-years", type=float, default=10_000.0, help="Return-period cap in years (guards against tiny probabilities).")
    p.add_argument("--small-prob-policy", type=str, default="clip", choices=["clip", "mask"], help="How to handle probs below the rp-cap floor.")

    # NEW reporting controls
    p.add_argument("--times", type=str, default="1900,1950,2020", help="Comma-separated times for reporting (e.g. 1950-07,2020-07 or numeric x).")
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
