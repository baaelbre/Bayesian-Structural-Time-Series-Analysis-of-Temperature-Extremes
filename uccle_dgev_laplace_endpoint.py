# %% simulator/uccle_dgev_laplace_endpoint.py
from __future__ import annotations
"""
Uccle wrapper for DGEV Laplace endpoint tracking
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore

from simulator.dgev_laplace_endpoint import (  # type: ignore
    DGEVLaplaceEndpoint,
    resolve_bundle,
    apply_burn_thin,
    _parse_date,
    _parse_csv_strings,
)

import matplotlib as mpl
mpl.use("Agg")
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
import matplotlib.pyplot as plt  # noqa: E402


# =============================================================================
# Uccle defaults
# =============================================================================
def _default_root(*, group: str, series: str, agg: str) -> str:
    g = str(group).strip()
    s = str(series).strip()
    a = str(agg).strip()
    return os.path.join("results", "uccle", g, s, a, "Laplace")


def _infer_group(series: str) -> str:
    s = str(series).strip()
    if s.startswith("TX"):
        return "TX"
    if s.startswith("TN"):
        return "TN"
    if s.startswith("Prec") or s.startswith("PR"):
        return "Prec"
    # fallback
    return "Misc"


def _uccle_color(series: str) -> Optional[str]:
    s = str(series).strip()
    if s.startswith("TX"):
        return "red"
    if s.startswith("TN"):
        return "blue"
    return None


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _nan_summarize_2d(draws_2d: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loq = (1.0 - float(level)) / 2.0
    hiq = 1.0 - loq
    med = np.nanquantile(draws_2d, 0.5, axis=0)
    lo = np.nanquantile(draws_2d, loq, axis=0)
    hi = np.nanquantile(draws_2d, hiq, axis=0)
    return med, lo, hi


def _safe_set_yscale(ax: plt.Axes, yscale: str, y: np.ndarray) -> str:
    ys = str(yscale).strip().lower()
    if ys not in ("linear", "log"):
        raise ValueError("yscale must be 'linear' or 'log'")
    if ys == "log":
        yy = np.asarray(y, float)
        ok = np.isfinite(yy) & (yy > 0)
        if not np.any(ok):
            print("[warn] yscale=log requested but values are not positive; falling back to linear.")
            return "linear"
        if float(np.nanmin(yy[ok])) <= 0:
            print("[warn] yscale=log requested but values are not strictly positive; falling back to linear.")
            return "linear"
        ax.set_yscale("log")
    return ys


def plot_fine_uccle(
    *,
    series: str,
    out_path: str,
    x: np.ndarray,
    draws: np.ndarray,   # (S,L)
    split_x: float,
    minima: bool,
    mode: str,
    xi_eps: float,
    p_finite: float,
    nonfinite_policy: str,
    level: float,
    window: Optional[int],
    yscale: str,
) -> None:
    L = int(x.size)
    i0 = 0
    if window is not None:
        w = max(1, int(window))
        i0 = max(0, L - w)
    xx = x[i0:]
    dd = draws[:, i0:]

    med, lo, hi = _nan_summarize_2d(dd, level=float(level))

    color = _uccle_color(series)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
    ax.plot(xx, med, lw=1.8, color=color if color else None)
    ax.fill_between(xx, lo, hi, alpha=0.25, color=color if color else None)
    ax.axvline(float(split_x), lw=1.0, alpha=0.8, color="black")

    kind = "lower endpoint" if (minima and str(mode) == "plot") else "upper endpoint"
    ax.set_ylabel(f"{kind} ({mode} scale)")
    ax.set_xlabel("time")
    ax.grid(True, alpha=0.25)

    _safe_set_yscale(ax, yscale, np.concatenate([med, lo, hi]))

    ax.set_title("")  # no title
    ax.legend_.remove() if ax.get_legend() is not None else None


    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def plot_annual_uccle(
    *,
    series: str,
    out_path: str,
    x_year: np.ndarray,
    draws_year: np.ndarray,  # (S,nY)
    split_x_year: float,
    minima: bool,
    mode: str,
    xi_eps: float,
    p_finite: float,
    nonfinite_policy: str,
    level: float,
    window_years: Optional[int],
    yscale: str,
) -> None:
    nY = int(x_year.size)
    j0 = 0
    if window_years is not None:
        w = max(1, int(window_years))
        j0 = max(0, nY - w)
    xx = x_year[j0:]
    dd = draws_year[:, j0:]

    med, lo, hi = _nan_summarize_2d(dd, level=float(level))

    color = _uccle_color(series)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
    ax.plot(xx, med, lw=1.8, color=color if color else None)
    ax.fill_between(xx, lo, hi, alpha=0.25, color=color if color else None)
    if np.isfinite(split_x_year):
        ax.axvline(float(split_x_year), lw=1.0, alpha=0.8, color="black")

    ax.set_ylabel(r"Endpoint $y^*$")
    ax.set_xlabel("year")
    ax.grid(True, alpha=0.25)

    _safe_set_yscale(ax, yscale, np.concatenate([med, lo, hi]))

    ax.set_title("")  # no title
    ax.legend_.remove() if ax.get_legend() is not None else None

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Uccle endpoint tracking wrapper for DGEV Laplace posterior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Uccle selectors
    p.add_argument("--series", type=str, default="TXx", help="Uccle series (TXx, TXn, TNx, TNn, Precx, ...).")
    p.add_argument("--group", type=str, default=None, help="Override group folder (TX, TN, Prec). Default inferred from --series.")
    p.add_argument("--agg", type=str, default="Monthly", help="Aggregation folder under series (e.g. Seasonal, Monthly).")

    # run discovery / explicit target
    p.add_argument("--target", type=str, default=None, help="Run dir or path to posterior.npz. If omitted, uses latest under --root.")
    p.add_argument("--root", type=str, default=None, help="Search root when --target is omitted. Default: results/uccle/<group>/<series>/<agg>/Laplace")

    # output
    p.add_argument("--out", type=str, default=None, help="Output directory. Default: <run>/endpoint")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively (mostly for debugging).")

    # post-hoc chain trimming
    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    # calendar axis
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY / YYYY-MM / YYYY-MM-DD). If omitted, tries meta['start_date'], else numeric index.",
    )

    # optional forecast extension
    p.add_argument("--horizon", type=int, default=0, help="Extra future steps to extend endpoints (simulate latent states).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed used only if --horizon>0.")

    # endpoint controls
    p.add_argument("--mode", type=str, default="plot", choices=["plot", "model"], help="Compute endpoints on plot/original scale or model scale.")
    p.add_argument("--xi-eps", type=float, default=0, help="Treat xi >= -xi_eps as non-finite (guards near-zero blow-ups).")
    p.add_argument("--nonfinite-policy", type=str, default="nan", choices=["nan", "inf"], help="How to represent xi>=-xi_eps draws.")

    # plotting
    p.add_argument("--level", type=float, default=0.90, help="Credible band level (over posterior draws).")
    p.add_argument("--window-months", type=int, default=240, help="Fine plot: last N points shown (includes forecast tail if any).")
    p.add_argument("--window-years", type=int, default=60, help="Annual plot: last N year groups shown.")
    p.add_argument("--yscale", type=str, default="linear", choices=["linear", "log"], help="y-axis scale for endpoint plots.")

    # reporting
    p.add_argument("--times", type=str, default="", help="Comma-separated times for reporting (e.g. 1950-07,2020-07 or numeric x).")
    p.add_argument("--report-scale", type=str, default="fine", choices=["fine", "annual"], help="Whether to report on fine or annual scale.")
    p.add_argument("--report-csv", type=str, default="", help="Optional CSV path to save report (default: <out>/endpoint_report_times_*.csv).")

    args = p.parse_args()

    group = args.group if args.group is not None else _infer_group(args.series)
    root = args.root if args.root is not None else _default_root(group=group, series=args.series, agg=args.agg)

    bundle = resolve_bundle(target=args.target, root=root)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    if args.burn > 0 or args.thin > 1:
        draws, meta = apply_burn_thin(draws, dict(meta), burn=int(args.burn), thin=int(args.thin))

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "endpoint")
    _ensure_dir(out_dir)

    print(f"[info] series={args.series} | group={group} | agg={args.agg}")
    print(f"[info] using posterior: {npz_path}")
    print(f"[info] saving outputs to: {out_dir}")

    sd = _parse_date(args.start_date)
    if sd is None:
        sd = _parse_date(meta.get("start_date")) if isinstance(meta, dict) else None

    ep = DGEVLaplaceEndpoint(draws, meta, npz_path=npz_path)

    er = ep.compute(
        horizon=int(args.horizon),
        seed=int(args.seed),
        start_date=sd,
        mode=str(args.mode),
        xi_eps=float(args.xi_eps),
        nonfinite_policy=str(args.nonfinite_policy),
    )

    # save raw
    npz_out = os.path.join(out_dir, "endpoint_over_time.npz")
    ep.save(er, out_path=npz_out)

    # Uccle-colored plots (replot instead of calling ep.plot_* directly)
    fine_path = os.path.join(out_dir, f"endpoint_fine_{er.mode}_{args.yscale}.png")
    plot_fine_uccle(
        series=str(args.series),
        out_path=fine_path,
        x=er.x_full,
        draws=er.endpoint_fine,
        split_x=float(er.split_x_fine),
        minima=bool(er.minima),
        mode=str(er.mode),
        xi_eps=float(er.xi_eps),
        p_finite=float(er.p_finite),
        nonfinite_policy=str(er.nonfinite_policy),
        level=float(args.level),
        window=int(args.window_months) if args.window_months is not None else None,
        yscale=str(args.yscale),
    )

    annual_path = os.path.join(out_dir, f"endpoint_annual_{er.mode}_{args.yscale}.png")
    plot_annual_uccle(
        series=str(args.series),
        out_path=annual_path,
        x_year=er.x_year,
        draws_year=er.endpoint_year,
        split_x_year=float(er.split_x_year),
        minima=bool(er.minima),
        mode=str(er.mode),
        xi_eps=float(er.xi_eps),
        p_finite=float(er.p_finite),
        nonfinite_policy=str(er.nonfinite_policy),
        level=float(args.level),
        window_years=int(args.window_years) if args.window_years is not None else None,
        yscale=str(args.yscale),
    )

    # reporting
    times = _parse_csv_strings(args.times)
    if times:
        report_csv = str(args.report_csv).strip()
        if not report_csv:
            report_csv = os.path.join(out_dir, f"endpoint_report_times_{str(args.report_scale).strip().lower()}.csv")
        ep.print_at_times(
            er,
            times=times,
            scale=str(args.report_scale),
            level=float(args.level),
            csv_path=report_csv,
        )

    print("[done] uccle endpoint tracking written.")
