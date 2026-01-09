# %% simulator/uccle_dlm_forecast.py
from __future__ import annotations
"""
Uccle DLM Forecast (TXm, TNm, Precm; Monthly)
============================================

Wrapper around simulator.dlm_forecast core logic with Uccle defaults.

Changes vs simulator/dlm_forecast.py
------------------------------------
- Forces Uccle monthly origin (start_date = 1892-01-01, period=12).
- Robust run discovery (same as uccle_dlm_plotter.py):
    1) optimization.posterior_bundle.find_latest_run (posterior.npz runs)
    2) fallback recursive search for posterior*.npz
- NO titles and NO legends on all plots.
- Axis labels configurable from CLI (single ylabel used everywhere).

NEW: printing predictions (stdout)
----------------------------------
Print posterior predictive summaries (median + interval) for:
  - fine scale (monthly),
  - annual averages,
  - seasonal averages (all seasons stacked; optional DJF/MAM/JJA/SON).

By default we print:
  --print-scales fine,annual,seasonal_all
  - fine: first --print-horizon forecast months (default 24)
  - annual/seasonal: forecast points only
Use --print-dates / --print-start / --print-end to select a window.

Outputs (default: <run>/forecast)
---------------------------------
  - <series>_dlm_forecast_fine.png (+ .npz)
  - <series>_dlm_forecast_annual_avg.png (+ .npz)
  - <series>_dlm_forecast_seasonal_all_avg.png (+ .npz)
  - <series>_dlm_forecast_seasonal_{DJF,MAM,JJA,SON}_avg.png (+ .npz)
"""

import os
import sys
import re
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Iterable

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore

# Core forecast routines (simulation + coarse graining)
from simulator.dlm_forecast import (  # type: ignore
    simulate_dlm_forecast,
    coarse_grain_annual_average,
    coarse_grain_meteo_seasons,
)


# =============================================================================
# Small utils
# =============================================================================
def _ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


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


def _parse_csv(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _summarize_ribbon(draws_2d: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """draws_2d: (S, L) -> (median, lo, hi) across S."""
    loq = (1.0 - level) / 2.0
    hiq = 1.0 - loq
    med = np.quantile(draws_2d, 0.5, axis=0)
    lo = np.quantile(draws_2d, loq, axis=0)
    hi = np.quantile(draws_2d, hiq, axis=0)
    return med, lo, hi


# =============================================================================
# Printing helpers (median + interval)
# =============================================================================
def _month_labels(L: int, *, start_year: int, start_month: int) -> List[str]:
    labels: List[str] = []
    y0, m0 = int(start_year), int(start_month)
    for i in range(L):
        mm = (m0 - 1) + i
        yy = y0 + (mm // 12)
        mo = (mm % 12) + 1
        labels.append(f"{yy:04d}-{mo:02d}")
    return labels


def _fine_index_from_date(dt: datetime, *, start_year: int, start_month: int) -> int:
    diff_months = (dt.year - start_year) * 12 + (dt.month - start_month)
    return int(round(diff_months))


def _clip_indices(idxs: Iterable[int], L: int) -> List[int]:
    out: List[int] = []
    for i in idxs:
        ii = int(i)
        if 0 <= ii < L:
            out.append(ii)
    return sorted(set(out))


def _select_fine_indices(
    *,
    L_full: int,
    T_obs: int,
    H: int,
    start_year: int,
    start_month: int,
    dates: List[str],
    start: Optional[str],
    end: Optional[str],
    include_observed: bool,
    print_horizon: int,
) -> List[int]:
    """
    Fine-scale selection logic:
      - If no dates/window: print first `print_horizon` forecast steps (cap at H).
        If print_horizon <= 0, print all H.
      - If dates: select those months.
      - If window: select all months in [start,end].
    """
    if (not dates) and (start is None) and (end is None):
        ph = int(print_horizon)
        if ph <= 0:
            ph = H
        ph = min(ph, H)
        return list(range(T_obs, min(L_full, T_obs + ph)))

    if dates:
        idxs: List[int] = []
        for s in dates:
            dt = _parse_date(s)
            if dt is None:
                continue
            k = _fine_index_from_date(dt, start_year=start_year, start_month=start_month)
            idxs.append(k)
        idxs = _clip_indices(idxs, L_full)
    else:
        if start is None or end is None:
            raise ValueError("For a fine-scale window, provide BOTH --print-start and --print-end.")
        dt0 = _parse_date(start)
        dt1 = _parse_date(end)
        if dt0 is None or dt1 is None:
            raise ValueError("Could not parse --print-start/--print-end.")
        a = _fine_index_from_date(dt0, start_year=start_year, start_month=start_month)
        b = _fine_index_from_date(dt1, start_year=start_year, start_month=start_month)
        if b < a:
            a, b = b, a
        idxs = _clip_indices(range(a, b + 1), L_full)

    if not include_observed:
        idxs = [i for i in idxs if i >= T_obs]
    return idxs


def _select_by_years(
    *,
    x: np.ndarray,
    forecast_mask: np.ndarray,
    dates: List[str],
    start: Optional[str],
    end: Optional[str],
    include_observed: bool,
) -> List[int]:
    """
    For annual/seasonal series with x on (decimal) year scale.
    - If no dates/window: select forecast points only (unless include_observed).
    - If dates: select years matching dt.year.
    - If window: select years in [y0,y1].
    """
    n = int(x.size)
    if n == 0:
        return []

    years = np.floor(x).astype(int)

    if (not dates) and (start is None) and (end is None):
        if include_observed:
            return list(range(n))
        return np.where(forecast_mask)[0].astype(int).tolist()

    if dates:
        want = sorted({int(_parse_date(s).year) for s in dates if _parse_date(s) is not None})
        idxs = [i for i in range(n) if int(years[i]) in want]
    else:
        if start is None or end is None:
            raise ValueError("For a year-window, provide BOTH --print-start and --print-end.")
        dt0 = _parse_date(start)
        dt1 = _parse_date(end)
        if dt0 is None or dt1 is None:
            raise ValueError("Could not parse --print-start/--print-end.")
        y0, y1 = sorted([int(dt0.year), int(dt1.year)])
        idxs = [i for i in range(n) if y0 <= int(years[i]) <= y1]

    if not include_observed:
        idxs = [i for i in idxs if bool(forecast_mask[i])]
    return idxs


def _print_block(
    *,
    name: str,
    labels: List[str],
    x: np.ndarray,
    draws: np.ndarray,  # (S, n)
    is_forecast: List[bool],
    level: float,
    max_lines: int,
) -> None:
    n = int(x.size)
    if n == 0:
        print(f"\n[print] {name}: (no points selected)\n")
        return

    loq = (1.0 - level) / 2.0
    hiq = 1.0 - loq
    med = np.quantile(draws, 0.5, axis=0)
    lo = np.quantile(draws, loq, axis=0)
    hi = np.quantile(draws, hiq, axis=0)

    print(f"\n[print] {name}: {n} points (median + {int(round(level*100))}% interval)")
    print(f"{'label':>10s}  {'x':>10s}  {'median':>10s}  {'lo':>10s}  {'hi':>10s}  {'forecast':>9s}")
    print("-" * 74)

    n_show = min(n, int(max_lines))
    for i in range(n_show):
        print(
            f"{labels[i]:>10s}  "
            f"{float(x[i]):10.3f}  "
            f"{float(med[i]):10.3f}  "
            f"{float(lo[i]):10.3f}  "
            f"{float(hi[i]):10.3f}  "
            f"{str(bool(is_forecast[i])):>9s}"
        )

    if n_show < n:
        print(f"... ({n - n_show} more rows truncated; increase --print-max-lines)")
    print("")


# =============================================================================
# Paths / discovery (mirrors uccle_dlm_plotter.py)
# =============================================================================
def default_root(series: str) -> str:
    base = "results/uccle"
    if series == "TXm":
        return os.path.join(base, "TX", "TXm", "Monthly")
    if series == "TNm":
        return os.path.join(base, "TN", "TNm", "Monthly")
    if series == "Precm":
        return os.path.join(base, "Prec", "Precm", "Monthly")
    raise ValueError(f"Unknown series {series!r} for default root.")


def _extract_ts_from_path(path_str: str) -> Optional[float]:
    m = re.search(r"(\d{8})_(\d{6})", path_str)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.timestamp()
    except Exception:
        return None


def find_latest_posterior_npz(root: str) -> Optional[str]:
    root_p = Path(root)
    if not root_p.exists():
        return None
    cands = list(root_p.rglob("posterior*.npz"))
    if not cands:
        return None

    def key(p: Path) -> Tuple[int, float]:
        ts = _extract_ts_from_path(str(p))
        if ts is not None:
            return (1, ts)
        return (0, p.stat().st_mtime)

    return str(max(cands, key=key))


def resolve_bundle(*, target: Optional[str], series: str, root: Optional[str]) -> Any:
    if target:
        return load_posterior(target)

    search_root = root or default_root(series)
    print(f"[info] searching latest posterior run under: {search_root!r}")

    run_path = find_latest_run(root=search_root)
    if run_path is not None:
        print(f"[info] using latest run: {run_path}")
        return load_posterior(run_path)

    npz_path = find_latest_posterior_npz(search_root)
    if npz_path is None:
        print(
            f"[error] No posterior runs found under {search_root!r}.\n"
            f"  → Tried find_latest_run() (posterior.npz) and recursive search (posterior*.npz).\n"
            f"  → Either run the sampler first, or provide --target."
        )
        raise SystemExit(1)

    print(f"[info] find_latest_run found nothing; using latest npz: {npz_path}")
    return load_posterior(npz_path)


# =============================================================================
# Post-processing: burn-in + thinning (post-hoc; same pattern as plotter)
# =============================================================================
def apply_burn_thin(
    draws: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    burn: int = 0,
    thin: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    burn = int(burn or 0)
    thin = int(thin or 1)
    if burn < 0:
        raise ValueError(f"--burn must be >= 0, got {burn}")
    if thin < 1:
        raise ValueError(f"--thin must be >= 1, got {thin}")

    n_samp: Optional[int] = None
    for k in ("mu", "x", "sigma", "sigma2", "Q_alpha", "s_alpha"):
        if k in draws and isinstance(draws[k], np.ndarray) and np.asarray(draws[k]).ndim >= 1:
            n_samp = int(np.asarray(draws[k]).shape[0])
            break
    if n_samp is None:
        print("[warn] could not infer chain length; skipping burn/thin.")
        return draws, meta

    if burn >= n_samp:
        raise ValueError(f"--burn={burn} ≥ number of saved samples ({n_samp}).")

    idx = slice(burn, None, thin)
    n_used = int(math.ceil((n_samp - burn) / thin))
    print(f"[info] post-processing chains: raw n={n_samp}, burn={burn}, thin={thin} → used n={n_used}")

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
            "n_samples_used": int(n_used),
        }
    )
    meta["postproc"] = postproc
    return draws, meta


# =============================================================================
# Color policy (fixed)
# =============================================================================
def series_color(series: str) -> str:
    s = str(series).upper()
    if s.startswith("TX"):
        return "red"
    if s.startswith("TN"):
        return "blue"
    return "C0"


# =============================================================================
# Plotting: NO TITLES, NO LEGENDS
# =============================================================================
def plot_forecast_minimal(
    *,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_fore: np.ndarray,
    fore_draws: np.ndarray,  # (S, len(x_fore))
    level: float,
    split_x: float,
    xlabel: str,
    ylabel: str,
    save_path: str,
    color: str,
    show: bool,
) -> None:
    med, lo, hi = _summarize_ribbon(fore_draws, level=level) if fore_draws.size else (
        np.array([]),
        np.array([]),
        np.array([]),
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))

    # observed
    ax.plot(x_obs, y_obs, lw=1.2, alpha=0.85, color="0.25")

    # forecast ribbon + median
    if x_fore.size:
        ax.fill_between(x_fore, lo, hi, alpha=0.20, color=color)
        ax.plot(x_fore, med, lw=1.6, color=color)

    ax.axvline(float(split_x), lw=1.0, alpha=0.85, color="0.35")

    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"[save] {save_path}")


# =============================================================================
# CLI
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Uccle DLM forecast (TXm/TNm/Precm; Monthly)\n"
            "- Loads latest posterior by default (like uccle_dlm_plotter.py).\n"
            "- Simulates fine-scale posterior predictive paths.\n"
            "- Coarse-grains per draw to annual averages and meteorological seasons.\n"
            "- Plots have NO titles and NO legends.\n"
            "- Can print prediction summaries (median + interval) to stdout.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to a run directory or directly to posterior.npz. If omitted, uses latest under --root.",
    )
    p.add_argument(
        "--series",
        type=str,
        choices=["TXm", "TNm", "Precm"],
        default="TNm",
        help="Series code when searching by default roots (ignored if --target is given).",
    )
    p.add_argument("--root", type=str, default=None, help="Search root when --target is omitted (defaults to Uccle layout).")

    p.add_argument("--horizon", type=int, default=12 * 20, help="Forecast horizon in months (fine-scale steps).")
    p.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for predictive simulation.")

    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help=(
            "Optional calendar start date (YYYY / YYYY-MM / YYYY-MM-DD). "
            "Default: forced meta['start_date']=1892-01-01."
        ),
    )

    p.add_argument("--out", type=str, default=None, help="Output directory. Default: <run>/forecast")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")

    # Post-hoc chain trimming
    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    # plot windows (last N points of the observed series; forecast always shown)
    p.add_argument("--window-months", type=int, default=12 * 50, help="Fine-scale plot: last N observed months.")
    p.add_argument("--window-years", type=int, default=60, help="Annual plot: last N observed years.")
    p.add_argument("--window-seasons", type=int, default=4 * 50, help="Seasonal plot: last N observed seasons.")

    # Axis labels
    p.add_argument("--xlabel", type=str, default="", help="x-axis label (applied to all plots).")
    p.add_argument("--ylabel", type=str, default="T (°C)", help="y-axis label (applied to all plots).")

    # ---- printing predictions (stdout) ----
    g = p.add_mutually_exclusive_group()
    g.add_argument("--print-forecast", dest="print_forecast", action="store_true", default=True,
                   help="Print prediction summaries to stdout.")
    g.add_argument("--no-print-forecast", dest="print_forecast", action="store_false",
                   help="Disable printing prediction summaries.")

    p.add_argument(
        "--print-scales",
        type=str,
        default="fine,annual,seasonal_all",
        help="Comma-separated: fine,annual,seasonal_all,DJF,MAM,JJA,SON",
    )
    p.add_argument(
        "--print-horizon",
        type=int,
        default=24,
        help="(fine) If no --print-dates/--print-start/--print-end are given, print first N forecast months (<=0 prints all).",
    )
    p.add_argument(
        "--print-dates",
        type=str,
        default=None,
        help="Comma-separated dates (YYYY or YYYY-MM or YYYY-MM-DD). fine uses months; annual/seasonal use years.",
    )
    p.add_argument("--print-start", type=str, default=None, help="Window start (YYYY or YYYY-MM or YYYY-MM-DD).")
    p.add_argument("--print-end", type=str, default=None, help="Window end (YYYY or YYYY-MM or YYYY-MM-DD).")
    p.add_argument(
        "--print-include-observed",
        action="store_true",
        default=False,
        help="If set, allow printing points that fall in the observed segment too.",
    )
    p.add_argument(
        "--print-max-lines",
        type=int,
        default=80,
        help="Maximum lines printed per scale (truncate beyond this).",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()

    bundle = resolve_bundle(target=args.target, series=args.series, root=args.root)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    # Force Uccle monthly calendar origin + period
    meta = dict(meta)
    meta["start_date"] = "1892-01-01"
    meta["freq"] = "Monthly"
    meta["period"] = 12

    # Post-hoc burn/thin
    if args.burn > 0 or args.thin > 1:
        draws, meta = apply_burn_thin(draws, meta, burn=args.burn, thin=args.thin)

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "forecast")
    _ensure_dir(out_dir)

    print(f"[info] using posterior: {npz_path}")
    print(f"[info] saving outputs to: {out_dir}")
    print(f"[info] calendar origin forced to: {meta['start_date']} (monthly)")

    # start date: forced meta unless user explicitly overrides
    sd = _parse_date(args.start_date) if args.start_date else _parse_date(meta.get("start_date"))
    if sd is None:
        sd = datetime(1892, 1, 1)

    # simulate fine-scale posterior predictive
    fr = simulate_dlm_forecast(
        draws=draws,
        meta=meta,
        horizon=int(args.horizon),
        seed=int(args.seed),
        start_date=sd,
    )

    y_obs = fr.y_obs
    y_future = fr.y_future
    y_full = fr.y_full
    x_full = fr.x_axis_full
    T = int(y_obs.size)
    H = int(args.horizon)
    L_full = T + H

    col = series_color(args.series)
    xlabel = str(args.xlabel or "")
    ylabel = str(args.ylabel or "")

    # -----------------------------
    # 1) Fine-scale plot
    # -----------------------------
    wM = max(1, int(args.window_months))
    i0 = max(0, T - wM)
    x_obs_m = x_full[i0:T]
    y_obs_m = y_obs[i0:T]

    x_fore_m = x_full[T:L_full]
    fore_draws_m = y_future  # (S, H)

    plot_forecast_minimal(
        x_obs=x_obs_m,
        y_obs=y_obs_m,
        x_fore=x_fore_m,
        fore_draws=fore_draws_m,
        level=float(args.level),
        split_x=float(fr.split_x),
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=os.path.join(out_dir, f"{args.series}_dlm_forecast_fine.png"),
        color=col,
        show=bool(args.show),
    )

    np.savez_compressed(
        os.path.join(out_dir, f"{args.series}_dlm_forecast_fine.npz"),
        series=str(args.series),
        npz_path=str(npz_path),
        y_obs=y_obs,
        y_future=y_future,
        x_full=x_full,
        T=T,
        H=H,
        level=float(args.level),
        seed=int(args.seed),
        start_date=str(sd.date()) if sd is not None else "",
        xlabel=xlabel,
        ylabel=ylabel,
    )

    # -----------------------------
    # 2) Annual averages
    # -----------------------------
    period = int(meta.get("period", 12))
    xY, yY_obs, drawsY, maskY, splitY = coarse_grain_annual_average(
        y_full=y_full,
        T_obs=T,
        period=period,
        start_year=fr.start_year,
        start_month=fr.start_month,
    )

    obs_idx = np.isfinite(yY_obs)
    x_obs_y = xY[obs_idx]
    y_obs_y = yY_obs[obs_idx]

    x_fore_y = xY[maskY]
    fore_draws_y = drawsY[:, maskY] if x_fore_y.size else np.zeros((drawsY.shape[0], 0))

    wY = max(1, int(args.window_years))
    if x_obs_y.size > wY:
        x_obs_y = x_obs_y[-wY:]
        y_obs_y = y_obs_y[-wY:]

    plot_forecast_minimal(
        x_obs=x_obs_y,
        y_obs=y_obs_y,
        x_fore=x_fore_y,
        fore_draws=fore_draws_y,
        level=float(args.level),
        split_x=float(splitY),
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=os.path.join(out_dir, f"{args.series}_dlm_forecast_annual_avg.png"),
        color=col,
        show=bool(args.show),
    )

    np.savez_compressed(
        os.path.join(out_dir, f"{args.series}_dlm_forecast_annual_avg.npz"),
        series=str(args.series),
        x_year=xY,
        y_year_obs=yY_obs,
        year_draws=drawsY,
        forecast_mask=maskY,
        split_x=float(splitY),
        level=float(args.level),
        xlabel=xlabel,
        ylabel=ylabel,
    )

    # -----------------------------
    # 3) Meteorological seasons (period=12 only)
    # -----------------------------
    per_season = None
    seasonal_all = None
    labelsA: Optional[List[str]] = None

    if period != 12 or fr.start_year is None or fr.start_month is None:
        print("[warn] meteorological seasons (DJF/MAM/JJA/SON) require period=12 and a known start-date. Skipping.")
    else:
        per_season, seasonal_all = coarse_grain_meteo_seasons(
            y_full=y_full,
            T_obs=T,
            start_year=int(fr.start_year),
            start_month=int(fr.start_month),
        )

        # Support both possible signatures:
        # - old: seasonal_all = (xA, obsA, drawsA, maskA, splitA)
        # - new: seasonal_all = (xA, obsA, drawsA, maskA, splitA, labelsA)
        if isinstance(seasonal_all, tuple) and len(seasonal_all) == 6:
            xA, obsA, drawsA, maskA, splitA, labelsA = seasonal_all  # type: ignore
        else:
            xA, obsA, drawsA, maskA, splitA = seasonal_all  # type: ignore
            yrs = np.floor(xA).astype(int)
            labelsA = [f"{int(yrs[i]):04d}-S{i:03d}" for i in range(int(xA.size))]

        obs_idxA = np.isfinite(obsA)
        x_obs_A = xA[obs_idxA]
        y_obs_A = obsA[obs_idxA]

        x_fore_A = xA[maskA]
        fore_draws_A = drawsA[:, maskA] if x_fore_A.size else np.zeros((drawsA.shape[0], 0))

        wS = max(1, int(args.window_seasons))
        if x_obs_A.size > wS:
            x_obs_A = x_obs_A[-wS:]
            y_obs_A = y_obs_A[-wS:]

        plot_forecast_minimal(
            x_obs=x_obs_A,
            y_obs=y_obs_A,
            x_fore=x_fore_A,
            fore_draws=fore_draws_A,
            level=float(args.level),
            split_x=float(splitA),
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=os.path.join(out_dir, f"{args.series}_dlm_forecast_seasonal_all_avg.png"),
            color=col,
            show=bool(args.show),
        )

        np.savez_compressed(
            os.path.join(out_dir, f"{args.series}_dlm_forecast_seasonal_all_avg.npz"),
            series=str(args.series),
            x_all=xA,
            y_obs_all=obsA,
            draws_all=drawsA,
            forecast_mask_all=maskA,
            split_x=float(splitA),
            level=float(args.level),
            labels_all=np.array(labelsA, dtype=object),
            xlabel=xlabel,
            ylabel=ylabel,
        )

        # separate season plots + npz
        for sname in ("DJF", "MAM", "JJA", "SON"):
            xS, obsS, drawsS, maskS, splitS = per_season[sname]

            obs_idxS = np.isfinite(obsS)
            x_obs_S = xS[obs_idxS]
            y_obs_S = obsS[obs_idxS]

            x_fore_S = xS[maskS]
            fore_draws_S = drawsS[:, maskS] if x_fore_S.size else np.zeros((drawsS.shape[0], 0))

            if x_obs_S.size > wY:
                x_obs_S = x_obs_S[-wY:]
                y_obs_S = y_obs_S[-wY:]

            plot_forecast_minimal(
                x_obs=x_obs_S,
                y_obs=y_obs_S,
                x_fore=x_fore_S,
                fore_draws=fore_draws_S,
                level=float(args.level),
                split_x=float(splitS),
                xlabel=xlabel,
                ylabel=ylabel,
                save_path=os.path.join(out_dir, f"{args.series}_dlm_forecast_seasonal_{sname}_avg.png"),
                color=col,
                show=bool(args.show),
            )

            np.savez_compressed(
                os.path.join(out_dir, f"{args.series}_dlm_forecast_seasonal_{sname}_avg.npz"),
                series=str(args.series),
                season=str(sname),
                x=xS,
                y_obs=obsS,
                draws=drawsS,
                forecast_mask=maskS,
                split_x=float(splitS),
                level=float(args.level),
                xlabel=xlabel,
                ylabel=ylabel,
            )

    # -----------------------------
    # 4) PRINT predictions (stdout)
    # -----------------------------
    if bool(args.print_forecast):
        scales = [s.strip() for s in str(args.print_scales).split(",") if s.strip()]
        dates = _parse_csv(args.print_dates)
        include_obs = bool(args.print_include_observed)
        max_lines = int(args.print_max_lines)

        # fine labels always available for Uccle monthly
        lab_fine = _month_labels(L_full, start_year=sd.year, start_month=sd.month)

        # ---- fine ----
        if "fine" in scales:
            idxs = _select_fine_indices(
                L_full=L_full,
                T_obs=T,
                H=H,
                start_year=sd.year,
                start_month=sd.month,
                dates=dates,
                start=args.print_start,
                end=args.print_end,
                include_observed=include_obs,
                print_horizon=int(args.print_horizon),
            )
            labs = [lab_fine[i] for i in idxs]
            xx = x_full[idxs] if idxs else np.array([], float)
            DD = y_full[:, idxs] if idxs else np.zeros((y_full.shape[0], 0), float)
            isF = [bool(i >= T) for i in idxs]
            _print_block(
                name="fine",
                labels=labs,
                x=xx,
                draws=DD,
                is_forecast=isF,
                level=float(args.level),
                max_lines=max_lines,
            )

        # ---- annual ----
        if "annual" in scales:
            idxsY = _select_by_years(
                x=xY,
                forecast_mask=maskY,
                dates=dates,
                start=args.print_start,
                end=args.print_end,
                include_observed=include_obs,
            )
            labs = [f"{int(np.floor(xY[i])):04d}" for i in idxsY]
            xx = xY[idxsY] if idxsY else np.array([], float)
            DD = drawsY[:, idxsY] if idxsY else np.zeros((drawsY.shape[0], 0), float)
            isF = [bool(maskY[i]) for i in idxsY]
            _print_block(
                name="annual",
                labels=labs,
                x=xx,
                draws=DD,
                is_forecast=isF,
                level=float(args.level),
                max_lines=max_lines,
            )

        # ---- seasonal_all ----
        if "seasonal_all" in scales:
            if seasonal_all is None or labelsA is None:
                print("\n[print] seasonal_all: not available (seasonal coarse-graining was skipped)\n")
            else:
                # unpack from above (guaranteed to exist here)
                if len(seasonal_all) == 6:
                    xA, obsA, drawsA, maskA, splitA, labelsA_ = seasonal_all  # type: ignore
                    labels_use = list(labelsA_)
                else:
                    xA, obsA, drawsA, maskA, splitA = seasonal_all  # type: ignore
                    labels_use = list(labelsA)

                idxsA = _select_by_years(
                    x=xA,
                    forecast_mask=maskA,
                    dates=dates,
                    start=args.print_start,
                    end=args.print_end,
                    include_observed=include_obs,
                )
                labs = [labels_use[i] for i in idxsA]
                xx = xA[idxsA] if idxsA else np.array([], float)
                DD = drawsA[:, idxsA] if idxsA else np.zeros((drawsA.shape[0], 0), float)
                isF = [bool(maskA[i]) for i in idxsA]
                _print_block(
                    name="seasonal_all",
                    labels=labs,
                    x=xx,
                    draws=DD,
                    is_forecast=isF,
                    level=float(args.level),
                    max_lines=max_lines,
                )

        # ---- individual seasons ----
        for sname in ("DJF", "MAM", "JJA", "SON"):
            if sname not in scales:
                continue
            if per_season is None or sname not in per_season:
                print(f"\n[print] {sname}: not available (seasonal coarse-graining was skipped)\n")
                continue
            xS, obsS, drawsS, maskS, splitS = per_season[sname]
            idxsS = _select_by_years(
                x=xS,
                forecast_mask=maskS,
                dates=dates,
                start=args.print_start,
                end=args.print_end,
                include_observed=include_obs,
            )
            labs = [f"{int(np.floor(xS[i])):04d}-{sname}" for i in idxsS]
            xx = xS[idxsS] if idxsS else np.array([], float)
            DD = drawsS[:, idxsS] if idxsS else np.zeros((drawsS.shape[0], 0), float)
            isF = [bool(maskS[i]) for i in idxsS]
            _print_block(
                name=sname,
                labels=labs,
                x=xx,
                draws=DD,
                is_forecast=isF,
                level=float(args.level),
                max_lines=max_lines,
            )

    print("[done] fine + annual + seasonal forecasts written.")


if __name__ == "__main__":
    main()
