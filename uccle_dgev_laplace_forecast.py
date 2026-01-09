# %% simulator/uccle_dgev_laplace_forecast.py
from __future__ import annotations
"""
Uccle DGEV Laplace Forecast (TXx, TXn, TNx, TNn, Precx; Monthly/Seasonal)
========================================================================

Uccle wrapper around the generic DGEV forecaster.

Key features
------------
- Mirrors uccle_dlm_forecast workflow:
  - --target loads a specific run dir or posterior(.npz)
  - otherwise robust "latest run" discovery under --root (or Uccle defaults)
    1) optimization.posterior_bundle.find_latest_run (posterior.npz runs)
    2) fallback recursive search for posterior*.npz
- Uccle defaults for root layout:
    results/uccle/<GROUP>/<SERIES>/<FREQ>/Laplace
  where GROUP inferred from SERIES prefix (TX/TN/Prec).
- Optional post-hoc --burn/--thin trimming.
- Minimal plots: NO titles and NO legends (labels only).
- A single --xlabel/--ylabel applied to all plots.
- NEW: prints prediction summaries (median + interval) to stdout, similarly to the generic forecaster.

Outputs
-------
Saved to <run>/forecast by default:
  - <series>_dgev_forecast_fine.png (+ .npz)
  - <series>_dgev_forecast_annual_extreme.png (+ .npz)
  - <series>_dgev_forecast_seasonal_all_extreme.png (+ .npz)
  - <series>_dgev_forecast_seasonal_{DJF,MAM,JJA,SON}_extreme.png (+ .npz)
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

# Generic DGEV forecasting logic (try canonical name first; fall back for older file name)
try:
    from simulator.dgev_forecast import (  # type: ignore
        simulate_dgev_forecast,
        coarse_grain_annual_extreme,
        coarse_grain_meteo_seasons_extreme,
    )
except Exception:
    from simulator.dgev_laplace_forecast import (  # type: ignore
        simulate_dgev_forecast,
        coarse_grain_annual_extreme,
        coarse_grain_meteo_seasons_extreme,
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


def _decimal_year_from_ym(y: int, m: int) -> float:
    return float(y) + (float(m) - 0.5) / 12.0


def _ym_from_decimal_year(x: float) -> Tuple[int, int]:
    y = int(np.floor(float(x)))
    frac = float(x) - float(y)
    m = int(np.round(frac * 12.0 + 0.5))
    m = int(np.clip(m, 1, 12))
    return y, m


def _season_name_from_end_month(m: int) -> str:
    if m == 2:
        return "DJF"
    if m == 5:
        return "MAM"
    if m == 8:
        return "JJA"
    if m == 11:
        return "SON"
    return "SEAS"


def _summarize_ribbon(draws_2d: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """draws_2d: (S, L) -> (median, lo, hi) across S."""
    loq = (1.0 - level) / 2.0
    hiq = 1.0 - loq
    med = np.quantile(draws_2d, 0.5, axis=0)
    lo = np.quantile(draws_2d, loq, axis=0)
    hi = np.quantile(draws_2d, hiq, axis=0)
    return med, lo, hi


# =============================================================================
# Printing predictions (median + interval)
# =============================================================================
def _clip_indices(idxs: Iterable[int], L: int) -> List[int]:
    out: List[int] = []
    for i in idxs:
        ii = int(i)
        if 0 <= ii < L:
            out.append(ii)
    return sorted(set(out))


def _fine_index_from_date(dt: datetime, *, start_year: int, start_month: int, step_months: int) -> int:
    diff_months = (dt.year - start_year) * 12 + (dt.month - start_month)
    k = int(round(diff_months / float(step_months)))
    return k


def _fine_labels_from_axis(x_full: np.ndarray, *, has_calendar: bool) -> List[str]:
    if not has_calendar:
        return [str(int(i)) for i in range(int(x_full.size))]
    labs: List[str] = []
    for x in x_full:
        y, m = _ym_from_decimal_year(float(x))
        labs.append(f"{y:04d}-{m:02d}")
    return labs


def _select_fine_indices(
    *,
    L_full: int,
    T_obs: int,
    H: int,
    dates: List[str],
    start: Optional[str],
    end: Optional[str],
    include_observed: bool,
    print_horizon: int,
    has_calendar: bool,
    start_year: Optional[int],
    start_month: Optional[int],
    step_months: Optional[int],
) -> List[int]:
    if (not dates) and (start is None) and (end is None):
        ph = int(print_horizon)
        if ph <= 0:
            ph = H
        ph = min(ph, H)
        return list(range(T_obs, min(L_full, T_obs + ph)))

    if not has_calendar or start_year is None or start_month is None or step_months is None:
        raise ValueError(
            "Date-based printing for fine scale requires a calendar axis "
            "(need --start-date or meta['start_date'] + a period dividing 12)."
        )

    if dates:
        idxs: List[int] = []
        for s in dates:
            dt = _parse_date(s)
            if dt is None:
                continue
            idxs.append(_fine_index_from_date(dt, start_year=start_year, start_month=start_month, step_months=step_months))
        idxs = _clip_indices(idxs, L_full)
    else:
        if start is None or end is None:
            raise ValueError("For a fine-scale window, provide BOTH --print-start and --print-end.")
        dt0 = _parse_date(start)
        dt1 = _parse_date(end)
        if dt0 is None or dt1 is None:
            raise ValueError("Could not parse --print-start/--print-end.")
        a = _fine_index_from_date(dt0, start_year=start_year, start_month=start_month, step_months=step_months)
        b = _fine_index_from_date(dt1, start_year=start_year, start_month=start_month, step_months=step_months)
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
    header: str,
    labels: List[str],
    x: np.ndarray,
    draws: np.ndarray,  # (S, n)
    is_forecast: List[bool],
    level: float,
    max_lines: int,
) -> None:
    n = int(x.size)
    if n == 0:
        print(f"\n[print] {header}: (no points selected)\n")
        return

    loq = (1.0 - level) / 2.0
    hiq = 1.0 - loq
    med = np.quantile(draws, 0.5, axis=0)
    lo = np.quantile(draws, loq, axis=0)
    hi = np.quantile(draws, hiq, axis=0)

    print(f"\n[print] {header}: {n} points (median + {int(round(level*100))}% interval)")
    print(f"{'label':>12s}  {'x':>10s}  {'median':>10s}  {'lo':>10s}  {'hi':>10s}  {'forecast':>9s}")
    print("-" * 78)

    n_show = min(n, int(max_lines))
    for i in range(n_show):
        print(
            f"{labels[i]:>12s}  "
            f"{float(x[i]):10.3f}  "
            f"{float(med[i]):10.3f}  "
            f"{float(lo[i]):10.3f}  "
            f"{float(hi[i]):10.3f}  "
            f"{str(bool(is_forecast[i])):>9s}"
        )
    if n_show < n:
        print(f"... ({n - n_show} more rows truncated; increase --print-max-lines)")
    print("")


def print_predictions(
    *,
    series: str,
    fr: Any,
    meta: Dict[str, Any],
    sd: Optional[datetime],
    x_full: np.ndarray,
    y_full: np.ndarray,  # (S, T+H)
    T_obs: int,
    H: int,
    level: float,
    xY: np.ndarray,
    drawsY: np.ndarray,
    maskY: np.ndarray,
    seasonal_all: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
    per_season: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]],
    args: argparse.Namespace,
) -> None:
    scales = [s.strip() for s in str(args.print_scales).split(",") if s.strip()]
    dates = _parse_csv(args.print_dates)
    include_obs = bool(args.print_include_observed)
    max_lines = int(args.print_max_lines)

    # calendar axis availability for fine printing
    period = int(getattr(fr, "period", meta.get("period", 12)))
    has_calendar = (sd is not None) and (period in (12, 6, 4, 3, 2, 1)) and ((12 % period) == 0)
    step_months = (12 // period) if has_calendar else None

    L_full = int(x_full.size)
    fine_labels = _fine_labels_from_axis(x_full, has_calendar=has_calendar)

    prefix = f"{series}"

    if "fine" in scales:
        idxs = _select_fine_indices(
            L_full=L_full,
            T_obs=T_obs,
            H=H,
            dates=dates,
            start=args.print_start,
            end=args.print_end,
            include_observed=include_obs,
            print_horizon=int(args.print_horizon),
            has_calendar=has_calendar,
            start_year=(sd.year if sd is not None else None),
            start_month=(sd.month if sd is not None else None),
            step_months=step_months,
        )
        labs = [fine_labels[i] for i in idxs]
        xx = x_full[idxs] if idxs else np.array([], float)
        DD = y_full[:, idxs] if idxs else np.zeros((y_full.shape[0], 0), float)
        isF = [bool(i >= T_obs) for i in idxs]
        _print_block(
            header=f"{prefix} | fine",
            labels=labs,
            x=xx,
            draws=DD,
            is_forecast=isF,
            level=float(level),
            max_lines=max_lines,
        )

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
            header=f"{prefix} | annual",
            labels=labs,
            x=xx,
            draws=DD,
            is_forecast=isF,
            level=float(level),
            max_lines=max_lines,
        )

    if "seasonal_all" in scales:
        if seasonal_all is None:
            print(f"\n[print] {prefix} | seasonal_all: not available (seasonal coarse-graining was skipped)\n")
        else:
            xA, obsA, drawsA, maskA, splitA = seasonal_all
            idxsA = _select_by_years(
                x=xA,
                forecast_mask=maskA,
                dates=dates,
                start=args.print_start,
                end=args.print_end,
                include_observed=include_obs,
            )
            labs: List[str] = []
            for i in idxsA:
                yy, mm = _ym_from_decimal_year(float(xA[i]))
                labs.append(f"{yy:04d}-{_season_name_from_end_month(mm)}")
            xx = xA[idxsA] if idxsA else np.array([], float)
            DD = drawsA[:, idxsA] if idxsA else np.zeros((drawsA.shape[0], 0), float)
            isF = [bool(maskA[i]) for i in idxsA]
            _print_block(
                header=f"{prefix} | seasonal_all",
                labels=labs,
                x=xx,
                draws=DD,
                is_forecast=isF,
                level=float(level),
                max_lines=max_lines,
            )

    for sname in ("DJF", "MAM", "JJA", "SON"):
        if sname not in scales:
            continue
        if per_season is None or sname not in per_season:
            print(f"\n[print] {prefix} | {sname}: not available (seasonal coarse-graining was skipped)\n")
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
            header=f"{prefix} | {sname}",
            labels=labs,
            x=xx,
            draws=DD,
            is_forecast=isF,
            level=float(level),
            max_lines=max_lines,
        )


# =============================================================================
# Uccle root selection + robust latest discovery
# =============================================================================
def _series_group(series: str) -> str:
    s = str(series).strip()
    if s.startswith("TX"):
        return "TX"
    if s.startswith("TN"):
        return "TN"
    if s.lower().startswith("prec"):
        return "Prec"
    head = "".join([c for c in s if c.isalpha()])
    return head if head else "misc"


def default_root(series: str, freq: str) -> str:
    return os.path.join("results", "uccle", _series_group(series), str(series), str(freq), "Laplace")


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


def resolve_bundle(*, target: Optional[str], series: str, root: Optional[str], freq: str) -> Any:
    if target:
        return load_posterior(target)

    search_root = root or default_root(series, freq)
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
# Post-processing: burn-in + thinning (post-hoc)
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
    for k in ("x", "sigma", "sigma2", "xi", "Q_alpha", "s_alpha", "gamma0"):
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
    if fore_draws.size:
        med, lo, hi = _summarize_ribbon(fore_draws, level=level)
    else:
        med = lo = hi = np.array([], float)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))

    ax.plot(x_obs, y_obs, lw=1.2, alpha=0.85, color="0.25")
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
# Run one series
# =============================================================================
def _run_one(series: str, args: argparse.Namespace) -> None:
    bundle = resolve_bundle(target=args.target, series=series, root=args.root, freq=args.freq)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    meta = dict(meta)

    # Uccle defaults: start_date
    meta.setdefault("start_date", "1892-01-01")

    # manual minima/maxima override (affects forecaster via meta detection)
    if args.minima:
        meta["minima"] = True
    if args.maxima:
        meta["minima"] = False

    # Post-hoc burn/thin
    if args.burn > 0 or args.thin > 1:
        draws, meta = apply_burn_thin(draws, meta, burn=args.burn, thin=args.thin)

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "forecast")
    _ensure_dir(out_dir)

    print(f"[info] series={series} | using posterior: {npz_path}")
    print(f"[info] series={series} | saving outputs to: {out_dir}")

    sd = _parse_date(args.start_date) if args.start_date else _parse_date(meta.get("start_date"))

    # simulate fine-scale posterior predictive
    fr = simulate_dgev_forecast(
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

    col = series_color(series)
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
        save_path=os.path.join(out_dir, f"{series}_dgev_forecast_fine.png"),
        color=col,
        show=bool(args.show),
    )

    if bool(args.save_npz):
        np.savez_compressed(
            os.path.join(out_dir, f"{series}_dgev_forecast_fine.npz"),
            series=str(series),
            npz_path=str(npz_path),
            minima=bool(getattr(fr, "minima", False)),
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
    # 2) Annual extremes (always)
    # -----------------------------
    period = int(meta.get("period", 12))
    xY, yY_obs, drawsY, maskY, splitY = coarse_grain_annual_extreme(
        y_full=y_full,
        T_obs=T,
        period=period,
        start_year=getattr(fr, "start_year", None),
        start_month=getattr(fr, "start_month", None),
        minima=bool(getattr(fr, "minima", False)),
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
        save_path=os.path.join(out_dir, f"{series}_dgev_forecast_annual_extreme.png"),
        color=col,
        show=bool(args.show),
    )

    if bool(args.save_npz):
        np.savez_compressed(
            os.path.join(out_dir, f"{series}_dgev_forecast_annual_extreme.npz"),
            series=str(series),
            minima=bool(getattr(fr, "minima", False)),
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

    if period != 12 or getattr(fr, "start_year", None) is None or getattr(fr, "start_month", None) is None:
        print(f"[warn] series={series} | seasons require period=12 + known start-date. Skipping seasonal plots.")
    else:
        per_season, seasonal_all = coarse_grain_meteo_seasons_extreme(
            y_full=y_full,
            T_obs=T,
            start_year=int(fr.start_year),
            start_month=int(fr.start_month),
            minima=bool(getattr(fr, "minima", False)),
        )

        # seasonal_all
        xA, obsA, drawsA, maskA, splitA = seasonal_all
        obs_idxA = np.isfinite(obsA)
        x_obs_A = xA[obs_idxA]
        y_obs_A = obsA[obs_idxA]

        x_fore_A = xA[maskA]
        fore_draws_A = drawsA[:, maskA] if x_fore_A.size else np.zeros((drawsA.shape[0], 0))

        wS = max(1, int(args.window_seasons))
        if x_obs_A.size > wS:
            x_obs_A = x_obs_A[-wS:]
            y_obs_A = y_obs_y = y_obs_A[-wS:]

        plot_forecast_minimal(
            x_obs=x_obs_A,
            y_obs=y_obs_A,
            x_fore=x_fore_A,
            fore_draws=fore_draws_A,
            level=float(args.level),
            split_x=float(splitA),
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=os.path.join(out_dir, f"{series}_dgev_forecast_seasonal_all_extreme.png"),
            color=col,
            show=bool(args.show),
        )

        if bool(args.save_npz):
            np.savez_compressed(
                os.path.join(out_dir, f"{series}_dgev_forecast_seasonal_all_extreme.npz"),
                series=str(series),
                minima=bool(getattr(fr, "minima", False)),
                x_all=xA,
                y_obs_all=obsA,
                draws_all=drawsA,
                forecast_mask_all=maskA,
                split_x=float(splitA),
                level=float(args.level),
                xlabel=xlabel,
                ylabel=ylabel,
            )

        # separate seasons
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
                save_path=os.path.join(out_dir, f"{series}_dgev_forecast_seasonal_{sname}_extreme.png"),
                color=col,
                show=bool(args.show),
            )

            if bool(args.save_npz):
                np.savez_compressed(
                    os.path.join(out_dir, f"{series}_dgev_forecast_seasonal_{sname}_extreme.npz"),
                    series=str(series),
                    season=str(sname),
                    minima=bool(getattr(fr, "minima", False)),
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
    # 4) Print predictions (stdout)
    # -----------------------------
    if bool(args.print_forecast):
        print_predictions(
            series=series,
            fr=fr,
            meta=meta,
            sd=sd,
            x_full=x_full,
            y_full=y_full,
            T_obs=T,
            H=H,
            level=float(args.level),
            xY=xY,
            drawsY=drawsY,
            maskY=maskY,
            seasonal_all=seasonal_all,
            per_season=per_season,
            args=args,
        )

    print(f"[done] series={series} | forecasts written.\n")


# =============================================================================
# CLI
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Uccle DGEV Laplace forecast (TXx/TXn/TNx/TNn/Precx).\n"
            "- Loads latest posterior by default (like uccle plotters).\n"
            "- Fine-scale posterior predictive via state propagation + GEV simulation.\n"
            "- Coarse-grains per draw to annual extremes and meteorological seasons.\n"
            "- Plots with NO titles and NO legends.\n"
            "- Prints prediction summaries to stdout by default.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to run directory or posterior.npz. If omitted uses latest under --root.",
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Search root when --target is omitted. Default: Uccle Laplace layout for the series.",
    )

    p.add_argument("--series", type=str, default="TXn", help="Series code (e.g. TXx, TXn, TNx, TNn, Precx).")
    p.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Run TXx, TXn, TNx, TNn, Precx sequentially (ignores --series).",
    )
    p.add_argument(
        "--freq",
        type=str,
        choices=["Monthly", "Seasonal"],
        default="Monthly",
        help="Folder level under series (matches results/<...>/<freq>/Laplace).",
    )

    p.add_argument("--horizon", type=int, default=12 * 20, help="Forecast horizon in native block units.")
    p.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for predictive simulation.")

    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional calendar start date (YYYY / YYYY-MM / YYYY-MM-DD). Default: meta['start_date'] or 1892-01-01.",
    )

    p.add_argument("--out", type=str, default=None, help="Output directory. Default: <run>/forecast")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")

    # save payloads
    gsave = p.add_mutually_exclusive_group()
    gsave.add_argument("--save-npz", dest="save_npz", action="store_true", default=True, help="Save per-plot .npz payloads.")
    gsave.add_argument("--no-save-npz", dest="save_npz", action="store_false", help="Do not save .npz payloads.")

    # Post-hoc chain trimming
    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    # Plot windows
    p.add_argument("--window-months", type=int, default=12 * 50, help="Fine-scale plot: last N observed months/blocks.")
    p.add_argument("--window-years", type=int, default=60, help="Annual plot: last N observed years.")
    p.add_argument("--window-seasons", type=int, default=4 * 50, help="Seasonal plot: last N observed seasons.")

    # Axis labels (single label for all plots)
    p.add_argument("--xlabel", type=str, default="", help="x-axis label (applied to all plots).")
    p.add_argument("--ylabel", type=str, default="T (°C)", help="y-axis label (applied to all plots).")

    # minima/maxima override
    g = p.add_mutually_exclusive_group()
    g.add_argument("--minima", action="store_true", default=False, help="Force minima=True for plotting/printing.")
    g.add_argument("--maxima", action="store_true", default=False, help="Force minima=False for plotting/printing.")

    # ---- printing predictions (stdout) ----
    gpr = p.add_mutually_exclusive_group()
    gpr.add_argument("--print-forecast", dest="print_forecast", action="store_true", default=True, help="Print prediction summaries.")
    gpr.add_argument("--no-print-forecast", dest="print_forecast", action="store_false", help="Disable printing prediction summaries.")

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
        help="(fine) If no --print-dates/--print-start/--print-end: print first N forecast steps (<=0 prints all).",
    )
    p.add_argument(
        "--print-dates",
        type=str,
        default=None,
        help="Comma-separated dates (YYYY or YYYY-MM or YYYY-MM-DD). fine uses calendar steps; annual/seasonal use years.",
    )
    p.add_argument("--print-start", type=str, default=None, help="Window start (YYYY or YYYY-MM or YYYY-MM-DD).")
    p.add_argument("--print-end", type=str, default=None, help="Window end (YYYY or YYYY-MM or YYYY-MM-DD).")
    p.add_argument("--print-include-observed", action="store_true", default=False, help="Allow printing observed-time points too.")
    p.add_argument("--print-max-lines", type=int, default=80, help="Max printed rows per scale (truncate beyond this).")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.all:
        for s in ["TXx", "TXn", "TNx", "TNn", "Precx"]:
            _run_one(s, args)
    else:
        _run_one(str(args.series), args)


if __name__ == "__main__":
    main()
