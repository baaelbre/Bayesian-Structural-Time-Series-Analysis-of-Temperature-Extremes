# %% simulator/uccle_dgev_laplace_forecast.py
from __future__ import annotations
"""
Uccle DGEV Laplace Forecast (TXx, TXn, TNx, TNn, Precx; Monthly/Seasonal)
========================================================================

Uccle wrapper around the *generic* DGEV forecaster:

    simulator.dgev_forecast.py

Key features
------------
- Mirrors the uccle_dlm_forecast workflow:
  - --target loads a specific run or posterior.npz
  - otherwise robust "latest run" discovery under --root (or Uccle defaults)
    1) optimization.posterior_bundle.find_latest_run (posterior.npz runs)
    2) fallback recursive search for posterior*.npz
- Uccle defaults for root layout:
    results/uccle/<GROUP>/<SERIES>/<FREQ>/Laplace
  where GROUP inferred from SERIES prefix (TX/TN/Prec).
- Optional post-hoc --burn/--thin trimming.
- Minimal plots: NO titles and NO legends (labels only).
- A *single* --xlabel/--ylabel applied to all plots (no annual/season-specific labels).

Outputs
-------
Saved to <run>/forecast by default:
  - <series>_dgev_forecast_fine.png (+ .npz)
  - <series>_dgev_forecast_annual_extreme.png (+ .npz)
  - <series>_dgev_forecast_seasonal_all_extreme.png (+ .npz)
  - <series>_dgev_forecast_seasonal_{DJF,MAM,JJA,SON}_extreme.png (+ .npz)

Notes on minima
---------------
Minima series (TXn/TNn) are typically modeled via sign-flip in the sampler meta.
The generic forecaster detects this from meta and returns plot-scale values.
Annual/seasonal *observed* block extremes are computed correctly as min for minima,
max for maxima (no extra patching needed).
"""

import os
import sys
import re
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Make project root importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Posterior I/O (same pattern as plotters)
from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore

# Generic DGEV forecasting logic (the "mirrors dlm_forecast" version)
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
    raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")


def _summarize_ribbon(draws_2d: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """draws_2d: (S, L) -> (median, lo, hi) across S."""
    loq = (1.0 - level) / 2.0
    hiq = 1.0 - loq
    med = np.quantile(draws_2d, 0.5, axis=0)
    lo = np.quantile(draws_2d, loq, axis=0)
    hi = np.quantile(draws_2d, hiq, axis=0)
    return med, lo, hi


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

    # infer chain length from a canonical key
    n_samp: Optional[int] = None
    for k in ("x", "sigma", "sigma2", "xi", "Q_alpha", "s_alpha"):
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

    # observed
    ax.plot(x_obs, y_obs, lw=1.2, alpha=0.85, color="0.25")

    # forecast ribbon + median
    if x_fore.size:
        ax.fill_between(x_fore, lo, hi, alpha=0.20, color=color)
        ax.plot(x_fore, med, lw=1.6, color=color)

    # split line
    ax.axvline(float(split_x), lw=1.0, alpha=0.85, color="0.35")

    # labels only
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

    # Uccle calendar defaults (monthly origin); do not fight explicit CLI start-date
    meta = dict(meta)
    if args.freq == "Monthly":
        meta["start_date"] = meta.get("start_date", "1892-01-01")
        meta["freq"] = "Monthly"
        meta["period"] = int(meta.get("period", 12))
    else:
        meta["start_date"] = meta.get("start_date", "1892-01-01")
        meta["freq"] = "Seasonal"

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

    print(f"[info] using posterior: {npz_path}")
    print(f"[info] saving outputs to: {out_dir}")

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

    fine_png = os.path.join(out_dir, f"{series}_dgev_forecast_fine.png")
    plot_forecast_minimal(
        x_obs=x_obs_m,
        y_obs=y_obs_m,
        x_fore=x_fore_m,
        fore_draws=fore_draws_m,
        level=float(args.level),
        split_x=float(fr.split_x),
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=fine_png,
        color=col,
        show=bool(args.show),
    )

    if args.save_npz:
        np.savez_compressed(
            os.path.join(out_dir, f"{series}_dgev_forecast_fine.npz"),
            series=str(series),
            npz_path=str(npz_path),
            minima=bool(fr.minima),
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
        start_year=fr.start_year,
        start_month=fr.start_month,
        minima=bool(fr.minima),
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

    annual_png = os.path.join(out_dir, f"{series}_dgev_forecast_annual_extreme.png")
    plot_forecast_minimal(
        x_obs=x_obs_y,
        y_obs=y_obs_y,
        x_fore=x_fore_y,
        fore_draws=fore_draws_y,
        level=float(args.level),
        split_x=float(splitY),
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=annual_png,
        color=col,
        show=bool(args.show),
    )

    if args.save_npz:
        np.savez_compressed(
            os.path.join(out_dir, f"{series}_dgev_forecast_annual_extreme.npz"),
            series=str(series),
            minima=bool(fr.minima),
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
    if period != 12 or fr.start_year is None or fr.start_month is None:
        print("[warn] meteorological seasons (DJF/MAM/JJA/SON) require period=12 and a known start-date. Skipping.")
        return

    per_season, seasonal_all = coarse_grain_meteo_seasons_extreme(
        y_full=y_full,
        T_obs=T,
        start_year=int(fr.start_year),
        start_month=int(fr.start_month),
        minima=bool(fr.minima),
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
        y_obs_A = y_obs_A[-wS:]

    seas_all_png = os.path.join(out_dir, f"{series}_dgev_forecast_seasonal_all_extreme.png")
    plot_forecast_minimal(
        x_obs=x_obs_A,
        y_obs=y_obs_A,
        x_fore=x_fore_A,
        fore_draws=fore_draws_A,
        level=float(args.level),
        split_x=float(splitA),
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=seas_all_png,
        color=col,
        show=bool(args.show),
    )

    if args.save_npz:
        np.savez_compressed(
            os.path.join(out_dir, f"{series}_dgev_forecast_seasonal_all_extreme.npz"),
            series=str(series),
            minima=bool(fr.minima),
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

        png = os.path.join(out_dir, f"{series}_dgev_forecast_seasonal_{sname}_extreme.png")
        plot_forecast_minimal(
            x_obs=x_obs_S,
            y_obs=y_obs_S,
            x_fore=x_fore_S,
            fore_draws=fore_draws_S,
            level=float(args.level),
            split_x=float(splitS),
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=png,
            color=col,
            show=bool(args.show),
        )

        if args.save_npz:
            np.savez_compressed(
                os.path.join(out_dir, f"{series}_dgev_forecast_seasonal_{sname}_extreme.npz"),
                series=str(series),
                season=str(sname),
                minima=bool(fr.minima),
                x=xS,
                y_obs=obsS,
                draws=drawsS,
                forecast_mask=maskS,
                split_x=float(splitS),
                level=float(args.level),
                xlabel=xlabel,
                ylabel=ylabel,
            )

    print("[done] fine + annual + seasonal forecasts written.\n")


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
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--target", type=str, default=None, help="Path to run directory or posterior.npz. If omitted uses latest under --root.")
    p.add_argument("--root", type=str, default=None, help="Search root when --target is omitted. Default: Uccle Laplace layout for the series.")

    p.add_argument("--series", type=str, default="TXn", help="Series code (e.g. TXx, TXn, TNx, TNn, Precx).")
    p.add_argument("--all", action="store_true", default=True, help="Run TXx, TXn, TNx, TNn sequentially (ignores --series).")
    p.add_argument("--freq", type=str, choices=["Monthly", "Seasonal"], default="Monthly", help="Folder level under series (matches results/<...>/<freq>/Laplace).")

    p.add_argument("--horizon", type=int, default=12*20, help="Forecast horizon in native block units.")
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
    p.add_argument("--save-npz", action="store_true", default=True, help="Save per-plot .npz payloads")

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
    g.add_argument("--minima", action="store_true", default=False, help="Force minima=True for plotting.")
    g.add_argument("--maxima", action="store_true", default=False, help="Force minima=False for plotting.")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.all:
        for s in ["TXx", "TXn", "TNx", "TNn"]:
            _run_one(s, args)
    else:
        _run_one(str(args.series), args)


if __name__ == "__main__":
    main()
