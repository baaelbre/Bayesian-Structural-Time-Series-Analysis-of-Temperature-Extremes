# simulator/dgev_laplace_forecast.py
from __future__ import annotations

"""
Uccle DGEV Laplace posterior predictive forecasting.

This is a thin *wrapper* around `simulator/dgev_forecast.py` that:
- uses the same forecasting core + plotting functions,
- mirrors your "latest run" discovery pattern (find_latest_run under a root),
- adds Uccle-friendly defaults: series-aware Laplace roots + --series/--all,
- FIXES the only real gotcha in the base script:
    for minima-series (TXn/TNn), the *observed* annual/seasonal block value must be MIN,
    not MAX (forecast distributions were already correct due to the sign-flip trick).

Run examples
------------
# latest TXn monthly Laplace run (Uccle defaults)
python -u simulator/dgev_laplace_forecast.py --series TXn --h 60

# explicit run directory
python -u simulator/dgev_laplace_forecast.py --target results/uccle/TX/TXn/Monthly/Laplace/<run_dir> --h 120

# run all common temperature indices
python -u simulator/dgev_laplace_forecast.py --all --h 60

Notes
-----
- Default root (if --root and --target are omitted) is:
    results/uccle/<GROUP>/<SERIES>/<FREQ>/Laplace
  where GROUP is inferred from SERIES (TX*, TN*, Prec*).
- Plot policy is inherited from dgev_forecast.py: observed + forecast median + band only.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

# ---------------------------------------------------------------------
# Make optimization + simulator packages visible (matches your repo style)
# ---------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------
# Posterior loader (EXACTLY like your plotters/forecast script)
# ---------------------------------------------------------------------
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e

# ---------------------------------------------------------------------
# Re-use the generic DGEV forecasting implementation
# ---------------------------------------------------------------------
try:
    from simulator import dgev_laplace_forecast as base  # type: ignore
except Exception:
    import dgev_laplace_forecast as base  # type: ignore


# =============================================================================
# Uccle roots
# =============================================================================
def _series_group(series: str) -> str:
    s = str(series).strip()
    if s.startswith("TX"):
        return "TX"
    if s.startswith("TN"):
        return "TN"
    if s.lower().startswith("prec"):
        return "Prec"
    # fallback: use leading alpha chunk if any
    head = "".join([c for c in s if c.isalpha()])
    return head if head else "misc"


def _default_uccle_root(series: str, freq: str) -> str:
    """
    Uccle default Laplace root for a given series/frequency.
    freq should be "Monthly" or "Seasonal" (capitalized like your folder names).
    """
    grp = _series_group(series)
    return os.path.join("results", "uccle", grp, str(series), str(freq), "Laplace")


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


# =============================================================================
# Patch: observed block extremes for minima-series
# =============================================================================
def _fix_observed_block_extremes_inplace(fc: Dict[str, Any]) -> None:
    """
    Base forecast script computes observed annual/seasonal block values using max(y_obs),
    which is wrong for minima-series. Fix by recomputing:
        minima: min over observed months in the block
        maxima: max over observed months in the block

    Forecast parts are already correct in base (due to sign-flip on model scale).
    """
    try:
        minima = bool(fc.get("minima", False))
        y_obs = np.asarray(fc["y_obs"], float)  # ORIGINAL scale, length T
        T = int(fc["T"])
        h = int(fc["h"])
        period = int(fc["period"])
    except Exception:
        return

    def reduce_obs(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        return float(np.min(arr)) if minima else float(np.max(arr))

    dates_full = fc.get("dates_full", None)

    # --- annual obs ---
    annual = fc.get("annual", None)
    if isinstance(annual, dict) and "years" in annual and "obs" in annual:
        years = np.asarray(annual["years"], int)
        new_obs = np.full(years.shape, np.nan, float)

        if dates_full is not None:
            # calendar-year blocks
            years_train = np.asarray(dates_full.year, int)[:T]
            for i, yy in enumerate(years):
                idx_train = np.where(years_train == int(yy))[0]
                new_obs[i] = reduce_obs(y_obs[idx_train])
        else:
            # fallback blocks of length=period
            year_id_train = (np.arange(T + h, dtype=int) // max(1, period)).astype(int)[:T]
            for i, yy in enumerate(years):
                idx_train = np.where(year_id_train == int(yy))[0]
                new_obs[i] = reduce_obs(y_obs[idx_train])

        annual["obs"] = new_obs

    # --- seasons obs (DJF/MAM/JJA/SON) ---
    seasons = fc.get("seasons", None)
    if (
        isinstance(seasons, dict)
        and seasons
        and "season_year" in seasons
        and "season_name" in seasons
        and "obs" in seasons
        and dates_full is not None
        and period == 12
    ):
        sy = np.asarray(seasons["season_year"], int)
        sn = np.asarray(seasons["season_name"], object)
        new_obs = np.full(sy.shape, np.nan, float)

        labels_train: List[Tuple[int, str]] = [
            (base._season_year(int(dt.year), int(dt.month)), base._season_label(int(dt.month)))
            for dt in dates_full[:T]
        ]
        for i in range(len(sy)):
            key = (int(sy[i]), str(sn[i]))
            idx_train = np.array([j for j, lab in enumerate(labels_train) if lab == key], dtype=int)
            new_obs[i] = reduce_obs(y_obs[idx_train])

        seasons["obs"] = new_obs


# =============================================================================
# Run resolution (mirrors plotter pattern)
# =============================================================================
def _resolve_run_path(series: str, *, target: Optional[str], root: Optional[str], freq: str) -> Optional[str]:
    if target is not None:
        return target

    search_root = root if root is not None else _default_uccle_root(series, freq)
    print(f"[info] --target not provided; searching for the latest posterior under --root={search_root!r} ...")
    run_path = find_latest_run(root=search_root)
    if run_path is None:
        print(f"[error] No posterior found under {search_root!r}. Provide --target or set --root.")
        return None
    print(f"[info] Using latest run: {run_path}")
    return run_path


def _run_one(series: str, args: argparse.Namespace) -> None:
    run_path = _resolve_run_path(series, target=args.target, root=args.root, freq=args.freq)
    if run_path is None:
        return

    bundle = load_posterior(run_path)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "forecasts")
    _ensure_dir(out_dir)
    print(f"[info] saving forecasts to: {out_dir}")

    # minima override (exactly like base)
    if args.minima:
        minima = True
    elif args.maxima:
        minima = False
    else:
        minima = base._detect_minima_from_meta(meta)
        if minima:
            print("[info] minima=True detected from meta → back-transforming forecasts for plotting.")

    rng = np.random.default_rng(int(args.seed))

    fc = base.forecast_from_bundle(
        draws=draws,
        meta=meta,
        h=int(args.h),
        rng=rng,
        n_draws=args.n_draws,
        rep_per_draw=int(args.rep_per_draw),
        rep_per_draw_max=int(args.rep_per_draw_max),
        alpha=float(args.alpha),
        minima=bool(minima),
    )

    # patch observed block extrema for minima-series
    _fix_observed_block_extremes_inplace(fc)

    ext_word = "min" if fc["minima"] else "max"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ser_lab = str(meta.get("series", series))

    # plots (reuse base plotting)
    base._plot_fine(
        fc,
        os.path.join(out_dir, f"forecast_fine_{ser_lab}_{stamp}.png"),
        title=f"{ser_lab}: DGEV fine forecast (h={int(args.h)})",
        alpha=float(args.alpha),
    )

    base._plot_annual(
        fc["annual"],
        os.path.join(out_dir, f"forecast_annual_{ser_lab}_{ext_word}_{stamp}.png"),
        title=f"{ser_lab}: DGEV annual-{ext_word} forecast",
    )

    seasons = fc.get("seasons", {})
    if seasons:
        base._plot_seasonal_all(
            seasons,
            os.path.join(out_dir, f"forecast_seasonal_all_{ser_lab}_{ext_word}_{stamp}.png"),
            title=f"{ser_lab}: DGEV seasonal-{ext_word} forecast (all seasons sequential)",
        )
        for s in ["DJF", "MAM", "JJA", "SON"]:
            base._plot_season_by_name(
                seasons,
                s,
                os.path.join(out_dir, f"forecast_seasonal_{s}_{ser_lab}_{ext_word}_{stamp}.png"),
                title=f"{ser_lab}: DGEV seasonal-{ext_word} forecast ({s})",
            )
    else:
        print("[info] Seasonal summaries not produced (need period==12 and meta.start_date).")

    # save NPZ summary (same style as base; dates_full not saved)
    if bool(args.save_npz):
        npz_out = os.path.join(out_dir, f"forecast_summary_{ser_lab}_{stamp}.npz")
        payload: Dict[str, Any] = {
            "series": np.asarray([ser_lab], object),
            "period": np.asarray([fc["period"]], int),
            "T": np.asarray([fc["T"]], int),
            "h": np.asarray([fc["h"]], int),
            "minima": np.asarray([fc["minima"]], bool),
            "x_train": np.asarray(fc["x_train"], float),
            "x_fore": np.asarray(fc["x_fore"], float),
            "y_obs": np.asarray(fc["y_obs"], float),
            "y_fore_samples": np.asarray(fc["y_fore_samples"], float),
            "mu_fore": np.asarray(fc["mu_fore"], float),
        }
        for k, v in fc["annual"].items():
            payload[f"annual_{k}"] = np.asarray(v)
        if seasons:
            for k, v in seasons.items():
                payload[f"seasons_{k}"] = np.asarray(v)

        np.savez_compressed(npz_out, **payload)

        meta_out = {
            "series": ser_lab,
            "source_run_path": str(run_path),
            "posterior_npz_path": str(npz_path),
            "created": stamp,
            "alpha": float(args.alpha),
            "seed": int(args.seed),
            "n_draws": None if args.n_draws is None else int(args.n_draws),
            "rep_per_draw": int(args.rep_per_draw),
            "rep_per_draw_max": int(args.rep_per_draw_max),
            "minima_used": bool(minima),
            "freq": str(args.freq),
            "notes": {
                "wrapper": "simulator/dgev_laplace_forecast.py (wrapper around simulator/dgev_forecast.py)",
                "seasonal_grouping": "meteorological seasons only when period==12 and dates are available",
                "annual_grouping": "calendar-year if dates available else blocks of length=period",
                "plot_policy": "observed + forecast median + forecast band only (no fit ribbons)",
                "obs_fix": "observed annual/seasonal extrema use min for minima-series, max otherwise",
            },
        }
        with open(npz_out.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)

        print(f"[save] {npz_out}")

    print("[done] forecasts written.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Uccle DGEV Laplace posterior predictive forecasting.\n"
            "If --target omitted, uses find_latest_run(root=--root).\n"
            "If --root omitted, uses Uccle default root inferred from --series and --freq."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--series", type=str, default="TXn",
                        help="Uccle series name (e.g. TXx, TXn, TNx, TNn, TXm, TNm, Precx, ...).")
    parser.add_argument("--all", action="store_true",
                        help="Run a standard set of series: TXx TXn TNx TNn (ignores --series).")

    parser.add_argument("--freq", choices=["Monthly", "Seasonal"], default="Monthly",
                        help="Folder level under series (matches your results/<...>/<freq>/Laplace structure).")

    parser.add_argument("--target", type=str, default=None,
                        help="Path to a run directory or directly to posterior.npz. If omitted, searches under --root.")
    parser.add_argument("--root", type=str, default=None,
                        help="Search root if --target omitted. If omitted, uses Uccle default root for the series.")

    parser.add_argument("--h", type=int, default=120, help="Forecast horizon in native block units.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Band tail prob (alpha=0.05 -> 90% band).")
    parser.add_argument("--n-draws", type=int, default=None, help="Subsample this many posterior draws (default all).")
    parser.add_argument("--rep-per-draw", type=int, default=1, help="Fine predictive replicates per posterior draw.")
    parser.add_argument("--rep-per-draw-max", type=int, default=1, help="Block-extreme replicates per posterior draw.")
    parser.add_argument("--seed", type=int, default=40)

    parser.add_argument("--out", type=str, default=None, help="Output dir. Default: <run>/forecasts")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--minima", action="store_true", help="Force minima=True (back-transform stored negated series).")
    g.add_argument("--maxima", action="store_true", help="Force minima=False (no back-transform).")

    parser.add_argument("--save-npz", action="store_true", default=True, help="Save forecast_summary_*.npz")
    args = parser.parse_args()

    if args.all:
        for s in ["TXx", "TXn", "TNx", "TNn"]:
            _run_one(s, args)
    else:
        _run_one(str(args.series), args)


if __name__ == "__main__":
    main()
