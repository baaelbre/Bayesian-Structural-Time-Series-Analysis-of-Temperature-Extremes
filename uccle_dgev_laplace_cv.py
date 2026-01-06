# %% simulator/uccle_dgev_laplace_cv.py
from __future__ import annotations
"""
Uccle DGEV Laplace Cross-Validation (rolling-origin; fine-scale forecast plots)
=============================================================================

This script *combines* the sampler runner pattern (fit DGEV Laplace NCP + lasso)
with the forecaster (simulate_dgev_forecast) to perform time-series CV.

Core idea
---------
For each fold k:
  - fit the model on y_{1:T_train(k)}
  - generate posterior predictive draws for the next H months
  - compare draws to the *actually observed* heldout months
  - save a fine-scale plot + simple proper scores (CRPS, coverage, PITs)

Minima
------
TXn and TNn are block MINIMA. We fit the sampler on the MODEL scale z_t = -y_t.
The forecast + plots + scoring are done on the PLOT scale (original units),
using meta["model_sign"] and meta["minima"] so that the generic forecaster
returns plot-scale values.

Example
-------
# One series, 5 folds, 10-year horizon, step 5 years between splits
python -u simulator/uccle_dgev_laplace_cv.py --series TXn --n-folds 5 --horizon 120 --step-months 60

# All four temperature extremes
python -u simulator/uccle_dgev_laplace_cv.py --all --n-folds 4 --horizon 120 --step-months 120
"""

import os
import sys
import re
import math
import json
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------
# Make project root importable (run from anywhere)
# ---------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Sampler
from optimization.dgev_laplace_2 import DGEVLaplaceNCP, Priors, SamplerConfig  # type: ignore

# Generic DGEV forecasting logic (already handles minima via meta)
from simulator.dgev_laplace_forecast import simulate_dgev_forecast  # type: ignore


# =============================================================================
# Series config + transforms
# =============================================================================
SERIES_FILES_DEFAULT: Dict[str, str] = {
    "TXx": "TXx.csv",
    "TXn": "TXn.csv",
    "TNx": "TNx.csv",
    "TNn": "TNn.csv",
}


def _tail_sign(series: str) -> float:
    """Sampler is written for GEV maxima; minima are sign-flipped on the MODEL scale."""
    return -1.0 if series in {"TXn", "TNn"} else 1.0


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _series_group(series: str) -> str:
    s = str(series).strip()
    if s.startswith("TX"):
        return "TX"
    if s.startswith("TN"):
        return "TN"
    head = "".join([c for c in s if c.isalpha()])
    return head if head else "misc"


def _series_out_root(series: str) -> Path:
    """
    Mirror your Laplace layout:
      results/uccle/<TX|TN>/<SERIES>/Monthly/Laplace
    """
    base = Path("results") / "uccle"
    return base / _series_group(series) / series / "Monthly" / "Laplace"


def _safe_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", s)


def _date_tag_from_index(idx: pd.PeriodIndex) -> str:
    if len(idx) == 0:
        return "NA-NA"
    return f"{idx[0]}-{idx[-1]}"


# =============================================================================
# CLI helpers
# =============================================================================
def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "":
        return None
    return [float(tok) for tok in ss.split(",") if tok.strip()]


# =============================================================================
# Seasonal pattern utilities (sum-to-zero parametrisation)
# =============================================================================
def _default_seasonal_pattern_full(period: int) -> np.ndarray:
    """Smooth seasonal template, mean-centered (sum-to-zero)."""
    p = int(period)
    g = np.cos(2.0 * np.pi * np.arange(p) / p)
    g = g - g.mean()
    g = g - (g.sum() / p)
    return g.astype(float)


def _to_gamma0_pminus1(vals: List[float], period: int) -> np.ndarray:
    """
    Convert to gamma0 length (p-1) for sum-to-zero parametrisation.
      seasonal_full = [gamma0...,  -sum(gamma0)]
    """
    p = int(period)
    arr = np.asarray(vals, float).ravel()
    if arr.size == p - 1:
        return arr.copy()
    if arr.size == p:
        arr = arr - arr.mean()
        arr = arr - (arr.sum() / p)
        return arr[: p - 1].copy()
    raise ValueError(f"Season pattern must have length {p-1} or {p}, got {arr.size}.")


# =============================================================================
# Data loading + model-scale transform
# =============================================================================
def load_monthly_series(
    csv_path: Path,
    *,
    start_year: int,
    end_year: int,
    force_start_january: bool = True,
    trim_full_years: bool = True,
) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0)

    if "value" in df.columns:
        vals = pd.to_numeric(df["value"], errors="coerce")
    else:
        df2 = df.copy()
        for c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="ignore")
        numcols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
        if not numcols:
            raise ValueError(f"No numeric column found in {csv_path}")
        vals = pd.to_numeric(df2[numcols[0]], errors="coerce")

    dt = pd.to_datetime(df.index, errors="coerce")
    ok = dt.notna() & vals.notna()
    dt = dt[ok]
    vals = vals[ok]
    idx = dt.to_period("M")

    ser = pd.Series(np.asarray(vals, float), index=idx, name=csv_path.stem).sort_index()
    ser = ser[(ser.index.year >= int(start_year)) & (ser.index.year <= int(end_year))]

    if force_start_january and len(ser) > 0 and int(ser.index[0].month) != 1:
        mask = (ser.index.month == 1)
        if mask.any():
            first_jan_pos = int(np.argmax(mask.to_numpy()))
            ser = ser.iloc[first_jan_pos:]

    if trim_full_years and len(ser) >= 12:
        n = len(ser) - (len(ser) % 12)
        ser = ser.iloc[:n]

    if len(ser) < 24:
        raise ValueError(f"Series {csv_path.name} too short after filtering (T={len(ser)}).")

    return ser


def transform_series_for_model(y_ser_orig: pd.Series, series: str) -> Tuple[pd.Series, float]:
    sign = _tail_sign(series)
    y_model = (sign * y_ser_orig.astype(float)).copy()
    y_model.name = f"{y_ser_orig.name}_model"
    return y_model, sign


def model_to_plot_scale(y_model: np.ndarray, sign: float) -> np.ndarray:
    """plot-scale (original units) = sign * model-scale."""
    return float(sign) * np.asarray(y_model, float)


# =============================================================================
# Scoring
# =============================================================================
def _summarize_ribbon(draws_2d: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loq = (1.0 - level) / 2.0
    hiq = 1.0 - loq
    med = np.quantile(draws_2d, 0.5, axis=0)
    lo = np.quantile(draws_2d, loq, axis=0)
    hi = np.quantile(draws_2d, hiq, axis=0)
    return med, lo, hi


def pit_from_draws(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    PIT via empirical CDF of posterior predictive draws.
    Continuous model ⇒ ties are negligible; we use mid-rank for safety.
    """
    draws = np.asarray(draws, float)
    y = np.asarray(y, float)
    S, H = draws.shape
    lt = np.sum(draws < y[None, :], axis=0)
    eq = np.sum(draws == y[None, :], axis=0)
    u = (lt + 0.5 * eq) / (S + 1.0)
    return u


def crps_from_draws(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sample CRPS per time point:
      CRPS = E|X - y| - 0.5 E|X - X'|.
    Efficient E|X - X'| via sorted-sample Gini mean difference formula.
    """
    draws = np.asarray(draws, float)
    y = np.asarray(y, float)
    S, H = draws.shape
    term1 = np.mean(np.abs(draws - y[None, :]), axis=0)

    xs = np.sort(draws, axis=0)
    i = (np.arange(1, S + 1, dtype=float)[:, None])
    w = 2.0 * i - S - 1.0
    gmd = (2.0 / (S * S)) * np.sum(w * xs, axis=0)  # E|X - X'|
    return term1 - 0.5 * gmd


# =============================================================================
# Plotting (fine-scale)
# =============================================================================
def series_color(series: str) -> str:
    s = str(series).upper()
    if s.startswith("TX"):
        return "red"
    if s.startswith("TN"):
        return "blue"
    return "C0"


def plot_cv_fine(
    *,
    series: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_future_draws: np.ndarray,  # (S,H) on plot scale
    level: float,
    xlabel: str,
    ylabel: str,
    save_path: str,
    color: str,
    window_months: int,
) -> None:
    S, H = y_future_draws.shape
    med, lo, hi = _summarize_ribbon(y_future_draws, level=level)

    # plot last window_months of training
    w = max(1, int(window_months))
    if x_train.size > w:
        x_tr = x_train[-w:]
        y_tr = y_train[-w:]
    else:
        x_tr, y_tr = x_train, y_train

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))

    # training history
    ax.plot(x_tr, y_tr, lw=1.2, alpha=0.85, color="0.35")

    # forecast ribbon + median
    ax.fill_between(x_test, lo, hi, alpha=0.20, color=color)
    ax.plot(x_test, med, lw=1.6, color=color)

    # heldout truth
    ax.plot(x_test, y_test, lw=1.2, alpha=0.95, color="0.10")

    # split marker (end of train)
    if x_train.size:
        ax.axvline(float(x_train[-1]), lw=1.0, alpha=0.85, color="0.35")

    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# One fold: fit -> forecast -> score
# =============================================================================
def fit_one_fold(
    *,
    series: str,
    y_train_model: pd.Series,  # PeriodIndex
    sign: float,
    period: int,
    priors: Priors,
    cfg: SamplerConfig,
    # initial values on MODEL scale
    alpha0_init: float,
    beta0_init: float,
    gamma0_init: Optional[np.ndarray],
    sigma_init: float,
    xi_init: float,
    s_alpha_init: float,
    s_beta_init: float,
    s_gamma_init: float,
    # knobs
    ffbs_C0_scale: float,
    ffbs_C0_A: float,
    sigma2_eff: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    y = y_train_model.to_numpy(dtype=float)
    idx = y_train_model.index  # PeriodIndex

    sampler = DGEVLaplaceNCP(
        y=y,
        period=int(period),
        alpha0=float(alpha0_init),
        beta0=float(beta0_init),
        gamma0=None if gamma0_init is None else np.asarray(gamma0_init, float),
        sigma_init=float(sigma_init),
        xi_init=float(xi_init),
        s_alpha_init=float(s_alpha_init),
        s_beta_init=float(s_beta_init),
        s_gamma_init=float(s_gamma_init),
        priors=priors,
        cfg=cfg,
        ffbs_C0_scale=float(ffbs_C0_scale),
        ffbs_C0_A=float(ffbs_C0_A),
        sigma2_eff=float(sigma2_eff),
    )

    post = sampler.run()

    # minimal meta for forecasting + bookkeeping
    meta = {
        "series": str(series),
        "period": int(period),
        "freq": "Monthly",
        "start_date": str(idx[0].to_timestamp().date()) if len(idx) else None,
        "index_type": type(idx).__name__,
        "date_tag": _date_tag_from_index(idx),
        "minima": bool(sign < 0),
        "model_sign": float(sign),
        "scale_note": "Stored states/μ etc are on MODEL scale (z = sign*y). Forecast returns PLOT scale.",
        "cfg": asdict(cfg),
        "priors": asdict(priors),
        "knobs": {"ffbs_C0_scale": float(ffbs_C0_scale), "ffbs_C0_A": float(ffbs_C0_A), "sigma2_eff": float(sigma2_eff)},
    }
    return post, meta


def forecast_from_posterior(
    *,
    post: Dict[str, Any],
    meta: Dict[str, Any],
    horizon: int,
    seed: int,
    start_date: datetime,
) -> np.ndarray:
    """
    Use the generic forecaster. It should return plot-scale y_future draws if meta indicates minima/sign.
    """
    fr = simulate_dgev_forecast(
        draws=post,
        meta=meta,
        horizon=int(horizon),
        seed=int(seed),
        start_date=start_date,
    )
    # y_future: (S,H) on PLOT scale (generic forecaster handles minima)
    return np.asarray(fr.y_future, float)


# =============================================================================
# CV driver
# =============================================================================
def run_cv_one_series(args: argparse.Namespace, series: str) -> None:
    period = int(args.period)
    K = period - 1
    sign = _tail_sign(series)

    # data
    file_map = {
        "TXx": args.txx_file,
        "TXn": args.txn_file,
        "TNx": args.tnx_file,
        "TNn": args.tnn_file,
    }
    csv_path = Path(args.data_dir) / str(file_map[series])
    y_orig = load_monthly_series(
        csv_path,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        force_start_january=True,
        trim_full_years=True,
    )
    y_model, _ = transform_series_for_model(y_orig, series)

    T_full = len(y_model)
    H = int(args.horizon)
    if T_full <= H + 12:
        raise ValueError(f"[{series}] Not enough data: T={T_full} must exceed horizon={H} by at least ~1 year.")

    # prior seasonal mean (ORIGINAL -> MODEL)
    m0_gamma_vals = _parse_csv_floats(args.prior_m0_gamma)
    if m0_gamma_vals is None:
        full = _default_seasonal_pattern_full(period)
        m0_gamma = full[:K].copy()
    else:
        m0_gamma = _to_gamma0_pminus1(m0_gamma_vals, period)
    m0_gamma = sign * m0_gamma  # MODEL scale

    # init gamma0 (ORIGINAL -> MODEL)
    gamma0_init_vals = _parse_csv_floats(args.gamma0_init)
    if gamma0_init_vals is None:
        gamma0_init = None
    else:
        gamma0_init = sign * _to_gamma0_pminus1(gamma0_init_vals, period)

    # sampler config
    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        xi_lower=float(args.prior_xi_lower),
        xi_upper=float(args.prior_xi_upper),
        m0_alpha=sign * float(args.prior_m0_alpha),
        P0_alpha=float(args.prior_P0_alpha),
        m0_beta=sign * float(args.prior_m0_beta),
        P0_beta=float(args.prior_P0_beta),
        m0_gamma=m0_gamma.tolist(),
        P0_gamma=float(args.prior_P0_gamma),
        a_lambda=float(args.prior_a_lambda),
        b_lambda=float(args.prior_b_lambda),
    )

    cfg = SamplerConfig(
        n_iter=int(args.n_iter),
        burn=int(args.burn),
        thin=int(args.thin),
        random_seed=int(args.seed),
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
    )

    # output root
    base_root = Path(args.out_dir) if args.out_dir else _series_out_root(series)
    cv_root = base_root / "CV"
    _ensure_dir(cv_root)

    color = series_color(series)
    level = float(args.level)

    # rolling-origin split indices (expanding window by default)
    # fold 0 uses latest possible split: train ends at t0 = T_full - H
    step = int(args.step_months)
    n_folds = int(args.n_folds)
    min_train = int(args.min_train_months)

    rows: List[Dict[str, Any]] = []

    for k in range(n_folds):
        t0 = T_full - H - k * step
        if t0 <= min_train:
            print(f"[{series}] fold {k}: stopping (t0={t0} <= min_train={min_train}).")
            break

        idx_train = y_model.index[:t0]
        idx_test = y_model.index[t0 : t0 + H]

        y_train_model = y_model.iloc[:t0]
        y_test_model = y_model.iloc[t0 : t0 + H]

        # deterministic fold directory (no timestamp) for reuse
        fold_tag = _safe_tag(f"fold{k:02d}_train_{idx_train[-1]}_H{H}")
        fold_dir = cv_root / fold_tag
        _ensure_dir(fold_dir)
        fig_dir = fold_dir / "figures"
        _ensure_dir(fig_dir)

        # init values (MODEL scale)
        y_train_arr = y_train_model.to_numpy(float)
        alpha0_init = float(args.alpha0_init)
        if np.isfinite(alpha0_init):
            alpha0_init = sign * alpha0_init  # ORIGINAL -> MODEL
        else:
            alpha0_init = float(np.median(y_train_arr))

        beta0_init = sign * float(args.beta0_init)  # ORIGINAL -> MODEL

        sigma_init = float(args.sigma_init)
        if (not np.isfinite(sigma_init)) or sigma_init <= 0.0:
            sigma_init = float(max(1e-3, np.std(y_train_arr, ddof=1)))

        xi_init = float(np.clip(float(args.xi_init), float(args.prior_xi_lower), float(args.prior_xi_upper)))

        # maybe reuse
        post_path = fold_dir / "posterior.npz"
        meta_path = fold_dir / "meta.json"
        if post_path.exists() and meta_path.exists() and (not args.overwrite):
            print(f"[{series}] fold {k}: reusing existing posterior at {post_path}")
            # load minimal posterior dict
            post = dict(np.load(post_path, allow_pickle=True))
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            print(f"[{series}] fold {k}: fitting on {idx_train[0]}..{idx_train[-1]} (T_train={t0})")
            t_start = time.time() if "time" in globals() else None  # avoid import just for this
            post, meta = fit_one_fold(
                series=series,
                y_train_model=y_train_model,
                sign=sign,
                period=period,
                priors=priors,
                cfg=cfg,
                alpha0_init=alpha0_init,
                beta0_init=beta0_init,
                gamma0_init=gamma0_init,
                sigma_init=sigma_init,
                xi_init=xi_init,
                s_alpha_init=float(args.s_alpha_init),
                s_beta_init=float(args.s_beta_init),
                s_gamma_init=float(args.s_gamma_init),
                ffbs_C0_scale=float(args.ffbs_C0_scale),
                ffbs_C0_A=float(args.ffbs_C0_A),
                sigma2_eff=float(args.sigma2_eff),
            )
            meta["train_end"] = str(idx_train[-1])
            meta["test_start"] = str(idx_test[0])
            meta["test_end"] = str(idx_test[-1])
            meta["horizon"] = int(H)
            meta["fold"] = int(k)
            meta["timestamp"] = datetime.now().isoformat()

            # save
            np.savez_compressed(post_path, **post)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        # forecasting (PLOT scale)
        start_dt = datetime(int(idx_train[0].year), int(idx_train[0].month), 1)
        y_future_draws_plot = forecast_from_posterior(
            post=post,
            meta=meta,
            horizon=H,
            seed=int(args.seed) + 10_000 * k + 17,
            start_date=start_dt,
        )

        # heldout truth (PLOT scale)
        y_test_plot = model_to_plot_scale(y_test_model.to_numpy(float), sign=sign)

        # scores
        med, lo, hi = _summarize_ribbon(y_future_draws_plot, level=level)
        cover = (y_test_plot >= lo) & (y_test_plot <= hi)
        pit = pit_from_draws(y_future_draws_plot, y_test_plot)
        crps = crps_from_draws(y_future_draws_plot, y_test_plot)

        # axes
        x_train = idx_train.to_timestamp().to_numpy()
        x_test = idx_test.to_timestamp().to_numpy()
        y_train_plot = model_to_plot_scale(y_train_model.to_numpy(float), sign=sign)

        # plot
        fine_png = fig_dir / f"{series}_cv_fold{k:02d}_fine.png"
        plot_cv_fine(
            series=series,
            x_train=x_train,
            y_train=y_train_plot,
            x_test=x_test,
            y_test=y_test_plot,
            y_future_draws=y_future_draws_plot,
            level=level,
            xlabel=str(args.xlabel or ""),
            ylabel=str(args.ylabel or ""),
            save_path=str(fine_png),
            color=color,
            window_months=int(args.window_months),
        )

        # save fold payload
        np.savez_compressed(
            fold_dir / f"{series}_cv_fold{k:02d}_payload.npz",
            series=str(series),
            fold=int(k),
            train_end=str(idx_train[-1]),
            test_start=str(idx_test[0]),
            test_end=str(idx_test[-1]),
            minima=bool(sign < 0),
            model_sign=float(sign),
            level=float(level),
            y_test=y_test_plot,
            y_draws=y_future_draws_plot,
            med=med,
            lo=lo,
            hi=hi,
            cover=cover.astype(np.int8),
            pit=pit,
            crps=crps,
        )

        row = {
            "series": series,
            "fold": int(k),
            "train_end": str(idx_train[-1]),
            "test_start": str(idx_test[0]),
            "test_end": str(idx_test[-1]),
            "H": int(H),
            "S": int(y_future_draws_plot.shape[0]),
            "level": float(level),
            "cover_rate": float(np.mean(cover)),
            "crps_mean": float(np.mean(crps)),
            "mae_median": float(np.mean(np.abs(med - y_test_plot))),
            "pit_mean": float(np.mean(pit)),
            "pit_var": float(np.var(pit)),
            "fold_dir": str(fold_dir),
            "fine_plot": str(fine_png),
        }
        rows.append(row)

        print(
            f"[{series}] fold {k}: cover={row['cover_rate']:.3f}  "
            f"CRPS={row['crps_mean']:.4g}  MAE(med)={row['mae_median']:.4g}  "
            f"PIT mean={row['pit_mean']:.3f}"
        )

    # write summary CSV
    df = pd.DataFrame(rows)
    out_csv = cv_root / f"{series}_cv_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{series}] wrote CV summary: {out_csv}")

    if len(df) > 0:
        agg = df[["cover_rate", "crps_mean", "mae_median"]].mean(numeric_only=True)
        print(f"[{series}] mean over folds: cover={agg['cover_rate']:.3f}, CRPS={agg['crps_mean']:.4g}, MAE={agg['mae_median']:.4g}")


# =============================================================================
# CLI
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Uccle monthly DGEV Laplace CV (rolling-origin) with fine-scale plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # data
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--start-year", type=int, default=1892)
    p.add_argument("--end-year", type=int, default=2022)
    p.add_argument("--series", choices=["TXx", "TXn", "TNx", "TNn"], default="TXn")
    p.add_argument("--all", action="store_true", default=False, help="Run TXx, TXn, TNx, TNn sequentially.")
    p.add_argument("--period", type=int, default=12)

    # files
    p.add_argument("--txx-file", type=str, default=SERIES_FILES_DEFAULT["TXx"])
    p.add_argument("--txn-file", type=str, default=SERIES_FILES_DEFAULT["TXn"])
    p.add_argument("--tnx-file", type=str, default=SERIES_FILES_DEFAULT["TNx"])
    p.add_argument("--tnn-file", type=str, default=SERIES_FILES_DEFAULT["TNn"])

    # CV controls
    p.add_argument("--horizon", type=int, default=12 * 10, help="Heldout horizon per fold (months).")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--step-months", type=int, default=12 * 5, help="Step backward between successive train endpoints.")
    p.add_argument("--min-train-months", type=int, default=12 * 30, help="Stop when training length goes below this.")
    p.add_argument("--level", type=float, default=0.90, help="Central predictive interval level for coverage.")

    # sampler initial values (interpreted on ORIGINAL scale; converted internally for minima)
    p.add_argument("--alpha0-init", type=float, default=float("nan"))
    p.add_argument("--beta0-init", type=float, default=0.0)
    p.add_argument("--gamma0-init", type=str, default=None, help="CSV length p-1 or p (ORIGINAL scale).")
    p.add_argument("--sigma-init", type=float, default=float("nan"))
    p.add_argument("--xi-init", type=float, default=-0.1)
    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init", type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-3)

    # priors
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=2.0)
    p.add_argument("--prior-xi-lower", type=float, default=-0.5)
    p.add_argument("--prior-xi-upper", type=float, default=0.5)

    p.add_argument("--prior-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-P0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m0-beta", type=float, default=0.3 / 120)
    p.add_argument("--prior-P0-beta", type=float, default=1e-6)
    p.add_argument("--prior-m0-gamma", type=str, default=None, help="CSV length p-1 or p (ORIGINAL scale).")
    p.add_argument("--prior-P0-gamma", type=float, default=5.0)

    p.add_argument("--prior-a-lambda", type=float, default=4.0)
    p.add_argument("--prior-b-lambda", type=float, default=0.0005)

    # sampler config
    p.add_argument("--n-iter", type=int, default=4000)
    p.add_argument("--burn", type=int, default=2000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", type=_str2bool, default=True)
    p.add_argument("--progress-every", type=int, default=25)

    # knobs
    p.add_argument("--ffbs-C0-scale", type=float, default=1e-6)
    p.add_argument("--ffbs-C0-A", type=float, default=1e-6)
    p.add_argument("--sigma2-eff", type=float, default=1.0)

    # output / plotting
    p.add_argument("--out-dir", type=str, default=None, help="Override output root (otherwise Uccle Laplace layout).")
    p.add_argument("--overwrite", action="store_true", default=False, help="Refit even if fold posterior already exists.")
    p.add_argument("--window-months", type=int, default=12 * 50, help="How much train history to show in fine plots.")
    p.add_argument("--xlabel", type=str, default="")
    p.add_argument("--ylabel", type=str, default="T (°C)")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    series_list = ["TXx", "TXn", "TNx", "TNn"] if args.all else [str(args.series)]
    for s in series_list:
        run_cv_one_series(args, s)


if __name__ == "__main__":
    main()
