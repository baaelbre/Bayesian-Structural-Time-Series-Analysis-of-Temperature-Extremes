# run_uccle_precx_monthly_dgev_laplace_ncp_lasso.py
# ------------------------------------------------------------
# Uccle monthly precipitation extremes (Precx, period=12) with:
#   - DGEV Laplace pseudo-observations
#   - non-centred parametrisation (NCP) + FFBS
#   - dummy seasonal state (sum-to-zero parametrisation, length p-1)
#   - hierarchical Bayesian lasso prior on signed process SDs
#
# Example:
#   python -u run_uccle_precx_monthly_dgev_laplace_ncp_lasso.py --n-iter 20000 --burn 10000 --thin 2
#
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Robust import (works whether launched from project root or from scripts/)
# ---------------------------------------------------------------------
try:
    from optimization.dgev_laplace_2 import DGEVLaplaceNCP, Priors, SamplerConfig  # type: ignore
except Exception:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from optimization.dgev_laplace_2 import DGEVLaplaceNCP, Priors, SamplerConfig  # type: ignore


# =============================================================================
# Series config
# =============================================================================
SERIES_FILES_DEFAULT: Dict[str, str] = {"Precx": "Precx.csv"}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _series_out_root(series: str) -> Path:
    """
    Output root for *monthly* precipitation extremes:

      Precx → results/uccle/Prec/Precx/Monthly/Laplace
    """
    base = Path("results") / "uccle"
    mapping = {"Precx": base / "Prec" / "Precx" / "Monthly" / "Laplace"}
    if series not in mapping:
        raise ValueError(f"Unknown series '{series}' for output mapping.")
    return mapping[series]


def _date_tag_from_index(idx: pd.Index) -> str:
    if len(idx) == 0:
        return "NA-NA"
    if isinstance(idx, pd.PeriodIndex):
        return f"{idx[0]}-{idx[-1]}"
    if isinstance(idx, pd.DatetimeIndex):
        return f"{idx[0].strftime('%Y-%m-%d')}-{idx[-1].strftime('%Y-%m-%d')}"
    return f"{str(idx[0])}-{str(idx[-1])}"


def _start_end_from_index(idx: pd.Index) -> Tuple[str, str]:
    if len(idx) == 0:
        return ("NA", "NA")
    if isinstance(idx, pd.PeriodIndex):
        return (str(idx[0]), str(idx[-1]))
    if isinstance(idx, pd.DatetimeIndex):
        return (idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d"))
    return (str(idx[0]), str(idx[-1]))


# =============================================================================
# CLI parsing helpers
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
def _to_gamma0_pminus1(vals: List[float], period: int) -> np.ndarray:
    """
    Convert a user-provided seasonal pattern to gamma0 of length (p-1)
    for the sum-to-zero parametrisation:
      seasonal_full = [gamma0...,  -sum(gamma0)]

    Accepted input lengths:
      - p-1: interpreted directly as gamma0
      - p  : mean-center to sum-to-zero, then take first p-1 as gamma0
    """
    p = int(period)
    arr = np.asarray(vals, float).ravel()
    if arr.size == p - 1:
        return arr.copy()
    if arr.size == p:
        arr = arr - arr.mean()
        arr = arr - (arr.sum() / p)  # enforce exact sum-to-zero numerically
        return arr[: p - 1].copy()
    raise ValueError(f"Season pattern must have length {p-1} or {p}, got {arr.size}.")


def _month_ids(idx: pd.Index, period: int) -> np.ndarray:
    """
    Month IDs in {0,...,period-1}, aligned with index.
    """
    n = len(idx)
    if isinstance(idx, pd.PeriodIndex):
        return (idx.month - 1).astype(int)
    if isinstance(idx, pd.DatetimeIndex):
        return (idx.month - 1).astype(int)
    return (np.arange(n) % int(period)).astype(int)


def _gamma0_init_from_monthly_means(y: np.ndarray, idx: pd.Index, period: int) -> np.ndarray:
    """
    Data-driven init for gamma0 (length p-1):
      - compute mean per month
      - center across months (sum-to-zero)
      - take first p-1 entries (last month implicit)
    """
    p = int(period)
    mid = _month_ids(idx, p)
    eff = np.zeros(p, float)
    overall = float(np.mean(y)) if y.size else 0.0
    for m in range(p):
        mask = (mid == m)
        eff[m] = float(np.mean(y[mask])) if np.any(mask) else overall
    eff = eff - float(np.mean(eff))
    eff = eff - (eff.sum() / p)
    return eff[: p - 1].copy()


# =============================================================================
# Data loading
# =============================================================================
def load_precx_monthly(
    csv_path: Path,
    *,
    start_year: int,
    end_year: int,
    force_start_january: bool = True,
    trim_full_years: bool = True,
) -> pd.Series:
    """
    Load MONTHLY precipitation maxima (Precx) from CSV into a Series with PeriodIndex('M').

    Robust to:
      - date column named 'date'/'Date'/'time'/... OR date stored in first column
      - 'value' column OR first numeric column
    """
    df = pd.read_csv(csv_path)

    # ---- find date-like source ----
    date_col = next(
        (c for c in ["date", "Date", "time", "Time", "datetime", "Datetime", "DATE", "TIME"] if c in df.columns),
        None,
    )
    if date_col is not None:
        dt_raw = df[date_col]
        non_date_cols = [c for c in df.columns if c != date_col]
    else:
        # common in your repo: first column is date-like
        dt_raw = df.iloc[:, 0]
        non_date_cols = list(df.columns[1:])

    # ---- parse datetimes (force DatetimeIndex) ----
    dt = pd.to_datetime(dt_raw, errors="coerce")
    dt_idx = pd.DatetimeIndex(dt)  # <- key line: guarantees DatetimeIndex

    # ---- pick values ----
    if "value" in df.columns:
        vals = pd.to_numeric(df["value"], errors="coerce")
    else:
        # pick first numeric among non-date columns
        tmp = df[non_date_cols].copy() if non_date_cols else df.copy()
        for c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        numcols = [c for c in tmp.columns if pd.api.types.is_numeric_dtype(tmp[c])]
        if not numcols:
            raise ValueError(f"No numeric column found in {csv_path}")
        vals = pd.to_numeric(tmp[numcols[0]], errors="coerce")

    # ---- drop missing ----
    ok = dt_idx.notna() & vals.notna().to_numpy()
    dt_idx = dt_idx[ok]
    vals = np.asarray(vals.to_numpy(dtype=float)[ok], float)

    # ---- monthly PeriodIndex ----
    idx = dt_idx.to_period("M")
    ser = pd.Series(vals, index=idx, name="Precx").sort_index()

    # ---- year filter ----
    ser = ser[(ser.index.year >= int(start_year)) & (ser.index.year <= int(end_year))]

    # ---- force start on January (optional) ----
    if force_start_january and len(ser) > 0 and int(ser.index[0].month) != 1:
        mask = (ser.index.month == 1)
        if mask.any():
            first_jan_pos = int(np.argmax(mask.to_numpy()))
            ser = ser.iloc[first_jan_pos:]

    # ---- trim to full years (multiple of 12) ----
    if trim_full_years and len(ser) >= 12:
        n = len(ser) - (len(ser) % 12)
        ser = ser.iloc[:n]

    if len(ser) < 20:
        raise ValueError(f"Series {csv_path.name} too short after filtering (T={len(ser)}).")

    return ser


# =============================================================================
# Core runner
# =============================================================================
def run_one(
    *,
    y_ser: pd.Series,
    out_root: Path,
    period: int,
    priors: Priors,
    cfg: SamplerConfig,
    # initial values
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
    # plotting
    plot: bool,
) -> None:
    series = "Precx"
    y = y_ser.to_numpy(dtype=float)
    idx = y_ser.index

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

    date_tag = _date_tag_from_index(idx)
    start_date, end_date = _start_end_from_index(idx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes_tag = "dynamic_dynamic_dynamic"
    tag = f"{series}_{modes_tag}"

    outdir = out_root / f"{tag}_{date_tag}_{timestamp}"
    figdir = outdir / "figures"
    _ensure_dir(outdir)
    _ensure_dir(figdir)

    print(f"[{series}] Running Uccle monthly DGEV Laplace (NCP + lasso) ...")
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"[{series}] run time {elapsed:.2f} seconds")

    # Save posterior
    out_npz = outdir / f"posterior_{series}_{date_tag}_{modes_tag}.npz"
    sampler.save_posterior(
        out_npz_path=str(out_npz),
        extra_meta={
            "series": series,
            "label": f"{series}_monthly",
            "period": int(period),
            "date_tag": date_tag,
            "start_date": start_date,
            "end_date": end_date,
            "T": int(len(y)),
            "elapsed_seconds": float(elapsed),
            "timestamp": datetime.now().isoformat(),
            "index_type": type(idx).__name__,
            "index_values_preview": [str(ix) for ix in idx[: min(10, len(idx))]],
            "model": "DGEV_LAPLACE_NCP_LASSO_DUMMIES",
            "data_transform": "identity",
            "model_sign": 1.0,
            "scale_note": "All stored states/μ/etc. are on the MODEL scale (here: y).",
            "cfg": asdict(cfg),
            "priors": asdict(priors),
            "knobs": {
                "ffbs_C0_scale": float(ffbs_C0_scale),
                "ffbs_C0_A": float(ffbs_C0_A),
                "sigma2_eff": float(sigma2_eff),
            },
        },
    )

    # Quick summaries
    if isinstance(post, dict):
        if "sigma" in post:
            print(f"Posterior mean sigma: {float(np.mean(post['sigma'])):.4f}")
        if "xi" in post:
            print(f"Posterior mean xi:    {float(np.mean(post['xi'])):.4f}")
        for k in ["alpha", "beta", "gamma"]:
            kk = f"Q_{k}"
            if kk in post:
                mQ = float(np.mean(post[kk]))
                print(f"Posterior mean {kk}: {mQ:.4g} (sqrt≈{math.sqrt(max(mQ, 0.0)):.4g})")
        if "lambda2" in post:
            print(f"Posterior mean lambda2: {float(np.mean(post['lambda2'])):.4g}")

    print(f"Saved posterior to {out_npz}")

    # Optional quick fit plot (MODEL scale)
    if plot and isinstance(post, dict) and ("mu" in post):
        mu_draws = post["mu"]
        mu_hat = mu_draws.mean(axis=0)
        lo, hi = np.quantile(mu_draws, [0.05, 0.95], axis=0)

        x = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else idx

        plt.figure(figsize=(12, 4))
        plt.plot(x, y, lw=1, label=f"{series}")
        plt.plot(x, mu_hat, "-.", lw=1.5, label="μ̂_t (post mean)")
        plt.fill_between(x, lo, hi, alpha=0.2, label="90% CI (μ_t)")
        plt.title(f"{series}: monthly DGEV Laplace (NCP + Bayesian lasso) — period={period}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fit_path = figdir / f"fit_mu_{series}_{date_tag}_{modes_tag}.png"
        plt.savefig(fit_path, dpi=180)
        plt.close()
        print(f"Saved fit plot to {fit_path}\n")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Uccle Precx MONTHLY: DGEV Laplace + FFBS in NCP with monthly seasonal dummies "
            "and hierarchical Bayesian lasso prior on signed process SDs."
        )
    )

    # --- data ---
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--precx-file", type=str, default=SERIES_FILES_DEFAULT["Precx"])
    p.add_argument("--start-year", type=int, default=1892)
    p.add_argument("--end-year", type=int, default=2022)
    p.add_argument("--period", type=int, default=12)

    # --- initial values ---
    p.add_argument("--alpha0-init", type=float, default=float("nan"))  # nan -> median(data)
    p.add_argument("--beta0-init", type=float, default=0.0)
    p.add_argument("--gamma0-init", type=str, default=None, help="CSV length p-1 or p (sum-to-zero handled).")
    p.add_argument("--sigma-init", type=float, default=float("nan"))   # nan/<=0 -> std(data)
    p.add_argument("--xi-init", type=float, default=0.1)

    p.add_argument("--s-alpha-init", type=float, default=1e-2)
    p.add_argument("--s-beta-init", type=float, default=1e-3)
    p.add_argument("--s-gamma-init", type=float, default=1e-3)

    # --- priors (obs) ---
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=2.0)
    p.add_argument("--prior-xi-lower", type=float, default=-0.5)
    p.add_argument("--prior-xi-upper", type=float, default=0.5)

    # --- priors (baselines) ---
    p.add_argument("--prior-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-P0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-P0-beta", type=float, default=10.0)
    p.add_argument(
        "--prior-m0-gamma",
        type=str,
        default=None,
        help="CSV length p-1 (gamma0) or p (full seasonal pattern). If omitted: zeros.",
    )
    p.add_argument("--prior-P0-gamma", type=float, default=5.0)

    # --- priors (lasso hyperprior on lambda^2) ---
    p.add_argument("--prior-a-lambda", type=float, default=4.0)
    p.add_argument("--prior-b-lambda", type=float, default=0.0005)

    # --- sampler config ---
    p.add_argument("--n-iter", type=int, default=20000)
    p.add_argument("--burn", type=int, default=5000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", type=_str2bool, default=True)
    p.add_argument("--progress-every", type=int, default=10, help="0=auto (~2%).")

    # --- knobs ---
    p.add_argument("--ffbs-C0-scale", type=float, default=1e-6)
    p.add_argument("--ffbs-C0-A", type=float, default=1e-6)
    p.add_argument("--sigma2-eff", type=float, default=1.0)

    # --- output / plotting ---
    p.add_argument("--out-dir", type=str, default=None, help="Override output run directory root (folder).")
    p.add_argument("--plot", type=_str2bool, default=True)

    args = p.parse_args()
    np.random.seed(int(args.seed))

    series = "Precx"
    period = int(args.period)
    K = period - 1

    # ---- load data ----
    csv_path = Path(args.data_dir) / str(args.precx_file)
    y_ser = load_precx_monthly(
        csv_path,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        force_start_january=True,
        trim_full_years=True,
    )
    y = y_ser.to_numpy(dtype=float)
    idx = y_ser.index

    if len(y) < 24:
        print(f"[warn] T={len(y)} is short for monthly DGEV. Expect wide posteriors.")

    # ---- gamma prior mean (length p-1) ----
    m0_gamma_vals = _parse_csv_floats(args.prior_m0_gamma)
    if m0_gamma_vals is None:
        m0_gamma = np.zeros(K, float)
    else:
        m0_gamma = _to_gamma0_pminus1(m0_gamma_vals, period)

    # ---- init gamma0 ----
    gamma0_init_vals = _parse_csv_floats(args.gamma0_init)
    if gamma0_init_vals is None:
        gamma0_init = _gamma0_init_from_monthly_means(y, idx, period)
    else:
        gamma0_init = _to_gamma0_pminus1(gamma0_init_vals, period)

    # ---- data-driven alpha0/sigma init ----
    alpha0_init = float(args.alpha0_init)
    if not np.isfinite(alpha0_init):
        alpha0_init = float(np.median(y))
    beta0_init = float(args.beta0_init)

    sigma_init = float(args.sigma_init)
    if (not np.isfinite(sigma_init)) or sigma_init <= 0.0:
        sigma_init = float(max(1e-3, np.std(y, ddof=1)))

    # ---- priors + cfg ----
    xi_lb = float(args.prior_xi_lower)
    xi_ub = float(args.prior_xi_upper)
    xi_init = float(np.clip(float(args.xi_init), xi_lb, xi_ub))

    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        xi_lower=xi_lb,
        xi_upper=xi_ub,
        m0_alpha=float(args.prior_m0_alpha),
        P0_alpha=float(args.prior_P0_alpha),
        m0_beta=float(args.prior_m0_beta),
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

    # ---- output root ----
    out_root = Path(args.out_dir) if args.out_dir is not None else _series_out_root(series)
    _ensure_dir(out_root)

    # ---- run ----
    run_one(
        y_ser=y_ser,
        out_root=out_root,
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
        plot=bool(args.plot),
    )


if __name__ == "__main__":
    main()
