# run_uccle_precx_monthly_dgev_laplace_ncp_lasso.py
# Monthly precipitation extremes (Precx, period=12) using:
#   - DGEV Laplace pseudo-observations
#   - non-centred parametrisation (NCP) + FFBS
#   - dummy seasonal state (newest-first rotation)
#   - hierarchical Bayesian lasso prior on signed process SDs
#
# Example:
#   python -u run_uccle_precx_monthly_dgev_laplace_ncp_lasso.py --n-iter 20000 --burn 10000 --thin 2

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Import from project root
# ---------------------------------------------------------------------
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from optimization.dgev_laplace import (
    DGEVLaplaceNCP,
    Priors,
    SamplerConfig,
)

DATA_DIR = Path("data")


# ======================================================================
# Small helpers
# ======================================================================
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _series_out_root(series: str) -> Path:
    """
    Output root for *monthly* precipitation extremes:

      Precx → results/uccle/Prec/Precx/Monthly/DGEV_NCP_LASSO_LAPLACE
    """
    base = Path("results/uccle")
    mapping = {
        "Precx": base / "Prec" / "Precx" / "Monthly" / "DGEV_NCP_LASSO_LAPLACE",
    }
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


def _to_monthly_index(idx: pd.Index) -> pd.Index:
    if isinstance(idx, pd.PeriodIndex):
        return idx.asfreq("M", how="end")
    if isinstance(idx, pd.DatetimeIndex):
        return idx.to_period("M")
    return idx


def _month_id_from_index(idx: pd.Index, period: int) -> np.ndarray:
    """
    Month IDs in {0,...,period-1}, aligned with index.
    """
    n = len(idx)
    if isinstance(idx, pd.PeriodIndex) and idx.freqstr and "M" in idx.freqstr:
        return (idx.month - 1).astype(int)
    if isinstance(idx, pd.DatetimeIndex):
        return (idx.month - 1).astype(int)
    return (np.arange(n) % period).astype(int)


def _gamma0_init_from_monthly_means(y: np.ndarray, idx: pd.Index, period: int) -> np.ndarray:
    """
    Data-driven init for gamma0 (length period-1, newest-first convention handled by sampler):
      - compute mean per month
      - center across months (sum-to-zero)
      - take first period-1 entries (last month implicit)
    """
    mid = _month_id_from_index(idx, period)
    eff = np.zeros(period, float)
    for m in range(period):
        mask = (mid == m)
        eff[m] = float(np.mean(y[mask])) if np.any(mask) else float(np.mean(y))
    eff = eff - float(np.mean(eff))
    return eff[: period - 1].copy()


def _jsonify(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.generic,)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


# ======================================================================
# Data loading
# ======================================================================
def load_precx_monthly(
    csv_path: Path,
    start_year: int = 1892,
    end_year: int = 2022,
) -> pd.Series:
    """
    Load MONTHLY precipitation maxima (Precx) from CSV and trim to full years (multiple of 12).

    Flexible formats:
      - 'value' column or first numeric column
      - index from 'date'/'time'/'Date'/'Time' OR first column parseable as datetime
      - will filter by year if index has year
    """
    df = pd.read_csv(csv_path)

    # values
    if "value" in df.columns:
        vals = pd.to_numeric(df["value"], errors="coerce")
    else:
        for c in df.columns[1:]:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            raise ValueError(f"No numeric columns found in {csv_path}")
        vals = df[numcols[0]]

    # index
    idx = None
    for cand in ["date", "time", "Date", "Time", "year", "Year"]:
        if cand in df.columns:
            idx = pd.to_datetime(df[cand], errors="coerce")
            break
    if idx is None:
        idx = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    ser = pd.Series(vals.to_numpy(dtype=float), index=idx, name="Precx").dropna()
    ser = ser.sort_index()

    # filter years
    if hasattr(ser.index, "year"):
        mask = (ser.index.year >= start_year) & (ser.index.year <= end_year)
        ser = ser.loc[mask]

    # trim to full years
    n = len(ser) - (len(ser) % 12)
    if n <= 0:
        raise ValueError(f"Series is shorter than 12 observations after filtering: {csv_path}")
    ser = ser.iloc[:n]

    ser.index = _to_monthly_index(ser.index)
    return ser


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Uccle Precx monthly: DGEV Laplace + NCP FFBS + dummy seasonality "
            "+ hierarchical Bayesian lasso prior on process SDs."
        )
    )

    # data
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--precx-file", type=str, default="Precx.csv")
    p.add_argument("--start-year", type=int, default=1892)
    p.add_argument("--end-year", type=int, default=2022)

    # model
    p.add_argument("--period", type=int, default=12)

    # sampler
    p.add_argument("--n-iter", type=int, default=20000)
    p.add_argument("--burn", type=int, default=10000)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", type=bool, default=True)
    p.add_argument("--progress-every", type=int, default=10)

    # priors
    p.add_argument("--prior-a-sigma", type=float, default=2.0)
    p.add_argument("--prior-b-sigma", type=float, default=2.0)
    p.add_argument("--prior-xi-lower", type=float, default=-0.5)
    p.add_argument("--prior-xi-upper", type=float, default=0.5)

    p.add_argument("--prior-m0-alpha", type=float, default=0.0)
    p.add_argument("--prior-P0-alpha", type=float, default=10.0)
    p.add_argument("--prior-m0-beta", type=float, default=0.0)
    p.add_argument("--prior-P0-beta", type=float, default=10.0)
    p.add_argument("--prior-m0-gamma", type=str, default=None)  # comma list length 11
    p.add_argument("--prior-P0-gamma", type=float, default=5.0)

    p.add_argument("--prior-a-lambda", type=float, default=0.001)
    p.add_argument("--prior-b-lambda", type=float, default=0.001)

    # initial values
    p.add_argument("--alpha0-init", type=float, default=None)
    p.add_argument("--beta0-init", type=float, default=0.0)
    p.add_argument("--gamma0-init", type=str, default=None)  # comma list length 11

    p.add_argument("--sigma-init", type=float, default=None)
    p.add_argument("--xi-init", type=float, default=0.1)

    p.add_argument("--s-alpha-init", type=float, default=None)
    p.add_argument("--s-beta-init", type=float, default=None)
    p.add_argument("--s-gamma-init", type=float, default=None)

    # output
    p.add_argument("--out-dir", type=str, default=None)

    args = p.parse_args()
    np.random.seed(args.seed)

    series = "Precx"
    period = int(args.period)

    # ---- load data ----
    csv_path = Path(args.data_dir) / args.precx_file
    s = load_precx_monthly(csv_path, start_year=args.start_year, end_year=args.end_year)
    y = s.to_numpy(dtype=float)
    idx = s.index

    if len(y) < 24:
        print(f"[warn] T={len(y)} is pretty short for monthly DGEV. Expect wide posteriors.")

    # ---- priors ----
    if args.prior_m0_gamma is not None and str(args.prior_m0_gamma).strip():
        m0_gamma = [float(z) for z in str(args.prior_m0_gamma).split(",")]
    else:
        m0_gamma = [0.0] * (period - 1)

    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        xi_lower=float(args.prior_xi_lower),
        xi_upper=float(args.prior_xi_upper),
        m0_alpha=float(args.prior_m0_alpha),
        P0_alpha=float(args.prior_P0_alpha),
        m0_beta=float(args.prior_m0_beta),
        P0_beta=float(args.prior_P0_beta),
        m0_gamma=m0_gamma,
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

    # ---- initial values ----
    y_sd = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
    alpha0_init = float(np.mean(y[:12])) if args.alpha0_init is None else float(args.alpha0_init)
    beta0_init = float(args.beta0_init)

    if args.gamma0_init is not None and str(args.gamma0_init).strip():
        gamma0_init = [float(z) for z in str(args.gamma0_init).split(",")]
    else:
        gamma0_init = _gamma0_init_from_monthly_means(y, idx, period)

    if args.sigma_init is None:
        sigma_init = float(max(1e-3, 0.5 * y_sd))
    else:
        sigma_init = float(args.sigma_init)

    xi_init = float(np.clip(float(args.xi_init), priors.xi_lower, priors.xi_upper))

    s_alpha_init = float(0.05 * y_sd) if args.s_alpha_init is None else float(args.s_alpha_init)
    s_beta_init = float(0.01 * y_sd) if args.s_beta_init is None else float(args.s_beta_init)
    s_gamma_init = float(0.05 * y_sd) if args.s_gamma_init is None else float(args.s_gamma_init)

    # ---- sampler ----
    sampler = DGEVLaplaceNCP(
        y=y,
        period=period,
        alpha0=alpha0_init,
        beta0=beta0_init,
        gamma0=gamma0_init,
        sigma_init=sigma_init,
        xi_init=xi_init,
        s_alpha_init=s_alpha_init,
        s_beta_init=s_beta_init,
        s_gamma_init=s_gamma_init,
        priors=priors,
        cfg=cfg,
    )

    # ---- outputs ----
    date_tag = _date_tag_from_index(idx)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes_tag = "dynamic-dynamic-dynamic"
    tag = f"{series}_{modes_tag}"

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = _series_out_root(series) / f"{tag}_{date_tag}_{timestamp}"

    _ensure_dir(out_dir)

    # ---- run ----
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"[done] elapsed {elapsed:.1f}s")

    # ---- save ----
    out_npz = out_dir / "posterior.npz"
    sampler.save_posterior(
        out_npz_path=str(out_npz),
        extra_meta={
            "series": "Precx_monthly",
            "T": int(len(y)),
            "period": int(period),
            "years": {"start": int(args.start_year), "end": int(args.end_year)},
            "index_type": type(idx).__name__,
            "index_values_preview": [str(ix) for ix in idx[: min(10, len(idx))]],
            "modes": {"level_mode": "dynamic", "trend_mode": "dynamic", "seasonal_mode": "dynamic"},
            "cfg": _jsonify(asdict(cfg)),
            "priors": _jsonify(asdict(priors)),
            "elapsed_seconds": float(elapsed),
            "timestamp": datetime.now().isoformat(),
            "description": (
                "Structural DGEV with dummy seasonal state (newest-first), "
                "Laplace pseudo-observations, NCP-FFBS latent update, "
                "hierarchical Bayesian lasso prior on signed process SDs."
            ),
        },
    )

    # quick summaries
    if post.get("sigma", np.array([])).size:
        print(f"Posterior mean sigma: {np.mean(post['sigma']):.4f}")
    if post.get("xi", np.array([])).size:
        print(f"Posterior mean xi:    {np.mean(post['xi']):.4f}")
    if post.get("loglike", np.array([])).size:
        le = post["loglike"]
        print(f"loglike: mean={np.nanmean(le):.2f} median={np.nanmedian(le):.2f} best={np.nanmax(le):.2f}")

    print("Saved results under:")
    print(f"  {out_dir}/*")
