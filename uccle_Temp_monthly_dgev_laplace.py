# uccle_Temp_monthly_dgev_dummies_laplace_ncp_lasso.py
# for s in TXx TXn TNx TNn; do python -u uccle_Temp_monthly_dgev_dummies_laplace_ncp_lasso.py --series "$s"; done

from __future__ import annotations

import os
import sys
import time
import json
import math
from dataclasses import asdict
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root on path
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(THIS_DIR, "..")
sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Import NCP Laplace Bayesian-lasso DGEV with seasonal dummies
# ---------------------------------------------------------------------------

from optimization.dgev_laplace import (  # type: ignore
    DGEVLaplaceNCP,
    Priors,
    SamplerConfig,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    return v in {"true", "1", "yes", "y"}


def _parse_csv_floats(val: str | None) -> Optional[List[float]]:
    """Parse '1,2,3' → [1.0, 2.0, 3.0] (or None if empty)."""
    if val is None:
        return None
    val = str(val).strip()
    if not val:
        return None
    return [float(tok) for tok in val.split(",") if tok.strip()]


def _jsonify_dict(d: dict) -> dict:
    """Convert numpy types to JSON-safe Python objects."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _series_out_root(series: str) -> str:
    """
    Map series name to required output root:

      TXx → results/uccle/TX/TXx/Monthly/Laplace/
      TXn → results/uccle/TX/TXn/Monthly/Laplace/
      TNx → results/uccle/TN/TNx/Monthly/Laplace/
      TNn → results/uccle/TN/TNn/Monthly/Laplace/
    """
    base = "results/uccle"
    mapping = {
        "TXx": os.path.join(base, "TX", "TXx", "Monthly", "Laplace"),
        "TXn": os.path.join(base, "TX", "TXn", "Monthly", "Laplace"),
        "TNx": os.path.join(base, "TN", "TNx", "Monthly", "Laplace"),
        "TNn": os.path.join(base, "TN", "TNn", "Monthly", "Laplace"),
    }
    if series not in mapping:
        raise ValueError(f"Unknown series '{series}' for output mapping.")
    return mapping[series]


def _default_seasonal_pattern_full(period: int) -> np.ndarray:
    """
    Default smooth seasonal pattern (length=period) with sum-to-zero.
    """
    g = np.cos(2.0 * np.pi * np.arange(period) / period)
    g = g - g.mean()  # sum-to-zero (up to fp)
    # enforce exact sum-to-zero numerically
    g = g - (g.sum() / period)
    return g.astype(float)


def _to_gamma0_pminus1(vals: List[float], period: int) -> np.ndarray:
    """
    Convert a user-provided seasonal pattern to gamma0 of length (period-1)
    for the sum-to-zero parametrisation:
        seasonal_full = [gamma0...,  -sum(gamma0)]
    Accepted input lengths:
      - period-1: interpreted directly as gamma0
      - period: mean-center to sum-to-zero, then take first period-1 as gamma0
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


# ---------------------------------------------------------------------------
# Data loading (monthly extremes)
# ---------------------------------------------------------------------------
def load_monthlies(
    start_year: int = 1892,
    end_year: int = 2022,
    data_dir: str = "data",
    txx_file: str = "TXx.csv",
    txn_file: str = "TXn.csv",
    tnx_file: str = "TNx.csv",
    tnn_file: str = "TNn.csv",
) -> Dict[str, pd.Series]:
    """
    Load four *monthly* CSVs with a monthly PeriodIndex:

      TXx: monthly max of daily TX
      TXn: monthly min of daily TX
      TNx: monthly max of daily TN
      TNn: monthly min of daily TN

    Align on the common monthly index and trim to [start_year, end_year].
    """

    def read_one(path: str) -> pd.Series:
        df = pd.read_csv(path, index_col=0)

        # Select numeric data column
        if df.shape[1] == 1:
            col = df.columns[0]
        else:
            col = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None)
        if col is None:
            raise ValueError(f"No numeric column found in {path}.")

        ts = pd.to_datetime(df.index)
        s = pd.Series(df[col].to_numpy(dtype=float), index=ts.to_period("M")).sort_index()
        s = s[(s.index.year >= start_year) & (s.index.year <= end_year)]
        return s

    series_dict = {
        "TXx": read_one(os.path.join(data_dir, txx_file)),
        "TXn": read_one(os.path.join(data_dir, txn_file)),
        "TNx": read_one(os.path.join(data_dir, tnx_file)),
        "TNn": read_one(os.path.join(data_dir, tnn_file)),
    }

    # Align to common PeriodIndex
    common_idx = None
    for s in series_dict.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    for k in series_dict:
        series_dict[k] = series_dict[k].reindex(common_idx).sort_index()

    return series_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description=(
            "Uccle MONTHLY TX/TN extremes: DGEV Laplace + FFBS in NCP with seasonal DUMMIES "
            "and hierarchical Bayesian lasso prior on process SDs."
        )
    )

    # ---------------- Data & selection ----------------
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--txx-file", type=str, default="TXx.csv")
    parser.add_argument("--txn-file", type=str, default="TXn.csv")
    parser.add_argument("--tnx-file", type=str, default="TNx.csv")
    parser.add_argument("--tnn-file", type=str, default="TNn.csv")
    parser.add_argument("--start-year", type=int, default=1892)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--series", choices=["TXx", "TXn", "TNx", "TNn"], default="TNx")
    parser.add_argument("--period", type=int, default=12, help="Seasonal period (12 for Jan–Dec).")

    # ---------------- Initial values ----------------
    parser.add_argument("--alpha0-init", type=float, default=0.0)
    parser.add_argument("--beta0-init", type=float, default=0.0)
    parser.add_argument("--gamma0-init", type=str, default=None, help="CSV, length p-1 or p.")
    parser.add_argument("--sigma-init", type=float, default=1.0)
    parser.add_argument("--xi-init", type=float, default=-0.1)
    parser.add_argument("--s-alpha-init", type=float, default=1e-2)
    parser.add_argument("--s-beta-init", type=float, default=1e-3)
    parser.add_argument("--s-gamma-init", type=float, default=1e-3)

    # ---------------- Priors: obs ----------------
    parser.add_argument("--prior-a-sigma", type=float, default=2.0)
    parser.add_argument("--prior-b-sigma", type=float, default=2.0)
    parser.add_argument("--prior-xi-lower", type=float, default=-0.5)
    parser.add_argument("--prior-xi-upper", type=float, default=0.5)

    # ---------------- Priors: baselines ----------------
    parser.add_argument("--prior-m0-alpha", type=float, default=0.0)
    parser.add_argument("--prior-P0-alpha", type=float, default=10.0)
    parser.add_argument("--prior-m0-beta", type=float, default=0.0)
    parser.add_argument("--prior-P0-beta", type=float, default=10.0)

    parser.add_argument(
        "--prior-m0-gamma",
        type=str,
        default=None,
        help="CSV dummy pattern mean. Length p-1 (gamma0 directly) or p (full seasonal pattern).",
    )
    parser.add_argument("--prior-P0-gamma", type=float, default=5.0)

    # ---------------- Priors: hierarchical Bayesian lasso ----------------
    parser.add_argument("--prior-a-lambda", type=float, default=0.001)
    parser.add_argument("--prior-b-lambda", type=float, default=0.001)

    # ---------------- Sampler config ----------------
    parser.add_argument("--n-iter", type=int, default=8000)
    parser.add_argument("--burn", type=int, default=4000)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", type=_str2bool, default=True)
    parser.add_argument("--progress-every", type=int, default=0, help="0=auto (~2%).")

    # ---------------- Output ----------------
    parser.add_argument("--out-dir", type=str, default=None, help="Override output run directory.")
    parser.add_argument("--plot", type=_str2bool, default=True)

    args = parser.parse_args()
    np.random.seed(args.seed)

    # ---------------- Load data ----------------
    all_series = load_monthlies(
        start_year=args.start_year,
        end_year=args.end_year,
        data_dir=args.data_dir,
        txx_file=args.txx_file,
        txn_file=args.txn_file,
        tnx_file=args.tnx_file,
        tnn_file=args.tnn_file,
    )
    s = all_series[args.series].dropna().sort_index()
    y = s.to_numpy(dtype=float)
    idx = s.index  # PeriodIndex("M")
    T = int(y.size)
    if T < 20:
        raise ValueError(f"Not enough observations after filtering; got T={T}.")

    # ---------------- Seasonal priors/initial gamma0 ----------------
    p = int(args.period)

    # prior mean for gamma0
    m0_gamma_vals = _parse_csv_floats(args.prior_m0_gamma)
    if m0_gamma_vals is None:
        # default smooth seasonal pattern (sum-to-zero), then convert to gamma0 (p-1)
        full = _default_seasonal_pattern_full(p)
        m0_gamma = full[: p - 1].copy()
    else:
        m0_gamma = _to_gamma0_pminus1(m0_gamma_vals, p)

    # initial gamma0
    gamma0_init_vals = _parse_csv_floats(args.gamma0_init)
    if gamma0_init_vals is None:
        gamma0_init = None
    else:
        gamma0_init = _to_gamma0_pminus1(gamma0_init_vals, p)

    # data-driven init for alpha0 if left at 0
    alpha0_init = float(args.alpha0_init)
    beta0_init = float(args.beta0_init)
    if abs(alpha0_init) < 1e-12:
        alpha0_init = float(np.median(y))

    # sigma init if left at 1
    sigma_init = float(args.sigma_init)
    if sigma_init <= 0 or abs(sigma_init - 1.0) < 1e-12:
        sigma_init = float(max(1e-3, np.std(y, ddof=1)))

    # xi init clipped into prior support
    xi_lb = float(args.prior_xi_lower)
    xi_ub = float(args.prior_xi_upper)
    xi_init = float(np.clip(float(args.xi_init), xi_lb, xi_ub))

    # ---------------- Priors + cfg ----------------
    priors = Priors(
        a_sigma=float(args.prior_a_sigma),
        b_sigma=float(args.prior_b_sigma),
        xi_lower=float(args.prior_xi_lower),
        xi_upper=float(args.prior_xi_upper),
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

    # ---------------- Construct sampler ----------------
    sampler = DGEVLaplaceNCP(
        y=y,
        period=p,
        alpha0=alpha0_init,
        beta0=beta0_init,
        gamma0=gamma0_init,
        sigma_init=sigma_init,
        xi_init=xi_init,
        s_alpha_init=float(args.s_alpha_init),
        s_beta_init=float(args.s_beta_init),
        s_gamma_init=float(args.s_gamma_init),
        priors=priors,
        cfg=cfg,
    )

    # ---------------- Output directories ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modes_tag = "dynamic_dynamic_dynamic"  # this sampler is NCP + RW/RW/seasonal-rotation
    tag = f"{args.series}_{modes_tag}_{args.start_year}-{args.end_year}"

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        root = _series_out_root(args.series)
        out_dir = os.path.join(root, f"{tag}_{timestamp}")

    fig_dir = os.path.join(out_dir, "figures")
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    # ---------------- Run sampler ----------------
    t0 = time.time()
    post = sampler.run()
    elapsed = time.time() - t0
    print(f"\n[Run completed in {elapsed:.1f}s]")

    # ---------------- Save posterior + metadata ----------------
    meta = {
        "series": args.series,
        "T": int(T),
        "period": int(p),
        "index_type": "PeriodIndex('M')",
        "years": {"start": int(args.start_year), "end": int(args.end_year)},
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": float(elapsed),
        "model": "DGEV_LAPLACE_NCP_LASSO_DUMMIES",
        "cfg": _jsonify_dict(asdict(cfg)),
        "priors": _jsonify_dict(asdict(priors)),
        "notes": {
            "seasonality": "dummy sum-to-zero baseline gamma0 + optional dynamic seasonal state shrunk by lasso",
            "trend": "random-walk slope via NCP (shrunk by lasso)",
        },
    }

    sampler.save_posterior(
        out_npz_path=os.path.join(out_dir, "posterior.npz"),
        extra_meta=meta,
    )

    # ---------------- Quick summaries ----------------
    if post.get("sigma", np.array([])).size:
        print(f"Posterior mean sigma: {np.mean(post['sigma']):.3f}")
    if post.get("xi", np.array([])).size:
        print(f"Posterior mean xi:    {np.mean(post['xi']):.3f}")
    for k in ["alpha", "beta", "gamma"]:
        key = f"Q_{k}"
        if key in post and post[key].size:
            mQ = float(np.mean(post[key]))
            print(f"Posterior mean {key}: {mQ:.4g} (sqrt≈{math.sqrt(max(mQ,0.0)):.4g})")
    if post.get("lambda2", np.array([])).size:
        print(f"Posterior mean lambda2: {np.mean(post['lambda2']):.4g}")

    # ---------------- Plot ----------------
    if bool(args.plot):
        mu_hat = post["mu"].mean(axis=0)
        lo = np.quantile(post["mu"], 0.05, axis=0)
        hi = np.quantile(post["mu"], 0.95, axis=0)

        x = np.arange(T)
        plt.figure(figsize=(11, 4))
        plt.plot(x, y, lw=1.0, label="y_t")
        plt.plot(x, mu_hat, lw=1.5, label="posterior mean μ_t")
        plt.fill_between(x, lo, hi, alpha=0.25, label="90% band (μ_t)")
        plt.title(f"Uccle {args.series}: DGEV Laplace NCP + Bayesian lasso (dummies)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fit_mu.png"), dpi=200)
        plt.show()
