# %% simulator/dlm_gof.py
from __future__ import annotations
"""
DLM Goodness-of-Fit Diagnostics
==============================

Goodness-of-fit (in-sample, model checking) for fitted Gaussian DLM runs saved by
optimization/dlm_lasso.py (posterior.npz + posterior.meta.json).

What this script does
---------------------
Given posterior draws of (mu_t, sigma), it computes *posterior predictive* PIT values:
  u_t = E[ Phi( (y_t - mu_t) / sigma ) | y_{1:T} ]  (Monte Carlo average over draws)

Then it produces:
  - PIT histogram + KS statistic (heuristic; time-series dependence)
  - PIT PP plot (empirical vs uniform)
  - Normal QQ/PP plots for r_t = Phi^{-1}(u_t)
  - ACF of r_t (detect remaining dependence / structure)

Notes
-----
- These are *smoothing / posterior predictive* diagnostics (conditioning on all y_{1:T}).
  One-step-ahead (filter) PITs would require storing or recomputing predictive moments
  from a forward filter per draw; this script focuses on practical GoF for saved runs.

Usage examples
--------------
# Use latest run under a root directory (via posterior_bundle)
python -u simulator/dlm_gof.py --root results/uccle/TX/TXm --outdir Figures/TXm/GoF

# Use an explicit run directory
python -u simulator/dlm_gof.py --run-dir results/uccle/TX/TXm/Seasonal/run_20260105_120000

# Provide start date for calendar axis (if not in meta)
python -u simulator/dlm_gof.py --run-dir ... --start-date 1892-01-01
"""

import os
import sys
import json
import math
import glob
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------
# Make optimization/ visible (mirrors plotter pattern)
# ---------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
except Exception:
    load_posterior = None  # type: ignore
    find_latest_run = None  # type: ignore


# =============================================================================
# Utilities
# =============================================================================
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "":
        return None
    parts = [int(p) for p in ss.split("-")]
    if len(parts) == 1:
        return datetime(parts[0], 1, 1)
    if len(parts) == 2:
        return datetime(parts[0], parts[1], 1)
    if len(parts) == 3:
        return datetime(parts[0], parts[1], parts[2])
    raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")


def _month_index(T: int, start_date: Optional[datetime]) -> np.ndarray:
    """Return array of datetime objects of length T (monthly)."""
    if start_date is None:
        # fallback: integer axis disguised as dates not needed
        return np.arange(1, T + 1)

    try:
        import pandas as pd  # type: ignore

        return pd.date_range(start=start_date, periods=T, freq="MS").to_pydatetime()
    except Exception:
        # manual month stepping
        out: List[datetime] = []
        y, m = start_date.year, start_date.month
        for _ in range(T):
            out.append(datetime(y, m, 1))
            m += 1
            if m == 13:
                m = 1
                y += 1
        return np.asarray(out, dtype=object)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF using math.erf (vectorized)."""
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """
    Approximate inverse standard normal CDF (Acklam's approximation).
    Accurate enough for diagnostics. Pure-numpy, no SciPy dependency.
    """
    p = np.asarray(p, float)
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)

    # Coefficients in rational approximations
    a = np.array([
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ], float)

    b = np.array([
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ], float)

    c = np.array([
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ], float)

    d = np.array([
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ], float)

    plow = 0.02425
    phigh = 1.0 - plow

    x = np.zeros_like(p)

    # lower region
    m = p < plow
    if np.any(m):
        q = np.sqrt(-2.0 * np.log(p[m]))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        x[m] = num / den

    # central region
    m = (p >= plow) & (p <= phigh)
    if np.any(m):
        q = p[m] - 0.5
        r = q*q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        x[m] = num / den

    # upper region
    m = p > phigh
    if np.any(m):
        q = np.sqrt(-2.0 * np.log(1.0 - p[m]))
        num = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        x[m] = num / den

    return x


def _ks_uniform(u: np.ndarray) -> Tuple[float, float]:
    """
    One-sample KS statistic vs Unif(0,1) and asymptotic p-value approximation.
    (Heuristic under time-series dependence.)
    """
    u = np.asarray(u, float)
    u = u[np.isfinite(u)]
    n = u.size
    if n == 0:
        return float("nan"), float("nan")

    us = np.sort(np.clip(u, 1e-12, 1.0 - 1e-12))
    i = np.arange(1, n + 1, dtype=float)

    d_plus = np.max(i / n - us)
    d_minus = np.max(us - (i - 1.0) / n)
    D = float(max(d_plus, d_minus))

    # asymptotic p-value (Kolmogorov distribution)
    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * D

    # Q_KS(lam) = 2 * sum_{k>=1} (-1)^{k-1} exp(-2 k^2 lam^2)
    # truncate when terms are tiny
    s = 0.0
    for k in range(1, 200):
        term = (-1.0) ** (k - 1) * math.exp(-2.0 * (k * k) * (lam * lam))
        s += term
        if abs(term) < 1e-10:
            break
    pval = float(max(0.0, min(1.0, 2.0 * s)))
    return D, pval


def _acf(x: np.ndarray, max_lag: int = 24) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.full(max_lag + 1, np.nan)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.full(max_lag + 1, np.nan)
    ac = np.empty(max_lag + 1, float)
    ac[0] = 1.0
    for k in range(1, max_lag + 1):
        ac[k] = float(np.dot(x[:-k], x[k:]) / denom)
    return ac


def _discover_latest_npz(run_dir: str) -> str:
    cand = glob.glob(os.path.join(run_dir, "posterior*.npz"))
    if not cand:
        raise FileNotFoundError(f"No posterior*.npz found in {run_dir}")
    cand.sort(key=lambda p: os.path.getmtime(p))
    return cand[-1]


def _load_npz_meta(npz_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    arr = dict(np.load(npz_path, allow_pickle=False))
    meta_path = npz_path.replace(".npz", ".meta.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return arr, meta


def _load_posterior_any(run_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], str]:
    """
    Return arrays, meta, and the npz path used.
    Prefers posterior_bundle if available; falls back to newest posterior*.npz.
    """
    if load_posterior is not None:
        try:
            bundle = load_posterior(run_dir)  # expected to return {"arrays":..., "meta":..., ...}
            arrays = bundle["arrays"]
            meta = bundle.get("meta", {})
            # best-effort: locate a .npz inside run_dir for bookkeeping
            npz_path = _discover_latest_npz(run_dir)
            return arrays, meta, npz_path
        except Exception:
            pass

    npz_path = _discover_latest_npz(run_dir)
    arrays, meta = _load_npz_meta(npz_path)
    return arrays, meta, npz_path


# =============================================================================
# Main GoF
# =============================================================================
@dataclass
class GoFResults:
    T: int
    n_draws: int
    ks_D: float
    ks_pvalue: float
    pit_mean: float
    pit_var: float
    r_mean: float
    r_var: float
    acf1_r: float


def dlm_gof(
    *,
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    outdir: str,
    start_date: Optional[datetime] = None,
    max_acf_lag: int = 36,
    n_draws_plot: int = 0,
) -> GoFResults:
    _ensure_dir(outdir)

    if "y" not in arrays:
        raise KeyError("posterior arrays must include 'y'")
    if "mu" not in arrays:
        raise KeyError("posterior arrays must include 'mu' (draws x T)")
    if "sigma" not in arrays:
        raise KeyError("posterior arrays must include 'sigma' (draws,)")

    y = np.asarray(arrays["y"], float).reshape(-1)
    mu = np.asarray(arrays["mu"], float)
    sig = np.asarray(arrays["sigma"], float).reshape(-1)

    if mu.ndim != 2:
        raise ValueError(f"'mu' must be 2D (n_draws, T), got shape {mu.shape}")

    M, T = mu.shape
    if y.size != T:
        raise ValueError(f"Length mismatch: y has {y.size}, but mu has T={T}")
    if sig.size != M:
        raise ValueError(f"Length mismatch: sigma has {sig.size}, but mu has n_draws={M}")

    # ------------------------------------------------------------------
    # Calendar axis
    # ------------------------------------------------------------------
    # try meta first
    if start_date is None:
        for key in ("start_date", "start", "t0"):
            if key in meta:
                start_date = _parse_date(meta[key])
                if start_date is not None:
                    break

    xaxis = _month_index(T, start_date=start_date)

    # ------------------------------------------------------------------
    # Posterior predictive PIT: u_t = mean_m Phi((y_t - mu_mt)/sigma_m)
    # ------------------------------------------------------------------
    # vectorize: (M,T) z = (y - mu) / sigma[:,None]
    z = (y[None, :] - mu) / np.maximum(sig[:, None], 1e-12)
    u = _norm_cdf(z).mean(axis=0)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)

    D, pval = _ks_uniform(u)

    # r_t = Phi^{-1}(u_t)
    r = _norm_ppf(u)

    # ACF
    ac = _acf(r, max_lag=max_acf_lag)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # PIT histogram
    plt.figure(figsize=(7.2, 4.0))
    plt.hist(u, bins=12, density=True)
    plt.axhline(1.0, lw=1.5, ls="--")
    plt.title(f"DLM posterior predictive PIT histogram (KS D={D:.3f}, p≈{pval:.3g})")
    plt.xlabel("u_t")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pit_hist.png"), dpi=200)
    plt.close()

    # PIT PP plot (Uniform)
    us = np.sort(u)
    n = us.size
    emp = (np.arange(1, n + 1) - 0.5) / n
    plt.figure(figsize=(4.8, 4.8))
    plt.plot(emp, us, lw=1.5)
    plt.plot([0, 1], [0, 1], ls="--", lw=1.0)
    plt.title("PIT PP plot (Uniform)")
    plt.xlabel("Empirical probability")
    plt.ylabel("Sorted PIT values")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pit_pp.png"), dpi=200)
    plt.close()

    # Normal QQ plot of r_t
    rs = np.sort(r[np.isfinite(r)])
    n2 = rs.size
    theo = _norm_ppf((np.arange(1, n2 + 1) - 0.5) / n2)

    plt.figure(figsize=(4.8, 4.8))
    plt.plot(theo, rs, lw=0, marker="o", markersize=3)
    lo = float(min(theo.min(), rs.min()))
    hi = float(max(theo.max(), rs.max()))
    plt.plot([lo, hi], [lo, hi], ls="--", lw=1.0)
    plt.title("Normal QQ plot of r_t = Φ^{-1}(u_t)")
    plt.xlabel("Theoretical N(0,1) quantiles")
    plt.ylabel("Empirical quantiles of r_t")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "r_qq.png"), dpi=200)
    plt.close()

    # Normal PP plot of r_t
    # PP: empirical CDF of r vs Phi(r)
    Phi_r = _norm_cdf(rs)
    emp2 = (np.arange(1, n2 + 1) - 0.5) / n2
    plt.figure(figsize=(4.8, 4.8))
    plt.plot(emp2, Phi_r, lw=1.5)
    plt.plot([0, 1], [0, 1], ls="--", lw=1.0)
    plt.title("Normal PP plot of r_t")
    plt.xlabel("Empirical probability")
    plt.ylabel("Φ(r_t)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "r_pp.png"), dpi=200)
    plt.close()

    # ACF of r_t
    plt.figure(figsize=(7.2, 3.6))
    lags = np.arange(ac.size)
    plt.stem(lags, ac, basefmt=" ", use_line_collection=True)
    # heuristic 95% bounds for white noise
    bound = 1.96 / math.sqrt(max(1, n2))
    plt.axhline(bound, ls="--", lw=1.0)
    plt.axhline(-bound, ls="--", lw=1.0)
    plt.title("ACF of r_t")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "r_acf.png"), dpi=200)
    plt.close()

    # Optional: time series of r_t
    plt.figure(figsize=(10.0, 3.2))
    plt.plot(xaxis, r, lw=0.8)
    plt.axhline(0.0, ls="--", lw=1.0)
    plt.title("r_t = Φ^{-1}(u_t) over time (posterior predictive PIT)")
    if not isinstance(xaxis[0], (int, np.integer, float, np.floating)):
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=10))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.gcf().autofmt_xdate()
    plt.ylabel("r_t")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "r_time.png"), dpi=200)
    plt.close()

    # Optional: overlay a few posterior predictive sample paths
    if n_draws_plot and n_draws_plot > 0:
        rng = np.random.default_rng(123)
        idx = rng.choice(M, size=min(n_draws_plot, M), replace=False)
        yrep = mu[idx, :] + sig[idx, None] * rng.standard_normal((idx.size, T))

        plt.figure(figsize=(10.0, 3.2))
        plt.plot(xaxis, y, lw=1.0, label="observed")
        for j in range(idx.size):
            plt.plot(xaxis, yrep[j], lw=0.6, alpha=0.5)
        plt.title("Posterior predictive replicated paths (subset)")
        if not isinstance(xaxis[0], (int, np.integer, float, np.floating)):
            plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=10))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.gcf().autofmt_xdate()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ppc_paths.png"), dpi=200)
        plt.close()

    # ------------------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------------------
    res = GoFResults(
        T=int(T),
        n_draws=int(M),
        ks_D=float(D),
        ks_pvalue=float(pval),
        pit_mean=float(np.mean(u)),
        pit_var=float(np.var(u)),
        r_mean=float(np.mean(r)),
        r_var=float(np.var(r)),
        acf1_r=float(ac[1]) if np.isfinite(ac[1]) else float("nan"),
    )

    with open(os.path.join(outdir, "gof_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res.__dict__, f, indent=2)

    return res


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Goodness-of-fit diagnostics for fitted DLM posterior runs.")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--run-dir", type=str, default=None, help="Explicit run directory containing posterior*.npz")
    g.add_argument("--root", type=str, default=None, help="Root directory; pick latest run under this root")

    p.add_argument("--outdir", type=str, default=None, help="Output directory for GoF figures (default: <run>/GoF)")
    p.add_argument("--start-date", type=str, default=None, help="Start date for monthly axis: YYYY or YYYY-MM or YYYY-MM-DD")
    p.add_argument("--max-acf-lag", type=int, default=36, help="Max lag for ACF plot")
    p.add_argument("--ppc-draws", type=int, default=0, help="If >0, overlay this many posterior predictive paths")

    args = p.parse_args()

    # Resolve run dir
    run_dir: Optional[str] = args.run_dir
    if run_dir is None:
        if args.root is None:
            # fallback: assume current directory is a run
            run_dir = os.getcwd()
        else:
            if find_latest_run is None:
                # naive: choose newest subdir
                subs = [os.path.join(args.root, d) for d in os.listdir(args.root)]
                subs = [d for d in subs if os.path.isdir(d)]
                if not subs:
                    raise FileNotFoundError(f"No subdirectories under {args.root}")
                subs.sort(key=lambda d: os.path.getmtime(d))
                run_dir = subs[-1]
            else:
                run_dir = find_latest_run(args.root)

    run_dir = os.path.abspath(run_dir)

    arrays, meta, npz_path = _load_posterior_any(run_dir)

    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(run_dir, "GoF")
    outdir = os.path.abspath(outdir)
    _ensure_dir(outdir)

    start_date = _parse_date(args.start_date)
    res = dlm_gof(
        arrays=arrays,
        meta=meta,
        outdir=outdir,
        start_date=start_date,
        max_acf_lag=int(args.max_acf_lag),
        n_draws_plot=int(args.ppc_draws),
    )

    print("\n[DLM GoF]")
    print(f"  run_dir  : {run_dir}")
    print(f"  npz      : {npz_path}")
    print(f"  outdir   : {outdir}")
    print(f"  T        : {res.T}")
    print(f"  draws    : {res.n_draws}")
    print(f"  KS D     : {res.ks_D:.4f} (p≈{res.ks_pvalue:.3g})  [heuristic]")
    print(f"  mean(u)  : {res.pit_mean:.4f}  var(u): {res.pit_var:.4f}")
    print(f"  mean(r)  : {res.r_mean:.4f}   var(r): {res.r_var:.4f}   acf1(r): {res.acf1_r:.4f}")
    print("  figures  : pit_hist.png, pit_pp.png, r_qq.png, r_pp.png, r_acf.png, r_time.png")
    if int(args.ppc_draws) > 0:
        print("             + ppc_paths.png")
    print("  summary  : gof_summary.json\n")


if __name__ == "__main__":
    main()
