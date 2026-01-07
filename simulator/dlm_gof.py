# %% simulator/dlm_gof.py
from __future__ import annotations
"""
DLM Goodness-of-Fit (Gaussian structural models)
==============================================

Implements the diagnostic workflow described in your manuscript (Section "Model diagnostics"):

1) PITs (probability integral transforms)
   For the Gaussian DLM,
     Y_t | mu_t, sigma ~ Normal(mu_t, sigma^2)
   so for posterior draws (mu_t^{(m)}, sigma^{(m)}),
     u_t^{(m)} = Phi( (y_t - mu_t^{(m)}) / sigma^{(m)} ).

   We treat {u_t^{(m)}} as Monte Carlo draws from the posterior of U_t
   (plug-in evaluation at (x_t^{(m)}, eta^{(m)})).

   Outputs:
   - PIT histogram with posterior median + credible band (over posterior draws)
   - PIT PP-plot (empirical PIT quantiles vs Uniform(0,1)) with posterior band

   Styling (per requirement):
   - PIT histogram: NO title, NO legend
   - PIT PP plot:   NO title, NO legend

2) Posterior predictive assessment (Gaetan & Grigoletto-style)
   Using the KS distance of PITs to Uniform(0,1) as discrepancy K(·).

   For each posterior draw m:
     K_obs^{(m)} = KS( {u_t^{(m)}}_{t=1}^T , U(0,1) )
     simulate y_rep^{(m)} ~ p(y | x^{(m)}, eta^{(m)})  (pointwise)
     compute u_rep^{(m)} similarly and K_rep^{(m)}.

   Output:
   - scatter plot of (K_obs^{(m)}, K_rep^{(m)}) with 45° line

   Styling (per requirement):
   - KS scatter: NO title, NO legend, axis labels "K_obs" and "K_rep"

3) JSON summary
   Writes a compact JSON file with key GOF summaries (p_B and KS summaries).

Robust posterior discovery mirrors simulator/dlm_plotter.py:
- optimization.posterior_bundle.load_posterior/find_latest_run
- fallback to newest posterior*.npz under --root

Expected posterior bundle content
--------------------------------
Required:
  - draws['mu'] : (S, T) posterior draws of mu_t
  - draws['y']  : (T,)   observed series on model scale

Needed for PIT:
  - draws['sigma'] : (S,) posterior draws of observation SD,
    or draws['sigma2'] : (S,) posterior draws of variance.

Optional:
  - meta['start_date'] : 'YYYY-MM-DD' for pretty axes (not needed here)

CLI
---
python -m simulator.dlm_gof --target <run_or_npz> --out <dir> --show
python -m simulator.dlm_gof --root results/... (auto-picks latest)

"""

import os
import sys
import math
import json
import argparse
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# Make optimization/ visible
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Posterior loader
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e

# Utilities (same module used by plotter)
try:
    from simulator.utils import _ensure_dir, find_latest_posterior_npz  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import simulator.utils.\n"
        "Make sure simulator/utils.py is on PYTHONPATH."
    ) from e


# =============================================================================
# Math helpers
# =============================================================================
_SQRT2 = float(math.sqrt(2.0))


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    Standard Normal CDF (vectorized).

    Tries SciPy's erf; falls back to a SciPy-free approximation if SciPy is unavailable.
    """
    x = np.asarray(x, float)
    try:
        from scipy.special import erf as _erf  # type: ignore
        return 0.5 * (1.0 + _erf(x / _SQRT2))
    except Exception:
        # Abramowitz & Stegun (7.1.26) approximation for Φ(x)
        a1 = 0.319381530
        a2 = -0.356563782
        a3 = 1.781477937
        a4 = -1.821255978
        a5 = 1.330274429
        p = 0.2316419

        ax = np.abs(x)
        t = 1.0 / (1.0 + p * ax)
        pdf = 0.3989422804014327 * np.exp(-0.5 * ax * ax)
        poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
        cdf_pos = 1.0 - pdf * poly
        cdf = np.where(x >= 0.0, cdf_pos, 1.0 - cdf_pos)
        return np.clip(cdf, 0.0, 1.0)


def _ks_to_uniform(u: np.ndarray) -> float:
    """
    One-sample Kolmogorov–Smirnov distance to U(0,1):
      D = sup_u |F_n(u) - u|.
    """
    v = np.asarray(u, float).ravel()
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return float("nan")
    v = np.sort(np.clip(v, 0.0, 1.0))
    i = np.arange(1, n + 1, dtype=float)
    d_plus = float(np.max(i / n - v))
    d_minus = float(np.max(v - (i - 1.0) / n))
    return float(max(d_plus, d_minus))


def _draw_subsample_index(S: int, *, max_draws: Optional[int], rng: np.random.Generator) -> Optional[np.ndarray]:
    """
    Choose a single subsample index (shared across all draw arrays) to keep draw alignment.
    """
    if max_draws is None or int(max_draws) <= 0 or int(S) <= int(max_draws):
        return None
    idx = rng.choice(int(S), size=int(max_draws), replace=False)
    idx.sort()
    return idx


def _summarize(x: np.ndarray, qs: Tuple[float, float] = (0.05, 0.95)) -> Dict[str, float]:
    v = np.asarray(x, float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "min": float("nan"), "max": float("nan"),
                "median": float("nan"), f"q{int(qs[0]*100):02d}": float("nan"), f"q{int(qs[1]*100):02d}": float("nan")}
    ql, qh = float(qs[0]), float(qs[1])
    out = {
        "n": float(v.size),
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "median": float(np.quantile(v, 0.5)),
        f"q{int(ql*100):02d}": float(np.quantile(v, ql)),
        f"q{int(qh*100):02d}": float(np.quantile(v, qh)),
    }
    return out


# =============================================================================
# PIT diagnostics
# =============================================================================
def compute_pit_draws(
    *,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute posterior PIT draws u^{(m)}_t = Phi((y_t - mu^{(m)}_t)/sigma^{(m)}).

    Returns:
      u    : (S, T_eff)
      mask : (T,) boolean mask of finite y (T_eff = sum(mask))
    """
    y = np.asarray(y, float).ravel()
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float).ravel()

    if mu.ndim != 2:
        raise ValueError("mu must be (S,T)")
    S, T = mu.shape
    if y.size != T:
        raise ValueError(f"y has length {y.size} but mu has T={T}")
    if sigma.size != S:
        raise ValueError(f"sigma has length {sigma.size} but mu has S={S}")

    mask = np.isfinite(y)
    if not np.any(mask):
        raise ValueError("y has no finite values; cannot compute PITs")

    y_eff = y[mask][None, :]                          # (1, T_eff)
    mu_eff = mu[:, mask]                              # (S, T_eff)
    sig_eff = np.clip(sigma, 1e-12, None)[:, None]    # (S, 1)

    z = (y_eff - mu_eff) / sig_eff
    u = _norm_cdf(z)

    return np.clip(u, 0.0, 1.0), mask


def figure_pit_hist(
    u: np.ndarray,
    *,
    level: float = 0.90,
    bins: int = 20,
    save_dir: Optional[str] = None,
    fname: str = "gof_pit_hist.png",
    show: bool = True,
) -> Dict[str, Any]:
    """
    PIT histogram:
      - compute per-draw histogram density over t
      - plot posterior median density + pointwise band across draws
      - overlay Uniform(0,1) reference (flat density = 1)

    Styling: NO title, NO legend.

    Returns small numeric summary (for JSON).
    """
    U = np.asarray(u, float)
    if U.ndim != 2:
        raise ValueError("u must be (S,T_eff)")
    S, T = U.shape

    lev = float(level)
    if not (0.0 < lev < 1.0):
        raise ValueError("level must be in (0,1)")

    B = int(bins)
    if B < 5:
        raise ValueError("bins must be >= 5")

    idx = np.minimum((U * B).astype(int), B - 1)  # (S,T)
    dens = np.zeros((S, B), float)
    for m in range(S):
        c = np.bincount(idx[m], minlength=B).astype(float)
        dens[m] = c * (B / float(T))  # density

    lo_q = (1.0 - lev) / 2.0
    hi_q = 1.0 - lo_q

    dens_med = np.quantile(dens, 0.5, axis=0)
    dens_lo = np.quantile(dens, lo_q, axis=0)
    dens_hi = np.quantile(dens, hi_q, axis=0)

    edges = np.linspace(0.0, 1.0, B + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))
    ax.fill_between(mids, dens_lo, dens_hi, alpha=0.25, step="mid")
    ax.step(mids, dens_med, where="mid", lw=2.0)
    ax.axhline(1.0, lw=1.2, ls="--", color="k", alpha=0.7)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_dir:
        _ensure_dir(save_dir)
        out = os.path.join(save_dir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[save] {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # a light numeric summary: max absolute deviation of median bin density from 1
    max_abs_dev = float(np.max(np.abs(dens_med - 1.0))) if dens_med.size else float("nan")
    return {"bins": B, "level": lev, "max_abs_dev_from_uniform_density_median": max_abs_dev}


def figure_pit_pp(
    u: np.ndarray,
    *,
    level: float = 0.90,
    save_dir: Optional[str] = None,
    fname: str = "gof_pit_pp.png",
    show: bool = True,
) -> Dict[str, Any]:
    """
    PIT PP plot:
      - for each draw m, sort u_m (empirical quantiles)
      - plot posterior median curve + band vs Uniform(0,1) theoretical quantiles
      - overlay 45° line

    Styling: NO title, NO legend.

    Returns small numeric summary (for JSON).
    """
    U = np.asarray(u, float)
    if U.ndim != 2:
        raise ValueError("u must be (S,T_eff)")
    S, T = U.shape

    lev = float(level)
    if not (0.0 < lev < 1.0):
        raise ValueError("level must be in (0,1)")

    lo_q = (1.0 - lev) / 2.0
    hi_q = 1.0 - lo_q

    U_sorted = np.sort(U, axis=1)  # (S,T)
    q_med = np.quantile(U_sorted, 0.5, axis=0)
    q_lo = np.quantile(U_sorted, lo_q, axis=0)
    q_hi = np.quantile(U_sorted, hi_q, axis=0)

    p = (np.arange(1, T + 1, dtype=float) - 0.5) / float(T)

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.0))
    ax.fill_between(p, q_lo, q_hi, alpha=0.25)
    ax.plot(p, q_med, lw=2.0)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="k", alpha=0.7)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Uniform quantiles")
    ax.set_ylabel("Empirical PIT quantiles")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_dir:
        _ensure_dir(save_dir)
        out = os.path.join(save_dir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[save] {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # max deviation of posterior median PP curve from 45° line
    max_abs_dev = float(np.max(np.abs(q_med - p))) if q_med.size else float("nan")
    return {"level": lev, "max_abs_dev_from_45deg_median": max_abs_dev}


# =============================================================================
# Posterior predictive assessment (KS scatter + p_B)
# =============================================================================
def posterior_predictive_ks_scatter(
    *,
    u_obs: np.ndarray,
    rng: np.random.Generator,
    save_dir: Optional[str] = None,
    fname: str = "gof_ppc_ks_scatter.png",
    show: bool = True,
) -> Dict[str, Any]:
    """
    For each posterior draw m:
      KS_obs^{(m)} = KS( PIT_obs^{(m)}, U(0,1) )
      simulate y_rep^{(m)} ~ N(mu^{(m)}, sigma^{(m)}^2)
      KS_rep^{(m)} = KS( PIT_rep^{(m)}, U(0,1) )

    Implementation note (Gaussian):
      If y_rep = mu + sigma*eps, then PIT_rep = Phi(eps), so we can simulate eps directly.

    Styling: NO title, NO legend.
    Labels: x="K_obs", y="K_rep"

    Returns arrays and p_B = mean(K_rep > K_obs).
    """
    U = np.asarray(u_obs, float)
    if U.ndim != 2:
        raise ValueError("u_obs must be (S,T_eff)")
    S, T_eff = U.shape

    ks_obs = np.zeros(S, float)
    ks_rep = np.zeros(S, float)

    for m in range(S):
        ks_obs[m] = _ks_to_uniform(U[m])
        eps = rng.standard_normal(T_eff)
        u_rep = _norm_cdf(eps)
        ks_rep[m] = _ks_to_uniform(u_rep)

    finite = np.isfinite(ks_obs) & np.isfinite(ks_rep)
    p_B = float(np.mean(ks_rep[finite] > ks_obs[finite])) if np.any(finite) else float("nan")

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.0))
    ax.scatter(ks_obs, ks_rep, s=14, alpha=0.6)

    if np.any(finite):
        mn = float(min(np.min(ks_obs[finite]), np.min(ks_rep[finite])))
        mx = float(max(np.max(ks_obs[finite]), np.max(ks_rep[finite])))
        pad = 0.02 * (mx - mn + 1e-12)
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], ls="--", color="k", alpha=0.7)

    ax.set_xlabel("K_obs")
    ax.set_ylabel("K_rep")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_dir:
        _ensure_dir(save_dir)
        out = os.path.join(save_dir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[save] {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"ks_obs": ks_obs, "ks_rep": ks_rep, "p_B": p_B}


# =============================================================================
# CLI
# =============================================================================
def _resolve_bundle(target: Optional[str], root: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], str]:
    run_path = target
    if run_path is None:
        print(f"[info] --target not provided; searching for the latest posterior under --root={root!r} ...")
        run_path = find_latest_run(root=root)
        if run_path is None:
            npz_path = find_latest_posterior_npz(root)
            if npz_path is None:
                raise FileNotFoundError(f"No posterior runs found under {root!r}. Provide --target or change --root.")
            run_path = npz_path
            print(f"[info] find_latest_run found nothing; using latest npz: {run_path}")
        else:
            print(f"[info] Using latest run: {run_path}")

    bundle = load_posterior(run_path)
    return bundle.draws, bundle.meta, bundle.npz_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Goodness-of-fit diagnostics for Gaussian DLM posterior bundles (PIT + posterior predictive KS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=str, default=None, help="Run directory or posterior.npz path.")
    parser.add_argument("--root", type=str, default="results/simulations/DLM", help="Search root if --target omitted.")
    parser.add_argument("--out", type=str, default=None, help="Directory to save figures (default: <run>/gof).")
    parser.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")

    parser.add_argument("--level", type=float, default=0.90, help="Credible band level for PIT plots.")
    parser.add_argument("--bins", type=int, default=20, help="Number of bins for PIT histogram.")

    parser.add_argument("--max-draws", type=int, default=None, help="Subsample posterior draws for speed (None = all).")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed (subsampling + PPC simulation).")

    parser.add_argument("--skip-pit", action="store_true", help="Skip PIT histogram and PP plot.")
    parser.add_argument("--skip-ppc", action="store_true", help="Skip posterior predictive KS scatter.")

    parser.add_argument("--json-name", type=str, default="gof_results.json", help="Filename for JSON summary.")
    parser.add_argument("--json-full", action="store_true", help="Also store ks_obs/ks_rep arrays in JSON (large).")

    args = parser.parse_args()
    rng = np.random.default_rng(int(args.seed))

    draws, meta, npz_path = _resolve_bundle(args.target, args.root)
    print(f"[info] using posterior: {npz_path}")

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "gof")
    _ensure_dir(out_dir)
    print(f"[info] saving GOF figures to: {out_dir}")

    # Required
    if "mu" not in draws:
        raise KeyError("draws must contain 'mu' of shape (S,T).")
    if "y" not in draws:
        raise KeyError("draws must contain 'y' of shape (T,).")

    mu_all = np.asarray(draws["mu"], float)
    y = np.asarray(draws["y"], float).ravel()

    if mu_all.ndim != 2:
        raise ValueError("draws['mu'] must be 2D (S,T).")

    S_all, T = mu_all.shape
    if y.size != T:
        raise ValueError(f"draws['y'] has length {y.size} but draws['mu'] has T={T}.")

    # sigma on SD scale
    sigma_all: Optional[np.ndarray] = None
    if "sigma" in draws:
        s = np.asarray(draws["sigma"], float).ravel()
        if s.ndim == 1 and s.shape[0] == S_all:
            sigma_all = s
    if sigma_all is None and "sigma2" in draws:
        s2 = np.asarray(draws["sigma2"], float).ravel()
        if s2.ndim == 1 and s2.shape[0] == S_all:
            sigma_all = np.sqrt(np.clip(s2, 0.0, None))
    if sigma_all is None:
        raise KeyError("Need draws['sigma'] or draws['sigma2'] aligned with draws['mu'].")

    # Optional subsample draws (IMPORTANT: shared index to keep draw alignment)
    idx = _draw_subsample_index(S_all, max_draws=args.max_draws, rng=rng)
    if idx is None:
        mu = mu_all
        sigma = sigma_all
    else:
        mu = mu_all[idx, :]
        sigma = sigma_all[idx]
        print(f"[info] subsampled draws: S={mu.shape[0]} (from {S_all})")

    # Compute PIT draws (S, T_eff)
    u_obs, mask = compute_pit_draws(y=y, mu=mu, sigma=sigma)
    S, T_eff = u_obs.shape
    n_missing = int(np.sum(~mask))
    print(f"[info] PIT draws computed: S={S}, T_eff={T_eff} (dropped {n_missing} non-finite y)")

    results: Dict[str, Any] = {
        "npz_path": str(npz_path),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "S_total": int(S_all),
        "S_used": int(S),
        "T": int(T),
        "T_eff": int(T_eff),
        "n_nonfinite_y": int(n_missing),
        "level": float(args.level),
        "bins": int(args.bins),
    }

    # PIT plots + summaries
    if args.skip_pit:
        results["pit"] = {"skipped": True}
    else:
        pit_hist_summary = figure_pit_hist(u_obs, level=float(args.level), bins=int(args.bins), save_dir=out_dir, show=args.show)
        pit_pp_summary = figure_pit_pp(u_obs, level=float(args.level), save_dir=out_dir, show=args.show)
        results["pit"] = {"skipped": False, "hist": pit_hist_summary, "pp": pit_pp_summary}

    # Posterior predictive KS scatter + summaries
    if args.skip_ppc:
        results["ppc"] = {"skipped": True}
    else:
        ppc = posterior_predictive_ks_scatter(u_obs=u_obs, rng=rng, save_dir=out_dir, show=args.show)
        ks_obs = np.asarray(ppc["ks_obs"], float)
        ks_rep = np.asarray(ppc["ks_rep"], float)

        results["ppc"] = {
            "skipped": False,
            "p_B": float(ppc["p_B"]),
            "ks_obs_summary": _summarize(ks_obs),
            "ks_rep_summary": _summarize(ks_rep),
        }
        if bool(args.json_full):
            # WARNING: can be large for big S
            results["ppc"]["ks_obs"] = ks_obs.tolist()
            results["ppc"]["ks_rep"] = ks_rep.tolist()

        print(f"[ppc] p_B = {results['ppc']['p_B']:.6f}")
        print(f"[ppc] mean(K_obs)={results['ppc']['ks_obs_summary']['mean']:.6f}, mean(K_rep)={results['ppc']['ks_rep_summary']['mean']:.6f}")

    # Write JSON summary
    json_path = os.path.join(out_dir, str(args.json_name))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[save] JSON summary -> {json_path}")

    print("[done] GOF diagnostics written.")


if __name__ == "__main__":
    main()
