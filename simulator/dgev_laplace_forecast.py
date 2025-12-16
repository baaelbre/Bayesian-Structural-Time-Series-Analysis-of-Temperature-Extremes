# %% simulator/dgev_return_forecast.py
from __future__ import annotations

import os
import math
import sys
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Make optimization package visible (mirrors dgev_plotter.py / dlm_plotter.py)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------
# I/O helpers: posterior loader
# ---------------------------------------------------------------------
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e


# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _parse_csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [float(z) for z in s.split(",")]


# ---------------------------------------------------------------------
# GEV helpers (standalone)
# ---------------------------------------------------------------------
def _gev_cdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Broadcasting-safe GEV CDF.

    - Stable near xi=0 (Gumbel limit).
    - Correct support handling:
        xi > 0: CDF = 0 for x below lower endpoint
        xi < 0: CDF = 1 for x above upper endpoint
    """
    x = np.asarray(x, float)
    mu = np.asarray(mu, float)
    sigma = np.clip(np.asarray(sigma, float), 1e-12, None)
    xi = np.asarray(xi, float)

    z = (x - mu) / sigma
    tol = 1e-8

    is_gumbel = np.abs(xi) < tol

    # Gumbel CDF
    F0 = np.exp(-np.exp(-z))

    # General xi != 0 branch (computed in a broadcast-safe way)
    t = 1.0 + xi * z

    # Safe placeholder where t<=0 (we overwrite those via support rules)
    t_pos = np.where(t > 0.0, t, 1.0)

    # raw formula where t>0
    F1_raw = np.exp(-(t_pos ** (-1.0 / xi)))

    # Support corrections for t<=0
    # xi>0  => below lower endpoint => F=0
    # xi<0  => above upper endpoint => F=1
    F1 = np.where(
        t > 0.0,
        F1_raw,
        np.where(xi > 0.0, 0.0, 1.0),
    )

    F = np.where(is_gumbel, F0, F1)
    return np.clip(F, 0.0, 1.0)



def _gev_return_level_block(mu: np.ndarray, sigma: np.ndarray, xi: np.ndarray, N: float) -> np.ndarray:
    """
    Block return level z_N such that P(X <= z_N) = 1 - 1/N.

    Works when mu, sigma, xi are arrays of the same shape and N is scalar.
    """
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    xi = np.asarray(xi, float)

    N_val = float(N)
    p = 1.0 - 1.0 / max(N_val, 1.0 + 1e-12)
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    tol = 1e-8

    z = np.empty_like(mu)
    g = -math.log(-math.log(p))  # scalar

    mask0 = np.abs(xi) < tol
    mask1 = ~mask0

    # Gumbel limit
    if np.any(mask0):
        z[mask0] = mu[mask0] + sigma[mask0] * g

    # xi != 0
    if np.any(mask1):
        base = -math.log(p)  # scalar
        z[mask1] = (
            mu[mask1]
            + (sigma[mask1] / xi[mask1]) * (np.power(base, -xi[mask1]) - 1.0)
        )

    return z


# ---------------------------------------------------------------------
# Seasonal baseline design + seasonal rotation
# ---------------------------------------------------------------------
def _build_season_design(T: int, period: int) -> np.ndarray:
    """
    Seasons indexed 0..p-1 with period=p.
    For seasons 0..p-2: one-hot in coordinates 0..p-2.
    For season p-1: -1 in all entries (sum-to-zero constraint).
    """
    if period < 2:
        return np.zeros((T, 0))
    p = period
    K = p - 1
    S = np.zeros((T, K), float)
    for t in range(T):
        season = t % p
        if season < K:
            S[t, season] = 1.0
        else:
            S[t, :] = -1.0
    return S


def _season_rotation_matrix(K: int) -> np.ndarray:
    """
    Rotation matrix R for dummy seasonal dynamics:
      R[0, :] = -1
      R[1:, :-1] = I_{K-1}
    """
    if K <= 0:
        return np.zeros((0, 0))
    R = np.zeros((K, K))
    R[0, :] = -1.0
    if K > 1:
        R[1:, :-1] = np.eye(K - 1)
    return R


# ---------------------------------------------------------------------
# Core: forecasting μ_t
# ---------------------------------------------------------------------
def _extract_state_indices(layout: List[str], period: int) -> Tuple[int, int, int, int]:
    """
    Extract indices for alpha, beta and the block of seasonal dynamic states g1..g_{p-1}.
    """
    if "alpha" not in layout or "beta" not in layout:
        raise ValueError("layout must contain 'alpha' and 'beta' entries.")
    idx_alpha = layout.index("alpha")
    idx_beta = layout.index("beta")

    g_indices = [i for i, nm in enumerate(layout) if nm.startswith("g")]
    if not g_indices:
        raise ValueError("layout must contain dynamic seasonal states 'g1', 'g2', ..., 'g{p-1}'.")
    idx_g_start = min(g_indices)
    idx_g_end = max(g_indices)

    expected_K = period - 1
    if expected_K > 0 and (idx_g_end - idx_g_start + 1) < expected_K:
        raise ValueError(
            f"layout appears inconsistent with period={period}: "
            f"expected ≥ {expected_K} seasonal coords, got {idx_g_end - idx_g_start + 1}."
        )

    return idx_alpha, idx_beta, idx_g_start, idx_g_end


def forecast_mu_and_params(
    draws: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    horizon: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From posterior draws and meta, build a forecast of μ_t for t = 1..T+H:

      - Structural CP dynamics:
          alpha_{t+1} = alpha_t + beta_t + η_alpha,t   (η ~ N(0, s_alpha^2))
          beta_{t+1}  = beta_t + η_beta,t              (η ~ N(0, s_beta^2))
          gamma_{t+1} = R gamma_t + η_gamma,t          (η ~ N(0, s_gamma^2 I))

      - Baseline seasonal design S[t,:] @ gamma0 (static γ0).
      - σ, ξ constant over t.

    Returns:
      mu_all: (S, T+H)
      sigma : (S,)
      xi    : (S,)
    """
    mu_draws = np.asarray(draws["mu"], float)  # (S, T)
    S_draws, T = mu_draws.shape

    if "x" not in draws or draws["x"].ndim != 3:
        raise ValueError("draws must contain 'x' with shape (S, T, dim) for forecasting.")
    x_draws = np.asarray(draws["x"], float)

    # GEV params
    if "sigma" in draws:
        sigma = np.asarray(draws["sigma"], float)
    elif "sigma2" in draws:
        sigma = np.sqrt(np.clip(np.asarray(draws["sigma2"], float), 0, None))
    else:
        raise ValueError("draws must contain 'sigma' or 'sigma2'.")
    if sigma.shape[0] != S_draws:
        raise ValueError("Length of sigma does not match number of draws S.")

    if "xi" not in draws:
        raise ValueError("draws must contain 'xi'.")
    xi = np.asarray(draws["xi"], float)
    if xi.shape[0] != S_draws:
        raise ValueError("Length of xi does not match number of draws S.")

    period = int(meta.get("period", 1))
    if period < 1:
        raise ValueError("meta['period'] must be >= 1.")
    K_gamma = max(period - 1, 0)

    layout = meta.get("layout")
    if layout is None:
        raise ValueError("meta must contain 'layout' as list of state names.")
    layout = list(layout)
    idx_alpha, idx_beta, idx_g_start, _idx_g_end = _extract_state_indices(layout, period)

    def _get_signed_sd(name_s: str, name_q: str) -> np.ndarray:
        if name_s in draws:
            arr = np.asarray(draws[name_s], float)
            if arr.shape[0] != S_draws:
                raise ValueError(f"{name_s} length mismatch.")
            return arr
        if name_q in draws:
            q = np.asarray(draws[name_q], float)
            if q.shape[0] != S_draws:
                raise ValueError(f"{name_q} length mismatch.")
            return np.sign(q) * np.sqrt(np.abs(q))
        raise ValueError(f"Neither '{name_s}' nor '{name_q}' found in draws.")

    s_alpha = _get_signed_sd("s_alpha", "Q_alpha")
    s_beta = _get_signed_sd("s_beta", "Q_beta")
    s_gamma = _get_signed_sd("s_gamma", "Q_gamma") if K_gamma > 0 else np.zeros(S_draws)

    # Static seasonal baselines gamma0
    if K_gamma > 0:
        if "gamma0" not in draws:
            raise ValueError("draws must contain 'gamma0' when period>1.")
        gamma0 = np.asarray(draws["gamma0"], float)  # (S, K_gamma)
        if gamma0.shape != (S_draws, K_gamma):
            raise ValueError("gamma0 shape mismatch.")
    else:
        gamma0 = np.zeros((S_draws, 0), float)

    T_total = T + max(horizon, 0)
    S_design_all = _build_season_design(T_total, period)  # (T_total, K_gamma)
    R_gamma = _season_rotation_matrix(K_gamma)            # (K_gamma, K_gamma)

    rng = np.random.default_rng(seed)

    mu_all = np.zeros((S_draws, T_total), float)
    mu_all[:, :T] = mu_draws  # historical

    for s in range(S_draws):
        x_last = x_draws[s, T - 1, :]

        alpha_curr = float(x_last[idx_alpha])
        beta_curr = float(x_last[idx_beta])
        gamma_curr = (
            x_last[idx_g_start: idx_g_start + K_gamma].astype(float).copy()
            if K_gamma > 0 else np.zeros(0, float)
        )

        s_a = float(s_alpha[s])
        s_b = float(s_beta[s])
        s_g = float(s_gamma[s]) if K_gamma > 0 else 0.0
        gamma0_s = gamma0[s, :] if K_gamma > 0 else np.zeros(0, float)

        for h in range(horizon):
            t = T + h

            eps_alpha = rng.normal(0.0, max(abs(s_a), 1e-16))
            eps_beta = rng.normal(0.0, max(abs(s_b), 1e-16))

            alpha_next = alpha_curr + beta_curr + eps_alpha
            beta_next = beta_curr + eps_beta

            if K_gamma > 0:
                eps_gamma = rng.normal(0.0, max(abs(s_g), 1e-16), size=K_gamma)
                gamma_next = R_gamma @ gamma_curr + eps_gamma
            else:
                gamma_next = gamma_curr

            g1_next = gamma_next[0] if K_gamma > 0 else 0.0
            base_next = float(S_design_all[t, :] @ gamma0_s) if K_gamma > 0 else 0.0
            mu_all[s, t] = alpha_next + g1_next + base_next

            alpha_curr, beta_curr, gamma_curr = alpha_next, beta_next, gamma_next

    return mu_all, sigma, xi


# ---------------------------------------------------------------------
# Return levels / return periods (block + yearly)
# ---------------------------------------------------------------------
def compute_return_levels_block(
    mu_all: np.ndarray,
    sigma: np.ndarray,
    xi: np.ndarray,
    Ns: List[float],
) -> Dict[float, np.ndarray]:
    """
    Block return levels: z_{N,t} with G_t(z_{N,t}) = 1 - 1/N.

    Returns:
      dict[N] -> zN (S, T_total)
    """
    S_draws, T_total = mu_all.shape
    sigma_exp = np.repeat(sigma[:, None], T_total, axis=1)
    xi_exp = np.repeat(xi[:, None], T_total, axis=1)

    out: Dict[float, np.ndarray] = {}
    for N in Ns:
        zN = _gev_return_level_block(mu_all, sigma_exp, xi_exp, N=float(N))
        out[float(N)] = zN
    return out

def compute_return_levels_yearly(
    mu_all: np.ndarray,
    sigma: np.ndarray,
    xi: np.ndarray,
    Ns_year: List[float],
    blocks_per_year: int,
    max_iter: int = 80,
) -> Dict[float, np.ndarray]:
    """
    Annual (coarse-grained) return levels for annual maxima M_j = max_{t in year j} Y_t:

        P(M_j > z_{N,j}^{ann}) = 1/N
        <=> prod_{t in year j} G_t(z_{N,j}^{ann}) = 1 - 1/N.

    Uses monotone bisection (vectorized over draws and years).

    Returns:
      dict[N] -> z_ann (S, Y_total), where Y_total = floor(T_total / blocks_per_year).
    """
    if blocks_per_year < 1:
        raise ValueError("blocks_per_year must be >= 1.")

    S_draws, T_total = mu_all.shape
    Y_total = T_total // blocks_per_year
    if Y_total <= 0:
        raise ValueError("Not enough blocks to form a single year.")

    # Truncate to full years for grouping
    T_use = Y_total * blocks_per_year
    mu_y = mu_all[:, :T_use].reshape(S_draws, Y_total, blocks_per_year)

    sig_y = sigma[:, None, None]
    xi_y = xi[:, None, None]

    # Helpful summaries for bracketing
    mu_min = np.min(mu_y, axis=2)  # (S, Y)
    mu_max = np.max(mu_y, axis=2)  # (S, Y)

    # Base brackets
    lo0 = mu_min - 20.0 * sigma[:, None]

    # Upper brackets depend on sign of xi (finite endpoint when xi<0)
    tol = 1e-8
    xi_s = xi[:, None]  # (S, 1)
    neg_xi = xi_s < -tol

    # endpoint per block when xi<0: mu - sigma/xi
    # (sigma/xi is negative, so mu - sigma/xi > mu)
    endpoints = mu_y - (sigma[:, None, None] / np.clip(xi[:, None, None], -np.inf, -tol))
    hi0 = mu_max + 20.0 * sigma[:, None]  # default for xi>=0
    if np.any(neg_xi):
        hi0[neg_xi] = np.max(endpoints[neg_xi[:, :, None]], axis=2) - 1e-10  # near max endpoint

    def _prod_cdf(z_sy: np.ndarray) -> np.ndarray:
        # z_sy: (S,Y) -> broadcast to (S,Y,B)
        F = _gev_cdf(z_sy[:, :, None], mu_y, sig_y, xi_y)
        return np.prod(F, axis=2)

    out: Dict[float, np.ndarray] = {}
    for N in Ns_year:
        N = float(N)
        target = 1.0 - 1.0 / max(N, 1.0 + 1e-12)
        target = float(np.clip(target, 1e-12, 1.0 - 1e-12))

        lo = lo0.copy()
        hi = hi0.copy()

        # Ensure hi brackets the target when xi>=0 by expanding upward if needed
        # (for xi<0, hi is already near the endpoint -> prod ~ 1)
        for k in range(15):
            prod_hi = _prod_cdf(hi)
            bad = (prod_hi < target) & (~neg_xi)  # only expand for nonnegative xi
            if not np.any(bad):
                break
            hi[bad] += (2.0 ** k) * 10.0 * sigma[bad[:, 0]]  # sigma broadcasted per draw

        # Bisection
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            prod_mid = _prod_cdf(mid)
            go_up = prod_mid < target  # need larger z -> move lo up
            lo[go_up] = mid[go_up]
            hi[~go_up] = mid[~go_up]

        out[N] = hi  # (S,Y)

    return out


def compute_return_periods_block(
    mu_all: np.ndarray,
    sigma: np.ndarray,
    xi: np.ndarray,
    us: List[float],
) -> Dict[float, np.ndarray]:
    """
    Block return periods for fixed thresholds u:

        p_t(u) = P(Y_t > u) = 1 - G_t(u)
        R_t(u) = 1 / p_t(u)

    Returns:
      dict[u] -> R_t(u) (S, T_total)
    """
    S_draws, T_total = mu_all.shape
    sigma_exp = np.repeat(sigma[:, None], T_total, axis=1)
    xi_exp = np.repeat(xi[:, None], T_total, axis=1)

    out: Dict[float, np.ndarray] = {}
    for u in us:
        u = float(u)
        u_arr = np.full((S_draws, T_total), u)
        F_u = _gev_cdf(u_arr, mu_all, sigma_exp, xi_exp)
        p_exc = np.clip(1.0 - F_u, 1e-12, 1.0)
        out[u] = 1.0 / p_exc
    return out


def compute_return_periods_yearly(
    mu_all: np.ndarray,
    sigma: np.ndarray,
    xi: np.ndarray,
    us: List[float],
    blocks_per_year: int,
) -> Dict[float, np.ndarray]:
    """
    Yearly exceedance probabilities and return periods for fixed thresholds u.

    For each year j (group of blocks_per_year consecutive blocks),
        p_j(u) = 1 - ∏_{t in year j} G_t(u),
        R_j(u) = 1 / p_j(u).

    Returns:
      dict[u] -> R_j(u) of shape (S, Y_total), Y_total = floor(T_total / blocks_per_year).
    """
    if blocks_per_year < 1:
        raise ValueError("blocks_per_year must be >= 1.")
    S_draws, T_total = mu_all.shape
    Y_total = T_total // blocks_per_year
    T_use = Y_total * blocks_per_year

    sigma_exp = np.repeat(sigma[:, None], T_use, axis=1)
    xi_exp = np.repeat(xi[:, None], T_use, axis=1)
    mu_use = mu_all[:, :T_use]

    out: Dict[float, np.ndarray] = {}
    for u in us:
        u = float(u)
        u_arr = np.full((S_draws, T_use), u)
        F_u = _gev_cdf(u_arr, mu_use, sigma_exp, xi_exp)  # (S, T_use)
        F_y = F_u.reshape(S_draws, Y_total, blocks_per_year)  # (S,Y,B)
        prod_F = np.prod(F_y, axis=2)  # (S,Y)
        p_y = np.clip(1.0 - prod_F, 1e-12, 1.0)
        out[u] = 1.0 / p_y
    return out


# ---------------------------------------------------------------------
# Summaries + plotting
# ---------------------------------------------------------------------
def summarize_ribbon(arr_2d: np.ndarray, level: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = (1 - level) / 2.0
    hi = 1.0 - lo
    return (
        np.quantile(arr_2d, 0.5, axis=0),
        np.quantile(arr_2d, lo, axis=0),
        np.quantile(arr_2d, hi, axis=0),
    )


def plot_return_level_ribbon(
    z: np.ndarray,
    label: str,
    level: float,
    hist_len: int,
    xlab: str,
    title: str,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    z: (S, L) return level trajectory (block or yearly).
    hist_len: number of historical indices (blocks or years) for shading forecast region.
    """
    _, L = z.shape
    ctr, lo, hi = summarize_ribbon(z, level=level)
    t = np.arange(L)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(t, ctr, lw=1.6, label=f"median {label}")
    ax.fill_between(t, lo, hi, alpha=0.25, label=f"{int(round(level * 100))}% band")

    if hist_len < L:
        ax.axvspan(hist_len - 0.5, L - 0.5, color="grey", alpha=0.15, label="forecast")

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel("return level")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[save] {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_return_period_ribbon(
    R: np.ndarray,
    label: str,
    level: float,
    hist_len: int,
    xlab: str,
    title: str,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    R: (S, L) return period trajectory (block or yearly).
    """
    _, L = R.shape
    ctr, lo, hi = summarize_ribbon(R, level=level)
    t = np.arange(L)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(t, ctr, lw=1.6, label=f"median {label}")
    ax.fill_between(t, lo, hi, alpha=0.25, label=f"{int(round(level * 100))}% band")
    ax.set_yscale("log")

    if hist_len < L:
        ax.axvspan(hist_len - 0.5, L - 0.5, color="grey", alpha=0.15, label="forecast")

    ax.set_xlabel(xlab)
    ax.set_ylabel("return period (log scale)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[save] {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute time-varying return levels and return periods from a structural DGEV posterior, "
            "including (i) block-scale 'instantaneous' quantities and (ii) yearly coarse-grained quantities."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to a run directory or directly to posterior.npz. If omitted, searches under --root.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results/simulations/DGEV_NCP_LASSO",
        help="Search root when --target is omitted.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="Forecast horizon H in blocks (H=0 → no forecasting).",
    )
    parser.add_argument(
        "--blocks-per-year",
        type=int,
        default=None,
        help="Override number of blocks per calendar year for coarse-graining (default: meta['period']).",
    )
    parser.add_argument(
        "--rl-N",
        type=str,
        default="20,50,100",
        help="Comma-separated return periods N for return levels (computed both block-scale and yearly).",
    )
    parser.add_argument(
        "--rp-u",
        type=str,
        default="10,20",
        help="Comma-separated thresholds u for return periods (computed both block-scale and yearly).",
    )

    parser.add_argument("--yearly", dest="yearly", action="store_true", help="Compute yearly coarse-grained quantities.")
    parser.add_argument("--block-only", dest="yearly", action="store_false", help="Skip yearly coarse-graining.")
    parser.set_defaults(yearly=True)

    parser.add_argument(
        "--level",
        type=float,
        default=0.90,
        help="Credible band level for ribbons.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for forecasting innovations.",
    )
    parser.add_argument("--show", dest="show", action="store_true", help="Show figures interactively.")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not show figures.")
    parser.set_defaults(show=True)

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Directory to save figures and derived arrays. Default: <run>/returns",
    )

    args = parser.parse_args()

    Ns = _parse_csv_floats(args.rl_N) or []
    us = _parse_csv_floats(args.rp_u) or []

    # Resolve run path via posterior_bundle
    run_path = args.target
    if run_path is None:
        print(f"[info] --target not provided; searching latest posterior under --root={args.root!r} ...")
        run_path = find_latest_run(root=args.root)
        if run_path is None:
            print(f"[error] No 'posterior.npz' found under {args.root!r}. Provide --target or change --root.")
            sys.exit(1)
        print(f"[info] Using latest run: {run_path}")

    bundle = load_posterior(run_path)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "returns")
    _ensure_dir(out_dir)
    print(f"[info] saving outputs to: {out_dir}")

    # Forecast μ, get σ, ξ
    print(f"[info] forecasting horizon H={int(args.horizon)} blocks ...")
    mu_all, sigma, xi = forecast_mu_and_params(
        draws=draws,
        meta=meta,
        horizon=int(args.horizon),
        seed=int(args.seed),
    )
    S_draws, T_total = mu_all.shape
    T_hist = int(draws["mu"].shape[1])

    # Coarse-graining settings
    season_period = int(meta.get("period", 1))
    blocks_per_year = int(args.blocks_per_year) if args.blocks_per_year is not None else season_period
    if blocks_per_year < 1:
        raise ValueError("blocks_per_year must be >= 1.")
    Y_total = T_total // blocks_per_year
    Y_hist = T_hist // blocks_per_year

    # -----------------------------------------------------------------
    # 1) Return levels (block + yearly)
    # -----------------------------------------------------------------
    if Ns:
        # Block
        print(f"[info] computing BLOCK return levels for N ∈ {Ns} ...")
        rl_block = compute_return_levels_block(mu_all, sigma, xi, Ns=Ns)

        for N in Ns:
            N = float(N)
            zN = rl_block[N]
            np.savez_compressed(
                os.path.join(out_dir, f"return_levels_block_N{int(round(N))}.npz"),
                z=zN,
                N=N,
                T_total=T_total,
                T_hist=T_hist,
            )
            plot_return_level_ribbon(
                z=zN,
                label=f"z_N (N={N:g})",
                level=float(args.level),
                hist_len=T_hist,
                xlab="time index (blocks)",
                title=f"Block return level z_N(t), N={N:g}",
                save_path=os.path.join(out_dir, f"return_levels_block_N{int(round(N))}.png"),
                show=bool(args.show),
            )

        # Yearly
        if args.yearly and blocks_per_year >= 1:
            print(f"[info] computing YEARLY return levels for N ∈ {Ns} (blocks_per_year={blocks_per_year}) ...")
            rl_year = compute_return_levels_yearly(
                mu_all, sigma, xi, Ns_year=Ns, blocks_per_year=blocks_per_year
            )

            for N in Ns:
                N = float(N)
                zAnn = rl_year[N]  # (S, Y_total)
                np.savez_compressed(
                    os.path.join(out_dir, f"return_levels_yearly_N{int(round(N))}.npz"),
                    z=zAnn,
                    N=N,
                    blocks_per_year=blocks_per_year,
                    Y_total=Y_total,
                    Y_hist=Y_hist,
                )
                plot_return_level_ribbon(
                    z=zAnn,
                    label=f"z_N^ann (N={N:g})",
                    level=float(args.level),
                    hist_len=Y_hist,
                    xlab="year index",
                    title=f"Annual return level (coarse-grained), N={N:g}",
                    save_path=os.path.join(out_dir, f"return_levels_yearly_N{int(round(N))}.png"),
                    show=bool(args.show),
                )

    # -----------------------------------------------------------------
    # 2) Return periods for thresholds (block + yearly)
    # -----------------------------------------------------------------
    if us:
        print(f"[info] computing BLOCK return periods for thresholds u ∈ {us} ...")
        rp_block = compute_return_periods_block(mu_all, sigma, xi, us=us)

        for u in us:
            u = float(u)
            R_t = rp_block[u]
            np.savez_compressed(
                os.path.join(out_dir, f"return_periods_block_u{u:g}.npz"),
                R=R_t,
                u=u,
                T_total=T_total,
                T_hist=T_hist,
            )
            plot_return_period_ribbon(
                R=R_t,
                label=f"R_t(u={u:g})",
                level=float(args.level),
                hist_len=T_hist,
                xlab="time index (blocks)",
                title=f"Block return period for threshold u={u:g}",
                save_path=os.path.join(out_dir, f"return_periods_block_u{u:g}.png"),
                show=bool(args.show),
            )

        if args.yearly and blocks_per_year >= 1:
            if Y_total <= 0:
                print("[warn] not enough blocks to form yearly aggregation; skipping yearly return periods.")
            else:
                print(f"[info] computing YEARLY return periods for thresholds u ∈ {us} (blocks_per_year={blocks_per_year}) ...")
                rp_year = compute_return_periods_yearly(mu_all, sigma, xi, us=us, blocks_per_year=blocks_per_year)

                for u in us:
                    u = float(u)
                    R_y = rp_year[u]  # (S, Y_total)
                    np.savez_compressed(
                        os.path.join(out_dir, f"return_periods_yearly_u{u:g}.npz"),
                        R=R_y,
                        u=u,
                        blocks_per_year=blocks_per_year,
                        Y_total=Y_total,
                        Y_hist=Y_hist,
                    )
                    plot_return_period_ribbon(
                        R=R_y,
                        label=f"R_j(u={u:g})",
                        level=float(args.level),
                        hist_len=Y_hist,
                        xlab="year index",
                        title=f"Annual return period (coarse-grained) for threshold u={u:g}",
                        save_path=os.path.join(out_dir, f"return_periods_yearly_u{u:g}.png"),
                        show=bool(args.show),
                    )

    print("[done] block + yearly return levels/periods computed and saved.")
