# %% simulator/uccle_dgev_laplace_forecast.py
from __future__ import annotations

import os
import sys
import math
import json
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Ensure project root on path
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(THIS_DIR, "..")
sys.path.append(PROJECT_ROOT)

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
    return [float(z) for z in s.split(",") if z.strip()]


def _series_root(series: str, freq: str = "Monthly") -> str:
    """
    Map series to Uccle default run roots.

      Monthly:
        TXx → results/uccle/TX/TXx/Monthly/Laplace
        TXn → results/uccle/TX/TXn/Monthly/Laplace
        TNx → results/uccle/TN/TNx/Monthly/Laplace
        TNn → results/uccle/TN/TNn/Monthly/Laplace
    """
    base = os.path.join("results", "uccle")
    mapping = {
        ("TXx", "Monthly"): os.path.join(base, "TX", "TXx", "Monthly", "Laplace"),
        ("TXn", "Monthly"): os.path.join(base, "TX", "TXn", "Monthly", "Laplace"),
        ("TNx", "Monthly"): os.path.join(base, "TN", "TNx", "Monthly", "Laplace"),
        ("TNn", "Monthly"): os.path.join(base, "TN", "TNn", "Monthly", "Laplace"),
        # optionally add Seasonal later if you want:
        ("TXx", "Seasonal"): os.path.join(base, "TX", "TXx", "Seasonal", "Laplace"),
        ("TXn", "Seasonal"): os.path.join(base, "TX", "TXn", "Seasonal", "Laplace"),
        ("TNx", "Seasonal"): os.path.join(base, "TN", "TNx", "Seasonal", "Laplace"),
        ("TNn", "Seasonal"): os.path.join(base, "TN", "TNn", "Seasonal", "Laplace"),
    }
    key = (series, freq)
    if key not in mapping:
        raise ValueError(f"Unknown series/freq: {series}/{freq}")
    return mapping[key]


# ---------------------------------------------------------------------
# GEV helpers (broadcast-safe)
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

    # xi != 0
    t = 1.0 + xi * z
    t_pos = np.where(t > 0.0, t, 1.0)  # placeholder where t<=0
    F1_raw = np.exp(-(t_pos ** (-1.0 / xi)))

    # Support corrections on t<=0
    F1 = np.where(t > 0.0, F1_raw, np.where(xi > 0.0, 0.0, 1.0))

    F = np.where(is_gumbel, F0, F1)
    return np.clip(F, 0.0, 1.0)


def _gev_return_level_block(mu: np.ndarray, sigma: np.ndarray, xi: np.ndarray, N: float) -> np.ndarray:
    """
    Block return level z_N such that P(X <= z_N) = 1 - 1/N.
    """
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    xi = np.asarray(xi, float)

    N_val = float(N)
    p = 1.0 - 1.0 / max(N_val, 1.0 + 1e-12)
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    tol = 1e-8

    z = np.empty_like(mu)
    g = -math.log(-math.log(p))

    mask0 = np.abs(xi) < tol
    mask1 = ~mask0

    if np.any(mask0):
        z[mask0] = mu[mask0] + sigma[mask0] * g

    if np.any(mask1):
        base = -math.log(p)
        z[mask1] = mu[mask1] + (sigma[mask1] / xi[mask1]) * (np.power(base, -xi[mask1]) - 1.0)

    return z


# ---------------------------------------------------------------------
# Seasonal design + rotation (dummy seasonal dynamics)
# ---------------------------------------------------------------------
def _build_season_design(T: int, period: int) -> np.ndarray:
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
    if K <= 0:
        return np.zeros((0, 0))
    R = np.zeros((K, K))
    R[0, :] = -1.0
    if K > 1:
        R[1:, :-1] = np.eye(K - 1)
    return R


# ---------------------------------------------------------------------
# Forecast μ_t (uses saved x-states + meta['layout'])
# ---------------------------------------------------------------------
def _extract_state_indices(layout: List[str], period: int) -> Tuple[int, int, int]:
    """
    Returns:
      idx_alpha, idx_beta, idx_g_start (g1..g_{p-1} contiguous)
    """
    if "alpha" not in layout or "beta" not in layout:
        raise ValueError("meta['layout'] must contain 'alpha' and 'beta'.")

    idx_alpha = layout.index("alpha")
    idx_beta = layout.index("beta")

    g_indices = [i for i, nm in enumerate(layout) if nm.startswith("g")]
    if not g_indices:
        raise ValueError("meta['layout'] must contain seasonal dynamic states 'g1', 'g2', ...")

    idx_g_start = min(g_indices)

    K = max(period - 1, 0)
    if K > 0:
        # only require that at least K positions exist from g_start
        if idx_g_start + K > len(layout):
            raise ValueError(
                f"layout too short for period={period}: need g1..g{K} contiguous starting at {idx_g_start}."
            )

    return idx_alpha, idx_beta, idx_g_start


def forecast_mu_and_params(
    draws: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    horizon: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build μ_t for t=1..T+H.

    Uses:
      alpha_{t+1} = alpha_t + beta_t + eps_alpha
      beta_{t+1}  = beta_t + eps_beta
      gamma_{t+1} = R gamma_t + eps_gamma
      μ_t = alpha_t + g1_t + S_t gamma0
    """
    mu_draws = np.asarray(draws["mu"], float)  # (S,T)
    S_draws, T = mu_draws.shape

    if "x" not in draws or np.asarray(draws["x"]).ndim != 3:
        raise ValueError("draws must contain 'x' with shape (S,T,dim) to forecast states.")
    x_draws = np.asarray(draws["x"], float)

    # static GEV params
    if "sigma" in draws:
        sigma = np.asarray(draws["sigma"], float)
    elif "sigma2" in draws:
        sigma = np.sqrt(np.clip(np.asarray(draws["sigma2"], float), 0.0, None))
    else:
        raise ValueError("draws must contain 'sigma' or 'sigma2'.")

    if "xi" not in draws:
        raise ValueError("draws must contain 'xi'.")
    xi = np.asarray(draws["xi"], float)

    if sigma.shape[0] != S_draws or xi.shape[0] != S_draws:
        raise ValueError("sigma/xi length mismatch with number of posterior draws.")

    period = int(meta.get("period", 1))
    if period < 1:
        raise ValueError("meta['period'] must be >= 1.")
    K_gamma = max(period - 1, 0)

    layout = meta.get("layout")
    if layout is None:
        raise ValueError(
            "meta does not contain 'layout'. Ensure sampler.save_posterior stores it (state names)."
        )
    layout = list(layout)
    idx_alpha, idx_beta, idx_g_start = _extract_state_indices(layout, period)

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

    if K_gamma > 0:
        if "gamma0" not in draws:
            raise ValueError("draws must contain 'gamma0' when period>1.")
        gamma0 = np.asarray(draws["gamma0"], float)  # (S,K)
        if gamma0.shape != (S_draws, K_gamma):
            raise ValueError("gamma0 shape mismatch.")
    else:
        gamma0 = np.zeros((S_draws, 0), float)

    T_total = T + max(int(horizon), 0)
    S_design_all = _build_season_design(T_total, period)  # (T_total,K)
    R_gamma = _season_rotation_matrix(K_gamma)            # (K,K)

    rng = np.random.default_rng(seed)

    mu_all = np.zeros((S_draws, T_total), float)
    mu_all[:, :T] = mu_draws

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

        for h in range(int(horizon)):
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
    mu_all: np.ndarray, sigma: np.ndarray, xi: np.ndarray, Ns: List[float]
) -> Dict[float, np.ndarray]:
    S_draws, T_total = mu_all.shape
    sigma_exp = np.repeat(sigma[:, None], T_total, axis=1)
    xi_exp = np.repeat(xi[:, None], T_total, axis=1)
    out: Dict[float, np.ndarray] = {}
    for N in Ns:
        out[float(N)] = _gev_return_level_block(mu_all, sigma_exp, xi_exp, N=float(N))
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
    Solve for each year j:
        prod_{t in year j} G_t(z) = 1 - 1/N
    """
    if blocks_per_year < 1:
        raise ValueError("blocks_per_year must be >= 1.")

    S_draws, T_total = mu_all.shape
    Y_total = T_total // blocks_per_year
    if Y_total <= 0:
        raise ValueError("Not enough blocks to form a single year.")

    T_use = Y_total * blocks_per_year
    mu_use = mu_all[:, :T_use].reshape(S_draws, Y_total, blocks_per_year)

    sig = sigma[:, None, None]  # (S,1,1)
    xis = xi[:, None, None]     # (S,1,1)

    mu_min = np.min(mu_use, axis=2)  # (S,Y)
    mu_max = np.max(mu_use, axis=2)  # (S,Y)

    lo0 = mu_min - 50.0 * sigma[:, None]
    hi0 = mu_max + 50.0 * sigma[:, None]

    tol = 1e-8
    neg_xi = xi < -tol
    if np.any(neg_xi):
        endpoint_blocks = mu_use[neg_xi, :, :] - (sigma[neg_xi, None, None] / xi[neg_xi, None, None])
        endpoint_year = np.max(endpoint_blocks, axis=2)  # (Sneg, Y)
        hi0[neg_xi, :] = endpoint_year - 1e-10

    def _prod_cdf(z_sy: np.ndarray) -> np.ndarray:
        F = _gev_cdf(z_sy[:, :, None], mu_use, sig, xis)
        return np.prod(F, axis=2)  # (S,Y)

    out: Dict[float, np.ndarray] = {}
    for N in Ns_year:
        N = float(N)
        target = 1.0 - 1.0 / max(N, 1.0 + 1e-12)
        target = float(np.clip(target, 1e-12, 1.0 - 1e-12))

        lo = lo0.copy()
        hi = hi0.copy()

        # Expand hi if needed for xi>=0
        if np.any(~neg_xi):
            for k in range(20):
                prod_hi = _prod_cdf(hi)
                bad = (prod_hi < target) & (~neg_xi)  # (S,Y)
                if not np.any(bad):
                    break
                step = (2.0 ** k) * 50.0
                hi = np.where(bad, hi + step * sigma[:, None], hi)

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            prod_mid = _prod_cdf(mid)
            go_up = prod_mid < target
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)

        out[N] = hi  # (S,Y)

    return out


def compute_return_periods_block(
    mu_all: np.ndarray, sigma: np.ndarray, xi: np.ndarray, us: List[float]
) -> Dict[float, np.ndarray]:
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
    mu_all: np.ndarray, sigma: np.ndarray, xi: np.ndarray, us: List[float], blocks_per_year: int
) -> Dict[float, np.ndarray]:
    if blocks_per_year < 1:
        raise ValueError("blocks_per_year must be >= 1.")
    S_draws, T_total = mu_all.shape
    Y_total = T_total // blocks_per_year
    T_use = Y_total * blocks_per_year

    mu_use = mu_all[:, :T_use]
    sigma_exp = np.repeat(sigma[:, None], T_use, axis=1)
    xi_exp = np.repeat(xi[:, None], T_use, axis=1)

    out: Dict[float, np.ndarray] = {}
    for u in us:
        u = float(u)
        u_arr = np.full((S_draws, T_use), u)
        F_u = _gev_cdf(u_arr, mu_use, sigma_exp, xi_exp)
        F_y = F_u.reshape(S_draws, Y_total, blocks_per_year)
        prod_F = np.prod(F_y, axis=2)
        p_y = np.clip(1.0 - prod_F, 1e-12, 1.0)
        out[u] = 1.0 / p_y
    return out


# ---------------------------------------------------------------------
# Summaries + plotting
# ---------------------------------------------------------------------
def summarize_ribbon(arr_2d: np.ndarray, level: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    return (
        np.quantile(arr_2d, 0.5, axis=0),
        np.quantile(arr_2d, lo, axis=0),
        np.quantile(arr_2d, hi, axis=0),
    )


def plot_ribbon(
    arr: np.ndarray,
    level: float,
    hist_len: int,
    xlab: str,
    ylab: str,
    title: str,
    ylog: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
):
    _, L = arr.shape
    ctr, lo, hi = summarize_ribbon(arr, level=level)
    x = np.arange(L)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x, ctr, lw=1.6, label="median")
    ax.fill_between(x, lo, hi, alpha=0.25, label=f"{int(round(level * 100))}% band")

    if hist_len < L:
        ax.axvspan(hist_len - 0.5, L - 0.5, color="grey", alpha=0.15, label="forecast")

    if ylog:
        ax.set_yscale("log")

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
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
# Main / CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Uccle wrapper: forecast μ_t from a Laplace/NCP DGEV posterior and compute "
            "block + yearly return levels and return periods."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--series", choices=["TXx", "TXn", "TNx", "TNn"], default="TXx")
    parser.add_argument("--freq", choices=["Monthly", "Seasonal"], default="Monthly")

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to a run directory or directly to posterior.npz. If omitted, uses latest under series root.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="Forecast horizon in BLOCKS (months if Monthly; seasons if Seasonal).",
    )
    parser.add_argument(
        "--blocks-per-year",
        type=int,
        default=None,
        help="Override blocks per year for coarse-graining (default: meta['period']).",
    )

    parser.add_argument("--rl-N", type=str, default="20,50,100")
    parser.add_argument("--rp-u", type=str, default="30,35,40")
    parser.add_argument("--level", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--yearly", dest="yearly", action="store_true", help="Compute yearly coarse-grained metrics.")
    parser.add_argument("--block-only", dest="yearly", action="store_false", help="Skip yearly metrics.")
    parser.set_defaults(yearly=True)

    parser.add_argument("--show", dest="show", action="store_true")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory. Default: <run>/forecast_returns",
    )

    args = parser.parse_args()

    Ns = _parse_csv_floats(args.rl_N) or []
    us = _parse_csv_floats(args.rp_u) or []

    # Resolve run path
    run_path = args.target
    if run_path is None:
        root = _series_root(args.series, args.freq)
        print(f"[info] --target not provided; searching latest posterior under: {root}")
        run_path = find_latest_run(root=root)
        if run_path is None:
            print(f"[error] No posterior.npz found under {root!r}.")
            sys.exit(1)
        print(f"[info] Using latest run: {run_path}")

    bundle = load_posterior(run_path)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "forecast_returns")
    _ensure_dir(out_dir)
    with open(os.path.join(out_dir, "forecast_returns_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "series": args.series,
                "freq": args.freq,
                "target": run_path,
                "horizon_blocks": int(args.horizon),
                "Ns": Ns,
                "thresholds_u": us,
                "level": float(args.level),
                "seed": int(args.seed),
                "yearly": bool(args.yearly),
                "blocks_per_year_override": args.blocks_per_year,
            },
            f,
            indent=2,
        )

    # Forecast μ and get σ, ξ
    print(f"[info] forecasting horizon H={int(args.horizon)} blocks ...")
    mu_all, sigma, xi = forecast_mu_and_params(draws=draws, meta=meta, horizon=int(args.horizon), seed=int(args.seed))
    S_draws, T_total = mu_all.shape
    T_hist = int(np.asarray(draws["mu"]).shape[1])

    blocks_per_year = int(args.blocks_per_year) if args.blocks_per_year is not None else int(meta.get("period", 1))
    if blocks_per_year < 1:
        raise ValueError("blocks_per_year must be >= 1.")
    Y_total = T_total // blocks_per_year
    Y_hist = T_hist // blocks_per_year

    print(f"[info] blocks_per_year={blocks_per_year} | T_total={T_total} (hist={T_hist}) | Y_total={Y_total} (hist={Y_hist})")

    # -------------------------
    # Return levels
    # -------------------------
    if Ns:
        print(f"[info] computing BLOCK return levels for N ∈ {Ns} ...")
        rl_block = compute_return_levels_block(mu_all, sigma, xi, Ns=Ns)

        for N in Ns:
            N = float(N)
            z = rl_block[N]
            np.savez_compressed(os.path.join(out_dir, f"return_levels_block_N{int(round(N))}.npz"), z=z, N=N)
            plot_ribbon(
                z, level=float(args.level), hist_len=T_hist,
                xlab="time index (blocks)", ylab="return level",
                title=f"{args.series} {args.freq}: block return level z_N(t), N={N:g}",
                save_path=os.path.join(out_dir, f"return_levels_block_N{int(round(N))}.png"),
                show=bool(args.show),
            )

        if args.yearly and Y_total > 0:
            print(f"[info] computing YEARLY return levels for N ∈ {Ns} (blocks_per_year={blocks_per_year}) ...")
            rl_year = compute_return_levels_yearly(mu_all, sigma, xi, Ns_year=Ns, blocks_per_year=blocks_per_year)

            for N in Ns:
                N = float(N)
                zA = rl_year[N]
                np.savez_compressed(os.path.join(out_dir, f"return_levels_yearly_N{int(round(N))}.npz"), z=zA, N=N)
                plot_ribbon(
                    zA, level=float(args.level), hist_len=Y_hist,
                    xlab="year index", ylab="return level",
                    title=f"{args.series} {args.freq}: annual return level (coarse-grained), N={N:g}",
                    save_path=os.path.join(out_dir, f"return_levels_yearly_N{int(round(N))}.png"),
                    show=bool(args.show),
                )

    # -------------------------
    # Return periods
    # -------------------------
    if us:
        print(f"[info] computing BLOCK return periods for thresholds u ∈ {us} ...")
        rp_block = compute_return_periods_block(mu_all, sigma, xi, us=us)

        for u in us:
            u = float(u)
            R = rp_block[u]
            np.savez_compressed(os.path.join(out_dir, f"return_periods_block_u{u:g}.npz"), R=R, u=u)
            plot_ribbon(
                R, level=float(args.level), hist_len=T_hist,
                xlab="time index (blocks)", ylab="return period (blocks)", ylog=True,
                title=f"{args.series} {args.freq}: block return period R_t(u), u={u:g}",
                save_path=os.path.join(out_dir, f"return_periods_block_u{u:g}.png"),
                show=bool(args.show),
            )

        if args.yearly and Y_total > 0:
            print(f"[info] computing YEARLY return periods for thresholds u ∈ {us} (blocks_per_year={blocks_per_year}) ...")
            rp_year = compute_return_periods_yearly(mu_all, sigma, xi, us=us, blocks_per_year=blocks_per_year)

            for u in us:
                u = float(u)
                RY = rp_year[u]
                np.savez_compressed(os.path.join(out_dir, f"return_periods_yearly_u{u:g}.npz"), R=RY, u=u)
                plot_ribbon(
                    RY, level=float(args.level), hist_len=Y_hist,
                    xlab="year index", ylab="return period (years)", ylog=True,
                    title=f"{args.series} {args.freq}: annual return period (coarse-grained) for u={u:g}",
                    save_path=os.path.join(out_dir, f"return_periods_yearly_u{u:g}.png"),
                    show=bool(args.show),
                )

    print(f"[done] saved forecast returns to: {out_dir}")
