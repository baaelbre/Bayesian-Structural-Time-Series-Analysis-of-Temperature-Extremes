# experiments/compare_dlm_ig_vs_pc.py
from __future__ import annotations

from datetime import datetime
import os, math, json, time, csv
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# --- repo imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Hyperprior versions (IG & PC)
from optimization.kalman_gibbs_hyper import (
    DLMGibbs as DLM_IG,
    Priors   as PriorsIG,
    SamplerConfig as CfgIG,
)
from optimization.kalman_gibbs_pc_hyper import (
    DLMGibbs as DLM_PC,
    Priors   as PriorsPC,
    PCPrior,
    SamplerConfig as CfgPC,
)

from simulator.mean_time_series import Mean_Time_Series

# Optional SciPy (only used for an IG calibration helper)
try:
    from scipy.stats import invgamma
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ===========================
# Utilities
# ===========================
def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _mad_sd(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) / 0.67448975)  # MAD -> sd

def _robust_sigma_from_d1(d1: np.ndarray) -> float:
    # sigma ≈ sd(Δy) / sqrt(2) (robust)
    return _mad_sd(np.asarray(d1, float)) / math.sqrt(2.0) if d1.size else 1.0

def seasonal_diff(y: np.ndarray, period: int) -> np.ndarray:
    if len(y) <= period:
        return np.array([], float)
    return np.asarray(y[period:] - y[:-period], float)

def calibrate_scales_from_y(
    y: np.ndarray,
    period: int,
    denoise_sigma: bool = True,
    ridge_frac: float = 0.05,
    floor_frac_of_y: float = 1e-4,
) -> Dict[str, float]:
    """
    Robust empirical scales for calibration:
      - sigma_hat ~ measurement noise (from Δy)
      - s_alpha   ~ level innovation sd (from Δy, minus noise; ridged)
      - s_beta    ~ trend innovation sd (from Δ²y, minus noise; ridged)
      - s_gamma   ~ seasonal closure sd (from seasonal Δ; minus noise; ridged)
    """
    y = np.asarray(y, float)
    d1 = np.diff(y)
    d2 = np.diff(y, n=2)
    ds = seasonal_diff(y, period)

    sd1 = _mad_sd(d1)
    sd2 = _mad_sd(d2)
    sdg = _mad_sd(ds) if ds.size else sd1

    Sy = max(_mad_sd(y), 1e-12)  # global scale for floors
    eps = floor_frac_of_y * Sy

    if denoise_sigma and d1.size:
        sigma_hat = _robust_sigma_from_d1(d1)

        # subtract iid noise contributions; ridge to avoid zero
        s2_a = max(sd1**2 - 2.0 * sigma_hat**2, ridge_frac * sd1**2)
        s2_b = max(sd2**2 - 6.0 * sigma_hat**2, ridge_frac * sd2**2)
        s2_g = max(sdg**2 - 2.0 * sigma_hat**2, ridge_frac * sdg**2)

        s_alpha = math.sqrt(s2_a)
        s_beta  = math.sqrt(s2_b)
        s_gamma = math.sqrt(s2_g)
    else:
        sigma_hat = sd1 / math.sqrt(2.0) if d1.size else 1.0
        s_alpha, s_beta, s_gamma = sd1, sd2, sdg

    return dict(
        sigma_hat=max(sigma_hat, eps),
        s_alpha=max(s_alpha, eps),
        s_beta=max(s_beta, eps),
        s_gamma=max(s_gamma, eps),
        Sy=Sy
    )

def pc_lambda_from_scale(
    s_hat: float,
    *,
    frac: float = 0.25,
    alpha_tail: float = 0.10,
    u_min: Optional[float] = None,
    y_scale: Optional[float] = None
) -> float:
    """
    PC prior p(s)=λ exp(-λ s) matched by P(s > u) = alpha_tail, u = max(frac*s_hat, u_min).
    """
    if (u_min is None or u_min <= 0) and y_scale is not None:
        u_min = 1e-4 * max(y_scale, 1e-12)
    u = max(frac * max(s_hat, 0.0), (u_min if u_min is not None else 0.0), 1e-12)
    return float(-math.log(max(alpha_tail, 1e-16)) / u)

def ig_params_from_scale_quantile(
    s_hat: float,
    *,
    frac: float = 0.25,
    alpha_tail: float = 0.10,
    a_shape: float = 2.0
) -> Tuple[float, float]:
    """
    Choose IG(a,b) prior on variance v so that P(s > frac*s_hat) = alpha_tail for s = sqrt(v).
    If SciPy not available, fallback to a weak centering.
    """
    u = max(frac * max(s_hat, 0.0), 1e-12)
    u2 = u * u
    if not _HAVE_SCIPY:
        a = float(max(1.1, a_shape))
        b = float((a - 1.0) * u2)
        return a, b
    q0 = float(invgamma.ppf(1.0 - alpha_tail, a_shape, scale=1.0))
    b = float(u2 / max(1e-24, q0))
    return float(a_shape), b

# ==================
# Evaluation metrics
# ==================
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))

def coverage_ratio(true_path: np.ndarray, draws: np.ndarray, cred: float = 0.90) -> float:
    lo = np.quantile(draws, (1.0 - cred)/2.0, axis=0)
    hi = np.quantile(draws, 1.0 - (1.0 - cred)/2.0, axis=0)
    inside = (true_path >= lo) & (true_path <= hi)
    return float(np.mean(inside))

def mean_log_predictive_density(y: np.ndarray, mu_draws: np.ndarray, sig_draws: np.ndarray) -> float:
    S, T = mu_draws.shape
    sig = sig_draws.reshape(-1, 1)
    var = sig**2
    const = -0.5 * np.log(2.0 * np.pi)
    ll = const - 0.5*np.log(var) - 0.5*((y.reshape(1, T) - mu_draws)**2)/var
    m = np.max(ll, axis=0, keepdims=True)
    logmean = m + np.log(np.mean(np.exp(ll - m), axis=0, keepdims=True))
    return float(np.mean(logmean))

# =========================
# One experiment, one run
# =========================
@dataclass
class Setting:
    T: int
    period: int
    sigma_true: float
    q_level_true: float
    seed: int

@dataclass
class MCMCSpec:
    n_iter: int = 3000
    burn: int = 1000
    thin: int = 2
    seed: int = 777
    progress: bool = False

def simulate_series(setting: Setting):
    """
    Level-only stochastic series: level='dynamic', trend='none', season='none'.
    """
    mts = Mean_Time_Series(
        sigma=setting.sigma_true,
        level_mode="dynamic",
        trend_mode="none",
        seasonal_mode="none",
        period=setting.period,
        q_level=setting.q_level_true,
        q_trend=0.0,
        q_season=0.0,
        m0_level=0.0, v0_level=1.0,
        m0_trend=0.0, v0_trend=1.0,
        m0_season=[0.0]*(setting.period-1),
        v0_season=[1.0]*(setting.period-1),
        start_date=None,
    )

    y = []
    for _ in range(setting.T):
        mts.move(); y.append(mts.measure())
    y = np.asarray(y, float)

    truths = mts.get_truth_paths(as_numpy=True)
    mu_T    = truths["mu"][1:1 + setting.T]
    alpha_T = truths["alpha"][1:1 + setting.T]
    return y, mu_T, alpha_T

def run_one(
    setting: Setting,
    # shared (but milder) tail statement for both PC & IG (only used for calibration/log-sigma centering)
    tail_alpha_sigma: float = 0.10,
    tail_frac_sigma: float = 0.25,
    tail_alpha_alpha: float = 0.10,
    tail_frac_alpha: float = 0.25,
    # IG shapes (used only if you pass base a,b — the IG-hyper module may ignore/override)
    ig_a_sigma: float = 2.0,
    ig_a_alpha: float = 2.0,
    mcmc: MCMCSpec = MCMCSpec(),
    out_dir: str = ".",
) -> Dict[str, Dict[str, float]]:

    ensure_dir(out_dir)

    # ----- simulate
    y, mu_true, alpha_true = simulate_series(setting)

    # ----- robust, data-driven calibration
    cal = calibrate_scales_from_y(
        y, period=setting.period,
        denoise_sigma=True, ridge_frac=0.05, floor_frac_of_y=1e-4
    )
    sigma_hat   = cal["sigma_hat"]
    s_alpha_hat = cal["s_alpha"]
    Sy          = cal["Sy"]

    # ================= IG (hyperprior version) =================
    # If your IG-hyper module accepts base a,b, you can seed them using the same tail statement:
    a_sigma, b_sigma = ig_params_from_scale_quantile(
        s_hat=sigma_hat, frac=tail_frac_sigma, alpha_tail=tail_alpha_sigma, a_shape=ig_a_sigma
    )
    a_alpha, b_alpha = ig_params_from_scale_quantile(
        s_hat=s_alpha_hat, frac=tail_frac_alpha, alpha_tail=tail_alpha_alpha, a_shape=ig_a_alpha
    )

    # Build priors; if the hyper module exposes additional hyper-hyper parameters,
    # leave them at defaults (weak) unless you want to pass them explicitly.
    pri_ig = PriorsIG(
        # Base IG parameters are optional if the hyper module samples them anyway.
        # Keeping them here "centers" the initial state; the hyperprior will adapt.
        a_sigma=float(a_sigma), b_sigma=float(b_sigma),
        a_alpha=float(a_alpha), b_alpha=float(b_alpha),
        # beta/gamma unused in this setting; safe defaults
        # If your hyper module uses hyperpriors for these too, it will just ignore them.
    )
    cfg_ig = CfgIG(
        n_iter=mcmc.n_iter, burn=mcmc.burn, thin=mcmc.thin,
        random_seed=mcmc.seed, progress=mcmc.progress, progress_every=50,
    )
    mdl_ig = DLM_IG(
        y=y, period=setting.period,
        level_mode="dynamic", trend_mode="none", seasonal_mode="none",
        m0_level=0.0, v0_level=1.0,
        sigma2_init=max(1e-12, setting.sigma_true**2),
        q_alpha_init=max(1e-12, setting.q_level_true),
        priors=pri_ig, cfg=cfg_ig
    )
    mdl_ig.set_truth(sigma=setting.sigma_true, Q=np.array([setting.q_level_true]))
    mdl_ig.set_truth_paths(mu=mu_true, alpha=alpha_true)

    t0 = time.time()
    post_ig = mdl_ig.run()
    t_ig = time.time() - t0

    mu_ig = post_ig["mu"]
    sig2_ig = post_ig["sigma2"]
    qalpha_ig = post_ig.get("q_alpha", np.full(mu_ig.shape[0], np.nan))

    mu_mean_ig = mu_ig.mean(axis=0)
    metrics_ig = {
        "time_sec": t_ig,
        "rmse_mu": rmse(mu_mean_ig, mu_true),
        "cov90_mu": coverage_ratio(mu_true, mu_ig, 0.90),
        "mlpd": mean_log_predictive_density(y, mu_ig, np.sqrt(sig2_ig)),
        "qalpha_mean": float(np.nanmean(qalpha_ig)),
        "qalpha_median": float(np.nanmedian(qalpha_ig)),
    }

    np.savez_compressed(os.path.join(out_dir, "posterior_ig.npz"),
                        y=y, mu_draws=mu_ig, sigma2_draws=sig2_ig, qalpha_draws=qalpha_ig,
                        mu_true=mu_true,
                        # calibration record
                        a_sigma=a_sigma, b_sigma=b_sigma, a_alpha=a_alpha, b_alpha=b_alpha,
                        sigma_hat=sigma_hat, s_alpha_hat=s_alpha_hat,
                        tail_frac_sigma=tail_frac_sigma, tail_alpha_sigma=tail_alpha_sigma,
                        tail_frac_alpha=tail_frac_alpha, tail_alpha_alpha=tail_alpha_alpha)

    # ================= PC (hyperprior on λ) =================
    # We DO NOT fix λ here (lambda_s=None) so the sampler learns it via Gamma hyperprior.
    lam_alpha_init = pc_lambda_from_scale(
        s_alpha_hat, frac=tail_frac_alpha, alpha_tail=tail_alpha_alpha, y_scale=Sy
    )

    pri_pc = PriorsPC(
        # "uninformative-ish" center for log-sigma: around log(sigma_hat)
        m_sigma=float(np.log(max(1e-12, sigma_hat))),
        s_sigma=1.0,  # broad
        # Leave lambda_s=None to sample λ; the sampler auto-initializes from data.
        pc_alpha=PCPrior(lambda_s=None, frac=tail_frac_alpha, alpha_prob=tail_alpha_alpha,
                         a_lambda=1.0, b_lambda=1.0),
        pc_beta=PCPrior(lambda_s=None),
        pc_gamma=PCPrior(lambda_s=None),
    )
    cfg_pc = CfgPC(
        n_iter=mcmc.n_iter, burn=mcmc.burn, thin=mcmc.thin,
        random_seed=mcmc.seed, progress=mcmc.progress, progress_every=50,
        step_logsigma=0.15, step_log_s_alpha=0.20,
        adapt_steps=True, adapt_every=25, adapt_until="burn",
        adapt_eta0=0.08, adapt_eta_decay=0.75, adapt_target_1d=0.44
    )
    mdl_pc = DLM_PC(
        y=y, period=setting.period,
        level_mode="dynamic", trend_mode="none", seasonal_mode="none",
        m0_level=0.0, v0_level=1.0,
        sigma2_init=max(1e-12, setting.sigma_true**2),
        q_alpha_init=max(1e-12, setting.q_level_true),
        priors=pri_pc, cfg=cfg_pc
    )
    mdl_pc.set_truth(sigma=setting.sigma_true, Q=np.array([setting.q_level_true]))
    mdl_pc.set_truth_paths(mu=mu_true, alpha=alpha_true)

    t0 = time.time()
    post_pc = mdl_pc.run()
    t_pc = time.time() - t0

    mu_pc = post_pc["mu"]
    sigma_pc = post_pc["sigma"]  # already σ
    if "Q" in post_pc and post_pc["Q"].size:
        qalpha_pc = post_pc["Q"][:, 0]
    else:
        qalpha_pc = (post_pc["sd"][:, 0]**2) if ("sd" in post_pc and post_pc["sd"].size) else np.full(mu_pc.shape[0], np.nan)

    mu_mean_pc = mu_pc.mean(axis=0)
    metrics_pc = {
        "time_sec": t_pc,
        "rmse_mu": rmse(mu_mean_pc, mu_true),
        "cov90_mu": coverage_ratio(mu_true, mu_pc, 0.90),
        "mlpd": mean_log_predictive_density(y, mu_pc, sigma_pc),
        "qalpha_mean": float(np.nanmean(qalpha_pc)),
        "qalpha_median": float(np.nanmedian(qalpha_pc)),
    }

    # Keep λ draws if present for quick diagnostics
    lambda_alpha_draws = post_pc.get("lambda_alpha", np.array([]))

    np.savez_compressed(os.path.join(out_dir, "posterior_pc.npz"),
                        y=y, mu_draws=mu_pc, sigma_draws=sigma_pc, qalpha_draws=qalpha_pc,
                        mu_true=mu_true,
                        lambda_alpha_draws=lambda_alpha_draws,
                        lambda_alpha_init=lam_alpha_init,
                        tail_frac_sigma=tail_frac_sigma, tail_alpha_sigma=tail_alpha_sigma,
                        tail_frac_alpha=tail_frac_alpha, tail_alpha_alpha=tail_alpha_alpha,
                        sigma_hat=sigma_hat, s_alpha_hat=s_alpha_hat)

    # minimal calibration diagnostics
    u_alpha = tail_frac_alpha * s_alpha_hat
    lam_str = (f"{np.mean(lambda_alpha_draws):.3g}" if lambda_alpha_draws.size else "learned")
    print(f"[cal] sd(Δy)~{_mad_sd(np.diff(y)):.3g}  σ̂={sigma_hat:.3g}  ŝ_α={s_alpha_hat:.3g}  "
          f"u_α={u_alpha:.3g}  λ_PC≈{lam_str}  (q_true={setting.q_level_true:g})")

    res = {"IG": metrics_ig, "PC": metrics_pc,
           "_truth": {"sigma_true": setting.sigma_true, "q_alpha_true": setting.q_level_true},
           "_calib": {"sigma_hat": sigma_hat, "s_alpha_hat": s_alpha_hat,
                      "ig": {"a_sigma": a_sigma, "b_sigma": b_sigma, "a_alpha": a_alpha, "b_alpha": b_alpha},
                      "pc": {"lambda_alpha_init": lam_alpha_init}}}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res

# =========================
# Experiment grid runner
# =========================
if __name__ == "__main__":
    OUT = os.path.join("results", "experiments", "DLM", "IG_vs_PC_HYPER_" + f"{datetime.now():%Y%m%d_%H%M%S}")
    ensure_dir(OUT)

    # Process-noise grid (level RW1)
    q_grid = [1e-8, 1e-6, 1e-4, 1e-2]
    T = 500
    period = 12
    sigma_true = 2.0
    reps = 3
    base_seed = 202409

    # Shared tail statements (M I L D E R) — only used for calibration/log-sigma center
    TAIL_ALPHA_SIGMA = 0.10
    TAIL_FRAC_SIGMA  = 0.25
    TAIL_ALPHA_ALPHA = 0.10
    TAIL_FRAC_ALPHA  = 0.25

    # IG base shapes (used to seed a,b if your hyper module allows)
    IG_A_SIGMA = 2.0
    IG_A_ALPHA = 2.0

    # MCMC settings
    mcmc = MCMCSpec(n_iter=5000, burn=1000, thin=2, seed=777, progress=True)

    rows = []
    for q in q_grid:
        for rep in range(reps):
            set_seed = base_seed + 101 * rep + int(1e6 * q) % 997
            setting = Setting(T=T, period=period, sigma_true=sigma_true,
                              q_level_true=q, seed=set_seed)
            tag = f"T{T}_P{period}_sig{sigma_true}_q{q:.0e}_rep{rep+1}"
            out_dir = os.path.join(OUT, tag)
            print(f"\n=== {tag} ===")

            res = run_one(
                setting=setting,
                tail_alpha_sigma=TAIL_ALPHA_SIGMA,
                tail_frac_sigma=TAIL_FRAC_SIGMA,
                tail_alpha_alpha=TAIL_ALPHA_ALPHA,
                tail_frac_alpha=TAIL_FRAC_ALPHA,
                ig_a_sigma=IG_A_SIGMA,
                ig_a_alpha=IG_A_ALPHA,
                mcmc=mcmc,
                out_dir=out_dir
            )

            rows.append({
                "tag": tag, "model": "IG-hyper", "T": T, "period": period,
                "sigma_true": sigma_true, "q_true": q,
                **res["IG"]
            })
            rows.append({
                "tag": tag, "model": "PC-hyper", "T": T, "period": period,
                "sigma_true": sigma_true, "q_true": q,
                **res["PC"]
            })

    # Save summary CSV
    csv_path = os.path.join(OUT, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        fields = ["tag","model","T","period","sigma_true","q_true",
                  "time_sec","rmse_mu","cov90_mu","mlpd","qalpha_mean","qalpha_median"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, np.nan) for k in fields})
    print(f"\nSaved: {csv_path}")

    # Quick plots
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        dfg = (df.groupby(["model","q_true"])
                 .agg(rmse_mu=("rmse_mu","mean"),
                      cov90_mu=("cov90_mu","mean"),
                      mlpd=("mlpd","mean"))
                 .reset_index())

        # RMSE vs q_true
        plt.figure()
        for model, sub in dfg.groupby("model"):
            sub = sub.sort_values("q_true")
            plt.plot(sub["q_true"], sub["rmse_mu"], marker="o", label=model)
        plt.xscale("log"); plt.xlabel("True Q_level"); plt.ylabel("RMSE of μ_t")
        plt.title("RMSE(μ) vs true Q (level-only, hyperprior models)"); plt.legend()
        p1 = os.path.join(OUT, "rmse_vs_q.png"); plt.tight_layout(); plt.savefig(p1, dpi=160)

        # Coverage vs q_true
        plt.figure()
        for model, sub in dfg.groupby("model"):
            sub = sub.sort_values("q_true")
            plt.plot(sub["q_true"], sub["cov90_mu"], marker="o", label=model)
        plt.xscale("log"); plt.axhline(0.90, linestyle="--", alpha=0.6)
        plt.xlabel("True Q_level"); plt.ylabel("90% coverage of μ_t")
        plt.title("Coverage(μ) vs true Q (level-only, hyperprior models)"); plt.legend()
        p2 = os.path.join(OUT, "coverage_vs_q.png"); plt.tight_layout(); plt.savefig(p2, dpi=160)

        # MLPD vs q_true
        plt.figure()
        for model, sub in dfg.groupby("model"):
            sub = sub.sort_values("q_true")
            plt.plot(sub["q_true"], sub["mlpd"], marker="o", label=model)
        plt.xscale("log"); plt.xlabel("True Q_level"); plt.ylabel("Mean log predictive density")
        plt.title("MLPD vs true Q (level-only, hyperprior models)"); plt.legend()
        p3 = os.path.join(OUT, "mlpd_vs_q.png"); plt.tight_layout(); plt.savefig(p3, dpi=160)

        print(f"Saved plots:\n - {p1}\n - {p2}\n - {p3}")
    except Exception as e:
        print(f"Plotting skipped ({e}).")
