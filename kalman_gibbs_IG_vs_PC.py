# experiments/compare_dlm_ig_vs_pc.py
from __future__ import annotations

import os, math, json, time, csv
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

# --- repo imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from optimization.kalman_gibbs import DLMGibbs as DLM_IG, Priors as PriorsIG, SamplerConfig as CfgIG
from optimization.kalman_gibbs_pc import DLMGibbs as DLM_PC, Priors as PriorsPC, PCPrior, SamplerConfig as CfgPC
from simulator.mean_time_series import Mean_Time_Series

# ===========================
# Utilities: dirs & sampling
# ===========================
def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def sample_inv_gamma(a: float, b: float, size: int, rng: np.random.Generator) -> np.ndarray:
    # If v ~ IG(a,b), then 1/v ~ Gamma(a, scale=1/b)
    g = rng.gamma(shape=a, scale=1.0 / b, size=size)
    return 1.0 / g

def pc_lambda_from_ig(a: float, b: float, alpha_tail: float = 0.05,
                      n:int = 200000, seed: int = 123) -> Tuple[float, float]:
    """
    Calibrate PC prior p(s)=λ exp(-λ s) by matching tail:
      Draw v~IG(a,b), s=sqrt(v); let u be the (1-α) quantile of s.
      Choose λ so that P_PC(s > u) = α  =>  λ = -log(α)/u.
    Returns (λ, u).
    """
    rng = np.random.default_rng(seed)
    v = sample_inv_gamma(a, b, size=n, rng=rng)
    s = np.sqrt(v)
    u = float(np.quantile(s, 1.0 - alpha_tail))
    lam = -math.log(max(alpha_tail, 1e-16)) / max(u, 1e-16)
    return lam, u

def logsigma_normal_from_ig(a_sigma: float, b_sigma: float,
                            n:int = 200000, seed: int = 123) -> Tuple[float, float]:
    """
    Monte-Carlo match Normal(log σ) to IG(a_sigma,b_sigma) on σ².
    """
    rng = np.random.default_rng(seed)
    s2 = sample_inv_gamma(a_sigma, b_sigma, size=n, rng=rng)
    s = np.sqrt(s2)
    ls = np.log(np.clip(s, 1e-20, None))
    return float(ls.mean()), float(ls.std(ddof=1))

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
    Use your Mean_Time_Series to generate a **level-only** stochastic series:
      level_mode='dynamic', trend_mode='none', seasonal_mode='none'
    """
    mts = Mean_Time_Series(
        sigma=setting.sigma_true,
        level_mode="dynamic",
        trend_mode="none",
        seasonal_mode="none",
        period=setting.period,
        q_level=setting.q_level_true,
        q_trend=0.0,            # ignored (trend none)
        q_season=0.0,           # ignored (season none)
        m0_level=0.0, v0_level=1.0,
        m0_trend=0.0, v0_trend=1.0,
        # season lists (required by simulator; unused under 'none')
        m0_season=[0.0]*(setting.period-1),
        v0_season=[1.0]*(setting.period-1),
        start_date=None,
    )

    # Generate T observations
    y = []
    for _ in range(setting.T):
        mts.move(); y.append(mts.measure())
    y = np.asarray(y, float)

    truths = mts.get_truth_paths(as_numpy=True)
    # Simulator records an initial state at t=0 then t=1..T; align to length T:
    mu_T    = truths["mu"][1:1 + setting.T]
    alpha_T = truths["alpha"][1:1 + setting.T]

    return y, mu_T, alpha_T

def run_one(setting: Setting,
            ig_prior: Dict[str, float],
            mcmc: MCMCSpec,
            out_dir: str) -> Dict[str, Dict[str, float]]:

    ensure_dir(out_dir)

    # ----- simulate with your simulator
    y, mu_true, alpha_true = simulate_series(setting)

    # ---------------- IG–conjugate sampler ----------------
    pri_ig = PriorsIG(
        a_sigma=float(ig_prior["a_sigma"]), b_sigma=float(ig_prior["b_sigma"]),
        a_alpha=float(ig_prior["a_alpha"]), b_alpha=float(ig_prior["b_alpha"]),
        a_beta=1.0, b_beta=1.0,   # unused here
        a_gamma=1.0, b_gamma=1.0, # unused here
    )
    cfg_ig = CfgIG(
        n_iter=mcmc.n_iter, burn=mcmc.burn, thin=mcmc.thin,
        random_seed=mcmc.seed, progress=mcmc.progress, progress_every=50,
    )
    mdl_ig = DLM_IG(
        y=y, period=setting.period,
        level_mode="dynamic", trend_mode="none", seasonal_mode="none",
        m0_level=0.0, v0_level=1.0,
        sigma2_init=setting.sigma_true**2,
        q_alpha_init=max(1e-12, setting.q_level_true),
        priors=pri_ig, cfg=cfg_ig
    )
    mdl_ig.set_truth(sigma=setting.sigma_true, Q=np.array([setting.q_level_true]))
    mdl_ig.set_truth_paths(mu=mu_true, alpha=alpha_true)

    t0 = time.time()
    post_ig = mdl_ig.run()
    t_ig = time.time() - t0

    mu_ig = post_ig["mu"]                   # (S,T)
    sig2_ig = post_ig["sigma2"]             # (S,)
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
                        mu_true=mu_true)

    # ---------------- PC–prior sampler ----------------
    # Match Normal(log σ) to IG prior on σ²
    m_logs, s_logs = logsigma_normal_from_ig(ig_prior["a_sigma"], ig_prior["b_sigma"],
                                             seed=setting.seed+11)
    # Match PC λ to IG prior on level sd tail at α=0.05
    lam_alpha, u_match = pc_lambda_from_ig(ig_prior["a_alpha"], ig_prior["b_alpha"],
                                           alpha_tail=0.05, seed=setting.seed+13)

    pri_pc = PriorsPC(
        m_sigma=m_logs, s_sigma=s_logs,
        pc_alpha=PCPrior(lambda_s=lam_alpha, frac=0.10, alpha_prob=0.05),
        pc_beta=PCPrior(lambda_s=None),
        pc_gamma=PCPrior(lambda_s=None),
    )
    cfg_pc = CfgPC(
        n_iter=mcmc.n_iter, burn=mcmc.burn, thin=mcmc.thin,
        random_seed=mcmc.seed, progress=mcmc.progress, progress_every=50,
        step_logsigma=0.15, step_log_s_alpha=0.2,
        adapt_steps=True, adapt_every=25, adapt_until="burn",
        adapt_eta0=0.08, adapt_eta_decay=0.75, adapt_target_1d=0.44
    )
    mdl_pc = DLM_PC(
        y=y, period=setting.period,
        level_mode="dynamic", trend_mode="none", seasonal_mode="none",
        m0_level=0.0, v0_level=1.0,
        sigma2_init=setting.sigma_true**2,
        q_alpha_init=max(1e-12, setting.q_level_true),
        priors=pri_pc, cfg=cfg_pc
    )
    mdl_pc.set_truth(sigma=setting.sigma_true, Q=np.array([setting.q_level_true]))
    mdl_pc.set_truth_paths(mu=mu_true, alpha=alpha_true)

    t0 = time.time()
    post_pc = mdl_pc.run()
    t_pc = time.time() - t0

    mu_pc = post_pc["mu"]                      # (S,T)
    sigma_pc = post_pc["sigma"]                # (S,) already σ
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

    np.savez_compressed(os.path.join(out_dir, "posterior_pc.npz"),
                        y=y, mu_draws=mu_pc, sigma_draws=sigma_pc, qalpha_draws=qalpha_pc,
                        mu_true=mu_true, lam_alpha=lam_alpha, u_match=u_match,
                        m_logs=m_logs, s_logs=s_logs)

    res = {"IG": metrics_ig, "PC": metrics_pc,
           "_truth": {"sigma_true": setting.sigma_true, "q_alpha_true": setting.q_level_true}}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res

# =========================
# Experiment grid runner
# =========================
def main():
    OUT = os.path.join("results", "experiments", "IG_vs_PC_level_only_simulator")
    ensure_dir(OUT)

    # Tiny process-noise grid
    q_grid = [1e-10, 1e-8, 1e-6, 1e-4]
    T = 500
    period = 12   # any >=2 is fine; season='none' so this is calendar only
    sigma_true = 2.0
    reps = 3
    base_seed = 202409

    # IG priors (weak-ish)
    ig_prior = {"a_sigma": 1.0, "b_sigma": 1.0,
                "a_alpha": 1.0, "b_alpha": 1.0}

    # MCMC settings
    mcmc = MCMCSpec(n_iter=3000, burn=1000, thin=2, seed=777, progress=False)

    rows = []
    for q in q_grid:
        for rep in range(reps):
            set_seed = base_seed + 101 * rep + int(1e6 * q) % 997
            setting = Setting(T=T, period=period, sigma_true=sigma_true,
                              q_level_true=q, seed=set_seed)
            tag = f"T{T}_P{period}_sig{sigma_true}_q{q:.0e}_rep{rep+1}"
            out_dir = os.path.join(OUT, tag)
            print(f"\n=== {tag} ===")
            res = run_one(setting, ig_prior, mcmc, out_dir)

            rows.append({
                "tag": tag, "model": "IG", "T": T, "period": period,
                "sigma_true": sigma_true, "q_true": q,
                **res["IG"]
            })
            rows.append({
                "tag": tag, "model": "PC", "T": T, "period": period,
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
        plt.title("RMSE(μ) vs true Q (level-only)"); plt.legend()
        p1 = os.path.join(OUT, "rmse_vs_q.png"); plt.tight_layout(); plt.savefig(p1, dpi=160)

        # Coverage vs q_true
        plt.figure()
        for model, sub in dfg.groupby("model"):
            sub = sub.sort_values("q_true")
            plt.plot(sub["q_true"], sub["cov90_mu"], marker="o", label=model)
        plt.xscale("log"); plt.axhline(0.90, linestyle="--", alpha=0.6)
        plt.xlabel("True Q_level"); plt.ylabel("90% coverage of μ_t")
        plt.title("Coverage(μ) vs true Q (level-only)"); plt.legend()
        p2 = os.path.join(OUT, "coverage_vs_q.png"); plt.tight_layout(); plt.savefig(p2, dpi=160)

        # MLPD vs q_true
        plt.figure()
        for model, sub in dfg.groupby("model"):
            sub = sub.sort_values("q_true")
            plt.plot(sub["q_true"], sub["mlpd"], marker="o", label=model)
        plt.xscale("log"); plt.xlabel("True Q_level"); plt.ylabel("Mean log predictive density")
        plt.title("MLPD vs true Q (level-only)"); plt.legend()
        p3 = os.path.join(OUT, "mlpd_vs_q.png"); plt.tight_layout(); plt.savefig(p3, dpi=160)

        print(f"Saved plots:\n - {p1}\n - {p2}\n - {p3}")
    except Exception as e:
        print(f"Plotting skipped ({e}).")

if __name__ == "__main__":
    main()
