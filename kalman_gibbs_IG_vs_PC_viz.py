# experiments/viz_from_posteriors.py
from __future__ import annotations
import os, json, glob, re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------
# Config / where to look
# -----------------------
# Point ROOT to your run root (folder that contains many run subfolders)
ROOT = os.path.join("results", "experiments", "DLM", "IG_vs_PC_HYPER_20251004_013113")
OUT_DIR = ROOT  # save figures here; change if you want a separate folder

# -----------------------
# Utilities
# -----------------------
def _ensure_dir(p: str) -> None:
    if p and not os.path.exists(p): os.makedirs(p, exist_ok=True)

def _load_run_dirs(root: str) -> List[str]:
    # A run dir is any directory below root that contains both posterior_ig.npz and posterior_pc.npz
    runs = []
    for d in glob.glob(os.path.join(root, "**"), recursive=True):
        if not os.path.isdir(d): continue
        if (os.path.exists(os.path.join(d, "posterior_ig.npz")) and
            os.path.exists(os.path.join(d, "posterior_pc.npz")) and
            os.path.exists(os.path.join(d, "metrics.json"))):
            runs.append(d)
    runs = sorted(runs)
    if not runs:
        raise FileNotFoundError(f"No posterior files found under {root}")
    return runs

def _credible_band(draws: np.ndarray, alpha: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.quantile(draws, alpha/2, axis=0)
    hi = np.quantile(draws, 1.0 - alpha/2, axis=0)
    return lo, hi

def _pit_values(y: np.ndarray, mu_draws: np.ndarray, sigma_draws: np.ndarray) -> np.ndarray:
    """
    Probability Integral Transform using the posterior predictive mixture:
      PIT_t = E_s,mu [ Phi( (y_t - mu)/s ) ]  ≈ average over MCMC draws.
    """
    S, T = mu_draws.shape
    s = sigma_draws.reshape(-1, 1)  # (S,1)
    z = (y.reshape(1, T) - mu_draws) / s
    return np.mean(norm.cdf(z), axis=0)  # (T,)

def _rank_of_truth(true: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """
    Rank histogram: for each t, rank true[t] among S draws.
    Returns integer ranks in 0..S (inclusive bins, S+1 categories).
    """
    S, T = draws.shape
    ranks = np.sum(draws <= true.reshape(1, T), axis=0)  # rank of truth among draws
    return ranks  # length T

def _acf(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """
    Simple unbiased ACF up to max_lag.
    """
    x = np.asarray(x, float)
    n = x.size
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom <= 0 or not np.isfinite(denom):
        return np.zeros(max_lag+1)
    ac = np.empty(max_lag + 1, float)
    for k in range(max_lag + 1):
        num = np.dot(x[:n-k], x[k:])
        ac[k] = num / denom
    return ac

def _ess_and_mcse(x: np.ndarray, max_lag: int = 200) -> Tuple[float, float, float]:
    """
    Effective sample size (ESS) via initial positive sequence:
        tau_int = 1 + 2*sum_{k>=1} rho_k (stop when rho_k<0)
        ESS = N / tau_int
        MCSE ≈ sd(x) * sqrt(tau_int/N)
    Returns (ESS, tau_int, MCSE).
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 3 or np.std(x) == 0:
        return float(n), 1.0, 0.0
    ac = _acf(x, max_lag=max_lag)
    pos_rhos = []
    for k in range(1, len(ac)):
        if not np.isfinite(ac[k]) or ac[k] <= 0:
            break
        pos_rhos.append(ac[k])
    tau_int = 1.0 + 2.0 * float(np.sum(pos_rhos))
    tau_int = max(1.0, tau_int)
    ess = n / tau_int
    sd = float(np.std(x, ddof=1))
    mcse = sd * np.sqrt(tau_int / n)
    return float(ess), float(tau_int), float(mcse)

def _plot_trace_and_acf(x: np.ndarray, title: str, out_path: str,
                        max_lag: int = 200, xlabel_trace: str = "iteration",
                        xscale: Optional[str] = None, log10_in_acf: bool = False) -> None:
    """
    Make side-by-side trace and ACF plots; optionally log10 for trace (handled outside).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    # trace
    axes[0].plot(x, lw=0.8)
    axes[0].set_title(f"{title} (trace)")
    axes[0].set_xlabel(xlabel_trace)
    axes[0].grid(True, alpha=0.3)
    if xscale:
        axes[0].set_yscale(xscale)

    # acf
    x_for_acf = np.log10(np.clip(x, 1e-30, None)) if log10_in_acf else x
    ac = _acf(x_for_acf, max_lag=max_lag)
    axes[1].stem(np.arange(len(ac)), ac, linefmt="C0-", markerfmt="C0o", basefmt="C0-")
    axes[1].set_title(f"{title} (ACF)")
    axes[1].set_xlabel("lag")
    axes[1].set_ylim(-0.1, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

def _load_one_run(run_dir: str) -> Dict:
    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        meta = json.load(f)

    ig = np.load(os.path.join(run_dir, "posterior_ig.npz"))
    pc = np.load(os.path.join(run_dir, "posterior_pc.npz"))

    # normalize keys
    d = {
        "dir": run_dir,
        "tag": os.path.basename(run_dir),
        "sigma_true": meta["_truth"]["sigma_true"],
        "q_true": meta["_truth"]["q_alpha_true"],
        "metrics": meta,  # keep the whole metrics.json for summaries
        "ig": {
            "y": ig["y"], "mu_draws": ig["mu_draws"], "sigma": np.sqrt(ig["sigma2_draws"]),
            "qalpha": ig["qalpha_draws"], "mu_true": ig["mu_true"],
        },
        "pc": {
            "y": pc["y"], "mu_draws": pc["mu_draws"], "sigma": pc["sigma_draws"],
            "qalpha": pc["qalpha_draws"], "mu_true": pc["mu_true"],
        },
    }
    # optional PC hyper draws (lambda)
    if "lambda_alpha_draws" in pc:
        d["pc"]["lambda_alpha"] = pc["lambda_alpha_draws"]
    return d

# -----------------------
# Main: aggregate & plot
# -----------------------
if __name__ == "__main__":
    _ensure_dir(OUT_DIR)
    run_dirs = _load_run_dirs(ROOT)
    runs = [_load_one_run(d) for d in run_dirs]

    # ---------------------------
    # 0) Summaries (print + save)
    # ---------------------------
    # Aggregate metrics.json (already has time_sec, rmse_mu, cov90_mu, mlpd, qalpha_mean/median)
    rows_metrics = []
    for r in runs:
        tag = r["tag"]
        q = r["q_true"]
        for mdl in ("IG", "PC"):
            m = r["metrics"][mdl]
            rows_metrics.append({
                "tag": tag,
                "model": mdl,
                "q_true": float(q),
                "time_sec": float(m.get("time_sec", np.nan)),
                "rmse_mu": float(m.get("rmse_mu", np.nan)),
                "cov90_mu": float(m.get("cov90_mu", np.nan)),
                "mlpd": float(m.get("mlpd", np.nan)),
                "qalpha_mean": float(m.get("qalpha_mean", np.nan)),
                "qalpha_median": float(m.get("qalpha_median", np.nan)),
            })
    dfm = pd.DataFrame(rows_metrics)
    df_summary = (dfm.groupby(["model","q_true"])
                    .agg(time_sec=("time_sec","mean"),
                         rmse_mu=("rmse_mu","mean"),
                         cov90_mu=("cov90_mu","mean"),
                         mlpd=("mlpd","mean"),
                         qalpha_mean=("qalpha_mean","mean"),
                         qalpha_median=("qalpha_median","mean"))
                    .reset_index())
    # Print side-by-side summaries
    print("\n=== Summary by model and q_true ===")
    print(df_summary.to_string(index=False, float_format=lambda z: f"{z:.4g}"))
    df_summary.to_csv(os.path.join(OUT_DIR, "viz_summary_by_model_q.csv"), index=False)

    # Overall model summary
    df_overall = (dfm.groupby(["model"])
                    .agg(time_sec=("time_sec","mean"),
                         rmse_mu=("rmse_mu","mean"),
                         cov90_mu=("cov90_mu","mean"),
                         mlpd=("mlpd","mean"))
                    .reset_index())
    print("\n=== Overall summary by model ===")
    print(df_overall.to_string(index=False, float_format=lambda z: f"{z:.4g}"))
    df_overall.to_csv(os.path.join(OUT_DIR, "viz_summary_by_model.csv"), index=False)

    # ---------------------------
    # 1) Posterior of σ (over runs), IG vs PC
    # ---------------------------
    rows = []
    for r in runs:
        q = r["q_true"]
        for model in ("ig", "pc"):
            sig = np.asarray(r[model]["sigma"]).reshape(-1)
            rows.append(pd.DataFrame({"sigma": sig, "model": model.upper(), "q_true": q}))
    df_sig = pd.concat(rows, ignore_index=True)

    plt.figure(figsize=(8, 5))
    for (model, q), sub in df_sig.groupby(["model", "q_true"]):
        x = np.sort(sub["sigma"].values)
        y = np.linspace(0, 1, len(x), endpoint=False)
        plt.plot(x, y, label=f"{model} | Q={q:.0e}")
    plt.xlabel("σ (posterior draws)"); plt.ylabel("ECDF")
    plt.title("Posterior ECDFs of σ by model and true Q")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "posterior_sigma_ecdf.png"), dpi=180)
    plt.close()

    # ---------------------------
    # 2) Posterior of Q_alpha (level noise) — IG vs PC
    # ---------------------------
    rows = []
    for r in runs:
        q = r["q_true"]
        for model in ("ig", "pc"):
            qd = np.asarray(r[model]["qalpha"]).reshape(-1)
            rows.append(pd.DataFrame({"Q_alpha": qd, "model": model.upper(), "q_true": q}))
    df_Q = pd.concat(rows, ignore_index=True)

    plt.figure(figsize=(8, 5))
    for (model, q), sub in df_Q.groupby(["model", "q_true"]):
        x = np.sort(sub["Q_alpha"].values)
        y = np.linspace(0, 1, len(x), endpoint=False)
        plt.plot(x, y, label=f"{model} | Q={q:.0e}")
    plt.xscale("log")
    plt.xlabel("Q_alpha (posterior draws, log scale)"); plt.ylabel("ECDF")
    plt.title("Posterior ECDFs of Q_alpha")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "posterior_Qalpha_ecdf.png"), dpi=180)
    plt.close()

    # ---------------------------
    # 3) Average credible-band width of μ vs true Q
    # ---------------------------
    rows = []
    for r in runs:
        q = r["q_true"]
        for model in ("ig", "pc"):
            lo, hi = _credible_band(r[model]["mu_draws"], alpha=0.10)
            width = np.mean(hi - lo)
            rows.append({"q_true": q, "model": model.upper(), "width90": width})
    df_w = pd.DataFrame(rows).groupby(["model","q_true"]).mean().reset_index()

    plt.figure(figsize=(7, 4.5))
    for model, sub in df_w.groupby("model"):
        sub = sub.sort_values("q_true")
        plt.plot(sub["q_true"], sub["width90"], marker="o", label=model)
    plt.xscale("log"); plt.xlabel("True Q_level"); plt.ylabel("Mean 90% band width of μ_t")
    plt.title("Credible Band Width vs true Q")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "width_vs_q.png"), dpi=180)
    plt.close()

    # ---------------------------
    # 4) One representative run: μ_t ribbon + truth (both models)
    # ---------------------------
    q_vals = sorted(set(r["q_true"] for r in runs))
    q_pick = q_vals[len(q_vals)//2]
    run_pick = [r for r in runs if np.isclose(r["q_true"], q_pick)][0]

    T = run_pick["ig"]["mu_draws"].shape[1]
    t = np.arange(1, T+1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    for model, color in (("ig", "#1f77b4"), ("pc", "#ff7f0e")):
        mu_draws = run_pick[model]["mu_draws"]
        mu_mean = mu_draws.mean(axis=0)
        lo, hi = _credible_band(mu_draws, alpha=0.10)
        ax.fill_between(t, lo, hi, alpha=0.20, label=f"{model.upper()} 90% band", color=color)
        ax.plot(t, mu_mean, lw=1.5, color=color, label=f"{model.upper()} mean")

    ax.plot(t, run_pick["ig"]["mu_true"], "k--", lw=1.2, label="truth μ_t")
    ax.set_title(f"Posterior μ_t vs truth (q_true={q_pick:.0e})")
    ax.set_xlabel("t"); ax.set_ylabel("μ_t")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"mu_ribbon_example_q{q_pick:.0e}.png"), dpi=180)
    plt.close()

    # ---------------------------
    # 5) PIT histograms (probabilistic calibration)
    # ---------------------------
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, len(q_vals), figsize=(3.2*len(q_vals), 3), sharey=True)
    if len(q_vals) == 1: axes = [axes]
    for ax, q in zip(axes, q_vals):
        sub_runs = [r for r in runs if np.isclose(r["q_true"], q)]
        for model, color in (("ig", "#1f77b4"), ("pc", "#ff7f0e")):
            pits = []
            for r in sub_runs:
                pits.append(_pit_values(r[model]["y"], r[model]["mu_draws"], r[model]["sigma"]))
            pits = np.concatenate(pits)  # over T and reps
            ax.hist(pits, bins=bins, density=True, alpha=0.4, label=model.upper(), edgecolor="none")
        ax.set_title(f"q_true={q:.0e}")
        ax.set_xlabel("PIT")
    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pit_hist_by_q.png"), dpi=180)
    plt.close()

    # ---------------------------
    # 6) Rank histograms for μ (discrete uniform if calibrated)
    # ---------------------------
    fig, axes = plt.subplots(1, len(q_vals), figsize=(3.2*len(q_vals), 3), sharey=True)
    if len(q_vals) == 1: axes = [axes]
    for ax, q in zip(axes, q_vals):
        sub_runs = [r for r in runs if np.isclose(r["q_true"], q)]
        for model, color in (("ig", "#1f77b4"), ("pc", "#ff7f0e")):
            ranks_all = []
            for r in sub_runs:
                ranks = _rank_of_truth(r[model]["mu_true"], r[model]["mu_draws"])
                ranks_all.append(ranks)
            ranks_all = np.concatenate(ranks_all)  # length (#runs*T)
            S = runs[0][model]["mu_draws"].shape[0]
            bins = np.arange(S+2) - 0.5
            ax.hist(ranks_all, bins=bins, density=True, alpha=0.4, label=model.upper(), edgecolor="none")
        ax.set_title(f"q_true={q:.0e}")
        ax.set_xlabel("rank")
    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rank_hist_by_q.png"), dpi=180)
    plt.close()

    # ---------------------------
    # 7) Time-wise RMSE and band width (single run)
    # ---------------------------
    r = run_pick
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    axs = ax if isinstance(ax, np.ndarray) else [ax]
    for i, model in enumerate(("ig","pc")):
        mu = r[model]["mu_draws"]
        mu_mean = mu.mean(axis=0)
        lo, hi = _credible_band(mu, alpha=0.10)
        width = (hi - lo)
        rmse_t = np.sqrt((mu_mean - r[model]["mu_true"])**2)  # pointwise abs err
        axs[0].plot(rmse_t, label=model.upper())
        axs[1].plot(width, label=model.upper())
    axs[0].set_title("Pointwise |μ̂-μ| (proxy for RMSE per t)"); axs[0].set_ylabel("|err|")
    axs[1].set_title("90% band width over time"); axs[1].set_ylabel("width"); axs[1].set_xlabel("t")
    for a in axs: a.grid(True, alpha=0.3); a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"timewise_rmse_width_q{q_pick:.0e}.png"), dpi=180)
    plt.close()

    print("Saved:")
    for f in ["posterior_sigma_ecdf.png","posterior_Qalpha_ecdf.png","width_vs_q.png",
              f"mu_ribbon_example_q{q_pick:.0e}.png","pit_hist_by_q.png","rank_hist_by_q.png",
              f"timewise_rmse_width_q{q_pick:.0e}.png"]:
        print(" -", os.path.join(OUT_DIR, f))

    # ---------------------------
    # 8) Posteriors of Q_alpha per (q_true, rep), IG vs PC + true Q line
    # ---------------------------
    from matplotlib.ticker import FuncFormatter

    # collect per-run info, including repetition parsed from the tag ("..._rep1", etc.)
    run_rows = []
    for r in runs:
        tag = r["tag"]
        m = re.search(r"rep(\d+)", tag)
        rep = int(m.group(1)) if m else 1
        run_rows.append({
            "q_true": float(r["q_true"]),
            "rep": rep,
            "ig_q": np.asarray(r["ig"]["qalpha"]).reshape(-1),
            "pc_q": np.asarray(r["pc"]["qalpha"]).reshape(-1),
        })

    # set up a common log10 range across all runs (robust to outliers)
    all_log_q = []
    for rr in run_rows:
        if rr["ig_q"].size: all_log_q.append(np.log10(np.clip(rr["ig_q"], 1e-20, None)))
        if rr["pc_q"].size: all_log_q.append(np.log10(np.clip(rr["pc_q"], 1e-20, None)))
    log_all = np.concatenate(all_log_q) if len(all_log_q) else np.array([-16, 0], float)
    x_min = float(np.nanpercentile(log_all, 0.5))
    x_max = float(np.nanpercentile(log_all, 99.5))
    bins = np.linspace(x_min, x_max, 60)

    q_vals = sorted({rr["q_true"] for rr in run_rows})
    reps = sorted({rr["rep"] for rr in run_rows})

    fig, axes = plt.subplots(len(q_vals), len(reps),
                            figsize=(3.6*len(reps), 2.8*len(q_vals)),
                            sharex=True, sharey=True)
    if len(q_vals) == 1 and len(reps) == 1:
        axes = np.array([[axes]])
    elif len(q_vals) == 1:
        axes = np.array([axes])

    def _pow10_fmt(x, pos):
        try:
            return f"1e{int(round(x))}"
        except Exception:
            return ""
    fmt = FuncFormatter(_pow10_fmt)

    for i, q in enumerate(q_vals):
        for j, rep in enumerate(reps):
            ax = axes[i, j]
            sub = [rr for rr in run_rows if np.isclose(rr["q_true"], q) and rr["rep"] == rep]
            if not sub:
                ax.set_visible(False)
                continue
            rr = sub[0]
            ig_log = np.log10(np.clip(rr["ig_q"], 1e-20, None))
            pc_log = np.log10(np.clip(rr["pc_q"], 1e-20, None))
            ax.hist(ig_log, bins=bins, density=True, alpha=0.45, label="IG", edgecolor="none")
            ax.hist(pc_log, bins=bins, density=True, alpha=0.45, label="PC", edgecolor="none")
            ax.axvline(np.log10(q), color="k", linestyle="--", linewidth=1.2, label="true Q")
            if i == 0: ax.set_title(f"rep {rep}")
            if j == 0: ax.set_ylabel(f"q_true={q:.0e}")
            ax.xaxis.set_major_formatter(fmt)
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, 0.02, "Q_alpha (log10 scale)", ha="center")
    fig.text(0.02, 0.5, "density", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.02, 0.04, 1.0, 0.94])
    out_path = os.path.join(OUT_DIR, "posterior_Qalpha_by_q_and_rep.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(" -", out_path)

    # ---------------------------
    # 9) TRACE + ACF diagnostics and ESS/MCSE (per run)
    # ---------------------------
    diag_rows = []
    for r in runs:
        tag = r["tag"]; q = float(r["q_true"])
        # Try to parse rep number from tag for nicer labels
        m = re.search(r"rep(\d+)", tag)
        rep = int(m.group(1)) if m else 1

        # Scalars to diagnose: sigma, Q_alpha; (PC) lambda_alpha if available
        for model_key, model_name in (("ig","IG"),("pc","PC")):
            # σ
            sigma = np.asarray(r[model_key]["sigma"]).reshape(-1)
            ess_s, tau_s, mcse_s = _ess_and_mcse(sigma, max_lag=min(200, max(10, len(sigma)//10)))
            diag_rows.append({"tag": tag, "model": model_name, "q_true": q, "rep": rep,
                              "param": "sigma", "mean": float(np.mean(sigma)),
                              "sd": float(np.std(sigma, ddof=1)),
                              "ess": ess_s, "tau_int": tau_s, "mcse": mcse_s})
            out_png = os.path.join(OUT_DIR, f"trace_acf_{model_name}_sigma_{tag}.png")
            _plot_trace_and_acf(sigma, f"{model_name} σ (q={q:.0e}, rep={rep})", out_png, max_lag=200)

            # Q_alpha (draws of variance)
            qalpha = np.asarray(r[model_key]["qalpha"]).reshape(-1)
            if qalpha.size:
                ess_q, tau_q, mcse_q = _ess_and_mcse(np.log10(np.clip(qalpha,1e-30,None)),
                                                     max_lag=min(200, max(10, len(qalpha)//10)))
                diag_rows.append({"tag": tag, "model": model_name, "q_true": q, "rep": rep,
                                  "param": "Q_alpha(log10)", "mean": float(np.mean(qalpha)),
                                  "sd": float(np.std(qalpha, ddof=1)),
                                  "ess": ess_q, "tau_int": tau_q, "mcse": mcse_q})
                out_png = os.path.join(OUT_DIR, f"trace_acf_{model_name}_Qalpha_{tag}.png")
                # trace on log scale for readability
                _plot_trace_and_acf(np.log10(np.clip(qalpha,1e-30,None)),
                                    f"{model_name} log10 Qα (q={q:.0e}, rep={rep})",
                                    out_png, max_lag=200)

            # PC hyper λ (if available)
            if model_key == "pc" and ("lambda_alpha" in r["pc"]):
                lam = np.asarray(r["pc"]["lambda_alpha"]).reshape(-1)
                if lam.size:
                    ess_l, tau_l, mcse_l = _ess_and_mcse(lam, max_lag=min(200, max(10, len(lam)//10)))
                    diag_rows.append({"tag": tag, "model": model_name, "q_true": q, "rep": rep,
                                      "param": "lambda_alpha", "mean": float(np.mean(lam)),
                                      "sd": float(np.std(lam, ddof=1)),
                                      "ess": ess_l, "tau_int": tau_l, "mcse": mcse_l})
                    out_png = os.path.join(OUT_DIR, f"trace_acf_PC_lambda_alpha_{tag}.png")
                    _plot_trace_and_acf(lam, f"PC λ (q={q:.0e}, rep={rep})", out_png, max_lag=200)

    df_diag = pd.DataFrame(diag_rows)
    if not df_diag.empty:
        df_diag.to_csv(os.path.join(OUT_DIR, "diagnostics_trace_acf_ess.csv"), index=False)
        print("\n=== MCMC diagnostics (per run) ===")
        # show median ESS/MCSE by model & parameter
        df_dsum = (df_diag.groupby(["model","param"])
                        .agg(ESS_median=("ess","median"),
                             tau_int_median=("tau_int","median"),
                             MCSE_median=("mcse","median"),
                             mean_of_means=("mean","mean"),
                             mean_of_sds=("sd","mean"))
                        .reset_index())
        print(df_dsum.to_string(index=False, float_format=lambda z: f"{z:.4g}"))
        df_dsum.to_csv(os.path.join(OUT_DIR, "diagnostics_summary.csv"), index=False)

    # ---------------------------
    # 10) Print compact comparative notes
    # ---------------------------
    # Bias and absolute error of Q_alpha mean vs truth, by model & q_true
    comp_rows = []
    for (model, q), sub in dfm.groupby(["model","q_true"]):
        q_true = float(q)
        # we’ll recompute from posteriors for better fidelity
        sub_runs = [r for r in runs if np.isclose(r["q_true"], q_true)]
        qa = []
        for r in sub_runs:
            qa.append(np.asarray(r[model.lower()]["qalpha"]).reshape(-1))
        qa = np.concatenate(qa) if qa else np.array([])
        if qa.size:
            mean_est = float(np.mean(qa))
            med_est  = float(np.median(qa))
            comp_rows.append({
                "model": model, "q_true": q_true,
                "Qalpha_mean": mean_est,
                "Qalpha_median": med_est,
                "bias_mean": mean_est - q_true,
                "rel_err_mean": (mean_est - q_true)/max(q_true, 1e-30)
            })
    df_comp = pd.DataFrame(comp_rows)
    if not df_comp.empty:
        print("\n=== Q_alpha posterior vs truth (pooled over reps) ===")
        print(df_comp.sort_values(["q_true","model"])
                    .to_string(index=False, float_format=lambda z: f"{z:.4g}"))
        df_comp.to_csv(os.path.join(OUT_DIR, "qalpha_vs_truth.csv"), index=False)

    print("\nAll figures and CSVs saved under:", OUT_DIR)
