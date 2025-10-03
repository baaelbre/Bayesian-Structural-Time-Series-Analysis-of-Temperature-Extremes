# experiments/viz_from_posteriors.py
from __future__ import annotations
import os, json, glob
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------
# Config / where to look
# -----------------------
ROOT = os.path.join("results", "experiments", "DLM", "IG_vs_PC_RW1_20251003_220655")
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
        "ig": {
            "y": ig["y"], "mu_draws": ig["mu_draws"], "sigma": np.sqrt(ig["sigma2_draws"]),
            "qalpha": ig["qalpha_draws"], "mu_true": ig["mu_true"],
        },
        "pc": {
            "y": pc["y"], "mu_draws": pc["mu_draws"], "sigma": pc["sigma_draws"],
            "qalpha": pc["qalpha_draws"], "mu_true": pc["mu_true"],
        },
    }
    return d

# -----------------------
# Main: aggregate & plot
# -----------------------
if __name__ == "__main__":
    _ensure_dir(OUT_DIR)
    run_dirs = _load_run_dirs(ROOT)
    runs = [_load_one_run(d) for d in run_dirs]

    # ---------------------------
    # 1) Posterior of σ (over runs), IG vs PC
    # ---------------------------
    # Build a tidy frame of sigma draws with labels & q_true
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

    # ---------------------------
    # 4) One representative run: μ_t ribbon + truth (both models)
    # ---------------------------
    # pick the median q_true block and the first run in it
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
            # normalize to a histogram on 0..S
            S = runs[0][model]["mu_draws"].shape[0]
            bins = np.arange(S+2) - 0.5
            ax.hist(ranks_all, bins=bins, density=True, alpha=0.4, label=model.upper(), edgecolor="none")
        ax.set_title(f"q_true={q:.0e}")
        ax.set_xlabel("rank")
    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rank_hist_by_q.png"), dpi=180)

    # ---------------------------
    # 7) Time-wise RMSE and band width (single run)
    # ---------------------------
    r = run_pick
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    axs = ax if isinstance(ax, np.ndarray) else [ax]
    for i, model in enumerate(("ig","pc")):
        mu = r[model]["mu_draws"]
        mu_mean = mu.mean(axis=0)
        width = (_credible_band(mu, alpha=0.10)[1] - _credible_band(mu, alpha=0.10)[0])
        rmse_t = np.sqrt((mu_mean - r[model]["mu_true"])**2)  # pointwise abs err
        axs[0].plot(rmse_t, label=model.upper())
        axs[1].plot(width, label=model.upper())
    axs[0].set_title("Pointwise |μ̂-μ| (proxy for RMSE per t)"); axs[0].set_ylabel("|err|")
    axs[1].set_title("90% band width over time"); axs[1].set_ylabel("width"); axs[1].set_xlabel("t")
    for a in axs: a.grid(True, alpha=0.3); a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"timewise_rmse_width_q{q_pick:.0e}.png"), dpi=180)

    print("Saved:")
    for f in ["posterior_sigma_ecdf.png","posterior_Qalpha_ecdf.png","width_vs_q.png",
              f"mu_ribbon_example_q{q_pick:.0e}.png","pit_hist_by_q.png","rank_hist_by_q.png",
              f"timewise_rmse_width_q{q_pick:.0e}.png"]:
        print(" -", os.path.join(OUT_DIR, f))
        
    # ---------------------------
    # 8) Posteriors of Q_alpha per (q_true, rep), IG vs PC + true Q line
    # ---------------------------
    import re
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
    reps = [1, 2, 3]

    fig, axes = plt.subplots(len(q_vals), len(reps),
                            figsize=(3.6*len(reps), 2.8*len(q_vals)),
                            sharex=True, sharey=True)
    if len(q_vals) == 1 and len(reps) == 1:
        axes = np.array([[axes]])
    elif len(q_vals) == 1:
        axes = np.array([axes])

    def _pow10_fmt(x, pos):
        # show ticks like 1e-8, 1e-6, ...
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

            # overlaid density histograms on log10 scale
            ax.hist(ig_log, bins=bins, density=True, alpha=0.45, label="IG", edgecolor="none")
            ax.hist(pc_log, bins=bins, density=True, alpha=0.45, label="PC", edgecolor="none")

            # true Q line (in log10)
            ax.axvline(np.log10(q), color="k", linestyle="--", linewidth=1.2, label="true Q")

            if i == 0:
                ax.set_title(f"rep {rep}")
            if j == 0:
                ax.set_ylabel(f"q_true={q:.0e}")

            ax.xaxis.set_major_formatter(fmt)
            ax.grid(True, alpha=0.25)

    # one shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, 0.02, "Q_alpha (log10 scale)", ha="center")
    fig.text(0.02, 0.5, "density", va="center", rotation="vertical")

    plt.tight_layout(rect=[0.02, 0.04, 1.0, 0.94])
    out_path = os.path.join(OUT_DIR, "posterior_Qalpha_by_q_and_rep.png")
    plt.savefig(out_path, dpi=180)
    print(" -", out_path)

