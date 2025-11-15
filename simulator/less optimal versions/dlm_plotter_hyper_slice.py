# %% simulator/dlm_plotter.py
from __future__ import annotations

"""
DLM Plotter — concise, truth-aware (now includes full m0/P0 plotting)

Assumes posterior bundle (npz) may include:
  true_sigma, true_Q, true_mu_t / true_alpha_t / true_beta_t / true_gamma_t,
  true_m0_alpha, true_m0_beta, true_m0_gamma,
  true_P0_alpha, true_P0_beta, true_P0_gamma,

Also supports legacy path keys: mu_t_truth, alpha_t_truth, ...
"""

import os, re, sys, math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

# allow optimization/ imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from optimization.posterior_bundle import load_posterior, find_latest_run  # noqa


# --------------------- small utils ---------------------
def _ensure_dir(p: str | None):
    os.makedirs(p, exist_ok=True) if p else None


def _san(s: str) -> str:
    """
    Sanitize string to be used as a filename.
    """
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))


def _maybe(d: Dict[str, Any], *ks):
    """
    Return d[k] for the first k in ks that exists in d and is not None; else None.
    """
    for k in ks:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


def _qtiles(x: np.ndarray, lvl: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute median and equal-tailed credible interval."""
    x = np.asarray(x, float)
    a = (1 - lvl) / 2
    b = 1 - a
    return np.quantile(x, 0.5, 0), np.quantile(x, a, 0), np.quantile(x, b, 0)


def _acf(x: np.ndarray, L: int = 200) -> np.ndarray:
    """Compute autocorrelation function up to lag L (inclusive)."""
    x = np.asarray(x, float).ravel()
    if x.size <= 1:
        return np.array([1.0 if x.size == 1 else np.nan])
    x = x - x.mean()
    d = float(x @ x) + 1e-300
    L = min(L, x.size - 1)
    return np.array([(x[: x.size - k] @ x[k:]) / d for k in range(L + 1)], float)


def _ess(x: np.ndarray, L: int = 200) -> float:
    """Estimate effective sample size using initial positive sequence."""
    ac = _acf(x, L)
    if ac.size <= 1 or not np.all(np.isfinite(ac)):
        return float(len(x))
    s = 0.0
    for k in range(1, ac.size):
        if ac[k] <= 0:
            break
        s += 2.0 * ac[k]
    return float(len(x)) / max(1e-12, 1.0 + s)


def _geweke(x: np.ndarray, a: float = 0.1, b: float = 0.5) -> float:
    """Geweke diagnostic z-score (difference of means scaled by variance)."""
    x = np.asarray(x, float).ravel()
    n = x.size
    if n < 8:
        return np.nan
    A, B = max(2, int(a * n)), max(2, int(b * n))
    xa, xb = x[:A], x[-B:]
    va = float(np.var(xa, ddof=1)) / max(1, xa.size)
    vb = float(np.var(xb, ddof=1)) / max(1, xb.size)
    return (float(xa.mean()) - float(xb.mean())) / math.sqrt(max(1e-300, va + vb))


def _layout_idxs(meta: Dict[str, Any], period: int) -> Dict[str, Optional[int]]:
    """
    From meta["layout"], find indices of alpha, beta, and last gamma component.
    If not found, idx_alpha and idx_beta default to None; idx_g_end defaults to last g* or None.
    """
    idx_alpha = idx_beta = idx_g_end = None
    lay = meta.get("layout") or []
    try:
        if "alpha" in lay:
            idx_alpha = lay.index("alpha")
        if "beta" in lay:
            idx_beta = lay.index("beta")
        gnames = [n for n in lay if re.fullmatch(r"g\d+", n)]
        if gnames:
            last = f"g{max(int(s[1:]) for s in gnames)}"
            if last in lay:
                idx_g_end = lay.index(last)
    except Exception:
        pass
    return {"idx_alpha": idx_alpha, "idx_beta": idx_beta, "idx_g_end": idx_g_end}


def _to_Q(draws: Dict[str, Any], w: str) -> Optional[np.ndarray]:
    """
    Extract process variance for component w in {"alpha","beta","gamma"}.
    """
    assert w in {"alpha", "beta", "gamma"}
    if f"Q_{w}" in draws:
        return np.asarray(draws[f"Q_{w}"]).ravel()
    if f"s_{w}" in draws:
        s = np.asarray(draws[f"s_{w}"]).ravel()
        return s * s
    return None


def _uniq_legend(ax):
    """Make legend with unique labels only."""
    h, l = ax.get_legend_handles_labels()
    if l:
        u = dict(zip(l, h))
        ax.legend(u.values(), u.keys(), fontsize=8, loc="best")


# --------------------- truth helpers ---------------------
def _truth_paths(draws: Dict[str, Any]) -> Dict[str, Optional[np.ndarray]]:
    return {
        "mu": _maybe(draws, "true_mu_t", "mu_t_truth"),
        "alpha": _maybe(draws, "true_alpha_t", "alpha_t_truth"),
        "beta": _maybe(draws, "true_beta_t", "beta_t_truth"),
        "gamma": _maybe(draws, "true_gamma_t", "gamma_t_truth"),
    }


def _truth_sigma(draws: Dict[str, Any]) -> Optional[float]:
    v = _maybe(draws, "true_sigma")
    return None if v is None else float(v)


def _truth_Q(draws: Dict[str, Any], comp: str, idxs: Dict[str, Optional[int]]) -> Optional[float]:
    QQ = _maybe(draws, "true_Q")
    if QQ is None:
        return None
    QQ = np.asarray(QQ, float)
    j = {"alpha": idxs["idx_alpha"], "beta": idxs["idx_beta"], "gamma": idxs["idx_g_end"]}[comp]
    if j is None:
        if QQ.ndim == 1:
            j = {"alpha": 0, "beta": (1 if QQ.size >= 2 else 0), "gamma": -1}[comp]
        else:
            j = {"alpha": 0, "beta": (1 if QQ.shape[0] >= 2 else 0), "gamma": -1}[comp]
    try:
        val = QQ[j, j] if QQ.ndim == 2 else QQ[j]
        return float(max(0.0, val))
    except Exception:
        return None


def _truth_m0(draws: Dict[str, Any], name: str, period: int) -> Optional[float]:
    if name == "m0_α":
        return _maybe(draws, "true_m0_alpha")
    if name == "m0_β":
        return _maybe(draws, "true_m0_beta")
    m = re.fullmatch(r"m0_γ\[(\d+)\]", name)
    if m:
        v = _maybe(draws, "true_m0_gamma")
        if v is None:
            return None
        v = np.asarray(v, float).ravel()  # expected length p-1
        j = int(m.group(1))
        if j < v.size:
            return float(v[j])
        if j == period - 1 and v.size == period - 1:
            # closure term
            return float(-np.sum(v))
    return None


def _truth_P0(draws: Dict[str, Any], name: str) -> Optional[float]:
    if name == "P0_α":
        return _maybe(draws, "true_P0_alpha")
    if name == "P0_β":
        return _maybe(draws, "true_P0_beta")
    if name == "P0_γ":
        v = _maybe(draws, "true_P0_gamma")
        if v is None:
            return None
        arr = np.asarray(v, float).ravel()
        return float(arr[0]) if arr.size else None
    return None


def _truth_det(draws: Dict[str, Any], name: str, period: int) -> Optional[float]:
    """Truths for deterministic aliases ('α_det','β_det','season_det[j]')."""
    if name == "α_det":
        return _maybe(draws, "true_m0_alpha")
    if name == "β_det":
        return _maybe(draws, "true_m0_beta")
    m = re.fullmatch(r"season_det\[(\d+)\]", name)
    if m:
        v = _maybe(draws, "true_m0_gamma")
        if v is None:
            return None
        v = np.asarray(v, float).ravel()  # length p-1
        j = int(m.group(1))
        if j < v.size:
            return float(v[j])
        if j == period - 1 and v.size == period - 1:
            return float(-np.sum(v))
    return None


# --------------------- plotter ---------------------
class DLMPlotter:
    def __init__(self, draws: Dict[str, np.ndarray], meta: Dict[str, Any], level: float = 0.90):
        self.d, self.meta, self.level = draws, meta, float(level)
        if not (0 < self.level < 1):
            raise ValueError("level in (0,1)")
        self.T = int(meta.get("T") or draws["mu"].shape[1])
        self.period = int(meta.get("period", 12))
        self.band = f"{int(round(self.level * 100))}% band"
        self.y = _maybe(draws, "y")

        paths = _truth_paths(draws)
        self.t_mu = None if paths["mu"] is None else np.asarray(paths["mu"], float)
        self.t_a = None if paths["alpha"] is None else np.asarray(paths["alpha"], float)
        self.t_b = None if paths["beta"] is None else np.asarray(paths["beta"], float)
        self.t_g = None if paths["gamma"] is None else np.asarray(paths["gamma"], float)

        # σ or σ²
        self.sigma = (
            np.asarray(draws["sigma"], float).ravel()
            if "sigma" in draws
            else (
                np.sqrt(np.clip(np.asarray(draws["sigma2"], float), 0, None)).ravel()
                if "sigma2" in draws
                else None
            )
        )
        self.Qa, self.Qb, self.Qg = _to_Q(draws, "alpha"), _to_Q(draws, "beta"), _to_Q(draws, "gamma")
        self.idxs = _layout_idxs(meta, self.period)

    # families
    def _families(self) -> Dict[str, List[Tuple[str, np.ndarray]]]:
        f: Dict[str, List[Tuple[str, np.ndarray]]] = {}

        def add(g, n, a):
            f.setdefault(g, []).append((n, np.asarray(a).ravel()))

        d = self.d
        if self.sigma is not None:
            add("sigma", "σ", self.sigma)
        if self.Qa is not None:
            add("Q", "Q_α", self.Qa)
        if self.Qb is not None:
            add("Q", "Q_β", self.Qb)
        if self.Qg is not None:
            add("Q", "Q_γ", self.Qg)
        for k, name in (("lambda_alpha", "λ_α"), ("lambda_beta", "λ_β"), ("lambda_gamma", "λ_γ")):
            if k in d:
                add("lambda", name, d[k])

        # m0 (dynamic or deterministic — both are stored under these names)
        if "m0_alpha" in d:
            add("m0", "m0_α", d["m0_alpha"])
        if "m0_beta" in d:
            add("m0", "m0_β", d["m0_beta"])
        if "m0_gamma" in d:
            mg = np.asarray(d["m0_gamma"])
            if mg.ndim == 2:
                for j in range(mg.shape[1]):
                    add("m0", f"m0_γ[{j}]", mg[:, j])

        # P0 (dynamic only; may be absent if deterministic)
        if "P0_alpha" in d:
            add("P0", "P0_α", d["P0_alpha"])
        if "P0_beta" in d:
            add("P0", "P0_β", d["P0_beta"])
        if "P0_gamma" in d:
            add("P0", "P0_γ", d["P0_gamma"])

        # Optional compatibility aliases for users who saved deterministic params separately
        if "m0_alpha_det" in d:
            add("deterministic", "α_det", d["m0_alpha_det"])
        if "m0_beta_det" in d:
            add("deterministic", "β_det", d["m0_beta_det"])
        if "season_det" in d:
            S = np.asarray(d["season_det"])
            if S.ndim == 2:
                for j in range(S.shape[1]):
                    add("season_det", f"season_det[{j}]", S[:, j])
        return f

    # grouped figs
    def _fig_traces(self, fam, items, outdir, show, L):
        if not items:
            return None
        R = len(items)
        fig, axs = plt.subplots(R, 2, figsize=(12, 3.0 * R), squeeze=False)
        for r, (name, s) in enumerate(items):
            s = np.asarray(s).ravel()
            ac = _acf(s, L)
            ess = _ess(s, L)
            gz = _geweke(s)
            axs[r, 0].plot(s, lw=1)
            axs[r, 0].set_title(f"{name} (trace)")
            axs[r, 0].set_xlabel("iter")
            axs[r, 1].bar(np.arange(ac.size), ac, width=0.9)
            axs[r, 1].set_xlim(-0.5, ac.size - 0.5)
            axs[r, 1].set_title(f"{name} (ACF, ESS≈{ess:.0f}, z≈{gz:.2f})")
            axs[r, 1].set_xlabel("lag")
        fig.suptitle(f"Trace + ACF — {fam}", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = None
        if outdir:
            _ensure_dir(outdir)
            path = os.path.join(outdir, f"traces_acf__{_san(fam)}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        plt.show() if show else plt.close(fig)
        return path

    def _fig_posts(self, fam, items, outdir, show):
        if not items:
            return None
        C = 2 if len(items) >= 4 else 1
        R = int(np.ceil(len(items) / C))
        fig, axs = plt.subplots(R, C, figsize=(6 * C + 1, 2.8 * R), squeeze=False)
        for k, (name, s) in enumerate(items):
            r, c = divmod(k, C)
            ax = axs[r, c]
            s = np.asarray(s).ravel()
            ax.hist(s, bins=40, density=True, alpha=0.85, label=name)
            ax.axvline(float(s.mean()), ls="--", lw=1.0, label="mean")
            ax.axvline(float(np.median(s)), ls=":", lw=1.0, label="median")
            # truths
            tv = None
            if name == "σ":
                tv = _truth_sigma(self.d)
            elif name == "Q_α":
                tv = _truth_Q(self.d, "alpha", self.idxs)
            elif name == "Q_β":
                tv = _truth_Q(self.d, "beta", self.idxs)
            elif name == "Q_γ":
                tv = _truth_Q(self.d, "gamma", self.idxs)
            elif name.startswith("m0_"):
                tv = _truth_m0(self.d, name, self.period)
            elif name.startswith("P0_"):
                tv = _truth_P0(self.d, name)
            elif name in {"α_det", "β_det"} or name.startswith("season_det"):
                tv = _truth_det(self.d, name, self.period)
            if tv is not None and np.isfinite(tv):
                ax.axvline(float(tv), color="k", lw=1.6, ls="-", label="truth")
            ax.set_title(name)
            _uniq_legend(ax)
        for k in range(len(items), R * C):
            r, c = divmod(k, C)
            axs[r, c].axis("off")
        fig.suptitle(f"Posteriors — {fam}", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = None
        if outdir:
            _ensure_dir(outdir)
            path = os.path.join(outdir, f"posteriors__{_san(fam)}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        plt.show() if show else plt.close(fig)
        return path

    # overview
    def figure_overview(self, save_dir: Optional[str] = None, fname_prefix="overview", show=True):
        mu = np.asarray(self.d["mu"], float)
        ctr, lo, hi = _qtiles(mu, self.level)
        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        axs = axs.ravel()
        t = np.arange(self.T)
        axs[0].plot(ctr, lw=1.6, label="μ median")
        axs[0].fill_between(t, lo, hi, alpha=0.25, label=self.band)
        if self.y is not None and len(self.y) == self.T:
            axs[0].plot(self.y, lw=1.0, alpha=0.6, label="y")
        if self.t_mu is not None and len(self.t_mu) == self.T:
            axs[0].plot(self.t_mu, lw=1.2, ls="--", label="true μ")
        axs[0].set_title("Posterior μ_t")
        axs[0].legend(loc="upper left")

        if self.sigma is not None:
            axs[1].plot(self.sigma, lw=1)
            axs[1].set_title("trace: σ")
            axs[2].hist(self.sigma, bins=40, density=True)
            es = _ess(self.sigma)
            gz = _geweke(self.sigma)
            ts = _truth_sigma(self.d)
            if ts is not None:
                axs[2].axvline(float(ts), color="k", lw=1.6, label="truth")
            axs[2].set_title(f"posterior: σ (ESS≈{es:.0f}, z≈{gz:.2f})")
            _uniq_legend(axs[2])
        else:
            axs[1].axis("off")
            axs[2].axis("off")

        # process variances (log10)
        ax = axs[3]
        plotted = False
        for Q, label in ((self.Qa, "α"), (self.Qb, "β"), (self.Qg, "γ")):
            if Q is not None:
                ax.hist(np.log10(np.clip(Q, 1e-20, None)), bins=40, density=True, alpha=0.55, label=f"log10 Q[{label}]")
                plotted = True
        for comp, tag in (("alpha", "α"), ("beta", "β"), ("gamma", "γ")):
            q = _truth_Q(self.d, comp, self.idxs)
            if q is not None and q > 0:
                ax.axvline(np.log10(float(q)), lw=1.6, color="k", ls="--", label=f"truth Q[{tag}]")
                plotted = True
        (ax.set_title("Process variances (log10)") or _uniq_legend(ax)) if plotted else ax.axis("off")

        # m0 quick view (if any)
        fams = self._families()
        m0_items = fams.get("m0", [])
        ax = axs[4]
        if m0_items:
            for i, (name, s) in enumerate(m0_items[:2]):
                ax.hist(np.asarray(s).ravel(), bins=40, density=True, alpha=0.55, label=name)
                tv = _truth_m0(self.d, name, self.period)
                if tv is not None:
                    ax.axvline(float(tv), color="k", lw=1.4, label=f"truth {name}")
            ax.set_title("m0 (subset)")
            _uniq_legend(ax)
        else:
            ax.axis("off")

        # P0 quick view (if any)
        ax = axs[5]
        P0_items = fams.get("P0", [])
        if P0_items:
            for i, (name, s) in enumerate(P0_items[:2]):
                ax.hist(np.asarray(s).ravel(), bins=40, density=True, alpha=0.55, label=name)
                tv = _truth_P0(self.d, name)
                if tv is not None:
                    ax.axvline(float(tv), color="k", lw=1.4, label=f"truth {name}")
            ax.set_title("P0 (subset)")
            _uniq_legend(ax)
        else:
            ax.axis("off")

        fig.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            p = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(p, dpi=200, bbox_inches="tight")
            print(f"[save] {p}")
        plt.show() if show else plt.close(fig)

    # states
    def figure_states(self, save_dir: Optional[str] = None, fname_prefix="states", show=True):
        has_x = ("x" in self.d) and getattr(self.d["x"], "ndim", 0) == 3
        get = lambda idx: (self.d["x"][:, :, idx] if (has_x and idx is not None) else None)
        A, B, G = get(self.idxs["idx_alpha"]), get(self.idxs["idx_beta"]), get(self.idxs["idx_g_end"])
        rows = 1 + sum(v is not None for v in (A, B, G))
        fig, axes = plt.subplots(rows, 1, figsize=(12, 3.0 * rows), sharex=True)
        axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
        t = np.arange(self.T)
        r = 0
        mu = np.asarray(self.d["mu"], float)
        c, lo, hi = _qtiles(mu, self.level)
        ax = axes[r]
        if self.y is not None and len(self.y) == self.T:
            ax.plot(self.y, lw=1.0, alpha=0.6, label="y")
        ax.plot(c, lw=1.6, label=r"$\mu$ median")
        ax.fill_between(t, lo, hi, alpha=0.25, label=self.band)
        if self.t_mu is not None and len(self.t_mu) == self.T:
            ax.plot(self.t_mu, lw=1.2, ls="--", label=r"true $\mu$")
        ax.set_title(r"Posterior $\mu_t$")
        ax.legend()
        r += 1
        for arr, lab, truth in ((A, "α", self.t_a), (B, "β", self.t_b), (G, "γ(last)", self.t_g)):
            if arr is None:
                continue
            c, lo, hi = _qtiles(arr, self.level)
            ax = axes[r]
            ax.plot(c, lw=1.6, label=f"{lab} median")
            ax.fill_between(t, lo, hi, alpha=0.25, label=self.band)
            if truth is not None and len(truth) == self.T:
                ax.plot(truth, lw=1.2, ls="--", label=f"true {lab.split('(')[0]}")
            ax.set_title(f"{'Level' if lab=='α' else ('Trend' if lab=='β' else 'Seasonal')} {lab}")
            ax.legend()
            r += 1
        fig.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            p = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(p, dpi=200, bbox_inches="tight")
            print(f"[save] {p}")
        plt.show() if show else plt.close(fig)

    def figure_m0_P0(self, save_dir: Optional[str] = None, fname_prefix="m0_P0", show=True):
        fams = self._families()
        m0_items = fams.get("m0", [])
        P0_items = fams.get("P0", [])

        # Nothing to plot?
        if not m0_items and not P0_items:
            print("[info] no m0/P0 draws to plot.")
            return None

        R = max(len(m0_items), len(P0_items))
        C = 2
        fig, axs = plt.subplots(R, C, figsize=(12, max(2.6 * R, 3.5)), squeeze=False)
        for i in range(R):
            # m0 column (left)
            axL = axs[i, 0]
            if i < len(m0_items):
                name, s = m0_items[i]
                s = np.asarray(s).ravel()
                axL.hist(s, bins=40, density=True, alpha=0.85, label=name)
                axL.axvline(float(np.mean(s)), ls="--", lw=1.0, label="mean")
                axL.axvline(float(np.median(s)), ls=":", lw=1.0, label="median")
                tv = _truth_m0(self.d, name, self.period)
                if tv is not None and np.isfinite(tv):
                    axL.axvline(float(tv), color="k", lw=1.6, label="truth")
                axL.set_title(name)
                _uniq_legend(axL)
            else:
                axL.axis("off")

            # P0 column (right)
            axR = axs[i, 1]
            if i < len(P0_items):
                name, s = P0_items[i]
                s = np.asarray(s).ravel()
                axR.hist(s, bins=40, density=True, alpha=0.85, label=name)
                axR.axvline(float(np.mean(s)), ls="--", lw=1.0, label="mean")
                axR.axvline(float(np.median(s)), ls=":", lw=1.0, label="median")
                tv = _truth_P0(self.d, name)
                if tv is not None and np.isfinite(tv):
                    axR.axvline(float(tv), color="k", lw=1.6, label="truth")
                axR.set_title(name)
                _uniq_legend(axR)
            else:
                axR.axis("off")

        fig.suptitle("Posteriors — m0 (left) and P0 (right)", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = None
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        plt.show() if show else plt.close(fig)
        return path

    # grouped APIs
    def figure_traces_grouped_all(self, save_dir: Optional[str] = None, show=False, max_lag: int = 200):
        outs = []
        fams = self._families()
        for fam, items in fams.items():
            p = self._fig_traces(fam, items, save_dir, show, max_lag)
            outs.append(p) if p else None
        if not outs:
            print("[warn] no parameter families for trace+ACF.")
        return outs

    def figure_posteriors_grouped_all(self, save_dir: Optional[str] = None, show=False):
        outs = []
        fams = self._families()
        for fam, items in fams.items():
            p = self._fig_posts(fam, items, save_dir, show)
            outs.append(p) if p else None
        if not outs:
            print("[warn] no parameter families for posterior histograms.")
        return outs

    # quick
    def quick_report(self, save_dir: Optional[str] = None, fname_prefix="quick_report", show=True):
        mu = np.asarray(self.d["mu"], float)
        c, lo, hi = _qtiles(mu, self.level)
        t = np.arange(self.T)
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].plot(c, lw=1.6, label="μ median")
        axs[0].fill_between(t, lo, hi, alpha=0.25, label=self.band)
        if self.t_mu is not None and len(self.t_mu) == self.T:
            axs[0].plot(self.t_mu, lw=1.2, ls="--", label="true μ")
        axs[0].set_title("μ_t")
        axs[0].legend()

        if self.sigma is not None:
            axs[1].hist(self.sigma, bins=40, density=True, label="σ")
            ts = _truth_sigma(self.d)
            if ts is not None:
                axs[1].axvline(float(ts), color="k", lw=1.6, label="truth")
            axs[1].set_title("σ | y")
            _uniq_legend(axs[1])
        else:
            axs[1].axis("off")

        # third panel preference: Q → m0 → P0 (first available)
        fams = self._families()
        panel_done = False
        for comp, series, tag in (("alpha", self.Qa, "Q_α"), ("beta", self.Qb, "Q_β"), ("gamma", self.Qg, "Q_γ")):
            if series is None:
                continue
            axs[2].hist(np.log10(np.clip(series, 1e-20, None)), bins=40, density=True, label=f"log10 {tag}")
            q = _truth_Q(self.d, comp, self.idxs)
            if q is not None and q > 0:
                axs[2].axvline(np.log10(float(q)), color="k", lw=1.6, label="truth")
            axs[2].set_title(f"log10 {tag} | y")
            _uniq_legend(axs[2])
            panel_done = True
            break
        if not panel_done and fams.get("m0"):
            name, s = fams["m0"][0]
            axs[2].hist(np.asarray(s).ravel(), bins=40, density=True, label=name)
            tv = _truth_m0(self.d, name, self.period)
            if tv is not None:
                axs[2].axvline(float(tv), color="k", lw=1.6, label="truth")
            axs[2].set_title(f"{name} | y")
            _uniq_legend(axs[2])
            panel_done = True
        if not panel_done and fams.get("P0"):
            name, s = fams["P0"][0]
            axs[2].hist(np.asarray(s).ravel(), bins=40, density=True, label=name)
            tv = _truth_P0(self.d, name)
            if tv is not None:
                axs[2].axvline(float(tv), color="k", lw=1.6, label="truth")
            axs[2].set_title(f"{name} | y")
            _uniq_legend(axs[2])
            panel_done = True
        if not panel_done:
            axs[2].axis("off")

        fig.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            p = os.path.join(save_dir, f"{fname_prefix}.png")
            fig.savefig(p, dpi=200, bbox_inches="tight")
            print(f"[save] {p}")
        plt.show() if show else plt.close(fig)


# --------------------- CLI ---------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="DLM plotter (concise, truth-aware)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", type=str, default=None, help="Run dir or posterior.npz. If omitted, search --root.")
    p.add_argument("--root", type=str, default="results/simulations/DLM", help="Search root.")
    p.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    p.add_argument("--out", type=str, default=None, help="Save dir (default: <run>/figures)")
    p.add_argument("--skip-overview", default=False, action="store_true")
    p.add_argument("--skip-states", default=False, action="store_true")
    p.add_argument("--skip-quick", default=False, action="store_true")
    p.add_argument("--skip-grouped-traces", default=False, action="store_true")
    p.add_argument("--skip-grouped-post", default=False, action="store_true")
    p.add_argument("--skip-m0p0", default=False, action="store_true", help="Skip the dedicated m0/P0 figure")
    p.add_argument("--max-lag", type=int, default=200, help="ACF/ESS max lag")
    a = p.parse_args()

    run = a.target or find_latest_run(root=a.root)
    if run is None:
        print(f"[error] no posterior.npz under {a.root!r}; provide --target or change --root.")
        sys.exit(1)
    bundle = load_posterior(run)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path
    if "layout" in meta:
        meta = dict(meta)
        meta.update(_layout_idxs(meta, meta.get("period", 12)))
    out_dir = a.out or os.path.join(os.path.dirname(npz_path), "figures")
    _ensure_dir(out_dir)
    print(f"[info] saving to: {out_dir}")

    pl = DLMPlotter(draws=draws, meta=meta, level=float(a.level))
    if not a.skip_overview:
        pl.figure_overview(out_dir, "overview", a.show)
    if not a.skip_states:
        pl.figure_states(out_dir, "states", a.show)
    if not a.skip_quick:
        pl.quick_report(out_dir, "quick_report", a.show)
    if not a.skip_m0p0:
        pl.figure_m0_P0(out_dir, "m0_P0", a.show)
    if not a.skip_grouped_traces:
        pl.figure_traces_grouped_all(out_dir, a.show, int(a.max_lag))
    if not a.skip_grouped_post:
        pl.figure_posteriors_grouped_all(out_dir, a.show)
    print("[done] plots written.")
