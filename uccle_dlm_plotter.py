from __future__ import annotations
"""
Uccle DLM Harmonic Plotter (TXm, TNm, Precm; Seasonal / Monthly)
================================================================

This is the *Uccle* wrapper around the official harmonic DLM plotter:

• Same figures, same internals:
  - overview.png
  - states.png
  - traces_acf__*.png
  - posteriors__*.png
  - quick_report.png

• Uccle-specific default roots:
  TXm, Seasonal  → results/uccle/TX/TXm/Seasonal
  TXm, Monthly   → results/uccle/TX/TXm/Monthly
  TNm, Seasonal  → results/uccle/TN/TNm/Seasonal
  TNm, Monthly   → results/uccle/TN/TNm/Monthly
  Precm, Seasonal→ results/uccle/Prec/Precm/Seasonal
  Precm, Monthly → results/uccle/Prec/Precm/Monthly

Usage
-----
# Automatically pick latest run for TXm / Seasonal:
python uccle_dlm_plotter.py --series TXm --freq Seasonal --show

# Latest TNm / Monthly:
python uccle_dlm_plotter.py --series TNm --freq Monthly

# Precm / Seasonal:
python uccle_dlm_plotter.py --series Precm --freq Seasonal

# Or explicitly point to a specific run:
python uccle_dlm_plotter.py --target results/uccle/TX/TXm/Seasonal/TXm_dynamic_dynamic_dynamic_20251116_123456

# Or a specific posterior.npz:
python uccle_dlm_plotter.py --target results/uccle/TX/TXm/Seasonal/.../posterior.npz
"""

import os, re, sys, json, math, argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------#
# Loading helpers
# -----------------------------------------------------------------------------#

def _ensure_dir(p: Optional[str]) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _san(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))


def _maybe(d: Dict[str, Any], *ks):
    for k in ks:
        if isinstance(d, dict) and (k in d) and (d[k] is not None):
            return d[k]
    return None


def _find_latest_npz(root: str) -> Optional[str]:
    """
    Recursively search under `root` for any *.npz file and return the newest one
    by modification time.

    This matches the DLM scripts which store runs as:

      root/<run_tag>/posterior.npz
    """
    best = None
    best_mtime = -1.0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".npz"):
                p = os.path.join(dirpath, f)
                mt = os.path.getmtime(p)
                if mt > best_mtime:
                    best_mtime = mt
                    best = p
    return best


def load_posterior(target_or_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Load npz (arrays) and sibling meta.json if present.

    - If `target_or_dir` is a directory:
        * Try 'posterior.npz' in that dir.
        * Else search recursively for the latest '*.npz' under it.
    - If it's a file, treat it as the npz path.
    """
    npz_path = target_or_dir
    if os.path.isdir(target_or_dir):
        cand = os.path.join(target_or_dir, "posterior.npz")
        if os.path.exists(cand):
            npz_path = cand
        else:
            npz_path = _find_latest_npz(target_or_dir)
            if npz_path is None:
                raise FileNotFoundError(
                    f"No .npz files found under directory {target_or_dir!r}"
                )
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    arrays = dict(np.load(npz_path, allow_pickle=True))
    meta_path = npz_path.replace(".npz", ".meta.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return arrays, meta, npz_path


def _default_root(series: str, freq: str) -> str:
    """
    Default root for given series & frequency.

    series in {TXm, TNm, Precm}
    freq   in {Seasonal, Monthly}

    Layout:
      TXm, Seasonal  → results/uccle/TX/TXm/Seasonal
      TXm, Monthly   → results/uccle/TX/TXm/Monthly
      TNm, Seasonal  → results/uccle/TN/TNm/Seasonal
      TNm, Monthly   → results/uccle/TN/TNm/Monthly
      Precm, Seasonal→ results/uccle/Prec/Precm/Seasonal
      Precm, Monthly → results/uccle/Prec/Precm/Monthly
    """
    base = "results/uccle"
    freq = str(freq)
    if series == "TXm":
        return os.path.join(base, "TX", "TXm", freq)
    if series == "TNm":
        return os.path.join(base, "TN", "TNm", freq)
    if series == "Precm":
        return os.path.join(base, "Prec", "Precm", freq)
    raise ValueError(f"Unknown series {series!r} for default root.")


# -----------------------------------------------------------------------------#
# Stats helpers
# -----------------------------------------------------------------------------#

def _qtiles(x: np.ndarray, lvl: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    a = (1 - lvl) / 2
    b = 1 - a
    return np.quantile(x, 0.5, 0), np.quantile(x, a, 0), np.quantile(x, b, 0)


def _acf(x: np.ndarray, L: int = 200) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    if x.size <= 1:
        return np.array([1.0 if x.size == 1 else np.nan])
    x = x - x.mean()
    d = float(x @ x) + 1e-300
    L = min(L, x.size - 1)
    return np.array([(x[: x.size - k] @ x[k:]) / d for k in range(L + 1)], float)


def _ess(x: np.ndarray, L: int = 200) -> float:
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
    x = np.asarray(x, float).ravel()
    n = x.size
    if n < 8:
        return np.nan
    A, B = max(2, int(a * n)), max(2, int(b * n))
    xa, xb = x[:A], x[-B:]
    va = float(np.var(xa, ddof=1)) / max(1, xa.size)
    vb = float(np.var(xb, ddof=1)) / max(1, xb.size)
    return (float(xa.mean()) - float(xb.mean())) / math.sqrt(max(1e-300, va + vb))


def _uniq_legend(ax):
    h, l = ax.get_legend_handles_labels()
    if l:
        u = dict(zip(l, h))
        ax.legend(u.values(), u.keys(), fontsize=8, loc="best")


# -----------------------------------------------------------------------------#
# Layout & truth helpers
# -----------------------------------------------------------------------------#

def _layout_from_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    lay = meta.get("layout", []) or []
    idx_alpha = lay.index("alpha") if "alpha" in lay else None
    idx_beta = lay.index("beta") if "beta" in lay else None
    # Harmonic pairs are named c1,s1,c2,s2,... and optional "nyq"
    pairs: List[Tuple[int, int]] = []
    k = 1
    while f"c{k}" in lay and f"s{k}" in lay:
        pairs.append((lay.index(f"c{k}"), lay.index(f"s{k}")))
        k += 1
    idx_nyq = lay.index("nyq") if "nyq" in lay else None
    return {"idx_alpha": idx_alpha, "idx_beta": idx_beta, "pairs": pairs, "idx_nyq": idx_nyq, "layout": lay}


def _truth_paths(d: Dict[str, Any]) -> Dict[str, Optional[np.ndarray]]:
    return {
        "mu": _maybe(d, "true_mu_t", "mu_t_truth"),
        "alpha": _maybe(d, "true_alpha_t", "alpha_t_truth"),
        "beta": _maybe(d, "true_beta_t", "beta_t_truth"),
        "gamma": _maybe(d, "true_gamma_t", "gamma_t_truth"),
    }


def _truth_sigma(d: Dict[str, Any]) -> Optional[float]:
    v = _maybe(d, "true_sigma")
    return None if v is None else float(v)


def _to_Q(draws: Dict[str, Any], w: str) -> Optional[np.ndarray]:
    if f"Q_{w}" in draws:
        return np.asarray(draws[f"Q_{w}"]).ravel()
    if f"s_{w}" in draws:
        s = np.asarray(draws[f"s_{w}"]).ravel()
        return s * s
    return None


def _truth_Q(d: Dict[str, Any], comp: str, layout: Dict[str, Any]) -> Optional[float]:
    QQ = _maybe(d, "true_Q")
    if QQ is None:
        return None
    QQ = np.asarray(QQ, float)
    if QQ.ndim == 1:
        # back-compat: [Q_alpha, Q_beta, Q_gamma]
        idx = {"alpha": 0, "beta": 1 if QQ.size > 1 else 0, "gamma": 2 if QQ.size > 2 else -1}.get(comp, -1)
        return float(QQ[idx]) if idx >= 0 else None
    # matrix: pull diagonal for the first relevant coord
    if comp == "alpha":
        j = layout.get("idx_alpha")
    elif comp == "beta":
        j = layout.get("idx_beta")
    else:  # gamma: use first cosine in pairs if present, else nyq
        pairs: List[Tuple[int, int]] = layout.get("pairs", [])
        j = pairs[0][0] if pairs else layout.get("idx_nyq")
    if j is None:
        return None
    try:
        return float(max(0.0, QQ[j, j]))
    except Exception:
        return None


# -----------------------------------------------------------------------------#
# Plotter class (official DLM harmonic plotter)
# -----------------------------------------------------------------------------#

class DLMHarmonicPlotter:
    def __init__(self, draws: Dict[str, Any], meta: Dict[str, Any], level: float = 0.90):
        self.d = draws
        self.meta = meta
        self.level = float(level)
        if not (0 < self.level < 1):
            raise ValueError("level in (0,1)")
        self.T = int(meta.get("T") or draws["mu"].shape[1])
        self.period = int(meta.get("period", 12))
        self.band = f"{int(round(self.level * 100))}% band"
        self.y = _maybe(draws, "y")

        self.layout = _layout_from_meta(meta)
        paths = _truth_paths(draws)
        self.t_mu = None if paths["mu"] is None else np.asarray(paths["mu"], float)
        self.t_a = None if paths["alpha"] is None else np.asarray(paths["alpha"], float)
        self.t_b = None if paths["beta"] is None else np.asarray(paths["beta"], float)
        self.t_g = None if paths["gamma"] is None else np.asarray(paths["gamma"], float)

        self.sigma = (
            np.asarray(draws["sigma"], float).ravel() if "sigma" in draws else (
                np.sqrt(np.clip(np.asarray(draws.get("sigma2", []), float), 0, None)).ravel()
                if "sigma2" in draws else None
            )
        )
        self.Qa, self.Qb, self.Qg = _to_Q(draws, "alpha"), _to_Q(draws, "beta"), _to_Q(draws, "gamma")

    # --------------- figure helpers ---------------

    def _fig_overview(self, save_dir: Optional[str], show: bool) -> Optional[str]:
        mu = np.asarray(self.d["mu"], float)
        ctr, lo, hi = _qtiles(mu, self.level)
        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        axs = axs.ravel()
        t = np.arange(self.T)

        # μ
        axs[0].plot(ctr, lw=1.6, label="μ median")
        axs[0].fill_between(t, lo, hi, alpha=0.25, label=self.band)
        if self.y is not None and len(self.y) == self.T:
            axs[0].plot(self.y, lw=1.0, alpha=0.6, label="y")
        if self.t_mu is not None and len(self.t_mu) == self.T:
            axs[0].plot(self.t_mu, lw=1.2, ls="--", label="true μ")
        axs[0].set_title("Posterior μ_t")
        _uniq_legend(axs[0])

        # σ
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

        # log10 Q
        ax = axs[3]
        plotted = False
        for Q, label in ((self.Qa, "α"), (self.Qb, "β"), (self.Qg, "γ")):
            if Q is not None:
                ax.hist(
                    np.log10(np.clip(Q, 1e-20, None)),
                    bins=40,
                    density=True,
                    alpha=0.55,
                    label=f"log10 Q[{label}]",
                )
                plotted = True
        for comp, tag in (("alpha", "α"), ("beta", "β"), ("gamma", "γ")):
            q = _truth_Q(self.d, comp, self.layout)
            if q is not None and q > 0:
                ax.axvline(np.log10(float(q)), lw=1.6, color="k", ls="--", label=f"truth Q[{tag}]")
                plotted = True
        if plotted:
            ax.set_title("Process variances (log10)")
            _uniq_legend(ax)
        else:
            ax.axis("off")

        # m0 subset
        ax = axs[4]
        plotted = False
        for k, title in (("m0_alpha", "m0_α"), ("m0_beta", "m0_β")):
            if k in self.d:
                s = np.asarray(self.d[k]).ravel()
                ax.hist(s, bins=40, density=True, alpha=0.55, label=title)
                plotted = True
        if plotted:
            ax.set_title("m0 (subset)")
            _uniq_legend(ax)
        else:
            ax.axis("off")

        # P0 subset
        ax = axs[5]
        plotted = False
        for k, title in (("P0_alpha", "P0_α"), ("P0_beta", "P0_β")):
            if k in self.d:
                s = np.asarray(self.d[k]).ravel()
                ax.hist(s, bins=40, density=True, alpha=0.55, label=title)
                plotted = True
        if plotted:
            ax.set_title("P0 (subset)")
            _uniq_legend(ax)
        else:
            ax.axis("off")

        fig.tight_layout()
        path = None
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, "overview.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        plt.show() if show else plt.close(fig)
        return path

    def _fig_states(self, save_dir: Optional[str], show: bool) -> Optional[str]:
        has_x = ("x" in self.d) and getattr(self.d["x"], "ndim", 0) == 3

        def get(idx):
            return (self.d["x"][:, :, idx] if (has_x and idx is not None) else None)

        A = get(self.layout["idx_alpha"]) if self.layout["idx_alpha"] is not None else None
        B = get(self.layout["idx_beta"]) if self.layout["idx_beta"] is not None else None
        # use first cosine coord if available; else Nyquist
        G = None
        if self.layout["pairs"]:
            G = get(self.layout["pairs"][0][0])
        elif self.layout["idx_nyq"] is not None:
            G = get(self.layout["idx_nyq"])

        rows = 1 + sum(v is not None for v in (A, B, G))
        fig, axes = plt.subplots(rows, 1, figsize=(12, 3.0 * rows), sharex=True)
        axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
        t = np.arange(self.T)

        # μ
        mu = np.asarray(self.d["mu"], float)
        c, lo, hi = _qtiles(mu, self.level)
        ax = axes[0]
        if self.y is not None and len(self.y) == self.T:
            ax.plot(self.y, lw=1.0, alpha=0.6, label="y")
        ax.plot(c, lw=1.6, label=r"$\mu$ median")
        ax.fill_between(t, lo, hi, alpha=0.25, label=self.band)
        if self.t_mu is not None and len(self.t_mu) == self.T:
            ax.plot(self.t_mu, lw=1.2, ls="--", label=r"true $\mu$")
        ax.set_title(r"Posterior $\mu_t$")
        _uniq_legend(ax)

        r = 1
        for arr, lab, truth in (
            (A, "α", self.t_a),
            (B, "β", self.t_b),
            (G, "season (cos1/nyq)", self.t_g),
        ):
            if arr is None:
                continue
            c, lo, hi = _qtiles(arr, self.level)
            ax = axes[r]
            ax.plot(c, lw=1.6, label=f"{lab} median")
            ax.fill_between(t, lo, hi, alpha=0.25, label=self.band)
            if truth is not None and len(truth) == self.T and lab != "season (cos1/nyq)":
                ax.plot(truth, lw=1.2, ls="--", label=f"true {lab}")
            ax.set_title("Level α" if lab == "α" else ("Trend β" if lab == "β" else "Seasonal (loaded coord)"))
            _uniq_legend(ax)
            r += 1

        fig.tight_layout()
        path = None
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, "states.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        plt.show() if show else plt.close(fig)
        return path

    def _families(self) -> Dict[str, List[Tuple[str, np.ndarray]]]:
        f: Dict[str, List[Tuple[str, np.ndarray]]] = {}

        def add(g, n, a):
            f.setdefault(g, []).append((n, np.asarray(a).ravel()))

        d = self.d
        if self.sigma is not None:
            add("sigma", "σ", self.sigma)
        for nm, tag in (("alpha", "α"), ("beta", "β"), ("gamma", "γ")):
            Q = _to_Q(d, nm)
            if Q is not None:
                add("Q", f"Q_{tag}", Q)
        if "m0_alpha" in d:
            add("m0", "m0_α", d["m0_alpha"])
        if "m0_beta" in d:
            add("m0", "m0_β", d["m0_beta"])
        if "P0_alpha" in d:
            add("P0", "P0_α", d["P0_alpha"])
        if "P0_beta" in d:
            add("P0", "P0_β", d["P0_beta"])
        if "m0_cos" in d and np.ndim(d["m0_cos"]) == 2:
            mgc = np.asarray(d["m0_cos"])  # (n, K)
            mgs = np.asarray(d.get("m0_sin", np.zeros_like(mgc)))
            K = mgc.shape[1]
            for j in range(K):
                add("m0", f"m0_cos[{j+1}]", mgc[:, j])
                add("m0", f"m0_sin[{j+1}]", mgs[:, j])
        if "P0_harm" in d:
            add("P0", "P0_harm", d["P0_harm"])
        if "m0_nyq" in d and np.size(d["m0_nyq"]) > 0:
            add("m0", "m0_nyq", d["m0_nyq"])
        return f

    def _fig_traces(self, fam, items, save_dir, show, L):
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
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, f"traces_acf__{_san(fam)}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        plt.show() if show else plt.close(fig)
        return path

    def _fig_posts(self, fam, items, save_dir, show):
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
            ax.axvline(float(np.mean(s)), ls="--", lw=1.0, label="mean")
            ax.axvline(float(np.median(s)), ls=":", lw=1.0, label="median")
            # truths where meaningful
            tv = None
            if name == "σ":
                tv = _truth_sigma(self.d)
            elif name in {"Q_α", "Q_β", "Q_γ"}:
                comp = {"Q_α": "alpha", "Q_β": "beta", "Q_γ": "gamma"}[name]
                tv = _truth_Q(self.d, comp, self.layout)
            if (tv is not None) and np.isfinite(tv):
                ax.axvline(float(tv), color="k", lw=1.6, ls="-", label="truth")
            ax.set_title(name)
            _uniq_legend(ax)
        for k in range(len(items), R * C):
            r, c = divmod(k, C)
            axs[r, c].axis("off")
        fig.suptitle(f"Posteriors — {fam}", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = None
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, f"posteriors__{_san(fam)}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        plt.show() if show else plt.close(fig)
        return path

    # --------------- public API ---------------

    def figure_overview(self, save_dir: Optional[str] = None, show: bool = True):
        return self._fig_overview(save_dir, show)

    def figure_states(self, save_dir: Optional[str] = None, show: bool = True):
        return self._fig_states(save_dir, show)

    def figure_traces_grouped_all(self, save_dir: Optional[str] = None, show: bool = False, max_lag: int = 200):
        outs = []
        fams = self._families()
        for fam, items in fams.items():
            p = self._fig_traces(fam, items, save_dir, show, max_lag)
            outs.append(p) if p else None
        if not outs:
            print("[warn] no parameter families for trace+ACF.")
        return outs

    def figure_posteriors_grouped_all(self, save_dir: Optional[str] = None, show: bool = False):
        outs = []
        fams = self._families()
        for fam, items in fams.items():
            p = self._fig_posts(fam, items, save_dir, show)
            outs.append(p) if p else None
        if not outs:
            print("[warn] no parameter families for posterior histograms.")
        return outs

    def quick_report(self, save_dir: Optional[str] = None, show: bool = True):
        mu = np.asarray(self.d["mu"], float)
        c, lo, hi = _qtiles(mu, self.level)
        t = np.arange(self.T)
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        # μ panel
        axs[0].plot(c, lw=1.6, label="μ median")
        axs[0].fill_between(t, lo, hi, alpha=0.25, label=self.band)
        if self.t_mu is not None and len(self.t_mu) == self.T:
            axs[0].plot(self.t_mu, lw=1.2, ls="--", label="true μ")
        axs[0].set_title("μ_t")
        _uniq_legend(axs[0])

        # σ panel
        if self.sigma is not None:
            axs[1].hist(self.sigma, bins=40, density=True, label="σ")
            ts = _truth_sigma(self.d)
            if ts is not None:
                axs[1].axvline(float(ts), color="k", lw=1.6, label="truth")
            axs[1].set_title("σ | y")
            _uniq_legend(axs[1])
        else:
            axs[1].axis("off")

        # third panel preference: Q_γ → Q_α → Q_β, else first m0/P0
        fams = self._families()
        panel_done = False
        for comp, series, tag in (("gamma", self.Qg, "Q_γ"),
                                  ("alpha", self.Qa, "Q_α"),
                                  ("beta", self.Qb, "Q_β")):
            if series is None:
                continue
            axs[2].hist(
                np.log10(np.clip(series, 1e-20, None)),
                bins=40,
                density=True,
                label=f"log10 {tag}",
            )
            q = _truth_Q(self.d, comp, self.layout)
            if q is not None and q > 0:
                axs[2].axvline(np.log10(float(q)), color="k", lw=1.6, label="truth")
            axs[2].set_title(f"log10 {tag} | y")
            _uniq_legend(axs[2])
            panel_done = True
            break
        if not panel_done:
            for fam in ("m0", "P0"):
                items = fams.get(fam, [])
                if items:
                    name, s = items[0]
                    axs[2].hist(np.asarray(s).ravel(), bins=40, density=True, label=name)
                    axs[2].set_title(f"{name} | y")
                    _uniq_legend(axs[2])
                    panel_done = True
                    break
        if not panel_done:
            axs[2].axis("off")

        fig.tight_layout()
        path = None
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, "quick_report.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        plt.show() if show else plt.close(fig)
        return path


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Uccle Harmonic DLM Plotter (TXm/TNm/Precm; Seasonal/Monthly)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Run dir or posterior.npz. If provided, overrides --series/--freq/--root.",
    )
    p.add_argument(
        "--series",
        type=str,
        choices=["TXm", "TNm", "Precm"],
        default="TXm",
        help="Series code when searching by default roots.",
    )
    p.add_argument(
        "--freq",
        type=str,
        choices=["Seasonal", "Monthly"],
        default="Seasonal",
        help="Frequency (Seasonal or Monthly) when searching by default roots.",
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Search root when --target is omitted. "
            "If not given, a default root is built from --series and --freq."
        ),
    )
    p.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    p.add_argument("--out", type=str, default=None, help="Save dir (default: <run>/figures)")
    p.add_argument("--skip-overview", action="store_true", default=False)
    p.add_argument("--skip-states", action="store_true", default=False)
    p.add_argument("--skip-grouped-traces", action="store_true", default=False)
    p.add_argument("--skip-grouped-post", action="store_true", default=False)
    p.add_argument("--skip-quick", action="store_true", default=False)
    p.add_argument("--max-lag", type=int, default=200, help="ACF/ESS max lag")
    a = p.parse_args()

    # ----- pick posterior path -----
    if a.target:
        # explicit path or run dir
        draws, meta, npz_path = load_posterior(a.target)
    else:
        if a.root:
            search_root = a.root
        else:
            search_root = _default_root(a.series, a.freq)

        print(f"[info] searching latest .npz under: {search_root}")
        npz = _find_latest_npz(search_root)
        if npz is None:
            print(
                f"[error] no .npz files under {search_root!r}; "
                f"provide --target or change --root/--series/--freq."
            )
            sys.exit(1)
        draws, meta, npz_path = load_posterior(npz)

    out_dir = a.out or os.path.join(os.path.dirname(npz_path), "figures")
    _ensure_dir(out_dir)
    print(f"[info] using posterior: {npz_path}")
    print(f"[info] saving plots to: {out_dir}")

    pl = DLMHarmonicPlotter(draws=draws, meta=meta, level=float(a.level))
    if not a.skip_overview:
        pl.figure_overview(out_dir, a.show)
    if not a.skip_states:
        pl.figure_states(out_dir, a.show)
    if not a.skip_grouped_traces:
        pl.figure_traces_grouped_all(out_dir, a.show, int(a.max_lag))
    if not a.skip_grouped_post:
        pl.figure_posteriors_grouped_all(out_dir, a.show)
    if not a.skip_quick:
        pl.quick_report(out_dir, a.show)
    print("[done] plots written.")
