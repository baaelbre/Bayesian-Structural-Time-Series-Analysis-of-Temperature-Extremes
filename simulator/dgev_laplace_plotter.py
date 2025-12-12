from __future__ import annotations
"""
DGEV Laplace NCP Plotter — dummy seasonality + lasso
====================================================

Designed for the Laplace–based structural DGEV sampler with seasonal dummies
(`DGEVLaplaceNCP`):

• Works with state layout: [alpha][beta][g1 g2 ... g_{p-1}]
  as stored in meta["layout"] by DGEVLaplaceNCP.save_posterior().
• Uses meta["layout"] to automatically locate alpha/beta/seasonal indices.
• Handles optional truth overlays and minimal calendar info (meta["years"]).
• Plots:
    - overview.png          : μ_t band + σ, ξ, and process variances
    - states.png            : μ_t + key state coordinates (α, β, first seasonal coord)
    - traces_acf__*.png     : grouped trace + ACF (with ESS, Geweke) for σ, ξ, Q, s, lasso, loglike
    - posteriors__*.png     : grouped posterior histograms (with means, medians, truth markers)
    - quick_report.png      : 3-panel diagnostic summary
    - return_levels_N*.png  : time-varying N-year return levels
    - return_periods_u*.png : time-varying return periods for threshold u

Expected posterior keys (subset, as produced by DGEVLaplaceNCP.save_posterior)
-------------------------------------------------------------------------------
Arrays (npz):
  - mu              : (n_kept, T)
  - y               : (T,)
  - x               : (n_kept, T, dim) [optional]
  - sigma           : (n_kept,)
  - xi              : (n_kept,)
  - loglike         : (n_kept,)  [recommended]
  - Q_alpha, Q_beta, Q_gamma : (n_kept,)
  - s_alpha, s_beta, s_gamma : (n_kept,)
  - tau_alpha, tau_beta, tau_gamma : (n_kept,)
  - lambda2         : (n_kept,)

Optional truth overlays:
  - true_mu_t, true_alpha_t, true_beta_t, true_gamma_t
  - true_sigma, true_xi
  - true_Q        : either 1D [Q_alpha, Q_beta, Q_gamma] or full covariance matrix

Meta (JSON, saved next to posterior.npz):
  - T             : length of series
  - dim           : state dimension
  - period        : seasonal period
  - layout        : list of state names, e.g. ["alpha","beta","g1","g2",...]
  - modes         : {"level_mode", "trend_mode", "seasonal_mode"}
  - cfg, priors   : SamplerConfig / Priors dicts (not required for plotting)
  - time_center   : centring constant used in regression step
  - years         : optional dict {"start": <int>} for calendar mapping
"""

import os
import re
import sys
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------#
# I/O helpers                                                                  #
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


def _find_latest_run(root: str) -> Optional[str]:
    """Return path to latest posterior.npz under root (any depth)."""
    best = None
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f == "posterior.npz":
                p = os.path.join(dirpath, f)
                if (best is None) or (os.path.getmtime(p) > os.path.getmtime(best)):
                    best = p
    return best


def load_posterior(target_or_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Load posterior arrays (npz) and sibling metadata (meta.json if present).

    Returns
    -------
    draws : dict
        All arrays from the npz file (converted to a plain dict).
    meta : dict
        Metadata parsed from the JSON file, or {} if missing.
    npz_path : str
        Resolved path to the npz that was loaded.
    """
    npz_path = target_or_dir
    if os.path.isdir(target_or_dir):
        cand = os.path.join(target_or_dir, "posterior.npz")
        npz_path = cand if os.path.exists(cand) else _find_latest_run(target_or_dir)
        if npz_path is None:
            raise FileNotFoundError(f"No posterior.npz under {target_or_dir!r}")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    arrays = dict(np.load(npz_path, allow_pickle=True))
    meta_path = npz_path.replace(".npz", ".meta.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return arrays, meta, npz_path


# -----------------------------------------------------------------------------#
# Simple stats helpers                                                         #
# -----------------------------------------------------------------------------#

def _qtiles(x: np.ndarray, lvl: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute median and symmetric credible band along axis 0.
    """
    x = np.asarray(x, float)
    a = (1.0 - lvl) / 2.0
    b = 1.0 - a
    med = np.quantile(x, 0.5, axis=0)
    lo = np.quantile(x, a, axis=0)
    hi = np.quantile(x, b, axis=0)
    return med, lo, hi


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
    A = max(2, int(a * n))
    B = max(2, int(b * n))
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
# Layout & truth helpers                                                       #
# -----------------------------------------------------------------------------#

def _layout_from_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract indices for alpha, beta and seasonal dummy states from meta["layout"].

    layout is expected to look like:
        ["alpha", "beta", "g1", "g2", ..., "g_{p-1}"]
    """
    lay: List[str] = meta.get("layout", []) or []
    idx_alpha = lay.index("alpha") if "alpha" in lay else None
    idx_beta = lay.index("beta") if "beta" in lay else None
    seasonal_idx: List[int] = [j for j, name in enumerate(lay) if name.startswith("g")]
    return {
        "idx_alpha": idx_alpha,
        "idx_beta": idx_beta,
        "idx_seasonal": seasonal_idx,
        "layout": lay,
    }


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


def _truth_xi(d: Dict[str, Any]) -> Optional[float]:
    v = _maybe(d, "true_xi")
    return None if v is None else float(v)


def _to_Q(draws: Dict[str, Any], w: str) -> Optional[np.ndarray]:
    """
    Return process variance draws for component w in {"alpha","beta","gamma"}.

    Prefers Q_w if present; otherwise squares s_w if available.
    """
    if f"Q_{w}" in draws:
        return np.asarray(draws[f"Q_{w}"]).ravel()
    if f"s_{w}" in draws:
        s = np.asarray(draws[f"s_{w}"]).ravel()
        return s * s
    return None


def _truth_Q(d: Dict[str, Any], comp: str, layout: Dict[str, Any]) -> Optional[float]:
    """
    Extract truth Q for component "alpha"/"beta"/"gamma" from true_Q if available.

    Supports:
      - 1D array [Q_alpha, Q_beta, Q_gamma]
      - 2D covariance matrix (take diagonal corresponding to first relevant coord).
    """
    QQ = _maybe(d, "true_Q")
    if QQ is None:
        return None
    QQ = np.asarray(QQ, float)
    if QQ.ndim == 1:
        idx_map = {"alpha": 0, "beta": 1, "gamma": 2}
        idx = idx_map.get(comp, -1)
        if idx < 0 or idx >= QQ.size:
            return None
        return float(max(0.0, QQ[idx]))
    if QQ.ndim == 2:
        if comp == "alpha":
            j = layout.get("idx_alpha")
        elif comp == "beta":
            j = layout.get("idx_beta")
        else:  # gamma: first seasonal index if any
            seasonal = layout.get("idx_seasonal") or []
            j = seasonal[0] if seasonal else None
        if j is None:
            return None
        try:
            return float(max(0.0, QQ[j, j]))
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------#
# Plotter class                                                                #
# -----------------------------------------------------------------------------#

class DGEVPlotter:
    """
    Plotter for Laplace–based structural DGEV with seasonal dummies and lasso.

    Assumes output format produced by DGEVLaplaceNCP.save_posterior().
    """

    def __init__(self, draws: Dict[str, Any], meta: Dict[str, Any], level: float = 0.90):
        self.d = draws
        self.meta = meta
        self.level = float(level)
        if not (0.0 < self.level < 1.0):
            raise ValueError("level must be in (0,1)")
        # T and basic series
        self.T = int(meta.get("T") or draws["mu"].shape[1])
        self.period = int(meta.get("period", 1))
        self.band = f"{int(round(self.level * 100))}% band"
        self.y = _maybe(draws, "y")

        # layout + truth paths
        self.layout = _layout_from_meta(meta)
        paths = _truth_paths(draws)
        self.t_mu = None if paths["mu"] is None else np.asarray(paths["mu"], float)
        self.t_a = None if paths["alpha"] is None else np.asarray(paths["alpha"], float)
        self.t_b = None if paths["beta"] is None else np.asarray(paths["beta"], float)
        self.t_g = None if paths["gamma"] is None else np.asarray(paths["gamma"], float)

        # GEV parameters
        self.sigma = np.asarray(draws["sigma"], float).ravel() if "sigma" in draws else None
        self.xi = np.asarray(draws["xi"], float).ravel() if "xi" in draws else None

        # process variances (or SDs)
        self.Qa = _to_Q(draws, "alpha")
        self.Qb = _to_Q(draws, "beta")
        self.Qg = _to_Q(draws, "gamma")

    # ---------------- time / calendar helpers ----------------

    def _time_to_year_season(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map t=0..T-1 to (year, season_index) using meta['period'] and meta['years'].

        If meta['years'] is missing or incomplete, fall back to a simple mapping:
          years = 0,0,...,1,1,... based on period.

        Returns
        -------
        years : (T,) array of calendar-ish years
        seasons : (T,) array with values in {0,..,period-1}
        """
        period = max(1, int(self.period))
        years_meta = self.meta.get("years", {}) or {}
        start_year = int(years_meta.get("start", 0))
        idx = np.arange(self.T)
        years = start_year + idx // period
        seasons = idx % period
        return years, seasons

    # ---------------- GEV helpers (CDF, return levels, RPs) ----------------

    def _gev_cdf(self, z: np.ndarray, mu: np.ndarray,
                 sigma: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """
        Vectorised GEV CDF G(z; mu, sigma, xi) with broadcasting.

        Parameters
        ----------
        z    : (..., T)
        mu   : (M, T) or broadcastable
        sigma: (M, 1) or (M, T), broadcastable
        xi   : (M, 1) or (M, T), broadcastable
        """
        z = np.asarray(z, float)
        mu = np.asarray(mu, float)
        sigma = np.asarray(sigma, float)
        xi = np.asarray(xi, float)

        # broadcast shapes
        z, mu, sigma, xi = np.broadcast_arrays(z, mu, sigma, xi)
        sigma = np.clip(sigma, 1e-12, None)

        G = np.zeros_like(z)

        # near-Gumbel
        near0 = np.abs(xi) < 1e-6
        if np.any(near0):
            z_std = (z[near0] - mu[near0]) / sigma[near0]
            G[near0] = np.exp(-np.exp(-z_std))

        # general ξ != 0
        non0 = ~near0
        if np.any(non0):
            t = 1.0 + xi[non0] * (z[non0] - mu[non0]) / sigma[non0]
            valid = t > 0
            xi_non0 = xi[non0]

            # valid t
            if np.any(valid):
                G_non0_valid = np.zeros_like(t)
                G_non0_valid[valid] = np.exp(-np.power(t[valid], -1.0 / xi_non0[valid]))
                G[non0][valid] = G_non0_valid[valid]

            # ξ<0 and invalid t ⇒ G=1
            invalid = ~valid
            if np.any(invalid):
                mask_neg = (xi_non0[invalid] < 0)
                if np.any(mask_neg):
                    idx = np.where(non0)[0]
                    bad_idx = idx[invalid][mask_neg]
                    G.flat[bad_idx] = 1.0

        return G

    def _gev_return_level_block(self, N: float) -> np.ndarray:
        """
        Compute *block-wise* N-year return level series z_N(t) for each posterior draw.

        Returns
        -------
        z : np.ndarray
            Shape (n_draws, T) with return levels per draw and time.
        """
        if self.sigma is None or self.xi is None:
            raise ValueError("Need sigma and xi in draws to compute return levels.")

        mu = np.asarray(self.d["mu"], float)            # (M, T)
        sigma = np.asarray(self.sigma, float)[:, None]  # (M, 1)
        xi = np.asarray(self.xi, float)[:, None]        # (M, 1)

        # p = 1 - 1/N ⇒ CDF at RL, c = -log(p) > 0
        c = -np.log(1.0 - 1.0 / float(N))

        # broadcast to (M,T)
        sigma = np.broadcast_to(sigma, mu.shape)
        xi = np.broadcast_to(xi, mu.shape)

        z = np.empty_like(mu)

        near0 = np.abs(xi) < 1e-6
        non0 = ~near0

        # Gumbel limit: z = mu - sigma * log(c)
        if np.any(near0):
            z[near0] = mu[near0] - sigma[near0] * np.log(c)

        # ξ != 0: z = mu + sigma/ξ * (c^{-ξ} - 1)
        if np.any(non0):
            z[non0] = (
                mu[non0]
                + sigma[non0] / xi[non0]
                * (np.power(c, -xi[non0]) - 1.0)
            )
        return z

    def compute_return_periods(
        self,
        u: float,
        yearly: bool = False,
        season: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute time-varying return periods for a fixed threshold u.

        Parameters
        ----------
        u : float
            Threshold in data units.
        yearly : bool
            If False: block-wise RP_t(u) for each time step t.
            If True : yearly RP_y(u) combining all seasons in a year
                      via p_y = 1 - prod_s G_{y,s}(u).
        season : int or None
            If yearly=False and season is not None, restrict to that season index.

        Returns
        -------
        x_axis : np.ndarray
            Years (if yearly=True) or "years per block" (if yearly=False).
        med, lo, hi : np.ndarray
            Posterior median and credible band for 1 / p (return period).
        """
        if self.sigma is None or self.xi is None:
            raise ValueError("Need sigma and xi in draws to compute return periods.")

        mu = np.asarray(self.d["mu"], float)  # (M,T)
        sigma = np.asarray(self.sigma, float)[:, None]
        xi = np.asarray(self.xi, float)[:, None]
        sigma = np.broadcast_to(sigma, mu.shape)
        xi = np.broadcast_to(xi, mu.shape)

        years, seasons = self._time_to_year_season()
        T = self.T
        M = mu.shape[0]

        # block-wise exceed prob per draw/time
        z = np.full((M, T), float(u))
        G = self._gev_cdf(z, mu, sigma, xi)
        p_ex = np.clip(1.0 - G, 1e-12, 1.0)  # (M,T)

        if not yearly:
            mask = np.ones(T, dtype=bool)
            if season is not None:
                mask &= (seasons == int(season))
            p_ex_sel = p_ex[:, mask]          # (M,T_sel)
            rp = 1.0 / p_ex_sel               # (M,T_sel)
            med, lo, hi = _qtiles(rp, self.level)
            return years[mask], med, lo, hi

        # yearly aggregation: p_y(u) = 1 - prod_s G_{y,s}(u)
        uniq_years = np.unique(years)
        Y = uniq_years.size
        rp_year = np.empty((M, Y), float)

        for j, yv in enumerate(uniq_years):
            mask_y = (years == yv)
            G_y = np.prod(G[:, mask_y], axis=1)    # (M,)
            p_y = np.clip(1.0 - G_y, 1e-12, 1.0)   # (M,)
            rp_year[:, j] = 1.0 / p_y

        med, lo, hi = _qtiles(rp_year, self.level)
        return uniq_years, med, lo, hi

    # ---------------- figure builders ----------------

    def _fig_overview(self, save_dir: Optional[str], show: bool) -> Optional[str]:
        mu = np.asarray(self.d["mu"], float)
        ctr, lo, hi = _qtiles(mu, self.level)
        t = np.arange(self.T)

        fig, axs = plt.subplots(2, 3, figsize=(14, 7))
        axs = axs.ravel()

        # (0) μ_t with band + data/truth
        ax = axs[0]
        if self.y is not None and len(self.y) == self.T:
            ax.plot(self.y, lw=1.0, alpha=0.6, label="y")
        ax.plot(ctr, lw=1.6, label="μ median")
        ax.fill_between(t, lo, hi, alpha=0.25, label=self.band)
        if self.t_mu is not None and len(self.t_mu) == self.T:
            ax.plot(self.t_mu, lw=1.2, ls="--", label="true μ")
        ax.set_title("Posterior μ_t")
        _uniq_legend(ax)

        # (1) σ trace
        ax = axs[1]
        if self.sigma is not None:
            ax.plot(self.sigma, lw=1)
            ax.set_title("trace: σ")
            ax.set_xlabel("iter")
        else:
            ax.axis("off")

        # (2) ξ trace
        ax = axs[2]
        if self.xi is not None:
            ax.plot(self.xi, lw=1)
            ax.set_title("trace: ξ")
            ax.set_xlabel("iter")
        else:
            ax.axis("off")

        # (3) σ posterior
        ax = axs[3]
        if self.sigma is not None:
            s = self.sigma
            es = _ess(s)
            gz = _geweke(s)
            ax.hist(s, bins=40, density=True)
            ts = _truth_sigma(self.d)
            if ts is not None:
                ax.axvline(float(ts), color="k", lw=1.6, label="truth")
            ax.set_title(f"posterior: σ (ESS≈{es:.0f}, z≈{gz:.2f})")
            _uniq_legend(ax)
        else:
            ax.axis("off")

        # (4) ξ posterior
        ax = axs[4]
        if self.xi is not None:
            s = self.xi
            es = _ess(s)
            gz = _geweke(s)
            ax.hist(s, bins=40, density=True)
            tx = _truth_xi(self.d)
            if tx is not None:
                ax.axvline(float(tx), color="k", lw=1.6, label="truth")
            ax.set_title(f"posterior: ξ (ESS≈{es:.0f}, z≈{gz:.2f})")
            _uniq_legend(ax)
        else:
            ax.axis("off")

        # (5) log10 Q histograms
        ax = axs[5]
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
                ax.axvline(
                    np.log10(float(q)),
                    lw=1.6,
                    color="k",
                    ls="--",
                    label=f"truth Q[{tag}]",
                )
                plotted = True
        if plotted:
            ax.set_title("Process variances (log10)")
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
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def _fig_states(self, save_dir: Optional[str], show: bool) -> Optional[str]:
        # x has shape (n_kept, T, dim) if present
        has_x = ("x" in self.d) and getattr(self.d["x"], "ndim", 0) == 3

        def _get(idx: Optional[int]) -> Optional[np.ndarray]:
            if not has_x or idx is None:
                return None
            return np.asarray(self.d["x"][:, :, idx], float)

        A = _get(self.layout["idx_alpha"]) if self.layout["idx_alpha"] is not None else None
        B = _get(self.layout["idx_beta"]) if self.layout["idx_beta"] is not None else None
        # use first seasonal dummy coord if available
        seasonal_idx = self.layout.get("idx_seasonal") or []
        G = _get(seasonal_idx[0]) if seasonal_idx else None

        rows = 1 + sum(v is not None for v in (A, B, G))
        fig, axes = plt.subplots(rows, 1, figsize=(12, 3.0 * rows), sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        t = np.arange(self.T)

        # row 0: μ_t
        mu = np.asarray(self.d["mu"], float)
        c, lo, hi = _qtiles(mu, self.level)
        ax = axes[0]
        if self.y is not None and len(self.y) == self.T:
            ax.plot(self.y, lw=1.0, alpha=0.6, label="y")
        ax.plot(c, lw=1.6, label="μ median")
        ax.fill_between(t, lo, hi, alpha=0.25, label=self.band)
        if self.t_mu is not None and len(self.t_mu) == self.T:
            ax.plot(self.t_mu, lw=1.2, ls="--", label="true μ")
        ax.set_title("Posterior μ_t")
        _uniq_legend(ax)

        # other rows: α, β, seasonal coord
        r = 1
        for arr, lab, truth in (
            (A, "α", self.t_a),
            (B, "β", self.t_b),
            (G, "season (g1)", self.t_g),
        ):
            if arr is None:
                continue
            c, lo, hi = _qtiles(arr, self.level)
            ax = axes[r]
            ax.plot(c, lw=1.6, label=f"{lab} median")
            ax.fill_between(t, lo, hi, alpha=0.25, label=self.band)

            if truth is not None:
                truth_arr = np.asarray(truth, float)
                if truth_arr.ndim == 2:   # e.g. gamma_t (T, p-1): take first column
                    truth_arr = truth_arr[:, 0]
                if truth_arr.size == self.T and lab != "season (g1)":
                    ax.plot(truth_arr, lw=1.2, ls="--", label=f"true {lab}")

            if lab == "α":
                ax.set_title("Level α")
            elif lab == "β":
                ax.set_title("Trend β")
            else:
                ax.set_title("Seasonal state (first dummy coord)")

            _uniq_legend(ax)
            r += 1

        fig.tight_layout()
        path = None
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, "states.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    # ----------- grouping helpers for traces / posteriors -----------

    def _families(self) -> Dict[str, List[Tuple[str, np.ndarray]]]:
        """
        Group parameters into families for trace/ACF and posterior plots.
        """
        f: Dict[str, List[Tuple[str, np.ndarray]]] = {}

        def add(group: str, name: str, a: Any) -> None:
            f.setdefault(group, []).append((name, np.asarray(a, float).ravel()))

        d = self.d

        # GEV parameters
        if self.sigma is not None:
            add("sigma", "σ", self.sigma)
        if self.xi is not None:
            add("xi", "ξ", self.xi)

        # process variances
        for nm, tag in (("alpha", "α"), ("beta", "β"), ("gamma", "γ")):
            Q = _to_Q(d, nm)
            if Q is not None:
                add("Q", f"Q_{tag}", Q)

        # process SDs (signed)
        if "s_alpha" in d:
            add("s", "s_α", d["s_alpha"])
        if "s_beta" in d:
            add("s", "s_β", d["s_beta"])
        if "s_gamma" in d:
            add("s", "s_γ", d["s_gamma"])

        # lasso local/global scales
        if "tau_alpha" in d:
            add("lasso", "τ_α", d["tau_alpha"])
        if "tau_beta" in d:
            add("lasso", "τ_β", d["tau_beta"])
        if "tau_gamma" in d:
            add("lasso", "τ_γ", d["tau_gamma"])
        if "lambda2" in d:
            add("lasso", "λ²", d["lambda2"])

        # log-likelihood
        if "loglike" in d:
            add("loglike", "loglike", d["loglike"])

        return f

    def _fig_traces(
        self,
        fam: str,
        items: List[Tuple[str, np.ndarray]],
        save_dir: Optional[str],
        show: bool,
        L: int,
    ) -> Optional[str]:
        if not items:
            return None
        R = len(items)
        fig, axs = plt.subplots(R, 2, figsize=(12, 3.0 * R), squeeze=False)
        for r, (name, s) in enumerate(items):
            s = np.asarray(s, float).ravel()
            ac = _acf(s, L)
            ess = _ess(s, L)
            gz = _geweke(s)
            # trace
            ax0 = axs[r, 0]
            ax0.plot(s, lw=1)
            ax0.set_title(f"{name} (trace)")
            ax0.set_xlabel("iter")
            # ACF
            ax1 = axs[r, 1]
            ax1.bar(np.arange(ac.size), ac, width=0.9)
            ax1.set_xlim(-0.5, ac.size - 0.5)
            ax1.set_title(f"{name} (ACF, ESS≈{ess:.0f}, z≈{gz:.2f})")
            ax1.set_xlabel("lag")
        fig.suptitle(f"Trace + ACF — {fam}", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = None
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, f"traces_acf__{_san(fam)}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def _fig_posts(
        self,
        fam: str,
        items: List[Tuple[str, np.ndarray]],
        save_dir: Optional[str],
        show: bool,
    ) -> Optional[str]:
        if not items:
            return None
        C = 2 if len(items) >= 4 else 1
        R = int(np.ceil(len(items) / C))
        fig, axs = plt.subplots(R, C, figsize=(6 * C + 1, 2.8 * R), squeeze=False)

        for k, (name, s) in enumerate(items):
            r, c = divmod(k, C)
            ax = axs[r, c]
            s = np.asarray(s, float).ravel()
            ax.hist(s, bins=40, density=True, alpha=0.85, label=name)
            ax.axvline(float(np.mean(s)), ls="--", lw=1.0, label="mean")
            ax.axvline(float(np.median(s)), ls=":", lw=1.0, label="median")

            # truths where meaningful
            tv = None
            if name == "σ":
                tv = _truth_sigma(self.d)
            elif name == "ξ":
                tv = _truth_xi(self.d)
            elif name in {"Q_α", "Q_β", "Q_γ"}:
                comp = {"Q_α": "alpha", "Q_β": "beta", "Q_γ": "gamma"}[name]
                tv = _truth_Q(self.d, comp, self.layout)
            if (tv is not None) and np.isfinite(tv):
                ax.axvline(float(tv), color="k", lw=1.6, ls="-", label="truth")

            ax.set_title(name)
            _uniq_legend(ax)

        # hide unused subplots
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
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    # ---------------- extra figures: return levels / periods ----------------

    def figure_return_levels(
        self,
        N: float = 10.0,
        season: Optional[int] = None,
        save_dir: Optional[str] = None,
        show: bool = True,
    ) -> Optional[str]:
        """
        Plot block-wise N-year return levels over time.

        Parameters
        ----------
        N : float
            Return period in block units (years if each block is yearly).
        season : int or None
            If not None, restrict to a specific season index within the period:
              e.g. 0=DJF,1=MAM,2=JJA,3=SON when period=4.
        """
        z_draws = self._gev_return_level_block(N)  # (M,T)
        med, lo, hi = _qtiles(z_draws, self.level)  # (T,)

        years, seasons = self._time_to_year_season()

        mask = np.ones(self.T, dtype=bool)
        if season is not None:
            mask &= (seasons == int(season))

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        ax.plot(years[mask], med[mask], lw=1.8, label=f"{N}-year RL (median)")
        ax.fill_between(years[mask], lo[mask], hi[mask], alpha=0.25,
                        label=f"{int(self.level * 100)}% band")

        if self.y is not None and len(self.y) == self.T:
            ax.scatter(years[mask], np.asarray(self.y)[mask], s=10, alpha=0.3,
                       label="observed block maxima")

        if season is not None:
            ax.set_title(f"{N}-year return level over time (season index {season})")
        else:
            ax.set_title(f"{N}-year return level over time (all blocks)")

        ax.set_xlabel("Year")
        ax.set_ylabel("Return level")
        _uniq_legend(ax)
        fig.tight_layout()

        path = None
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, f"return_levels_N{N:g}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def figure_return_periods(
        self,
        u: float,
        yearly: bool = False,
        season: Optional[int] = None,
        save_dir: Optional[str] = None,
        show: bool = True,
    ) -> Optional[str]:
        """
        Plot time-varying return periods for threshold u.

        Parameters
        ----------
        u : float
            Threshold in data units.
        yearly : bool
            If True, combine seasons in each year (at least one exceedance in year).
            If False, block-wise RP_t(u).
        season : int or None
            Only used when yearly=False; season index within period.
        """
        x, med, lo, hi = self.compute_return_periods(u=u, yearly=yearly, season=season)

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(x, med, lw=1.8, label="Return period (median)")
        ax.fill_between(x, lo, hi, alpha=0.25,
                        label=f"{int(self.level * 100)}% band")

        ax.set_xlabel("Year")
        ax.set_ylabel("Return period (years)")
        title = f"Return period for threshold u={u:g}"
        if yearly:
            title += " (yearly aggregated)"
        elif season is not None:
            title += f" (season index {season})"
        ax.set_title(title)
        ax.set_yscale("log")
        _uniq_legend(ax)
        fig.tight_layout()

        path = None
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, f"return_periods_u{u:g}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    # ---------------- public API ----------------

    def figure_overview(self, save_dir: Optional[str] = None, show: bool = True) -> Optional[str]:
        return self._fig_overview(save_dir, show)

    def figure_states(self, save_dir: Optional[str] = None, show: bool = True) -> Optional[str]:
        return self._fig_states(save_dir, show)

    def figure_traces_grouped_all(
        self,
        save_dir: Optional[str] = None,
        show: bool = False,
        max_lag: int = 200,
    ) -> List[Optional[str]]:
        outs: List[Optional[str]] = []
        fams = self._families()
        for fam, items in fams.items():
            p = self._fig_traces(fam, items, save_dir, show, max_lag)
            if p:
                outs.append(p)
        if not outs:
            print("[warn] no parameter families for trace+ACF.")
        return outs

    def figure_posteriors_grouped_all(
        self,
        save_dir: Optional[str] = None,
        show: bool = False,
    ) -> List[Optional[str]]:
        outs: List[Optional[str]] = []
        fams = self._families()
        for fam, items in fams.items():
            p = self._fig_posts(fam, items, save_dir, show)
            if p:
                outs.append(p)
        if not outs:
            print("[warn] no parameter families for posterior histograms.")
        return outs

    def quick_report(self, save_dir: Optional[str] = None, show: bool = True) -> Optional[str]:
        mu = np.asarray(self.d["mu"], float)
        c, lo, hi = _qtiles(mu, self.level)
        t = np.arange(self.T)

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        # μ panel
        ax = axs[0]
        ax.plot(c, lw=1.6, label="μ median")
        ax.fill_between(t, lo, hi, alpha=0.25, label=self.band)
        if self.t_mu is not None and len(self.t_mu) == self.T:
            ax.plot(self.t_mu, lw=1.2, ls="--", label="true μ")
        ax.set_title("μ_t")
        _uniq_legend(ax)

        # σ panel
        ax = axs[1]
        if self.sigma is not None:
            ax.hist(self.sigma, bins=40, density=True, label="σ")
            ts = _truth_sigma(self.d)
            if ts is not None:
                ax.axvline(float(ts), color="k", lw=1.6, label="truth")
            ax.set_title("σ | y")
            _uniq_legend(ax)
        else:
            ax.axis("off")

        # ξ or Q panel
        ax = axs[2]
        panel_done = False

        if self.xi is not None:
            ax.hist(self.xi, bins=40, density=True, label="ξ")
            tx = _truth_xi(self.d)
            if tx is not None:
                ax.axvline(float(tx), color="k", lw=1.6, label="truth")
            ax.set_title("ξ | y")
            _uniq_legend(ax)
            panel_done = True
        else:
            # fall back to log10 Q_γ/α/β if available
            for comp, series, tag in (
                ("gamma", self.Qg, "Q_γ"),
                ("alpha", self.Qa, "Q_α"),
                ("beta", self.Qb, "Q_β"),
            ):
                if series is None:
                    continue
                ax.hist(
                    np.log10(np.clip(series, 1e-20, None)),
                    bins=40,
                    density=True,
                    label=f"log10 {tag}",
                )
                q = _truth_Q(self.d, comp, self.layout)
                if q is not None and q > 0:
                    ax.axvline(np.log10(float(q)), color="k", lw=1.6, label="truth")
                ax.set_title(f"log10 {tag} | y")
                _uniq_legend(ax)
                panel_done = True
                break

        if not panel_done:
            ax.axis("off")

        fig.tight_layout()
        path = None
        if save_dir:
            _ensure_dir(save_dir)
            path = os.path.join(save_dir, "quick_report.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[save] {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path


# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DGEV Laplace NCP Plotter (dummy seasonality + lasso)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Run dir or posterior.npz. If omitted, search under --root.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results/simulations/DGEV_NCP_LASSO",
        help="Search root when --target omitted.",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.90,
        help="Credible band level for μ and states.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show figures interactively.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Save dir (default: <run>/figures).",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        default=False,
        help="Skip overview figure.",
    )
    parser.add_argument(
        "--skip-states",
        action="store_true",
        default=False,
        help="Skip states figure.",
    )
    parser.add_argument(
        "--skip-grouped-traces",
        action="store_true",
        default=False,
        help="Skip grouped trace+ACF figures.",
    )
    parser.add_argument(
        "--skip-grouped-post",
        action="store_true",
        default=False,
        help="Skip grouped posterior figures.",
    )
    parser.add_argument(
        "--skip-quick",
        action="store_true",
        default=False,
        help="Skip quick_report figure.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=200,
        help="ACF/ESS max lag for trace plots.",
    )
    # Return levels / periods
    parser.add_argument(
        "--rl-N",
        type=float,
        default=100,
        help="If set, plot N-year (block-wise) return level time series.",
    )
    parser.add_argument(
        "--rl-season",
        type=int,
        default=1,
        help="Season index for block-wise return levels (0..period-1).",
    )
    parser.add_argument(
        "--rp-u",
        type=float,
        default=1,
        help="If set, plot return period time series for threshold u.",
    )
    parser.add_argument(
        "--rp-yearly",
        action="store_true",
        default=True,
        help="If set, combine seasons within each year when computing return periods.",
    )
    parser.add_argument(
        "--rp-season",
        type=int,
        default=1,
        help="Season index for block-wise return periods (0..period-1). Ignored if --rp-yearly.",
    )

    args = parser.parse_args()

    if args.target:
        draws, meta, npz_path = load_posterior(args.target)
    else:
        npz = _find_latest_run(args.root)
        if npz is None:
            print(f"[error] no posterior.npz under {args.root!r}; provide --target or change --root.")
            sys.exit(1)
        draws, meta, npz_path = load_posterior(npz)

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "figures")
    _ensure_dir(out_dir)
    print(f"[info] saving to: {out_dir}")

    pl = DGEVPlotter(draws=draws, meta=meta, level=float(args.level))

    if not args.skip_overview:
        pl.figure_overview(out_dir, args.show)
    if not args.skip_states:
        pl.figure_states(out_dir, args.show)
    if not args.skip_grouped_traces:
        pl.figure_traces_grouped_all(out_dir, args.show, int(args.max_lag))
    if not args.skip_grouped_post:
        pl.figure_posteriors_grouped_all(out_dir, args.show)
    if not args.skip_quick:
        pl.quick_report(out_dir, args.show)

    # Return levels / periods
    if args.rl_N is not None:
        pl.figure_return_levels(
            N=float(args.rl_N),
            season=args.rl_season,
            save_dir=out_dir,
            show=args.show,
        )
    if args.rp_u is not None:
        pl.figure_return_periods(
            u=float(args.rp_u),
            yearly=bool(args.rp_yearly),
            season=None if args.rp_yearly else args.rp_season,
            save_dir=out_dir,
            show=args.show,
        )

    print("[done] plots written.")
