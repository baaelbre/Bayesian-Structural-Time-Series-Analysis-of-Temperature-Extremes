# %% simulator/dgev_laplace_waiting_times_annual.py
from __future__ import annotations
"""
Annual expected waiting times for the DGEV (Laplace NCP) posterior
=================================================================

This script ONLY computes **annual-scale expected waiting time** trajectories
for exceedances of fixed thresholds y* on the ORIGINAL (data) scale.

Definitions
-----------
Let M_j be the annual maximum (or annual minimum after the internal sign convention).
Define the annual exceedance hazard

    p_j(y*) = P(M_j > y* | x_{0:T}, θ)
            = 1 - ∏_{t in S_j} G_t(y*)

where S_j are the within-year blocks (e.g., months) and G_t is the block-level GEV CDF.

Define the waiting time (in YEARS) from year j:

    X_j = inf{k >= 1 : M_{j+k-1} > y*}

Then the expected waiting time E_j = E[X_j | x_{0:T}, θ] satisfies the recursion

    E_j = 1 + (1 - p_j) E_{j+1}

with a tail convention beyond the computed horizon. We compute E_j for each
posterior draw, then summarise pointwise into medians + credible bands.

Minima
------
If the series is a minima index (TXn/TNn), the sampler stores Z_t = -Y_t on MODEL scale.
Thresholds are specified on ORIGINAL scale and mapped internally via z* = -y*.

Expected posterior bundle content (MODEL scale)
----------------------------------------------
  - draws['mu']    : (S, T) location path
  - draws['sigma'] : (S,)   GEV scale   (or draws['sigma2'])
  - draws['xi']    : (S,)   GEV shape
Meta fields (optional):
  - meta['period']     : blocks per year (default 12)
  - meta['start_date'] : calendar grouping (YYYY-MM-DD)
  - meta hints for minima detection (minima/model_sign/series/data_transform)

Outputs (default: <run>/waiting_times_annual)
--------------------------------------------
For each threshold thr:
  - waiting_time_annual_thr<thr>.npz  (med/lo/hi + labels + settings)
  - waiting_time_annual_thr<thr>.png  (median + band)
Combined plot (if multiple thresholds):
  - waiting_time_annual_all_thresholds.png
Also:
  - metrics.json
"""

import os
import re
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Mapping

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
except Exception as e:
    raise ImportError("Could not import optimization.posterior_bundle.load_posterior/find_latest_run.") from e


# -----------------------------------------------------------------------------
# Robust run discovery
# -----------------------------------------------------------------------------
def _extract_ts_from_path(path_str: str) -> Optional[float]:
    m = re.search(r"(\d{8})_(\d{6})", path_str)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.timestamp()
    except Exception:
        return None


def find_latest_posterior_npz(root: str) -> Optional[str]:
    root_p = Path(root)
    if not root_p.exists():
        return None
    cands = list(root_p.rglob("posterior*.npz"))
    if not cands:
        return None

    def key(p: Path) -> Tuple[int, float]:
        ts = _extract_ts_from_path(str(p))
        if ts is not None:
            return (1, ts)
        return (0, p.stat().st_mtime)

    return str(max(cands, key=key))


def resolve_bundle(*, target: Optional[str], root: str):
    if target:
        return load_posterior(target)

    print(f"[info] searching latest posterior run under: {root!r}")
    run_path = find_latest_run(root=root)
    if run_path is not None:
        print(f"[info] using latest run: {run_path}")
        return load_posterior(run_path)

    npz_path = find_latest_posterior_npz(root)
    if npz_path is None:
        raise SystemExit(f"[error] No posterior runs found under {root!r}. Provide --target or run the sampler.")
    print(f"[info] find_latest_run found nothing; using latest npz: {npz_path}")
    return load_posterior(npz_path)


# -----------------------------------------------------------------------------
# Post-hoc burn/thin + optional subsample
# -----------------------------------------------------------------------------
def apply_burn_thin(
    draws: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    burn: int = 0,
    thin: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    burn = int(burn or 0)
    thin = int(thin or 1)
    if burn < 0:
        raise ValueError(f"--burn must be >= 0, got {burn}")
    if thin < 1:
        raise ValueError(f"--thin must be >= 1, got {thin}")

    n_samp: Optional[int] = None
    for k in ("mu", "sigma", "sigma2", "xi"):
        if k in draws and isinstance(draws[k], np.ndarray) and np.asarray(draws[k]).ndim >= 1:
            n_samp = int(np.asarray(draws[k]).shape[0])
            break
    if n_samp is None:
        return draws, meta

    if burn >= n_samp:
        raise ValueError(f"--burn={burn} ≥ number of saved samples ({n_samp}).")

    idx = slice(burn, None, thin)
    n_used = int(math.ceil((n_samp - burn) / thin))
    print(f"[info] post-processing chains: raw n={n_samp}, burn={burn}, thin={thin} → used n={n_used}")

    for k, v in list(draws.items()):
        if not isinstance(v, np.ndarray):
            continue
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n_samp:
            draws[k] = arr[idx, ...]

    postproc = meta.get("postproc", {})
    if not isinstance(postproc, dict):
        postproc = {}
    postproc.update(
        {
            "extra_burn": int(burn),
            "thin": int(thin),
            "n_samples_raw": int(n_samp),
            "n_samples_used": int(n_used),
        }
    )
    meta["postproc"] = postproc
    return draws, meta


def subsample_draws_first_dim(
    draws: Dict[str, Any],
    meta: Dict[str, Any],
    *,
    max_draws: Optional[int],
    seed: int = 123,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if max_draws is None:
        return draws, meta
    if "mu" not in draws:
        return draws, meta
    mu = np.asarray(draws["mu"])
    if mu.ndim != 2:
        return draws, meta

    S = int(mu.shape[0])
    M = int(max_draws)
    if M <= 0 or S <= M:
        return draws, meta

    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(S, size=M, replace=False))
    for k, v in list(draws.items()):
        if not isinstance(v, np.ndarray):
            continue
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == S:
            draws[k] = arr[idx, ...]

    postproc = meta.get("postproc", {})
    if not isinstance(postproc, dict):
        postproc = {}
    postproc.update({"subsample_draws": int(M), "subsample_seed": int(seed)})
    meta["postproc"] = postproc
    print(f"[info] subsampled posterior draws: S={S} → {M}")
    return draws, meta


# -----------------------------------------------------------------------------
# Meta helpers
# -----------------------------------------------------------------------------
def _coerce_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def detect_minima(meta: Dict[str, Any]) -> bool:
    if not isinstance(meta, dict):
        return False

    for k in ("minima", "is_minima", "minima_series"):
        b = _coerce_bool(meta.get(k, None))
        if b is not None:
            return b

    ms = meta.get("model_sign", None)
    try:
        if ms is not None and float(ms) < 0:
            return True
    except Exception:
        pass

    ser = meta.get("series", None)
    if isinstance(ser, str) and ser.strip() in {"TXn", "TNn"}:
        return True

    dt = meta.get("data_transform", None)
    if isinstance(dt, str):
        s = dt.lower()
        if any(tok in s for tok in ("negate", "signflip", "flip_sign", "minus")):
            return True

    return False


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def summarize_ci(arr: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loq = (1.0 - float(level)) / 2.0
    hiq = 1.0 - loq
    med = np.quantile(arr, 0.5, axis=0)
    lo = np.quantile(arr, loq, axis=0)
    hi = np.quantile(arr, hiq, axis=0)
    return med, lo, hi


# -----------------------------------------------------------------------------
# Broadcast-safe GEV log-CDF (we only need CDF here)
# -----------------------------------------------------------------------------
def gev_logcdf(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    xi: np.ndarray,
    eps_xi: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(x, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    xi = np.asarray(xi, float)

    sigma = np.clip(sigma, 1e-12, None)

    z = (x - mu) / sigma
    xi_b = np.broadcast_to(xi, z.shape)
    gmask = np.abs(xi_b) < eps_xi

    out = np.empty_like(z, dtype=float)

    # Gumbel: log G = -exp(-z)
    negz = -z
    out_g = np.where(negz > 709.0, -np.inf, -np.exp(negz))

    # Non-Gumbel
    t = 1.0 + xi_b * z
    good = t > 0.0

    expo = np.empty_like(z, dtype=float)
    expo[good] = (-1.0 / xi_b[good]) * np.log(t[good])

    out_ng = np.empty_like(z, dtype=float)
    out_ng[good] = np.where(expo[good] > 709.0, -np.inf, -np.exp(expo[good]))

    bad = ~good
    if np.any(bad):
        # outside support: for xi>0, CDF->0; for xi<0, CDF->1 (log->0)
        out_ng[bad] = np.where(xi_b[bad] > 0.0, -np.inf, 0.0)

    out[:] = np.where(gmask, out_g, out_ng)
    out = np.where(np.isfinite(out), np.clip(out, -745.0, 0.0), out)
    return out


# -----------------------------------------------------------------------------
# Year grouping helper (+ optional pandas calendar)
# -----------------------------------------------------------------------------
def build_year_groups(T: int, period: int, start_date: Optional[str]) -> Tuple[List[np.ndarray], List[str]]:
    if start_date:
        try:
            import pandas as pd
            dates = pd.date_range(start=pd.to_datetime(start_date), periods=T, freq="MS")
            years = dates.year.values
            uniq = np.unique(years)
            groups: List[np.ndarray] = []
            labels: List[str] = []
            for y in uniq:
                idx = np.where(years == y)[0]
                if idx.size > 0:
                    groups.append(idx.astype(int))
                    labels.append(str(int(y)))
            return groups, labels
        except Exception:
            pass

    groups, labels = [], []
    n_years = T // period
    for j in range(n_years):
        idx = np.arange(j * period, (j + 1) * period, dtype=int)
        groups.append(idx)
        labels.append(str(j))
    return groups, labels


# -----------------------------------------------------------------------------
# Core engine: annual hazards + annual expected waiting time
# -----------------------------------------------------------------------------
@dataclass
class SummaryBand:
    med: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


class DGEVAnnualWaitingTime:
    def __init__(
        self,
        *,
        mu: np.ndarray,        # (S,T)
        sigma: np.ndarray,     # (S,)
        xi: np.ndarray,        # (S,)
        period: int = 12,
        start_date: Optional[str] = None,
        minima: bool = False,
    ):
        self.mu = np.asarray(mu, float)
        if self.mu.ndim != 2:
            raise ValueError("mu must be (S,T)")
        self.S, self.T = self.mu.shape

        self.sigma = np.asarray(sigma, float).reshape(-1)
        self.xi = np.asarray(xi, float).reshape(-1)
        if self.sigma.size != self.S or self.xi.size != self.S:
            raise ValueError("sigma/xi must have length S matching mu")

        self.period = int(period)
        self.start_date = start_date
        self.minima = bool(minima)

        self._sigma2 = np.clip(self.sigma, 1e-12, None).reshape(self.S, 1)  # broadcast helper
        self._xi2 = self.xi.reshape(self.S, 1)

        self.year_groups, self.year_labels = build_year_groups(self.T, self.period, self.start_date)
        self.J = len(self.year_groups)

    def to_model_scalar(self, y_orig: float) -> float:
        return float(-y_orig if self.minima else y_orig)

    # ---- annual exceedance hazards p_j(y*) ----
    def annual_exceedance_prob_draws(self, threshold_orig: float) -> Tuple[np.ndarray, List[str]]:
        y_model = self.to_model_scalar(float(threshold_orig))
        out = np.empty((self.S, self.J), dtype=float)

        for j, idx in enumerate(self.year_groups):
            mu_j = self.mu[:, idx]  # (S,K)
            x = np.full_like(mu_j, y_model, dtype=float)
            logG = gev_logcdf(x, mu=mu_j, sigma=self._sigma2, xi=self._xi2)  # (S,K)
            sum_logG = np.sum(logG, axis=1)                                  # log ∏ G_t
            out[:, j] = -np.expm1(sum_logG)                                  # 1 - exp(sum logG)

        out = np.clip(out, 0.0, 1.0)
        return out, self.year_labels

    # ---- expected waiting time (years) ----
    @staticmethod
    def expected_waiting_time_from_hazards(
        p: np.ndarray,
        *,
        tail: str = "carry",
        eps: float = 1e-15,
        max_wait: Optional[float] = None,
    ) -> np.ndarray:
        """
        p: (S,J) annual hazards in [0,1]
        returns E: (S,J) expected waiting time in YEARS from each year index.
        Tail:
          - 'carry' : use p_tail = p[:, -1]
          - float   : constant tail hazard
        """
        p = np.asarray(p, float)
        if p.ndim != 2:
            raise ValueError("p must be (S,J)")
        S, J = p.shape
        p = np.clip(p, 0.0, 1.0)

        if tail == "carry":
            p_tail = p[:, -1]
        else:
            try:
                const = float(tail)
            except Exception as e:
                raise ValueError(f"Invalid --tail value {tail!r}. Use 'carry' or a float.") from e
            p_tail = np.full((S,), np.clip(const, 0.0, 1.0), dtype=float)

        E = np.empty((S, J + 1), dtype=float)
        E[:, J] = 1.0 / np.clip(p_tail, eps, 1.0)

        for j in range(J - 1, -1, -1):
            E[:, j] = 1.0 + (1.0 - p[:, j]) * E[:, j + 1]

        out = E[:, :J]
        if max_wait is not None:
            out = np.clip(out, 0.0, float(max_wait))
        return out

    def annual_expected_waiting_time_draws(
        self,
        threshold_orig: float,
        *,
        tail: str = "carry",
        max_wait: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        p_ann, labels = self.annual_exceedance_prob_draws(threshold_orig)  # (S,J)
        E = self.expected_waiting_time_from_hazards(p_ann, tail=tail, max_wait=max_wait)
        return E, labels


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
@dataclass
class PlotConfig:
    figsize: Tuple[float, float] = (10.5, 3.8)
    alpha_band: float = 0.18
    lw: float = 1.6
    grid_alpha: float = 0.25
    legend: bool = True
    legend_loc: str = "best"
    wt_logy: bool = False

    xlabel_year: str = "year index"
    ylabel_annual_wt: str = "expected waiting time (years)"
    label_template_thr: str = "thr={thr:g}"


class WaitingTimePlotter:
    def __init__(self, cfg: PlotConfig | None = None):
        self.cfg = cfg or PlotConfig()

    def plot_single_band(
        self,
        x: np.ndarray,
        band: SummaryBand,
        *,
        xlabel: Optional[str],
        ylabel: Optional[str],
        path: str,
        title: Optional[str] = None,
        logy: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=self.cfg.figsize)
        ax.fill_between(x, band.lo, band.hi, alpha=self.cfg.alpha_band)
        ax.plot(x, band.med, lw=self.cfg.lw)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=self.cfg.grid_alpha)
        if logy:
            ax.set_yscale("log")
        ensure_dir(os.path.dirname(path))
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {path}")

    def plot_multi_bands(
        self,
        x: np.ndarray,
        bands: Mapping[str, SummaryBand],
        *,
        xlabel: Optional[str],
        ylabel: Optional[str],
        path: str,
        title: Optional[str] = None,
        legend: Optional[bool] = None,
        logy: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=self.cfg.figsize)
        for lab, band in bands.items():
            ax.fill_between(x, band.lo, band.hi, alpha=self.cfg.alpha_band)
            ax.plot(x, band.med, lw=self.cfg.lw, label=lab)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=self.cfg.grid_alpha)
        if logy:
            ax.set_yscale("log")
        use_leg = self.cfg.legend if legend is None else bool(legend)
        if use_leg:
            ax.legend(loc=self.cfg.legend_loc, frameon=False)
        ensure_dir(os.path.dirname(path))
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {path}")


# -----------------------------------------------------------------------------
# Helpers: parsing + printing
# -----------------------------------------------------------------------------
def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_tokens(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def fmt_wt(x: float) -> str:
    if not np.isfinite(x):
        return str(x)
    return f"{x:.3g}"


def print_wt_annual_points(labels: List[str], band: SummaryBand, tokens: List[str], title: str) -> None:
    mp = {lab: i for i, lab in enumerate(labels)}
    print(f"\n{title}")
    print("  label |     med      lo      hi")
    for tok in tokens:
        idx = mp.get(tok, None)
        if idx is None:
            try:
                ii = int(tok)
                if 0 <= ii < len(labels):
                    idx = ii
            except Exception:
                idx = None
        if idx is None:
            print(f"  {tok:>5} |  (not found)")
            continue
        lab = labels[idx]
        print(f"  {lab:>5} | {fmt_wt(float(band.med[idx])):>7} {fmt_wt(float(band.lo[idx])):>7} {fmt_wt(float(band.hi[idx])):>7}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Annual expected waiting times for Laplace DGEV posterior (threshold-based).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", type=str, default=None,
                   help="Path to run dir or directly to posterior.npz. If omitted, uses latest under --root.")
    p.add_argument("--root", type=str, default="results/simulations/DGEV_NCP_LASSO",
                   help="Search root when --target is omitted.")
    p.add_argument("--out", type=str, default=None,
                   help="Output directory. Default: <run>/waiting_times_annual")

    p.add_argument("--period", type=int, default=None,
                   help="Blocks per year (default: meta['period'] or 12).")
    p.add_argument("--start-date", type=str, default=None,
                   help="Start date (YYYY-MM-DD) for calendar-year grouping.")

    p.add_argument("--level", type=float, default=0.90, help="Credible interval level.")
    p.add_argument("--burn", type=int, default=0, help="Extra burn-in (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning (post-hoc).")
    p.add_argument("--max-draws", type=int, default=None, help="Optional posterior draw subsample for speed.")
    p.add_argument("--seed", type=int, default=123, help="Seed used for subsampling.")

    p.add_argument("--thresholds", type=str, default="0.0",
                   help="Comma-separated thresholds y* (ORIGINAL scale) for annual expected waiting times.")
    p.add_argument("--tail", type=str, default="carry",
                   help="Tail convention for recursion: 'carry' or a float (constant tail hazard).")
    p.add_argument("--max-wait", type=float, default=None,
                   help="Optional cap on waiting times (years), mainly for plotting stability.")
    p.add_argument("--wt-logy", action="store_true", default=False,
                   help="Use log-scale on y-axis for waiting time plots.")

    p.add_argument("--print-years", type=str, default=None,
                   help="Comma-separated calendar years OR year-indices to print at (e.g. 1892,1950,2020).")

    p.add_argument("--xlabel-year", type=str, default=None)
    p.add_argument("--ylabel-annual-wt", type=str, default=None)
    p.add_argument("--legend", action="store_true", default=True, help="Show legends on combined plots.")
    p.add_argument("--no-legend", action="store_false", dest="legend", help="Disable legends.")
    p.add_argument("--no-combined", action="store_true", default=False,
                   help="Do not create combined plot (all thresholds on one plot).")
    p.add_argument("--no-individual", action="store_true", default=False,
                   help="Do not create individual plot per threshold.")

    args = p.parse_args()

    bundle = resolve_bundle(target=args.target, root=args.root)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    if args.burn > 0 or args.thin > 1:
        draws, meta = apply_burn_thin(draws, dict(meta), burn=args.burn, thin=args.thin)
    draws, meta = subsample_draws_first_dim(draws, dict(meta), max_draws=args.max_draws, seed=args.seed)

    if "mu" not in draws:
        raise SystemExit("[error] posterior draws must contain 'mu' (S,T).")
    mu = np.asarray(draws["mu"], float)

    if "sigma" in draws:
        sigma = np.asarray(draws["sigma"], float).reshape(-1)
    elif "sigma2" in draws:
        sigma = np.sqrt(np.clip(np.asarray(draws["sigma2"], float).reshape(-1), 0.0, None))
    else:
        raise SystemExit("[error] posterior draws must contain 'sigma' or 'sigma2'.")

    if "xi" not in draws:
        raise SystemExit("[error] posterior draws must contain 'xi'.")
    xi = np.asarray(draws["xi"], float).reshape(-1)

    period = int(args.period or meta.get("period", 12))
    start_date = args.start_date or meta.get("start_date", None)
    minima = detect_minima(meta)

    engine = DGEVAnnualWaitingTime(mu=mu, sigma=sigma, xi=xi, period=period, start_date=start_date, minima=minima)

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "waiting_times_annual")
    ensure_dir(out_dir)

    print(f"[info] using posterior: {npz_path}")
    print(f"[info] period={period}, start_date={start_date}, minima_detected={minima}")
    print(f"[info] saving outputs to: {out_dir}")

    cfg = PlotConfig(
        legend=bool(args.legend),
        wt_logy=bool(args.wt_logy),
    )
    if args.xlabel_year is not None:
        cfg.xlabel_year = args.xlabel_year
    if args.ylabel_annual_wt is not None:
        cfg.ylabel_annual_wt = args.ylabel_annual_wt

    plotter = WaitingTimePlotter(cfg)

    metrics: Dict[str, Any] = {
        "npz_path": str(npz_path),
        "period": int(period),
        "start_date": start_date,
        "minima_detected": bool(minima),
        "S": int(engine.S),
        "T": int(engine.T),
        "J_years": int(engine.J),
        "level": float(args.level),
        "tail": str(args.tail),
        "max_wait": args.max_wait,
        "plots": {
            "legend": bool(cfg.legend),
            "wt_logy": bool(cfg.wt_logy),
            "combined": (not bool(args.no_combined)),
            "individual": (not bool(args.no_individual)),
        },
    }

    thr_list = parse_csv_floats(args.thresholds)
    print_year_tokens = parse_csv_tokens(args.print_years)

    # Compute waiting-time bands
    ann_wt_bands: Dict[str, SummaryBand] = {}
    wt_meta: Dict[str, Any] = {}

    last_labels: Optional[List[str]] = None
    for thr in thr_list:
        wt_ann, labels = engine.annual_expected_waiting_time_draws(thr, tail=args.tail, max_wait=args.max_wait)  # (S,J)
        med, lo, hi = summarize_ci(wt_ann, level=args.level)
        band = SummaryBand(med=med, lo=lo, hi=hi)

        labT = cfg.label_template_thr.format(thr=thr)
        ann_wt_bands[labT] = band
        wt_meta[str(thr)] = {"annual_labels": labels}
        last_labels = labels

        # Save NPZ
        np.savez_compressed(
            os.path.join(out_dir, f"waiting_time_annual_thr{thr}.npz"),
            thr=float(thr),
            wt_ann_med=med,
            wt_ann_lo=lo,
            wt_ann_hi=hi,
            labels=np.array(labels, dtype=object),
            level=float(args.level),
            minima_detected=bool(minima),
            tail=str(args.tail),
            max_wait=args.max_wait,
        )

        # Individual plot
        if not args.no_individual:
            plotter.plot_single_band(
                np.arange(len(labels)),
                band,
                xlabel=cfg.xlabel_year,
                ylabel=cfg.ylabel_annual_wt,
                path=os.path.join(out_dir, f"waiting_time_annual_thr{thr}.png"),
                logy=cfg.wt_logy,
            )

        # Printing
        if print_year_tokens:
            print_wt_annual_points(labels, band, print_year_tokens, f"[print] Annual expected waiting time ({labT})")

    metrics["waiting_times"] = wt_meta

    # Combined plot
    if (not args.no_combined) and len(thr_list) >= 2 and last_labels is not None:
        plotter.plot_multi_bands(
            np.arange(len(last_labels)),
            ann_wt_bands,
            xlabel=cfg.xlabel_year,
            ylabel=cfg.ylabel_annual_wt,
            path=os.path.join(out_dir, "waiting_time_annual_all_thresholds.png"),
            legend=cfg.legend,
            logy=cfg.wt_logy,
        )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] {os.path.join(out_dir, 'metrics.json')}")
    print("[done] annual expected waiting times computed.")
