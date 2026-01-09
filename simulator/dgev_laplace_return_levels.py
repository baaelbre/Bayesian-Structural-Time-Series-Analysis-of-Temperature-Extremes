# %% simulator/dgev_return_levels.py
from __future__ import annotations
"""
Return levels & return periods for DGEV posterior draws (block + annual).

Key changes vs your version
---------------------------
- Annual grouping + annual return periods are fully vectorized (no per-year Python loop).
- Annual exceedance uses stable math: p = -expm1(sum(log G)).
- Annual return levels solved by *parallel* bisection over (draw, year) grid.
- Return periods can be computed/stored on log10-scale to avoid 1e15 caps.
- Robust run discovery kept (find_latest_run + fallback rglob posterior*.npz).

Conventions
-----------
- Posterior is on MODEL scale. For minima-series runs stored as z=-y, we detect minima and
  transform thresholds/outputs to PLOT scale (original sign) automatically.
"""

import os
import re
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
except Exception as e:
    raise ImportError("Could not import optimization.posterior_bundle.{load_posterior,find_latest_run}.") from e


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def _savefig(fig: plt.Figure, path: str, *, dpi: int = 200, show: bool = False) -> None:
    _ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[save] {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    parts = [int(x) for x in ss.split("-")]
    if len(parts) == 1:
        return datetime(parts[0], 1, 1)
    if len(parts) == 2:
        return datetime(parts[0], parts[1], 1)
    if len(parts) == 3:
        return datetime(parts[0], parts[1], parts[2])
    raise ValueError("start_date must be YYYY / YYYY-MM / YYYY-MM-DD")

def _coerce_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        t = x.strip().lower()
        if t in ("1", "true", "t", "yes", "y"): return True
        if t in ("0", "false", "f", "no", "n"): return False
    return None

def _detect_minima(meta: Dict[str, Any]) -> bool:
    for k in ("minima", "is_minima", "minima_series"):
        b = _coerce_bool(meta.get(k))
        if b is not None:
            return b
    ms = meta.get("model_sign", None)
    try:
        if ms is not None and float(ms) < 0:
            return True
    except Exception:
        pass
    dt = meta.get("data_transform", None)
    if isinstance(dt, str) and any(tok in dt.lower() for tok in ("negate", "signflip", "flip_sign", "minus")):
        return True
    ser = meta.get("series", None)
    if isinstance(ser, str) and ser.strip() in ("TXn", "TNn"):
        return True
    return False

def _extract_ts(path_str: str) -> Optional[float]:
    m = re.search(r"(\d{8})_(\d{6})", path_str)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.timestamp()
    except Exception:
        return None

def _find_latest_npz(root: str) -> Optional[str]:
    rp = Path(root)
    if not rp.exists():
        return None
    cands = list(rp.rglob("posterior*.npz"))
    if not cands:
        return None

    def key(p: Path) -> Tuple[int, float]:
        ts = _extract_ts(str(p))
        if ts is not None:
            return (1, ts)
        return (0, p.stat().st_mtime)

    return str(max(cands, key=key))

def resolve_bundle(*, target: Optional[str], root: str):
    if target:
        print(f"[info] loading posterior from --target: {target}")
        return load_posterior(target)

    print(f"[info] --target not set; searching latest run under --root: {root}")
    run = find_latest_run(root=root)
    if run is not None:
        print(f"[info] find_latest_run -> {run}")
        return load_posterior(run)

    npz = _find_latest_npz(root)
    if npz is None:
        raise SystemExit(f"No posterior found under {root!r}. Provide --target.")
    print(f"[info] find_latest_run found nothing; fallback newest posterior*.npz -> {npz}")
    return load_posterior(npz)


# -----------------------------------------------------------------------------
# GEV CDF/PPF (broadcast-safe)
# -----------------------------------------------------------------------------
def gev_cdf(z, mu, sigma, xi):
    z = np.asarray(z, float)
    mu = np.asarray(mu, float)
    sigma = np.clip(np.asarray(sigma, float), 1e-12, None)
    xi = np.asarray(xi, float)

    x = (z - mu) / sigma
    eps = 1e-12
    gumbel = np.abs(xi) < eps

    cdf_g = np.exp(-np.exp(-x))

    t = 1.0 + xi * x
    valid = t > 0.0
    cdf_ng = np.exp(-(t ** (-1.0 / xi)))

    xi_neg = xi < -eps
    cdf_ng = np.where(valid, cdf_ng, np.where(xi_neg, 1.0, 0.0))

    out = np.where(gumbel, cdf_g, cdf_ng)
    return np.clip(out, 0.0, 1.0)

def gev_ppf(p: float, mu, sigma, xi):
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    mu = np.asarray(mu, float)
    sigma = np.clip(np.asarray(sigma, float), 1e-12, None)
    xi = np.asarray(xi, float)

    t = -math.log(p)
    eps = 1e-12
    gumbel = np.abs(xi) < eps

    z_g = mu - sigma * math.log(t)
    z_ng = mu + (sigma / xi) * (t ** (-xi) - 1.0)
    return np.where(gumbel, z_g, z_ng)


def _summ(draws: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loq = (1.0 - level) / 2.0
    hiq = 1.0 - loq
    med = np.nanquantile(draws, 0.5, axis=0)
    lo = np.nanquantile(draws, loq, axis=0)
    hi = np.nanquantile(draws, hiq, axis=0)
    return med, lo, hi


# -----------------------------------------------------------------------------
# Annual grouping
# -----------------------------------------------------------------------------
def build_year_index(T: int, period: int, start_date: Optional[datetime]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      x_year: (nY,) decimal years (end-of-year) if start_date else 0..nY-1
      idx2d : (nY, period) integer indices mapping year -> block indices
    Only keeps years with exactly `period` blocks observed.
    """
    T = int(T)
    period = int(period)
    idx = np.arange(T, dtype=int)

    if start_date is None or period <= 0:
        nY = T // period
        idx2d = idx[: nY * period].reshape(nY, period)
        return np.arange(nY, dtype=float), idx2d

    if 12 % period != 0:
        nY = T // period
        idx2d = idx[: nY * period].reshape(nY, period)
        return np.arange(nY, dtype=float), idx2d

    step = 12 // period
    years = np.empty(T, dtype=int)

    y0, m0 = start_date.year, start_date.month
    for i in range(T):
        mm = (m0 - 1) + i * step
        yy = y0 + (mm // 12)
        years[i] = int(yy)

    uniq = np.unique(years)
    starts: List[int] = []
    x_year: List[float] = []
    for y in uniq:
        I = idx[years == y]
        if I.size == period:
            starts.append(int(I.min()))
            x_year.append(float(y) + (12 - 0.5) / 12.0)

    if not starts:
        return np.array([], float), np.zeros((0, period), int)

    starts = np.array(starts, dtype=int)
    idx2d = starts[:, None] + np.arange(period, dtype=int)[None, :]
    return np.array(x_year, float), idx2d


# -----------------------------------------------------------------------------
# Parallel bisection for annual return levels
# -----------------------------------------------------------------------------
def annual_return_level_bisect(mu_year: np.ndarray, sigma: np.ndarray, xi: np.ndarray, N: float,
                               max_iter: int = 70, max_expand: int = 60) -> np.ndarray:
    mu_year = np.asarray(mu_year, float)
    S, nY, p = mu_year.shape
    sigma = np.clip(np.asarray(sigma, float).reshape(S), 1e-12, None)
    xi = np.asarray(xi, float).reshape(S)

    target = float(np.clip(1.0 - 1.0 / float(N), 1e-12, 1.0 - 1e-12))
    log_target = math.log(target)

    def g(z: np.ndarray) -> np.ndarray:
        G = gev_cdf(z[:, :, None], mu_year, sigma[:, None, None], xi[:, None, None])
        logG = np.log(np.clip(G, 1e-300, 1.0))
        return np.sum(logG, axis=2) - log_target

    step_scale = np.maximum(sigma, 1.0)[:, None]
    lo = np.min(mu_year, axis=2) - 10.0 * step_scale
    hi = np.max(mu_year, axis=2) + 10.0 * step_scale

    eps_xi = 1e-12
    has_lb = xi > eps_xi
    if np.any(has_lb):
        lb = np.max(mu_year[has_lb] - (sigma[has_lb, None, None] / xi[has_lb, None, None]), axis=2)
        lo[has_lb] = np.maximum(lo[has_lb], lb + 1e-10 * np.maximum(1.0, np.abs(lb)))

    g_lo = g(lo)
    g_hi = g(hi)

    need = np.isfinite(g_hi) & (g_hi < 0.0)
    k = 0
    while np.any(need) and k < int(max_expand):
        bump = (2.0 ** k) * 5.0 * step_scale
        hi = np.where(need, hi + bump, hi)
        g_hi = np.where(need, g(hi), g_hi)
        need = np.isfinite(g_hi) & (g_hi < 0.0)
        k += 1

    ok = np.isfinite(g_lo) & np.isfinite(g_hi) & (g_lo <= 0.0) & (g_hi >= 0.0)
    z = np.full((S, nY), np.nan, float)
    if not np.any(ok):
        return z

    lo_ok = lo.copy()
    hi_ok = hi.copy()
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo_ok + hi_ok)
        g_mid = g(mid)
        left = g_mid <= 0.0
        lo_ok = np.where(ok & left, mid, lo_ok)
        hi_ok = np.where(ok & (~left), mid, hi_ok)

    return np.where(ok, 0.5 * (lo_ok + hi_ok), z)


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def plot_ribbon(x, med, lo, hi, y_obs, ylabel, save_path, *,
                color="C0", obs_color="0.25", show=False, no_title=True, title=""):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
    if y_obs is not None:
        ax.plot(x, y_obs, lw=1.0, alpha=0.35, color=obs_color)
    ax.fill_between(x, lo, hi, alpha=0.20, color=color)
    ax.plot(x, med, lw=1.6, color=color)
    if (not no_title) and title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, save_path, show=show)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class DGEVReturnLevelsConfig:
    level: float = 0.90
    Ns: Tuple[int, ...] = (20, 50, 100)
    thresholds: Tuple[float, ...] = ()
    period_unit: str = "years"          # "blocks" or "years" (block R scaling only)
    start_date: Optional[str] = None
    skip_annual: bool = False
    window_blocks: int = 0
    window_years: int = 0
    log10_period: bool = True           # default ON: avoids unreadable 1e15 axes
    show: bool = False
    no_title: bool = True

    # numeric floors
    p_floor: float = 1e-300

    # plotting style
    color: str = "C0"
    obs_color: str = "0.25"
    prefix: str = ""


class DGEVReturnLevels:
    def __init__(self, *, draws: Dict[str, Any], meta: Dict[str, Any], npz_path: str,
                 cfg: Optional[DGEVReturnLevelsConfig] = None, out_dir: Optional[str] = None):
        self.draws = draws
        self.meta = dict(meta) if isinstance(meta, dict) else {}
        self.npz_path = str(npz_path)
        self.cfg = cfg or DGEVReturnLevelsConfig()
        self.out_dir = out_dir or os.path.join(os.path.dirname(self.npz_path), "return_levels")

        for k in ("mu", "sigma", "xi", "y"):
            if k not in self.draws:
                raise ValueError(f"Posterior must contain draws['{k}'].")

        self.mu = np.asarray(self.draws["mu"], float)          # (S,T)
        self.sigma = np.asarray(self.draws["sigma"], float).ravel()
        self.xi = np.asarray(self.draws["xi"], float).ravel()
        self.y_model = np.asarray(self.draws["y"], float).ravel()

        self.S, self.T = self.mu.shape
        if self.y_model.size != self.T:
            raise ValueError("draws['y'] length mismatch with draws['mu'].")
        if self.sigma.size != self.S or self.xi.size != self.S:
            raise ValueError("draws['sigma'], draws['xi'] must have length S.")

        self.period = int(self.meta.get("period", 12))
        self.minima = _detect_minima(self.meta)

        sd = _parse_date(self.cfg.start_date) or _parse_date(self.meta.get("start_date"))
        self.start_date = sd

        # plot/model transform: model = sgn * plot, plot = sgn * model
        self.sgn = -1.0 if self.minima else 1.0
        self.y_plot = self.sgn * self.y_model

        # x axis
        self.x_fine = np.arange(self.T, dtype=float)
        if self.start_date is not None and (12 % self.period == 0):
            step = 12 // self.period
            y0, m0 = self.start_date.year, self.start_date.month
            for i in range(self.T):
                mm = (m0 - 1) + i * step
                yy = y0 + (mm // 12)
                mo = (mm % 12) + 1
                self.x_fine[i] = float(yy) + (float(mo) - 0.5) / 12.0

    @classmethod
    def from_target_root(cls, *, target: Optional[str], root: str,
                         cfg: Optional[DGEVReturnLevelsConfig] = None, out_dir: Optional[str] = None):
        b = resolve_bundle(target=target, root=root)
        return cls(draws=b.draws, meta=b.meta, npz_path=b.npz_path, cfg=cfg, out_dir=out_dir)

    def _i0(self) -> int:
        w = int(self.cfg.window_blocks)
        return max(0, self.T - w) if w > 0 else 0

    def _j0(self, nY: int) -> int:
        w = int(self.cfg.window_years)
        return max(0, nY - w) if w > 0 else 0

    def compute(self) -> Dict[str, Any]:
        cfg = self.cfg
        i0 = self._i0()

        Ns = tuple(int(n) for n in cfg.Ns if int(n) > 1)
        thr = tuple(float(x) for x in cfg.thresholds)

        payload: Dict[str, Any] = {
            "x_fine": self.x_fine,
            "y_obs": self.y_plot,
            "period": int(self.period),
            "minima": bool(self.minima),
            "credible_level": float(cfg.level),
            "Ns": np.array(Ns, int),
            "thresholds": np.array(thr, float),
            "period_unit": str(cfg.period_unit),
            "npz_path": self.npz_path,
            "out_dir": self.out_dir,
        }

        # -------- Block return levels
        for N in Ns:
            pN = 1.0 - 1.0 / float(N)
            z_model = gev_ppf(pN, self.mu, self.sigma[:, None], self.xi[:, None])
            z_plot = self.sgn * z_model
            med, lo, hi = _summ(z_plot[:, i0:], cfg.level)
            payload[f"z_block_N{N}_med"] = med
            payload[f"z_block_N{N}_lo"] = lo
            payload[f"z_block_N{N}_hi"] = hi

        # -------- Block return periods
        blocks_per_year = float(self.period)
        block_scale = (1.0 / blocks_per_year) if cfg.period_unit == "years" else 1.0

        if thr:
            for y_plot in thr:
                y_model = self.sgn * y_plot
                G = gev_cdf(y_model, self.mu, self.sigma[:, None], self.xi[:, None])
                p = np.clip(1.0 - G, cfg.p_floor, 1.0)
                if cfg.log10_period:
                    log10R = -np.log10(p) + math.log10(block_scale)
                    med, lo, hi = _summ(log10R[:, i0:], cfg.level)
                    payload[f"log10R_block_y{y_plot:g}_med"] = med
                    payload[f"log10R_block_y{y_plot:g}_lo"] = lo
                    payload[f"log10R_block_y{y_plot:g}_hi"] = hi
                else:
                    R = (1.0 / p) * block_scale
                    med, lo, hi = _summ(R[:, i0:], cfg.level)
                    payload[f"R_block_y{y_plot:g}_med"] = med
                    payload[f"R_block_y{y_plot:g}_lo"] = lo
                    payload[f"R_block_y{y_plot:g}_hi"] = hi

        # -------- Annual scale
        if not cfg.skip_annual:
            x_year, idx2d = build_year_index(self.T, self.period, self.start_date)
            nY = int(idx2d.shape[0])
            payload["x_year"] = x_year
            payload["n_years"] = nY

            if nY > 0:
                j0 = self._j0(nY)
                mu_year = self.mu[:, idx2d]  # (S,nY,period)

                for N in Ns:
                    z_ann_model = annual_return_level_bisect(mu_year, self.sigma, self.xi, float(N))
                    z_ann_plot = self.sgn * z_ann_model
                    med, lo, hi = _summ(z_ann_plot[:, j0:], cfg.level)
                    payload[f"z_ann_N{N}_med"] = med
                    payload[f"z_ann_N{N}_lo"] = lo
                    payload[f"z_ann_N{N}_hi"] = hi

                if thr:
                    for y_plot in thr:
                        y_model = self.sgn * y_plot
                        G_full = gev_cdf(y_model, self.mu, self.sigma[:, None], self.xi[:, None])
                        Gj = G_full[:, idx2d]
                        logprod = np.sum(np.log(np.clip(Gj, 1e-300, 1.0)), axis=2)
                        p_ann = np.clip(-np.expm1(logprod), cfg.p_floor, 1.0)
                        if cfg.log10_period:
                            log10R = -np.log10(p_ann)
                            med, lo, hi = _summ(log10R[:, j0:], cfg.level)
                            payload[f"log10R_ann_y{y_plot:g}_med"] = med
                            payload[f"log10R_ann_y{y_plot:g}_lo"] = lo
                            payload[f"log10R_ann_y{y_plot:g}_hi"] = hi
                        else:
                            R = 1.0 / p_ann
                            med, lo, hi = _summ(R[:, j0:], cfg.level)
                            payload[f"R_ann_y{y_plot:g}_med"] = med
                            payload[f"R_ann_y{y_plot:g}_lo"] = lo
                            payload[f"R_ann_y{y_plot:g}_hi"] = hi

        return payload

    def save_plots(self, payload: Dict[str, Any]) -> None:
        cfg = self.cfg
        out = self.out_dir
        _ensure_dir(out)

        # banner
        print(f"[info] posterior npz   : {self.npz_path}")
        print(f"[info] output dir      : {out}")
        print(f"[info] minima detected  : {self.minima} (sgn={self.sgn:+.0f})")
        print(f"[info] period          : {self.period}")
        print(f"[info] start_date       : {self.start_date}")
        print(f"[info] draws S, T       : {self.S}, {self.T}")
        print(f"[info] Ns              : {payload['Ns'].tolist()}")
        print(f"[info] thresholds       : {payload['thresholds'].tolist()}")
        print(f"[info] annual enabled   : {not cfg.skip_annual}")
        print(f"[info] log10_period     : {cfg.log10_period}")

        i0 = self._i0()
        x_block = payload["x_fine"][i0:]
        y_obs = payload["y_obs"][i0:]
        pref = cfg.prefix or ""

        # block return levels
        for N in payload["Ns"].tolist():
            path = os.path.join(out, f"{pref}return_level_block_N{int(N)}.png")
            plot_ribbon(
                x_block,
                payload[f"z_block_N{int(N)}_med"],
                payload[f"z_block_N{int(N)}_lo"],
                payload[f"z_block_N{int(N)}_hi"],
                y_obs,
                ylabel=f"z(N={int(N)})",
                save_path=path,
                color=cfg.color, obs_color=cfg.obs_color, show=cfg.show, no_title=cfg.no_title,
            )

        # block return periods
        for y in payload["thresholds"].tolist():
            if cfg.log10_period:
                path = os.path.join(out, f"{pref}return_period_block_log10_y{y:g}.png")
                plot_ribbon(
                    x_block,
                    payload[f"log10R_block_y{y:g}_med"],
                    payload[f"log10R_block_y{y:g}_lo"],
                    payload[f"log10R_block_y{y:g}_hi"],
                    None,
                    ylabel=f"log10 Return period @ y*={y:g} ({cfg.period_unit})",
                    save_path=path,
                    color=cfg.color, obs_color=cfg.obs_color, show=cfg.show, no_title=cfg.no_title,
                )
            else:
                path = os.path.join(out, f"{pref}return_period_block_y{y:g}.png")
                plot_ribbon(
                    x_block,
                    payload[f"R_block_y{y:g}_med"],
                    payload[f"R_block_y{y:g}_lo"],
                    payload[f"R_block_y{y:g}_hi"],
                    None,
                    ylabel=f"Return period @ y*={y:g} ({cfg.period_unit})",
                    save_path=path,
                    color=cfg.color, obs_color=cfg.obs_color, show=cfg.show, no_title=cfg.no_title,
                )

        # annual
        if ("x_year" in payload) and int(payload.get("n_years", 0)) > 0:
            x_year = payload["x_year"]
            nY = int(payload["n_years"])
            j0 = self._j0(nY)
            x_ann = x_year[j0:]

            for N in payload["Ns"].tolist():
                path = os.path.join(out, f"{pref}return_level_annual_N{int(N)}.png")
                plot_ribbon(
                    x_ann,
                    payload[f"z_ann_N{int(N)}_med"],
                    payload[f"z_ann_N{int(N)}_lo"],
                    payload[f"z_ann_N{int(N)}_hi"],
                    None,
                    ylabel=f"z_ann(N={int(N)})",
                    save_path=path,
                    color=cfg.color, obs_color=cfg.obs_color, show=cfg.show, no_title=cfg.no_title,
                )

            for y in payload["thresholds"].tolist():
                if cfg.log10_period:
                    path = os.path.join(out, f"{pref}return_period_annual_log10_y{y:g}.png")
                    plot_ribbon(
                        x_ann,
                        payload[f"log10R_ann_y{y:g}_med"],
                        payload[f"log10R_ann_y{y:g}_lo"],
                        payload[f"log10R_ann_y{y:g}_hi"],
                        None,
                        ylabel=f"log10 Annual return period @ y*={y:g} (years)",
                        save_path=path,
                        color=cfg.color, obs_color=cfg.obs_color, show=cfg.show, no_title=cfg.no_title,
                    )
                else:
                    path = os.path.join(out, f"{pref}return_period_annual_y{y:g}.png")
                    plot_ribbon(
                        x_ann,
                        payload[f"R_ann_y{y:g}_med"],
                        payload[f"R_ann_y{y:g}_lo"],
                        payload[f"R_ann_y{y:g}_hi"],
                        None,
                        ylabel=f"Annual return period @ y*={y:g} (years)",
                        save_path=path,
                        color=cfg.color, obs_color=cfg.obs_color, show=cfg.show, no_title=cfg.no_title,
                    )

        payload_path = os.path.join(out, "return_levels_payload.npz")
        np.savez_compressed(payload_path, **payload)
        print(f"[save] {payload_path}")
        print("[done] return levels / return periods written.")

    def run(self) -> Dict[str, Any]:
        payload = self.compute()
        self.save_plots(payload)
        return payload


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_csv_floats(s: Optional[str]) -> List[float]:
    if s is None:
        return []
    out: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Compute DGEV return levels / return periods from posterior draws.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", type=str, default=None, help="Run dir or posterior.npz.")
    p.add_argument("--root", type=str, default="results/simulations/DGEV_NCP_LASSO", help="Search root if --target omitted.")
    p.add_argument("--out", type=str, default=None, help="Output directory (default: <run>/return_levels).")

    p.add_argument("--level", type=float, default=0.90)
    p.add_argument("--N", type=str, default="20,50,100")
    p.add_argument("--threshold", type=str, default=None)

    p.add_argument("--period-unit", choices=["blocks", "years"], default="years")
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--skip-annual", action="store_true", default=False)

    p.add_argument("--window-blocks", type=int, default=0)
    p.add_argument("--window-years", type=int, default=0)

    p.add_argument("--log10-period", action="store_true", default=False)
    p.add_argument("--show", action="store_true", default=False)

    args = p.parse_args()

    Ns = tuple(int(round(x)) for x in _parse_csv_floats(args.N) if x > 1)
    thr = tuple(_parse_csv_floats(args.threshold)) if args.threshold else ()

    cfg = DGEVReturnLevelsConfig(
        level=float(args.level),
        Ns=Ns,
        thresholds=thr,
        period_unit=str(args.period_unit),
        start_date=str(args.start_date) if args.start_date else None,
        skip_annual=bool(args.skip_annual),
        window_blocks=int(args.window_blocks),
        window_years=int(args.window_years),
        log10_period=bool(args.log10_period),
        show=bool(args.show),
    )

    b = resolve_bundle(target=args.target, root=args.root)
    out_dir = args.out or os.path.join(os.path.dirname(b.npz_path), "return_levels")
    print(f"[info] resolved output dir: {out_dir}")
    rl = DGEVReturnLevels(draws=b.draws, meta=b.meta, npz_path=b.npz_path, cfg=cfg, out_dir=out_dir)
    rl.run()

if __name__ == "__main__":
    main()
