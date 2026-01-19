# %% simulator/dgev_laplace_risk.py
from __future__ import annotations
"""
Bayesian exceedance risk for the DGEV (Laplace NCP) posterior
============================================================

This module computes *event probabilities* (fine scale / block scale) and their
annual aggregates **for every posterior draw**, for thresholds provided on the
PLOT/original scale.

It also provides:
- plotting on linear OR log y-scale,
- plotting either event probability OR return period (years),
- overlaying multiple thresholds on one plot,
- printing (and saving) exceedance probabilities / return periods at selected times,
- safe handling of extremely small probabilities via an RP cap (default 10,000 years).

Definitions
-----------
For a threshold y* on the PLOT scale, for posterior draw s:

Fine scale (block t):
  p_t^{(s)}(y*) = P(event at t | draw s)

Maxima series:
  event="gt": p_t = P(Y_t > y*) = 1 - G_t(y*)

Minima (sign-flip run where model fits Z_t=-Y_t as maxima):
  thresholds are given on PLOT scale y*, we map to model scale z*=-y* and compute
  the PLOT-scale event probability exactly.

Annual aggregation (conditional independence within year group S_j):
  p_year^{(s)}(y*) = 1 - ∏_{t in S_j} (1 - p_t^{(s)}(y*))
implemented stably via log1p/expm1.

Return periods (years)
----------------------
Let period = blocks per year (e.g. 12 for monthly maxima, 4 for seasonal, ...)

Fine scale (block t):
  RP_t(years) = 1 / (p_t * period)

Annual:
  RP_year(years) = 1 / p_year

IMPORTANT: Very small probabilities cause huge RPs and log(0) issues.
We enforce a cap rp_cap_years (default 10,000 years) by flooring probabilities:
  - annual: p >= 1 / rp_cap_years
  - fine:   p >= 1 / (rp_cap_years * period)
You can choose policy:
  - "clip": floor p (default; keeps curves continuous)
  - "mask": set too-small values to NaN (breaks curves/bands)

Public API used by your Uccle wrapper
-------------------------------------
- resolve_bundle(target, root)
- apply_burn_thin(draws, meta, burn=..., thin=...)
- class DGEVLaplaceRisk with:
    compute(...)
    save(...)
    plot_fine(...)
    plot_annual(...)
    print_at_times(...)

Outputs (typical)
-----------------
<run>/risk/
  exceedance_probs.npz
  risk_monthly_thr_<thr>.png or risk_monthly_multi_*.png
  risk_annual_thr_<thr>.png   or risk_annual_multi_*.png
  risk_report_times_<fine|annual>_<prob|rp>.csv  (optional)
"""

import os
import re
import math
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Make optimization package visible (mirrors your plotter/forecast pattern)
import sys  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from optimization.posterior_bundle import load_posterior, find_latest_run
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e


# =============================================================================
# Small I/O + parsing utils
# =============================================================================
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    """Accepts YYYY, YYYY-MM, YYYY-MM-DD."""
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    parts = [int(p) for p in ss.split("-")]
    if len(parts) == 1:
        return datetime(parts[0], 1, 1)
    if len(parts) == 2:
        return datetime(parts[0], parts[1], 1)
    if len(parts) == 3:
        return datetime(parts[0], parts[1], parts[2])
    raise ValueError("date must be YYYY, YYYY-MM, or YYYY-MM-DD")


def _parse_csv_floats(s: Optional[str]) -> List[float]:
    if s is None:
        return []
    out: List[float] = []
    for tok in str(s).split(","):
        tt = tok.strip()
        if tt:
            out.append(float(tt))
    return out


def _parse_csv_strings(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    out: List[str] = []
    for tok in str(s).split(","):
        tt = tok.strip()
        if tt:
            out.append(tt)
    return out


def _decimal_year_from_ym(y: int, m: int) -> float:
    return float(y) + (float(m) - 0.5) / 12.0


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


def resolve_bundle(*, target: Optional[str], root: str) -> Any:
    """
    Mirrors your other tooling:
      1) if target is given: load it
      2) else try find_latest_run(root) for standard posterior.npz layout
      3) else fallback to recursive search for posterior*.npz
    """
    if target:
        return load_posterior(target)

    print(f"[info] searching latest posterior run under: {root!r}")
    run_path = find_latest_run(root=root)
    if run_path is not None:
        print(f"[info] using latest run: {run_path}")
        return load_posterior(run_path)

    npz_path = find_latest_posterior_npz(root)
    if npz_path is None:
        print(
            f"[error] No posterior runs found under {root!r}.\n"
            f"  → Tried find_latest_run() (posterior.npz) and recursive search (posterior*.npz).\n"
            f"  → Either run the sampler first, or provide --target."
        )
        raise SystemExit(1)

    print(f"[info] find_latest_run found nothing; using latest npz: {npz_path}")
    return load_posterior(npz_path)


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
    for k in ("mu", "x", "sigma", "sigma2", "xi", "Q_alpha", "s_alpha", "gamma0"):
        if k in draws and isinstance(draws[k], np.ndarray) and np.asarray(draws[k]).ndim >= 1:
            n_samp = int(np.asarray(draws[k]).shape[0])
            break
    if n_samp is None:
        print("[warn] could not infer chain length; skipping burn/thin.")
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


# =============================================================================
# Meta helpers (minima detection) + state indexing
# =============================================================================
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


def _detect_minima_from_meta(meta: Dict[str, Any]) -> bool:
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

    dt = meta.get("data_transform", None)
    if isinstance(dt, str):
        s = dt.strip().lower()
        if any(tok in s for tok in ("negate", "minus", "signflip", "flip_sign", "neg")):
            return True

    ser = meta.get("series", None)
    if isinstance(ser, str):
        ss = ser.strip()
        if ss in {"TNn", "TXn"}:
            return True
        if len(ss) >= 2 and ss.endswith("n") and ss[:-1].isalpha():
            return True

    return False


def _extract_state_indices(layout: List[str]) -> Tuple[int, int, int]:
    if "alpha" not in layout or "beta" not in layout:
        raise ValueError("meta['layout'] must contain 'alpha' and 'beta'.")
    ia = layout.index("alpha")
    ib = layout.index("beta")
    g_indices = [i for i, nm in enumerate(layout) if str(nm).startswith("g")]
    if not g_indices:
        raise ValueError("meta['layout'] must contain seasonal states g1..g{p-1}.")
    ig1 = min(g_indices)
    return ia, ib, ig1


def _get_sigma(draws: Dict[str, np.ndarray], S: int) -> np.ndarray:
    if "sigma" in draws:
        sig = np.asarray(draws["sigma"], float).ravel()
    elif "sigma2" in draws:
        sig = np.sqrt(np.clip(np.asarray(draws["sigma2"], float).ravel(), 0.0, None))
    else:
        raise ValueError("Posterior draws must include 'sigma' or 'sigma2'.")
    if sig.size != S:
        raise ValueError("sigma length mismatch.")
    return np.clip(sig, 1e-12, None)


def _get_xi(draws: Dict[str, np.ndarray], S: int) -> np.ndarray:
    if "xi" not in draws:
        raise ValueError("Posterior draws must include 'xi'.")
    xi = np.asarray(draws["xi"], float).ravel()
    if xi.size != S:
        raise ValueError("xi length mismatch.")
    return xi


def _get_Q(draws: Dict[str, np.ndarray], S: int, name_Q: str, name_s: str) -> np.ndarray:
    if name_Q in draws:
        q = np.asarray(draws[name_Q], float).ravel()
        if q.size != S:
            raise ValueError(f"{name_Q} length mismatch.")
        return np.clip(q, 0.0, None)
    if name_s in draws:
        s = np.asarray(draws[name_s], float).ravel()
        if s.size != S:
            raise ValueError(f"{name_s} length mismatch.")
        return np.clip(s * s, 0.0, None)
    raise ValueError(f"Need '{name_Q}' or '{name_s}' in posterior draws.")


# =============================================================================
# Seasonal design / rotation + annual grouping
# =============================================================================
def _build_season_design(L: int, period: int) -> np.ndarray:
    p = int(period)
    if p < 2:
        return np.zeros((L, 0), float)
    K = p - 1
    S = np.zeros((L, K), float)
    for t in range(L):
        season = t % p
        if season < K:
            S[t, season] = 1.0
        else:
            S[t, :] = -1.0
    return S


def _season_rotation_matrix(K: int) -> np.ndarray:
    if K <= 0:
        return np.zeros((0, 0), float)
    R = np.zeros((K, K), float)
    R[0, :] = -1.0
    if K > 1:
        R[1:, :-1] = np.eye(K - 1)
    return R


def _annual_groups_from_index(
    L_full: int,
    T_obs: int,
    period: int,
    *,
    start_year: Optional[int],
    start_month: Optional[int],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Returns:
      x_year (n_years,) : x-coordinate per year group
      groups : list of arrays of indices in each year group

    If (start_year,start_month) known AND period divides 12, group by calendar year.
    Otherwise group into consecutive blocks of length `period` from the start.
    """
    p = int(period)
    if p <= 0:
        raise ValueError("period must be >= 1")

    idx = np.arange(L_full)

    if start_year is not None and start_month is not None and p in (12, 6, 4, 3, 2, 1) and (12 % p == 0):
        step_months = 12 // p
        year = np.zeros(L_full, int)
        month = np.zeros(L_full, int)
        y0, m0 = int(start_year), int(start_month)
        for i in range(L_full):
            mm = (m0 - 1) + i * step_months
            yy = y0 + (mm // 12)
            mo = (mm % 12) + 1
            year[i] = yy
            month[i] = mo

        years = np.unique(year)
        groups: List[np.ndarray] = []
        x_year: List[float] = []
        for yy in years:
            I = idx[year == yy]
            if I.size == p:
                groups.append(I)
                x_year.append(_decimal_year_from_ym(int(yy), 12))  # anchor at Dec
        return np.array(x_year, float), groups

    n_years = L_full // p
    groups = [idx[j * p : (j + 1) * p] for j in range(n_years)]
    x_year = np.arange(n_years, dtype=float)
    return x_year, groups


# =============================================================================
# GEV CDF + event-prob mapping (supports minima sign-flip)
# =============================================================================
def _gev_cdf_broadcast(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Broadcast-safe GEV CDF: G(y) = P(Y <= y).

    Support handling:
      - xi < 0 : upper endpoint at mu - sigma/xi  -> CDF=1 for y >= endpoint
      - xi > 0 : lower endpoint at mu - sigma/xi  -> CDF=0 for y <= endpoint
    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    sigma = np.clip(np.asarray(sigma, float), 1e-12, None)
    xi = np.asarray(xi, float)

    z = (y - mu) / sigma
    out = np.empty(np.broadcast(y, mu, sigma, xi).shape, float)

    mask0 = np.abs(xi) < 1e-12
    if np.any(mask0):
        z0 = np.where(mask0, z, 0.0)
        c0 = np.exp(-np.exp(-z0))
        out = np.where(mask0, c0, out)

    if np.any(~mask0):
        zn = np.where(~mask0, z, 0.0)
        xin = np.where(~mask0, xi, 1.0)

        t = 1.0 + xin * zn  # must be > 0
        a = np.power(t, -1.0 / xin, where=(t > 0.0), out=np.full_like(t, np.inf))
        cn = np.exp(-a)

        cn = np.where((t <= 0.0) & (xin > 0.0), 0.0, cn)  # below lower endpoint
        cn = np.where((t <= 0.0) & (xin < 0.0), 1.0, cn)  # above upper endpoint

        out = np.where(~mask0, cn, out)

    return np.clip(out, 0.0, 1.0)


def _event_prob_from_model_cdf(cdf_model: np.ndarray, *, minima: bool, event: str) -> np.ndarray:
    """
    Map model-scale CDF evaluated at z* to event probability on PLOT/original scale.

    - minima=False (model Y): cdf_model = P(Y<=y*)
         event="gt": P(Y>y*) = 1-cdf
         event="lt": P(Y<y*) = cdf

    - minima=True (model Z=-Y): cdf_model = P(Z<=z*) with z*=-y*
         event="gt": P(Y>y*) = P(Z<-y*) = cdf_model
         event="lt": P(Y<y*) = P(Z>-y*) = 1-cdf_model
    """
    ev = str(event).strip().lower()
    if ev not in ("gt", "lt"):
        raise ValueError("event must be 'gt' or 'lt'.")
    if not minima:
        return (1.0 - cdf_model) if (ev == "gt") else cdf_model
    return cdf_model if (ev == "gt") else (1.0 - cdf_model)


# =============================================================================
# Probability floors / return-period conversion with 10k-year cap
# =============================================================================
def _p_floor_from_rp_cap(*, rp_cap_years: float, period: int, annual: bool) -> float:
    """
    Floor probability so return period does not exceed rp_cap_years.
    annual: p >= 1/rp_cap_years
    fine:   p >= 1/(rp_cap_years * period)
    """
    cap = float(rp_cap_years)
    if cap <= 0:
        raise ValueError("rp_cap_years must be > 0")
    if annual:
        return 1.0 / cap
    p = int(period)
    if p <= 0:
        raise ValueError("period must be >= 1")
    return 1.0 / (cap * float(p))


def _apply_small_prob_policy(p: np.ndarray, *, p_floor: float, policy: str) -> np.ndarray:
    """
    policy:
      - 'clip': replace p < p_floor by p_floor
      - 'mask': replace p < p_floor by NaN
    """
    policy = str(policy).strip().lower()
    if policy not in ("clip", "mask"):
        raise ValueError("small_prob_policy must be 'clip' or 'mask'")
    p = np.asarray(p, float)
    if policy == "clip":
        return np.clip(p, p_floor, 1.0 - 1e-15)
    out = p.copy()
    out[out < p_floor] = np.nan
    # also avoid exactly 1 for log safety
    out[out >= 1.0] = 1.0 - 1e-15
    return out


def _prob_to_rp_years(
    p: np.ndarray,
    *,
    period: int,
    annual: bool,
    rp_cap_years: float,
    small_prob_policy: str,
) -> np.ndarray:
    """
    Convert probability array to return period in YEARS, applying rp cap policy
    through a probability floor (or NaN masking).

    annual: RP = 1/p
    fine:   RP = 1/(p*period)
    """
    p_floor = _p_floor_from_rp_cap(rp_cap_years=rp_cap_years, period=period, annual=annual)
    p_eff = _apply_small_prob_policy(p, p_floor=p_floor, policy=small_prob_policy)
    denom = 1.0 if annual else float(int(period))
    rp = 1.0 / (p_eff * denom)
    return rp


def _nan_summarize(draws_3d: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    draws_3d: (S, M, L) possibly with NaNs
    returns median/lo/hi over S -> (M, L)
    """
    loq = (1.0 - float(level)) / 2.0
    hiq = 1.0 - loq
    med = np.nanquantile(draws_3d, 0.5, axis=0)
    lo = np.nanquantile(draws_3d, loq, axis=0)
    hi = np.nanquantile(draws_3d, hiq, axis=0)
    return med, lo, hi


def _thr_tag(thresholds: np.ndarray) -> str:
    toks = []
    for t in thresholds:
        s = f"{float(t):g}".replace("-", "m").replace(".", "p")
        toks.append(s)
    return "_".join(toks)


def _set_yscale(ax: plt.Axes, yscale: str) -> None:
    ys = str(yscale).strip().lower()
    if ys not in ("linear", "log"):
        raise ValueError("yscale must be 'linear' or 'log'")
    if ys == "log":
        ax.set_yscale("log")


# =============================================================================
# Main class
# =============================================================================
@dataclass(frozen=True)
class RiskResult:
    thresholds: np.ndarray        # (M,) on plot scale
    event: str                    # 'gt' or 'lt' on plot scale
    minima: bool

    x_full: np.ndarray            # (L,)
    p_fine: np.ndarray            # (S, M, L) draw-level fine probs
    split_x_fine: float

    x_year: np.ndarray            # (nY,)
    p_year: np.ndarray            # (S, M, nY) draw-level annual probs
    split_x_year: float

    start_year: Optional[int]
    start_month: Optional[int]
    period: int


class DGEVLaplaceRisk:
    """
    Compute Bayesian exceedance probabilities for each posterior draw.
    """

    def __init__(self, draws: Dict[str, Any], meta: Dict[str, Any], *, npz_path: str = ""):
        self.draws = draws
        self.meta = meta
        self.npz_path = str(npz_path)

        if "y" not in draws:
            raise ValueError("Posterior draws must include 'y'.")
        self.y_model = np.asarray(draws["y"], float).ravel()
        self.T = int(self.y_model.size)

        if "mu" not in draws:
            raise ValueError("Posterior draws must include 'mu' with shape (S, T).")
        self.mu_obs = np.asarray(draws["mu"], float)  # (S, T) on MODEL scale
        if self.mu_obs.ndim != 2:
            raise ValueError("draws['mu'] must be 2D (S, T).")

        self.S = int(self.mu_obs.shape[0])
        if int(self.mu_obs.shape[1]) != self.T:
            raise ValueError("draws['mu'] T mismatch with draws['y'].")

        self.period = int(meta.get("period", 12))
        self.layout = list(meta.get("layout", [])) if isinstance(meta.get("layout", None), (list, tuple)) else None
        self.minima = _detect_minima_from_meta(meta)

        self.sigma = _get_sigma(draws, self.S)
        self.xi = _get_xi(draws, self.S)

        # Optional forecast extension needs x + gamma0 + Q's
        self._has_state = ("x" in draws and np.asarray(draws["x"]).ndim == 3 and self.layout is not None)
        self._has_gamma0 = ("gamma0" in draws)
        self._has_Q = (
            any(k in draws for k in ("Q_alpha", "s_alpha"))
            and any(k in draws for k in ("Q_beta", "s_beta"))
            and any(k in draws for k in ("Q_gamma", "s_gamma"))
        )

    # ----------------------------- time axis ----------------------------- #
    def _build_x_axis(
        self,
        *,
        L_full: int,
        start_date: Optional[datetime],
    ) -> Tuple[np.ndarray, float, Optional[int], Optional[int]]:
        """
        Returns:
          x_full (L_full,)
          split_x (float): location of last observed point
          start_year, start_month if calendar-based, else None,None
        """
        if start_date is not None and self.period in (12, 6, 4, 3, 2, 1) and (12 % self.period == 0):
            step_months = 12 // self.period
            years = np.zeros(L_full, int)
            months = np.zeros(L_full, int)
            y0, m0 = start_date.year, start_date.month
            for i in range(L_full):
                mm = (m0 - 1) + i * step_months
                yy = y0 + (mm // 12)
                mo = (mm % 12) + 1
                years[i] = yy
                months[i] = mo
            x_full = np.array([_decimal_year_from_ym(int(yy), int(mo)) for yy, mo in zip(years, months)], float)
            split_x = float(x_full[self.T - 1])
            return x_full, split_x, int(y0), int(m0)

        x_full = np.arange(L_full, dtype=float)
        split_x = float(self.T - 1)
        return x_full, split_x, None, None

    # ----------------------------- forecast (optional) ----------------------------- #
    def _simulate_mu_future(
        self,
        *,
        horizon: int,
        seed: int,
        start_year: Optional[int],
        start_month: Optional[int],
    ) -> np.ndarray:
        """
        Simulate future mu_{T:T+H-1} on MODEL scale for each posterior draw.

        Requires draws['x'], draws['gamma0'], and Q_*.
        """
        H = int(horizon)
        if H <= 0:
            return np.zeros((self.S, 0), float)

        if not (self._has_state and self._has_gamma0 and self._has_Q):
            raise ValueError(
                "Forecast extension requires posterior draws: 'x' (S,T,dim), 'gamma0' (S,K), "
                "and Q_alpha/Q_beta/Q_gamma (or s_*)."
            )

        x = np.asarray(self.draws["x"], float)
        S_draws, T_x, dim = x.shape
        if S_draws != self.S or T_x != self.T:
            raise ValueError("draws['x'] shape mismatch.")

        period = int(self.period)
        K = period - 1
        if K <= 0:
            raise ValueError("period must be >= 2 for seasonal component.")

        ia, ib, ig1 = _extract_state_indices(list(self.layout))  # type: ignore[arg-type]
        if ig1 + K > dim:
            raise ValueError("State vector does not contain a full seasonal block g1..gK.")

        Q_alpha = _get_Q(self.draws, self.S, "Q_alpha", "s_alpha")
        Q_beta = _get_Q(self.draws, self.S, "Q_beta", "s_beta")
        Q_gamma = _get_Q(self.draws, self.S, "Q_gamma", "s_gamma")

        gamma0 = np.asarray(self.draws["gamma0"], float)
        if gamma0.shape != (self.S, K):
            raise ValueError(f"gamma0 must have shape (S,{K}).")

        L_full = self.T + H
        S_design_full = _build_season_design(L_full, period)  # (L, K)
        baseline_full = np.einsum("sk,tk->st", gamma0, S_design_full)  # (S, L)

        alpha = x[:, -1, ia].copy()
        beta = x[:, -1, ib].copy()
        gamma_dyn = x[:, -1, ig1 : ig1 + K].copy()

        R = _season_rotation_matrix(K)
        RT = R.T
        e1 = np.zeros((K,), float)
        e1[0] = 1.0

        rng = np.random.default_rng(int(seed))
        mu_future = np.zeros((self.S, H), float)
        for h in range(H):
            t = self.T + h
            alpha = alpha + beta + rng.normal(0.0, np.sqrt(Q_alpha), size=self.S)
            beta = beta + rng.normal(0.0, np.sqrt(Q_beta), size=self.S)

            epsg = rng.normal(0.0, np.sqrt(Q_gamma), size=self.S)
            gamma_dyn = gamma_dyn @ RT + epsg[:, None] * e1[None, :]

            mu_future[:, h] = alpha + gamma_dyn[:, 0] + baseline_full[:, t]

        return mu_future

    # ----------------------------- compute ----------------------------- #
    def compute(
        self,
        *,
        thresholds: Sequence[float],
        event: Optional[str] = None,
        horizon: int = 0,
        seed: int = 123,
        start_date: Optional[datetime] = None,
    ) -> RiskResult:
        """
        Compute draw-level fine probabilities for all time points (observed + optional future),
        and annual aggregates.

        thresholds: on PLOT/original scale
        event: 'gt' or 'lt' on PLOT scale (default: maxima->'gt', minima->'lt')
        """
        thr = np.asarray(list(thresholds), float).ravel()
        if thr.size == 0:
            raise ValueError("Provide at least one threshold.")
        M = int(thr.size)

        if event is None:
            event = "lt" if self.minima else "gt"
        event = str(event).strip().lower()
        if event not in ("gt", "lt"):
            raise ValueError("event must be 'gt' or 'lt'.")

        H = int(horizon)
        if H < 0:
            raise ValueError("horizon must be >= 0.")
        L_full = self.T + H

        x_full, split_x_fine, sy, sm = self._build_x_axis(L_full=L_full, start_date=start_date)

        # mu_full on MODEL scale: observed is stored; future optional via simulation
        mu_future = self._simulate_mu_future(horizon=H, seed=seed, start_year=sy, start_month=sm) if H > 0 else np.zeros((self.S, 0))
        mu_full = np.concatenate([self.mu_obs, mu_future], axis=1)  # (S, L)

        # map thresholds to model-scale for minima sign-flip runs
        thr_model = (-thr) if self.minima else thr

        # Broadcast shapes to (S, M, L)
        muSL = mu_full[:, None, :]             # (S,1,L)
        sigS = self.sigma[:, None, None]       # (S,1,1)
        xiS = self.xi[:, None, None]           # (S,1,1)
        yM = thr_model[None, :, None]          # (1,M,1)

        cdf_model = _gev_cdf_broadcast(yM, muSL, sigS, xiS)  # (S,M,L)
        p_fine = _event_prob_from_model_cdf(cdf_model, minima=self.minima, event=event)  # (S,M,L)
        p_fine = np.clip(p_fine, 0.0, 1.0)

        # Annual aggregation
        x_year, groups = _annual_groups_from_index(L_full, self.T, self.period, start_year=sy, start_month=sm)
        nY = len(groups)
        p_year = np.zeros((self.S, M, nY), float)

        last_obs_year = -1
        for j, I in enumerate(groups):
            I = np.asarray(I, int)
            pI = p_fine[:, :, I]  # (S,M,p)

            # stable: log(prod(1-p)) then 1-exp(log_surv)
            log_surv = np.sum(np.log1p(-pI), axis=2)  # (S,M)
            p_year[:, :, j] = -np.expm1(log_surv)

            if np.all(I < self.T):
                last_obs_year = j

        split_x_year = float(x_year[last_obs_year]) if (last_obs_year >= 0 and x_year.size) else float("nan")

        return RiskResult(
            thresholds=thr,
            event=event,
            minima=bool(self.minima),
            x_full=x_full,
            p_fine=p_fine,
            split_x_fine=float(split_x_fine),
            x_year=x_year,
            p_year=p_year,
            split_x_year=float(split_x_year),
            start_year=sy,
            start_month=sm,
            period=int(self.period),
        )

    # ----------------------------- save raw ----------------------------- #
    def save(self, rr: RiskResult, *, out_path: str) -> None:
        _ensure_dir(os.path.dirname(out_path))
        np.savez_compressed(
            out_path,
            thresholds=rr.thresholds,
            event=rr.event,
            minima=rr.minima,
            period=rr.period,
            start_year=(-1 if rr.start_year is None else int(rr.start_year)),
            start_month=(-1 if rr.start_month is None else int(rr.start_month)),
            x_full=rr.x_full,
            p_fine=rr.p_fine,     # (S,M,L)
            split_x_fine=float(rr.split_x_fine),
            x_year=rr.x_year,
            p_year=rr.p_year,     # (S,M,nY)
            split_x_year=float(rr.split_x_year),
        )
        print(f"[save] {out_path}")

    # =============================================================================
    # Plotting (prob or RP; linear or log; single or multi-threshold overlay)
    # =============================================================================
    def plot_fine(
        self,
        rr: RiskResult,
        *,
        out_dir: str,
        level: float = 0.90,
        window: Optional[int] = 240,
        show: bool = False,
        # NEW:
        y_mode: str = "prob",                 # "prob" or "rp"
        yscale: str = "linear",               # "linear" or "log"
        combine: bool = False,                # overlay all thresholds on one plot
        band: Optional[bool] = None,          # bands (default True when single; False when combine)
        legend: Optional[bool] = None,        # legend (default False when single; True when combine)
        rp_cap_years: float = 10_000.0,       # 10k-year guardrail
        small_prob_policy: str = "clip",      # "clip" or "mask"
    ) -> None:
        _ensure_dir(out_dir)

        y_mode = str(y_mode).strip().lower()
        if y_mode not in ("prob", "rp"):
            raise ValueError("y_mode must be 'prob' or 'rp'")
        yscale = str(yscale).strip().lower()
        if yscale not in ("linear", "log"):
            raise ValueError("yscale must be 'linear' or 'log'")

        if band is None:
            band = (not bool(combine))
        if legend is None:
            legend = bool(combine)

        draws = rr.p_fine  # (S,M,L)

        # Windowing
        L = int(rr.x_full.size)
        i0 = 0
        if window is not None:
            w = max(1, int(window))
            i0 = max(0, L - w)
        x = rr.x_full[i0:]

        # Convert to y-draws (prob or RP)
        if y_mode == "prob":
            p_floor = _p_floor_from_rp_cap(rp_cap_years=rp_cap_years, period=rr.period, annual=False)
            y_draws = _apply_small_prob_policy(draws[:, :, i0:], p_floor=p_floor, policy=small_prob_policy)
            ylabel = "event probability"
        else:
            y_draws = _prob_to_rp_years(
                draws[:, :, i0:],
                period=rr.period,
                annual=False,
                rp_cap_years=rp_cap_years,
                small_prob_policy=small_prob_policy,
            )
            ylabel = "return period (years)"

        med, lo, hi = _nan_summarize(y_draws, level=float(level))  # (M, Lwin)

        if combine:
            fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
            for m, thr in enumerate(rr.thresholds):
                ax.plot(x, med[m, :], lw=1.6, label=f"{thr:g}")
                if band:
                    ax.fill_between(x, lo[m, :], hi[m, :], alpha=0.18)

            ax.axvline(float(rr.split_x_fine), lw=1.0, alpha=0.8)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("time")
            ax.grid(True, alpha=0.25)
            _set_yscale(ax, yscale)
            if legend:
                ax.legend(title="threshold", fontsize=9, title_fontsize=9, frameon=False)

            plt.tight_layout()
            tag = _thr_tag(rr.thresholds)
            path = os.path.join(out_dir, f"risk_monthly_multi_{y_mode}_{yscale}_thr_{tag}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)
            print(f"[save] {path}")
            return

        # one figure per threshold
        for m, thr in enumerate(rr.thresholds):
            fig, ax = plt.subplots(1, 1, figsize=(12, 3.6))
            ax.plot(x, med[m, :], lw=1.6)
            if band:
                ax.fill_between(x, lo[m, :], hi[m, :], alpha=0.25)
            ax.axvline(float(rr.split_x_fine), lw=1.0, alpha=0.8)

            ax.set_ylabel(ylabel)
            ax.set_xlabel("time")
            ax.grid(True, alpha=0.25)
            _set_yscale(ax, yscale)

            plt.tight_layout()
            path = os.path.join(out_dir, f"risk_monthly_thr_{thr:g}_{y_mode}_{yscale}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)
            print(f"[save] {path}")

    def plot_annual(
        self,
        rr: RiskResult,
        *,
        out_dir: str,
        level: float = 0.90,
        window_years: Optional[int] = 60,
        show: bool = False,
        # NEW:
        y_mode: str = "prob",                 # "prob" or "rp"
        yscale: str = "linear",               # "linear" or "log"
        combine: bool = False,
        band: Optional[bool] = None,
        legend: Optional[bool] = None,
        rp_cap_years: float = 10_000.0,
        small_prob_policy: str = "clip",
    ) -> None:
        _ensure_dir(out_dir)

        y_mode = str(y_mode).strip().lower()
        if y_mode not in ("prob", "rp"):
            raise ValueError("y_mode must be 'prob' or 'rp'")
        yscale = str(yscale).strip().lower()
        if yscale not in ("linear", "log"):
            raise ValueError("yscale must be 'linear' or 'log'")

        if band is None:
            band = (not bool(combine))
        if legend is None:
            legend = bool(combine)

        draws = rr.p_year  # (S,M,nY)

        # Windowing
        nY = int(rr.x_year.size)
        j0 = 0
        if window_years is not None:
            w = max(1, int(window_years))
            j0 = max(0, nY - w)
        x = rr.x_year[j0:]

        # Convert to y-draws
        if y_mode == "prob":
            p_floor = _p_floor_from_rp_cap(rp_cap_years=rp_cap_years, period=rr.period, annual=True)
            y_draws = _apply_small_prob_policy(draws[:, :, j0:], p_floor=p_floor, policy=small_prob_policy)
            ylabel = "annual event probability"
        else:
            y_draws = _prob_to_rp_years(
                draws[:, :, j0:],
                period=rr.period,
                annual=True,
                rp_cap_years=rp_cap_years,
                small_prob_policy=small_prob_policy,
            )
            ylabel = "return period (years)"

        med, lo, hi = _nan_summarize(y_draws, level=float(level))  # (M, nYwin)

        if combine:
            fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
            for m, thr in enumerate(rr.thresholds):
                ax.plot(x, med[m, :], lw=1.6, label=f"{thr:g}")
                if band:
                    ax.fill_between(x, lo[m, :], hi[m, :], alpha=0.18)

            if np.isfinite(rr.split_x_year):
                ax.axvline(float(rr.split_x_year), lw=1.0, alpha=0.8)

            ax.set_ylabel(ylabel)
            ax.set_xlabel("year")
            ax.grid(True, alpha=0.25)
            _set_yscale(ax, yscale)
            if legend:
                ax.legend(title="threshold", fontsize=9, title_fontsize=9, frameon=False)

            plt.tight_layout()
            tag = _thr_tag(rr.thresholds)
            path = os.path.join(out_dir, f"risk_annual_multi_{y_mode}_{yscale}_thr_{tag}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)
            print(f"[save] {path}")
            return

        for m, thr in enumerate(rr.thresholds):
            fig, ax = plt.subplots(1, 1, figsize=(12, 3.6))
            ax.plot(x, med[m, :], lw=1.6)
            if band:
                ax.fill_between(x, lo[m, :], hi[m, :], alpha=0.25)

            if np.isfinite(rr.split_x_year):
                ax.axvline(float(rr.split_x_year), lw=1.0, alpha=0.8)

            ax.set_ylabel(ylabel)
            ax.set_xlabel("year")
            ax.grid(True, alpha=0.25)
            _set_yscale(ax, yscale)

            plt.tight_layout()
            path = os.path.join(out_dir, f"risk_annual_thr_{thr:g}_{y_mode}_{yscale}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)
            print(f"[save] {path}")

    # =============================================================================
    # Reporting at selected times (print + optional CSV)
    # =============================================================================
    def _time_to_target_float(self, t: Union[str, float, int, datetime]) -> float:
        if isinstance(t, (float, int, np.floating, np.integer)):
            return float(t)
        if isinstance(t, datetime):
            return _decimal_year_from_ym(t.year, t.month)
        s = str(t).strip()
        dt = _parse_date(s)
        if dt is not None:
            return _decimal_year_from_ym(dt.year, dt.month)
        return float(s)

    def _nearest_index(self, xgrid: np.ndarray, target: float) -> int:
        xg = np.asarray(xgrid, float)
        return int(np.argmin(np.abs(xg - float(target))))

    def print_at_times(
        self,
        rr: RiskResult,
        *,
        times: Sequence[Union[str, float, int, datetime]],
        scale: str = "fine",                 # "fine" or "annual"
        y_mode: str = "prob",                # "prob" or "rp"
        level: float = 0.90,
        rp_cap_years: float = 10_000.0,
        small_prob_policy: str = "clip",
        csv_path: Optional[str] = None,
    ) -> None:
        """
        Print posterior summaries (median, CI) for each threshold at selected times.

        - scale="fine": uses rr.x_full and rr.p_fine
        - scale="annual": uses rr.x_year and rr.p_year

        y_mode:
          - "prob": event probability
          - "rp": return period (years) with the 10k-year safeguard (rp_cap_years)

        If csv_path is given, writes a CSV with rows:
          time_label, time_x, threshold, median, lo, hi
        """
        scale = str(scale).strip().lower()
        if scale not in ("fine", "annual"):
            raise ValueError("scale must be 'fine' or 'annual'")
        y_mode = str(y_mode).strip().lower()
        if y_mode not in ("prob", "rp"):
            raise ValueError("y_mode must be 'prob' or 'rp'")

        if scale == "fine":
            xgrid = rr.x_full
            draws = rr.p_fine  # (S,M,L)
            annual = False
            time_name = "time"
        else:
            xgrid = rr.x_year
            draws = rr.p_year  # (S,M,nY)
            annual = True
            time_name = "year"

        # Convert draws to the requested y-mode
        if y_mode == "prob":
            p_floor = _p_floor_from_rp_cap(rp_cap_years=rp_cap_years, period=rr.period, annual=annual)
            y_draws = _apply_small_prob_policy(draws, p_floor=p_floor, policy=small_prob_policy)
            y_label = "prob"
        else:
            y_draws = _prob_to_rp_years(
                draws,
                period=rr.period,
                annual=annual,
                rp_cap_years=rp_cap_years,
                small_prob_policy=small_prob_policy,
            )
            y_label = "rp_years"

        med, lo, hi = _nan_summarize(y_draws, level=float(level))  # (M, Lgrid)

        rows: List[List[Any]] = []
        print(f"\n[report] scale={scale} | y_mode={y_mode} | level={level:.3f} | rp_cap_years={rp_cap_years:g} | policy={small_prob_policy}")
        for t in times:
            try:
                target = self._time_to_target_float(t)
            except Exception:
                print(f"[warn] could not parse time {t!r}; skipping.")
                continue
            j = self._nearest_index(xgrid, target)
            xj = float(np.asarray(xgrid, float)[j])
            label = str(t)

            print(f"\n  {time_name}={label}  (nearest x={xj:g}, idx={j})")
            for m, thr in enumerate(rr.thresholds):
                mmed = float(med[m, j])
                mlo = float(lo[m, j])
                mhi = float(hi[m, j])
                print(f"    thr={thr:g}  {y_label}: {mmed:.6g}  [{mlo:.6g}, {mhi:.6g}]")
                rows.append([label, xj, float(thr), mmed, mlo, mhi])

        if csv_path is not None:
            _ensure_dir(os.path.dirname(csv_path))
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time_label", "time_x", "threshold", "median", "lo", "hi"])
                w.writerows(rows)
            print(f"\n[save] {csv_path}\n")

    # Convenience wrappers you’ll likely use a lot:
    def print_probs_at_times(self, rr: RiskResult, *, times: Sequence[Any], scale: str = "fine", level: float = 0.90, csv_path: Optional[str] = None) -> None:
        self.print_at_times(rr, times=times, scale=scale, y_mode="prob", level=level, csv_path=csv_path)

    def print_rp_at_times(
        self,
        rr: RiskResult,
        *,
        times: Sequence[Any],
        scale: str = "fine",
        level: float = 0.90,
        rp_cap_years: float = 10_000.0,
        small_prob_policy: str = "clip",
        csv_path: Optional[str] = None,
    ) -> None:
        self.print_at_times(
            rr,
            times=times,
            scale=scale,
            y_mode="rp",
            level=level,
            rp_cap_years=rp_cap_years,
            small_prob_policy=small_prob_policy,
            csv_path=csv_path,
        )


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description=(
            "Compute Bayesian exceedance probabilities for DGEV Laplace posterior:\n"
            "- fine-scale (per time point) and annual aggregates,\n"
            "- for EVERY posterior draw,\n"
            "- optional: plot prob or return period, linear or log, single or multi-threshold,\n"
            "- optional: print summaries at chosen times.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--target", type=str, default=None, help="Run dir or path to posterior.npz. If omitted, uses latest under --root.")
    p.add_argument("--root", type=str, default="results/simulations/DGEV_NCP_LASSO", help="Search root when --target is omitted.")
    p.add_argument("--out", type=str, default=None, help="Output directory. Default: <run>/risk")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")

    # post-hoc chain trimming
    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    # thresholds + event
    p.add_argument("--thresholds", type=str, default="10,12", help="Comma-separated thresholds on plot/original scale, e.g. 35,37,39")
    p.add_argument("--event", type=str, default="auto", help="auto, gt, or lt (event on plot scale). auto: maxima->gt, minima->lt")

    # calendar axis
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY / YYYY-MM / YYYY-MM-DD). If omitted, tries meta['start_date'], else numeric index.",
    )

    # optional forecast extension
    p.add_argument("--horizon", type=int, default=0, help="Extra future steps to extend risk (simulate latent states).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed used only if --horizon>0.")

    # plotting options
    p.add_argument("--level", type=float, default=0.90, help="Credible band level (over posterior draws).")
    p.add_argument("--window-months", type=int, default=240, help="Fine plot: last N points shown (includes forecast tail if any).")
    p.add_argument("--window-years", type=int, default=60, help="Annual plot: last N year groups shown.")

    # NEW plotting controls
    p.add_argument("--y-mode", type=str, default="prob", choices=["prob", "rp"], help="Plot y as probability or return period (years).")
    p.add_argument("--yscale", type=str, default="log", choices=["linear", "log"], help="Plot y-axis scale.")
    p.add_argument("--combine", action="store_true", default=True, help="Overlay multiple thresholds on the same plot.")
    p.add_argument("--no-band", action="store_true", default=False, help="Disable credible bands (useful when combining).")
    p.add_argument("--rp-cap-years", type=float, default=10_000.0, help="Return-period cap in years (guards against tiny probabilities).")
    p.add_argument("--small-prob-policy", type=str, default="clip", choices=["clip", "mask"], help="How to handle probs below the rp-cap floor.")

    # NEW reporting controls
    p.add_argument("--times", type=str, default="", help="Comma-separated times for reporting (e.g. 1950-07,2020-07 or numeric x).")
    p.add_argument("--report-scale", type=str, default="fine", choices=["fine", "annual"], help="Whether to report on fine or annual scale.")
    p.add_argument("--report-csv", type=str, default="", help="Optional CSV path to save report (default: <out>/risk_report_times_*.csv).")

    args = p.parse_args()

    bundle = resolve_bundle(target=args.target, root=args.root)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    if args.burn > 0 or args.thin > 1:
        draws, meta = apply_burn_thin(draws, dict(meta), burn=int(args.burn), thin=int(args.thin))

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "risk")
    _ensure_dir(out_dir)
    print(f"[info] using posterior: {npz_path}")
    print(f"[info] saving outputs to: {out_dir}")

    thr = _parse_csv_floats(args.thresholds)

    sd = _parse_date(args.start_date)
    if sd is None:
        sd = _parse_date(meta.get("start_date")) if isinstance(meta, dict) else None

    risk = DGEVLaplaceRisk(draws, meta, npz_path=npz_path)

    ev = str(args.event).strip().lower()
    if ev == "auto":
        evv = None
    else:
        if ev not in ("gt", "lt"):
            raise ValueError("--event must be auto, gt, or lt.")
        evv = ev

    rr = risk.compute(
        thresholds=thr,
        event=evv,
        horizon=int(args.horizon),
        seed=int(args.seed),
        start_date=sd,
    )

    # save raw draw-level probabilities
    risk.save(rr, out_path=os.path.join(out_dir, "exceedance_probs.npz"))

    # plots
    risk.plot_fine(
        rr,
        out_dir=out_dir,
        level=float(args.level),
        window=int(args.window_months) if args.window_months is not None else None,
        show=bool(args.show),
        y_mode=str(args.y_mode),
        yscale=str(args.yscale),
        combine=bool(args.combine),
        band=(not bool(args.no_band)),
        rp_cap_years=float(args.rp_cap_years),
        small_prob_policy=str(args.small_prob_policy),
    )
    risk.plot_annual(
        rr,
        out_dir=out_dir,
        level=float(args.level),
        window_years=int(args.window_years) if args.window_years is not None else None,
        show=bool(args.show),
        y_mode=str(args.y_mode),
        yscale=str(args.yscale),
        combine=bool(args.combine),
        band=(not bool(args.no_band)),
        rp_cap_years=float(args.rp_cap_years),
        small_prob_policy=str(args.small_prob_policy),
    )

    # reporting at times
    times = _parse_csv_strings(args.times)
    if times:
        y_mode = str(args.y_mode).strip().lower()
        scale = str(args.report_scale).strip().lower()
        report_csv = str(args.report_csv).strip()
        if not report_csv:
            report_csv = os.path.join(out_dir, f"risk_report_times_{scale}_{y_mode}.csv")

        risk.print_at_times(
            rr,
            times=times,
            scale=scale,
            y_mode=y_mode,
            level=float(args.level),
            rp_cap_years=float(args.rp_cap_years),
            small_prob_policy=str(args.small_prob_policy),
            csv_path=report_csv,
        )

    print("[done] risk probabilities + plots written.")
