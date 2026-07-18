# %% simulator/dgev_laplace_endpoint.py
from __future__ import annotations
"""
Endpoint trajectories for the DGEV (Laplace NCP) posterior
=========================================================

Tracks the *finite endpoint over time* implied by GEV support when xi < 0.

For each posterior draw s and block t (MODEL scale):
  endpoint_model_{s,t} = mu_{s,t} - sigma_s / xi_s   if xi_s < 0
  endpoint_model_{s,t} = +inf (or NaN)               if xi_s >= 0

On PLOT scale:
- maxima series (no sign flip): endpoint_plot = endpoint_model (upper bound)
- minima series (sign flip model Z=-Y): endpoint_plot = -endpoint_model (lower bound)

Annual aggregation (year group S_j):
- maxima: annual endpoint = max_{t in S_j} endpoint_plot_{s,t}   (endpoint of annual max)
- minima: annual endpoint = min_{t in S_j} endpoint_plot_{s,t}   (endpoint of annual min)

Features
--------
- Uses the same bundle discovery helpers as your risk module:
    resolve_bundle(target, root)
    apply_burn_thin(draws, meta, burn, thin)
- Optional forecast extension (horizon>0) by simulating future mu_t using saved states.
- Conditional endpoint by default: draws with xi >= -xi_eps are treated as NaN (so summaries
  are conditional on "finite endpoint regime").
  Optionally, set nonfinite_policy="inf" to represent them as +/-inf.

Outputs
-------
<run>/endpoint/
  endpoint_over_time.npz
  endpoint_fine_<plot|model>_<linear|log>.png
  endpoint_annual_<plot|model>_<linear|log>.png
  endpoint_report_times_<fine|annual>.csv (optional)

Notes
-----
- "log" yscale is only applied if all plotted values are positive; otherwise we fall back to linear.
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


def _nan_summarize_2d(draws_2d: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    draws_2d: (S, L) possibly with NaNs
    returns median/lo/hi over S -> (L,)
    """
    loq = (1.0 - float(level)) / 2.0
    hiq = 1.0 - loq
    med = np.nanquantile(draws_2d, 0.5, axis=0)
    lo = np.nanquantile(draws_2d, loq, axis=0)
    hi = np.nanquantile(draws_2d, hiq, axis=0)
    return med, lo, hi


def _safe_set_yscale(ax: plt.Axes, yscale: str, y: np.ndarray) -> str:
    ys = str(yscale).strip().lower()
    if ys not in ("linear", "log"):
        raise ValueError("yscale must be 'linear' or 'log'")
    if ys == "log":
        yy = np.asarray(y, float)
        ok = np.isfinite(yy) & (yy > 0)
        if not np.any(ok) or np.nanmin(yy[ok]) <= 0:
            print("[warn] yscale=log requested but values are not strictly positive; falling back to linear.")
            return "linear"
        ax.set_yscale("log")
    return ys


# =============================================================================
# Result container
# =============================================================================
@dataclass(frozen=True)
class EndpointResult:
    minima: bool
    mode: str                 # "plot" or "model"
    nonfinite_policy: str     # "nan" or "inf"
    xi_eps: float

    p_finite: float           # P(xi < -xi_eps) across draws

    x_full: np.ndarray        # (L,)
    endpoint_fine: np.ndarray # (S, L) (plot or model scale per mode)
    split_x_fine: float

    x_year: np.ndarray        # (nY,)
    endpoint_year: np.ndarray # (S, nY) annual aggregated endpoint
    split_x_year: float

    start_year: Optional[int]
    start_month: Optional[int]
    period: int


# =============================================================================
# Main class
# =============================================================================
class DGEVLaplaceEndpoint:
    """
    Track the implied (finite) GEV endpoint over time for each posterior draw.
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

    # ----------------------------- compute endpoint ----------------------------- #
    def compute(
        self,
        *,
        horizon: int = 0,
        seed: int = 123,
        start_date: Optional[datetime] = None,
        mode: str = "plot",                 # "plot" or "model"
        xi_eps: float = 1e-6,               # treat xi >= -xi_eps as non-finite
        nonfinite_policy: str = "nan",      # "nan" or "inf"
    ) -> EndpointResult:
        """
        Build endpoint trajectories on either plot-scale or model-scale.

        mode="plot":
          - maxima: upper endpoint on plot scale
          - minima: lower endpoint on plot scale (because model is Z=-Y)

        nonfinite_policy:
          - "nan": xi >= -xi_eps -> NaN (conditional summaries)
          - "inf": xi >= -xi_eps -> +inf for maxima (mode=model/plot), -inf for minima on plot-scale
        """
        mode = str(mode).strip().lower()
        if mode not in ("plot", "model"):
            raise ValueError("mode must be 'plot' or 'model'")
        nonfinite_policy = str(nonfinite_policy).strip().lower()
        if nonfinite_policy not in ("nan", "inf"):
            raise ValueError("nonfinite_policy must be 'nan' or 'inf'")

        H = int(horizon)
        if H < 0:
            raise ValueError("horizon must be >= 0.")
        L_full = self.T + H

        x_full, split_x_fine, sy, sm = self._build_x_axis(L_full=L_full, start_date=start_date)

        mu_future = self._simulate_mu_future(horizon=H, seed=seed) if H > 0 else np.zeros((self.S, 0))
        mu_full = np.concatenate([self.mu_obs, mu_future], axis=1)  # (S, L)

        xi = self.xi.astype(float)
        sig = self.sigma.astype(float)
        xi_eps = float(abs(xi_eps))

        finite = xi < (-xi_eps)
        p_finite = float(np.mean(finite))

        # endpoint on MODEL scale (upper endpoint of the *modeled* maxima variable)
        # only meaningful as a finite upper endpoint when xi < 0
        endpoint_model = mu_full - (sig[:, None] / xi[:, None])  # (S,L)
        if nonfinite_policy == "nan":
            endpoint_model[~finite, :] = np.nan
        else:
            endpoint_model[~finite, :] = np.inf  # "upper endpoint is infinite" on model maxima scale

        # map to plot scale if requested
        if mode == "model":
            endpoint_use = endpoint_model
        else:
            if self.minima:
                # model Z=-Y; a finite upper endpoint in Z implies a finite lower endpoint in Y
                endpoint_use = -endpoint_model
                if nonfinite_policy == "inf":
                    # if Z is unbounded above, Y is unbounded below
                    endpoint_use[~finite, :] = -np.inf
            else:
                endpoint_use = endpoint_model

        # annual grouping / aggregation
        x_year, groups = _annual_groups_from_index(L_full, self.period, start_year=sy, start_month=sm)
        nY = len(groups)
        endpoint_year = np.full((self.S, nY), np.nan, float)

        last_obs_year = -1
        for j, I in enumerate(groups):
            I = np.asarray(I, int)
            A = endpoint_use[:, I]  # (S,p)

            if self.minima and mode == "plot":
                # annual minimum endpoint (lower bound): min of block lower bounds
                endpoint_year[:, j] = np.nanmin(A, axis=1)
            else:
                # annual maximum endpoint (upper bound): max of block upper bounds
                endpoint_year[:, j] = np.nanmax(A, axis=1)

            if np.all(I < self.T):
                last_obs_year = j

        split_x_year = float(x_year[last_obs_year]) if (last_obs_year >= 0 and x_year.size) else float("nan")

        return EndpointResult(
            minima=bool(self.minima),
            mode=str(mode),
            nonfinite_policy=str(nonfinite_policy),
            xi_eps=float(xi_eps),
            p_finite=float(p_finite),
            x_full=x_full,
            endpoint_fine=endpoint_use,
            split_x_fine=float(split_x_fine),
            x_year=x_year,
            endpoint_year=endpoint_year,
            split_x_year=float(split_x_year),
            start_year=sy,
            start_month=sm,
            period=int(self.period),
        )

    # ----------------------------- save ----------------------------- #
    def save(self, er: EndpointResult, *, out_path: str) -> None:
        _ensure_dir(os.path.dirname(out_path))
        np.savez_compressed(
            out_path,
            minima=bool(er.minima),
            mode=str(er.mode),
            nonfinite_policy=str(er.nonfinite_policy),
            xi_eps=float(er.xi_eps),
            p_finite=float(er.p_finite),
            period=int(er.period),
            start_year=(-1 if er.start_year is None else int(er.start_year)),
            start_month=(-1 if er.start_month is None else int(er.start_month)),
            x_full=er.x_full,
            endpoint_fine=er.endpoint_fine,  # (S,L)
            split_x_fine=float(er.split_x_fine),
            x_year=er.x_year,
            endpoint_year=er.endpoint_year,  # (S,nY)
            split_x_year=float(er.split_x_year),
        )
        print(f"[save] {out_path}")

    # ----------------------------- plotting ----------------------------- #
    def plot_fine(
        self,
        er: EndpointResult,
        *,
        out_dir: str,
        level: float = 0.90,
        window: Optional[int] = 240,
        yscale: str = "linear",
        show: bool = False,
    ) -> None:
        _ensure_dir(out_dir)

        L = int(er.x_full.size)
        i0 = 0
        if window is not None:
            w = max(1, int(window))
            i0 = max(0, L - w)
        x = er.x_full[i0:]
        draws = er.endpoint_fine[:, i0:]  # (S,Lwin)

        med, lo, hi = _nan_summarize_2d(draws, level=float(level))

        fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
        ax.plot(x, med, lw=1.8)
        ax.fill_between(x, lo, hi, alpha=0.25)
        ax.axvline(float(er.split_x_fine), lw=1.0, alpha=0.8)

        kind = "lower endpoint" if (er.minima and er.mode == "plot") else "upper endpoint"
        ax.set_ylabel(f"{kind} ({er.mode} scale)")
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.25)

        _safe_set_yscale(ax, yscale, np.concatenate([med, lo, hi]))

        ax.text(
            0.01,
            0.98,
            f"P(finite endpoint)=P(xi<-{er.xi_eps:g}) ≈ {er.p_finite:.3f}\npolicy={er.nonfinite_policy}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        plt.tight_layout()
        path = os.path.join(out_dir, f"endpoint_fine_{er.mode}_{yscale}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"[save] {path}")

    def plot_annual(
        self,
        er: EndpointResult,
        *,
        out_dir: str,
        level: float = 0.90,
        window_years: Optional[int] = 60,
        yscale: str = "linear",
        show: bool = False,
    ) -> None:
        _ensure_dir(out_dir)

        nY = int(er.x_year.size)
        j0 = 0
        if window_years is not None:
            w = max(1, int(window_years))
            j0 = max(0, nY - w)
        x = er.x_year[j0:]
        draws = er.endpoint_year[:, j0:]  # (S,nYwin)

        med, lo, hi = _nan_summarize_2d(draws, level=float(level))

        fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
        ax.plot(x, med, lw=1.8)
        ax.fill_between(x, lo, hi, alpha=0.25)
        if np.isfinite(er.split_x_year):
            ax.axvline(float(er.split_x_year), lw=1.0, alpha=0.8)

        kind = "annual lower endpoint" if (er.minima and er.mode == "plot") else "annual upper endpoint"
        ax.set_ylabel(f"{kind} ({er.mode} scale)")
        ax.set_xlabel("year")
        ax.grid(True, alpha=0.25)

        _safe_set_yscale(ax, yscale, np.concatenate([med, lo, hi]))

        ax.text(
            0.01,
            0.98,
            f"P(finite endpoint)=P(xi<-{er.xi_eps:g}) ≈ {er.p_finite:.3f}\npolicy={er.nonfinite_policy}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        plt.tight_layout()
        path = os.path.join(out_dir, f"endpoint_annual_{er.mode}_{yscale}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"[save] {path}")

    # ----------------------------- reporting ----------------------------- #
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
        er: EndpointResult,
        *,
        times: Sequence[Union[str, float, int, datetime]],
        scale: str = "fine",     # "fine" or "annual"
        level: float = 0.90,
        csv_path: Optional[str] = None,
    ) -> None:
        scale = str(scale).strip().lower()
        if scale not in ("fine", "annual"):
            raise ValueError("scale must be 'fine' or 'annual'")

        if scale == "fine":
            xgrid = er.x_full
            draws = er.endpoint_fine  # (S,L)
            time_name = "time"
        else:
            xgrid = er.x_year
            draws = er.endpoint_year  # (S,nY)
            time_name = "year"

        med, lo, hi = _nan_summarize_2d(draws, level=float(level))

        kind = "lower endpoint" if (er.minima and er.mode == "plot") else "upper endpoint"
        print(f"\n[endpoint report] scale={scale} | {kind} ({er.mode}) | level={level:.3f}")
        print(f"  P(finite endpoint)=P(xi<-{er.xi_eps:g}) ≈ {er.p_finite:.3f} | policy={er.nonfinite_policy}")

        rows: List[List[Any]] = []
        for t in times:
            try:
                target = self._time_to_target_float(t)
            except Exception:
                print(f"[warn] could not parse time {t!r}; skipping.")
                continue
            j = self._nearest_index(xgrid, target)
            xj = float(np.asarray(xgrid, float)[j])
            label = str(t)

            print(f"  {time_name}={label} (nearest x={xj:g}, idx={j})  endpoint: {med[j]:.6g}  [{lo[j]:.6g}, {hi[j]:.6g}]")
            rows.append([label, xj, float(med[j]), float(lo[j]), float(hi[j])])

        if csv_path is not None:
            _ensure_dir(os.path.dirname(csv_path))
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time_label", "time_x", "median", "lo", "hi"])
                w.writerows(rows)
            print(f"[save] {csv_path}\n")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    p = __import__("argparse").ArgumentParser(
        description=(
            "Track implied GEV endpoint over time for DGEV Laplace posterior.\n"
            "- Computes fine-scale endpoint trajectories (per block) and annual aggregated endpoints.\n"
            "- By default conditions on xi < 0 (finite endpoint regime) by treating xi>=-xi_eps as NaN.\n"
        ),
        formatter_class=__import__("argparse").ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--target", type=str, default=None, help="Run dir or path to posterior.npz. If omitted, uses latest under --root.")
    p.add_argument("--root", type=str, default="results/simulations/DGEV_NCP_LASSO", help="Search root when --target is omitted.")
    p.add_argument("--out", type=str, default=None, help="Output directory. Default: <run>/endpoint")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")

    p.add_argument("--burn", type=int, default=0, help="Extra burn-in draws (post-hoc).")
    p.add_argument("--thin", type=int, default=1, help="Extra thinning factor (post-hoc).")

    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY / YYYY-MM / YYYY-MM-DD). If omitted, tries meta['start_date'], else numeric index.",
    )

    p.add_argument("--horizon", type=int, default=0, help="Extra future steps to extend endpoint trajectories (simulate latent states).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed used only if --horizon>0.")

    p.add_argument("--mode", type=str, default="plot", choices=["plot", "model"], help="Compute endpoints on plot/original scale or model scale.")
    p.add_argument("--xi-eps", type=float, default=1e-6, help="Treat xi >= -xi_eps as non-finite (guards near-zero blow-ups).")
    p.add_argument("--nonfinite-policy", type=str, default="nan", choices=["nan", "inf"], help="How to represent xi>=-xi_eps draws.")
    p.add_argument("--level", type=float, default=0.90, help="Credible band level (over posterior draws).")
    p.add_argument("--window-months", type=int, default=240, help="Fine plot: last N points shown (includes forecast tail if any).")
    p.add_argument("--window-years", type=int, default=60, help="Annual plot: last N year groups shown.")
    p.add_argument("--yscale", type=str, default="linear", choices=["linear", "log"], help="y-axis scale for endpoint plots.")

    p.add_argument("--times", type=str, default="", help="Comma-separated times for reporting (e.g. 1950-07,2020-07 or numeric x).")
    p.add_argument("--report-scale", type=str, default="fine", choices=["fine", "annual"], help="Whether to report on fine or annual scale.")
    p.add_argument("--report-csv", type=str, default="", help="Optional CSV path to save report (default: <out>/endpoint_report_times_*.csv).")

    args = p.parse_args()

    bundle = resolve_bundle(target=args.target, root=args.root)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    if args.burn > 0 or args.thin > 1:
        draws, meta = apply_burn_thin(draws, dict(meta), burn=int(args.burn), thin=int(args.thin))

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "endpoint")
    _ensure_dir(out_dir)
    print(f"[info] using posterior: {npz_path}")
    print(f"[info] saving outputs to: {out_dir}")

    sd = _parse_date(args.start_date)
    if sd is None:
        sd = _parse_date(meta.get("start_date")) if isinstance(meta, dict) else None

    ep = DGEVLaplaceEndpoint(draws, meta, npz_path=npz_path)

    er = ep.compute(
        horizon=int(args.horizon),
        seed=int(args.seed),
        start_date=sd,
        mode=str(args.mode),
        xi_eps=float(args.xi_eps),
        nonfinite_policy=str(args.nonfinite_policy),
    )

    ep.save(er, out_path=os.path.join(out_dir, "endpoint_over_time.npz"))

    ep.plot_fine(
        er,
        out_dir=out_dir,
        level=float(args.level),
        window=int(args.window_months) if args.window_months is not None else None,
        yscale=str(args.yscale),
        show=bool(args.show),
    )
    ep.plot_annual(
        er,
        out_dir=out_dir,
        level=float(args.level),
        window_years=int(args.window_years) if args.window_years is not None else None,
        yscale=str(args.yscale),
        show=bool(args.show),
    )

    times = _parse_csv_strings(args.times)
    if times:
        report_csv = str(args.report_csv).strip()
        if not report_csv:
            report_csv = os.path.join(out_dir, f"endpoint_report_times_{str(args.report_scale).strip().lower()}.csv")
        ep.print_at_times(
            er,
            times=times,
            scale=str(args.report_scale),
            level=float(args.level),
            csv_path=report_csv,
        )

    print("[done] endpoint trajectories + plots written.")
