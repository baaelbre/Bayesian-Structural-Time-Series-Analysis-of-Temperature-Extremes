# %% simulator/dlm_forecast.py
from __future__ import annotations
"""
Posterior predictive forecasting for the Gaussian DLM (fine-scale + coarse-grained)

Posterior predictive recipe
---------------------------
  1) Draw (x_T, theta) from the posterior (stored MCMC draws).
  2) Propagate the latent state forward conditional on (x_T, theta).
  3) Simulate future observations conditional on propagated states and theta.
  4) Coarse-grain *each* simulated fine-scale predictive path (annual / seasonal means).

Printing selected forecasts (stdout)
------------------------------------
Optionally print posterior predictive summaries (median + interval) for selected
dates / windows:

  --print-forecast
  --print-scales fine,annual,seasonal_all,DJF,MAM,JJA,SON
  --print-dates  (comma-separated YYYY or YYYY-MM or YYYY-MM-DD)
  --print-start / --print-end (window)
  --print-include-observed (allow printing observed segment too)
  --print-max-lines (truncate output per scale)
  --print-horizon (FINE ONLY): if no dates/window are given, print the first
                  N forecast steps ahead (median + interval)

Notes
-----
- Coarse-graining per draw is correct: it samples from the push-forward of the
  fine-scale posterior predictive distribution under the aggregation map.
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Iterable

import numpy as np
import matplotlib

matplotlib.use("Agg")  # safe for batch runs
import matplotlib.pyplot as plt

# Make optimization package visible (mirrors dlm_plotter.py)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------
# I/O helpers: posterior loader (same pattern as dlm_plotter.py)
# ---------------------------------------------------------------------
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e


# =============================================================================
# Small utils
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


def _parse_csv(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _decimal_year_from_ym(y: int, m: int) -> float:
    # place month roughly at its center within the year
    return float(y) + (float(m) - 0.5) / 12.0


def _summarize_ribbon(draws_2d: np.ndarray, level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    draws_2d: (S, L)
    returns: (median, lo, hi) across S
    """
    loq = (1.0 - level) / 2.0
    hiq = 1.0 - loq
    med = np.quantile(draws_2d, 0.5, axis=0)
    lo = np.quantile(draws_2d, loq, axis=0)
    hi = np.quantile(draws_2d, hiq, axis=0)
    return med, lo, hi


def _build_season_design(L: int, period: int) -> np.ndarray:
    """
    Static seasonal design S[t,:] mapping gamma0 (length p-1) to seasonal baseline.
    Season p is represented as -sum_{j=1}^{p-1} gamma_j (sum-to-zero parametrisation).
    """
    p = int(period)
    if p < 2:
        return np.zeros((L, 0), float)
    K = p - 1
    S = np.zeros((L, K), float)
    for t in range(L):
        season = t % p  # 0..p-1
        if season < K:
            S[t, season] = 1.0
        else:
            S[t, :] = -1.0
    return S


def _season_rotation_matrix(K: int) -> np.ndarray:
    """
    Dummy seasonal rotation on (p-1)-vector (Durbin-Koopman seasonal form):
      g1_{t+1} = -sum_{j=1}^{K} g_j
      g_{j+1,t+1} = g_{j,t},  j=1..K-1
    """
    if K <= 0:
        return np.zeros((0, 0), float)
    R = np.zeros((K, K), float)
    R[0, :] = -1.0
    if K > 1:
        R[1:, :-1] = np.eye(K - 1)
    return R


def _extract_state_indices(layout: List[str]) -> Tuple[int, int, int]:
    """
    Returns indices for alpha, beta, and the first seasonal dynamic component g1
    in the *centred* state vector x (stored as (S, T, dim)).
    """
    if "alpha" not in layout or "beta" not in layout:
        raise ValueError("meta['layout'] must contain 'alpha' and 'beta'.")
    ia = layout.index("alpha")
    ib = layout.index("beta")
    g_indices = [i for i, nm in enumerate(layout) if str(nm).startswith("g")]
    if not g_indices:
        raise ValueError("meta['layout'] must contain seasonal states g1..g{p-1}.")
    ig1 = min(g_indices)
    return ia, ib, ig1


def _get_sd(draws: Dict[str, np.ndarray], S: int) -> np.ndarray:
    """Observation sd: accept sigma or sigma2."""
    if "sigma" in draws:
        sd = np.asarray(draws["sigma"], float).ravel()
    elif "sigma2" in draws:
        sd = np.sqrt(np.clip(np.asarray(draws["sigma2"], float).ravel(), 0.0, None))
    else:
        raise ValueError("Posterior draws must include 'sigma' or 'sigma2'.")
    if sd.size != S:
        raise ValueError("sigma length does not match number of posterior draws.")
    return sd


def _get_Q(draws: Dict[str, np.ndarray], S: int, name_Q: str, name_s: str) -> np.ndarray:
    """
    Process variance per draw:
      - prefer Q_* if present (already variance)
      - else use s_* and square
    """
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
# Printing helpers
# =============================================================================
def _calendar_labels(
    L_full: int, *, start_year: Optional[int], start_month: Optional[int], period: int
) -> Optional[List[str]]:
    """Fine-scale labels like 'YYYY-MM' if start date is known and period divides 12."""
    if start_year is None or start_month is None:
        return None
    if period not in (12, 6, 4, 3, 2, 1) or (12 % period != 0):
        return None
    step_months = 12 // period
    labels: List[str] = []
    y0, m0 = int(start_year), int(start_month)
    for i in range(L_full):
        mm = (m0 - 1) + i * step_months
        yy = y0 + (mm // 12)
        mo = (mm % 12) + 1
        labels.append(f"{yy:04d}-{mo:02d}")
    return labels


def _fine_index_from_date(dt: datetime, *, start_year: int, start_month: int, period: int) -> int:
    """
    Map a calendar date to the closest fine-scale index.
    Requires period divides 12 and known start (year, month).
    """
    if period not in (12, 6, 4, 3, 2, 1) or (12 % period != 0):
        raise ValueError("Date-based fine selection requires period dividing 12.")
    step_months = 12 // period
    diff_months = (dt.year - start_year) * 12 + (dt.month - start_month)
    k = int(round(diff_months / step_months))
    return k


def _clip_indices(idxs: Iterable[int], L: int) -> List[int]:
    out: List[int] = []
    for i in idxs:
        ii = int(i)
        if 0 <= ii < L:
            out.append(ii)
    return sorted(set(out))


def _select_fine_indices(
    *,
    L_full: int,
    T_obs: int,
    period: int,
    start_year: Optional[int],
    start_month: Optional[int],
    labels: Optional[List[str]],
    # selection args
    dates: List[str],
    start: Optional[str],
    end: Optional[str],
    include_observed: bool,
    print_horizon: Optional[int],
    H: int,
) -> List[int]:
    """
    Fine-scale selection logic:
      - If no dates/window: print first `print_horizon` forecast steps (default: all H).
      - If dates: select those dates (requires calendar).
      - If window: select all points in [start,end] (requires calendar).
    """
    # default when printing: forecast horizon only
    if (not dates) and (start is None) and (end is None):
        ph = H if print_horizon is None else max(0, int(print_horizon))
        ph = min(ph, H)
        return list(range(T_obs, min(L_full, T_obs + ph)))

    # date-based selection requires calendar
    if start_year is None or start_month is None or labels is None:
        raise ValueError(
            "Date-based printing requires a known start-date and period dividing 12.\n"
            "Provide --start-date (and ensure meta['period'] divides 12), or omit date selection."
        )

    if dates:
        idxs: List[int] = []
        for s in dates:
            dt = _parse_date(s)
            if dt is None:
                continue
            k = _fine_index_from_date(dt, start_year=int(start_year), start_month=int(start_month), period=period)
            idxs.append(k)
        idxs = _clip_indices(idxs, L_full)
    else:
        if start is None or end is None:
            raise ValueError("For a fine-scale window, provide BOTH --print-start and --print-end.")
        dt0 = _parse_date(start)
        dt1 = _parse_date(end)
        if dt0 is None or dt1 is None:
            raise ValueError("Could not parse --print-start/--print-end.")
        a = _fine_index_from_date(dt0, start_year=int(start_year), start_month=int(start_month), period=period)
        b = _fine_index_from_date(dt1, start_year=int(start_year), start_month=int(start_month), period=period)
        if b < a:
            a, b = b, a
        idxs = _clip_indices(range(a, b + 1), L_full)

    if not include_observed:
        idxs = [i for i in idxs if i >= T_obs]
    return idxs


def _select_by_years(
    *,
    x: np.ndarray,
    forecast_mask: np.ndarray,
    dates: List[str],
    start: Optional[str],
    end: Optional[str],
    include_observed: bool,
) -> List[int]:
    """
    For annual/seasonal series with x on (decimal) year scale.
    - If no dates/window: select forecast points only (unless include_observed).
    - If dates: select years matching dt.year.
    - If window: select years in [y0,y1].
    """
    n = int(x.size)
    if n == 0:
        return []

    years = np.floor(x).astype(int)

    if (not dates) and (start is None) and (end is None):
        if include_observed:
            return list(range(n))
        return np.where(forecast_mask)[0].astype(int).tolist()

    if dates:
        want = sorted({int(_parse_date(s).year) for s in dates if _parse_date(s) is not None})
        idxs = [i for i in range(n) if int(years[i]) in want]
    else:
        if start is None or end is None:
            raise ValueError("For a year-window, provide BOTH --print-start and --print-end.")
        dt0 = _parse_date(start)
        dt1 = _parse_date(end)
        if dt0 is None or dt1 is None:
            raise ValueError("Could not parse --print-start/--print-end.")
        y0, y1 = sorted([int(dt0.year), int(dt1.year)])
        idxs = [i for i in range(n) if y0 <= int(years[i]) <= y1]

    if not include_observed:
        idxs = [i for i in idxs if bool(forecast_mask[i])]
    return idxs


def _print_block(
    *,
    name: str,
    labels: List[str],
    x: np.ndarray,
    draws: np.ndarray,  # (S, n)
    is_forecast: List[bool],
    level: float,
    max_lines: int,
) -> None:
    n = int(x.size)
    if n == 0:
        print(f"\n[print] {name}: (no points selected)\n")
        return

    loq = (1.0 - level) / 2.0
    hiq = 1.0 - loq
    med = np.quantile(draws, 0.5, axis=0)
    lo = np.quantile(draws, loq, axis=0)
    hi = np.quantile(draws, hiq, axis=0)

    print(f"\n[print] {name}: {n} points (median + {int(round(level*100))}% interval)")
    print(f"{'label':>10s}  {'x':>10s}  {'median':>10s}  {'lo':>10s}  {'hi':>10s}  {'forecast':>9s}")
    print("-" * 74)

    n_show = min(n, int(max_lines))
    for i in range(n_show):
        print(
            f"{labels[i]:>10s}  "
            f"{float(x[i]):10.3f}  "
            f"{float(med[i]):10.3f}  "
            f"{float(lo[i]):10.3f}  "
            f"{float(hi[i]):10.3f}  "
            f"{str(bool(is_forecast[i])):>9s}"
        )

    if n_show < n:
        print(f"... ({n - n_show} more rows truncated; increase --print-max-lines)")
    print("")


# =============================================================================
# Forecast simulation (posterior predictive recipe)
# =============================================================================
@dataclass(frozen=True)
class ForecastResult:
    y_obs: np.ndarray           # (T,)
    y_future: np.ndarray        # (S, H)
    y_full: np.ndarray          # (S, T+H)
    x_axis_full: np.ndarray     # (T+H,)
    split_x: float              # x-value of last observed time point
    start_year: Optional[int]
    start_month: Optional[int]


def simulate_dlm_forecast(
    draws: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    horizon: int,
    *,
    seed: int = 123,
    start_date: Optional[datetime] = None,
) -> ForecastResult:
    """
    Simulate y_{T+1:T+H} from the posterior predictive using stored posterior draws.

    Vectorized over posterior draws m:
      - take (x_T^{(m)}, theta^{(m)}) ~ p(x_T, theta | y_{1:T})
      - propagate state forward with transition noise
      - simulate observations with observation noise
    """
    if horizon <= 0:
        raise ValueError("--horizon must be > 0.")

    if "y" not in draws:
        raise ValueError("Posterior draws must include 'y' (observations).")
    y_obs = np.asarray(draws["y"], float).ravel()
    T = int(y_obs.size)

    if "x" not in draws or np.asarray(draws["x"]).ndim != 3:
        raise ValueError("Posterior draws must include centred state draws 'x' with shape (S, T, dim).")
    x = np.asarray(draws["x"], float)
    S_draws, T_x, dim = x.shape
    if T_x != T:
        raise ValueError(f"Shape mismatch: draws['x'] has T={T_x} but draws['y'] has T={T}.")

    period = int(meta.get("period", 12))
    layout = meta.get("layout", None)
    if not isinstance(layout, (list, tuple)):
        raise ValueError("meta must include a list 'layout'.")
    layout = list(layout)

    ia, ib, ig1 = _extract_state_indices(layout)
    K = int(period) - 1
    if K <= 0:
        raise ValueError("This seasonal DLM requires period >= 2 (so K=period-1 >= 1).")
    if ig1 + K > dim:
        raise ValueError("State dimension does not contain a full seasonal block g1..g{K}.")

    # theta^{(m)} pieces
    sigma = _get_sd(draws, S_draws)
    Q_alpha = _get_Q(draws, S_draws, "Q_alpha", "s_alpha")
    Q_beta = _get_Q(draws, S_draws, "Q_beta", "s_beta")
    Q_gamma = _get_Q(draws, S_draws, "Q_gamma", "s_gamma")

    if "gamma0" not in draws:
        raise ValueError("Posterior draws must include 'gamma0' (baseline seasonal coefficients).")
    gamma0 = np.asarray(draws["gamma0"], float)
    if gamma0.ndim != 2 or gamma0.shape != (S_draws, K):
        raise ValueError(f"gamma0 must have shape (S, {K}), got {gamma0.shape}.")

    # x_T^{(m)} (last observed state per draw)
    alpha = x[:, -1, ia].copy()                 # (S,)
    beta = x[:, -1, ib].copy()                  # (S,)
    gamma_dyn = x[:, -1, ig1:ig1 + K].copy()    # (S, K)

    # seasonal design for baseline gamma0, and seasonal rotation for gamma_dyn
    L_full = T + int(horizon)
    S_design_full = _build_season_design(L_full, period)  # (T+H, K)
    R = _season_rotation_matrix(K)                        # (K, K)
    RT = R.T

    rng = np.random.default_rng(int(seed))

    y_future = np.zeros((S_draws, horizon), float)

    # innovations only enter first seasonal component in this representation
    e1 = np.zeros((K,), float)
    e1[0] = 1.0

    for h in range(horizon):
        t = T + h  # absolute time index in the full path (0-based)

        # state propagation
        alpha = alpha + beta + rng.normal(0.0, np.sqrt(Q_alpha), size=S_draws)
        beta = beta + rng.normal(0.0, np.sqrt(Q_beta), size=S_draws)

        epsg = rng.normal(0.0, np.sqrt(Q_gamma), size=S_draws)
        gamma_dyn = gamma_dyn @ RT + epsg[:, None] * e1[None, :]

        # observation mean
        base_row = S_design_full[t, :]  # (K,)
        baseline = np.einsum("sk,k->s", gamma0, base_row)  # (S,)
        mu = alpha + gamma_dyn[:, 0] + baseline

        # observation simulation
        y_future[:, h] = mu + sigma * rng.normal(0.0, 1.0, size=S_draws)

    # Full path per draw: observed past is fixed, future is simulated
    y_full = np.concatenate([np.repeat(y_obs[None, :], S_draws, axis=0), y_future], axis=1)

    # time axis
    start_year = start_date.year if start_date is not None else None
    start_month = start_date.month if start_date is not None else None

    if start_date is not None and period in (12, 6, 4, 3, 2, 1) and (12 % period == 0):
        step_months = 12 // period
        years = np.zeros(L_full, int)
        months = np.zeros(L_full, int)
        y0, m0 = start_date.year, start_date.month
        for i in range(L_full):
            mm = (m0 - 1) + i * step_months
            yy = y0 + (mm // 12)
            mo = (mm % 12) + 1
            years[i] = yy
            months[i] = mo
        x_axis_full = np.array([_decimal_year_from_ym(int(yy), int(mo)) for yy, mo in zip(years, months)], float)
        split_x = float(x_axis_full[T - 1])
    else:
        x_axis_full = np.arange(L_full, dtype=float)
        split_x = float(T - 1)

    return ForecastResult(
        y_obs=y_obs,
        y_future=y_future,
        y_full=y_full,
        x_axis_full=x_axis_full,
        split_x=split_x,
        start_year=start_year,
        start_month=start_month,
    )


# =============================================================================
# Coarse-graining (annual averages)
# =============================================================================
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
      x_year (n_years,) : x-coordinate per year
      groups : list of arrays of indices for each year block

    If (start_year,start_month) are known AND period divides 12, group by calendar year.
    Otherwise, group into consecutive blocks of length `period` from the start.
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
    groups = [idx[j * p:(j + 1) * p] for j in range(n_years)]
    x_year = np.arange(n_years, dtype=float)
    return x_year, groups


def coarse_grain_annual_average(
    y_full: np.ndarray,  # (S, L_full)
    T_obs: int,
    period: int,
    *,
    start_year: Optional[int],
    start_month: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute annual averages per year group by coarse-graining each fine-scale predictive path.

    Returns:
      x_year         : (n_years,)
      y_year_obs     : (n_years,) observed annual averages (NaN where not fully observed)
      year_draws     : (S, n_years) posterior predictive draws for annual averages
      forecast_mask  : (n_years,) True if the year uses any forecasted points
      split_x        : x-coordinate of last fully observed year
    """
    S, L_full = y_full.shape
    x_year, groups = _annual_groups_from_index(
        L_full, T_obs, period, start_year=start_year, start_month=start_month
    )

    nY = len(groups)
    year_draws = np.full((S, nY), np.nan, float)
    y_year_obs = np.full((nY,), np.nan, float)
    forecast_mask = np.zeros((nY,), bool)

    last_obs_idx = -1
    for j, I in enumerate(groups):
        I = np.asarray(I, int)
        year_draws[:, j] = np.mean(y_full[:, I], axis=1)

        if np.all(I < T_obs):
            y_year_obs[j] = float(year_draws[0, j])  # identical across draws
            last_obs_idx = j
        else:
            forecast_mask[j] = True

    split_x = float(x_year[last_obs_idx]) if last_obs_idx >= 0 else float(x_year[0]) if nY else float("nan")
    return x_year, y_year_obs, year_draws, forecast_mask, split_x


# =============================================================================
# Coarse-graining (meteorological seasons; monthly data only)
# =============================================================================
def coarse_grain_meteo_seasons(
    y_full: np.ndarray,  # (S, L_full)
    T_obs: int,
    *,
    start_year: int,
    start_month: int,
) -> Tuple[
    Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, List[str]],
]:
    """
    Meteorological seasons for MONTHLY data (period=12):
      DJF (Dec-Jan-Feb), MAM, JJA, SON.

    Returns:
      per_season dict: name -> (x, y_obs, draws, forecast_mask, split_x)
      seasonal_all    : (x_all, y_obs_all, draws_all, forecast_mask_all, split_x_all, labels_all)
        where seasonal_all stacks all seasons chronologically.
    """
    S, L_full = y_full.shape

    # build (year,month) sequences
    year = np.zeros(L_full, int)
    month = np.zeros(L_full, int)
    y0, m0 = int(start_year), int(start_month)
    for i in range(L_full):
        mm = (m0 - 1) + i
        yy = y0 + (mm // 12)
        mo = (mm % 12) + 1
        year[i] = yy
        month[i] = mo
    idx_map = {(int(yy), int(mo)): int(i) for i, (yy, mo) in enumerate(zip(year, month))}

    seasons = {
        "DJF": [(-1, 12), (0, 1), (0, 2)],
        "MAM": [(0, 3), (0, 4), (0, 5)],
        "JJA": [(0, 6), (0, 7), (0, 8)],
        "SON": [(0, 9), (0, 10), (0, 11)],
    }
    end_month = {"DJF": 2, "MAM": 5, "JJA": 8, "SON": 11}

    years = np.unique(year)

    per_season: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}
    all_entries: List[Tuple[float, str, np.ndarray]] = []  # (x, label, indices)

    for sname, trip in seasons.items():
        xs: List[float] = []
        groups: List[np.ndarray] = []
        labels: List[str] = []

        for yy in years:
            keys = [(int(yy) + int(dy), int(mo)) for dy, mo in trip]
            if all(k in idx_map for k in keys):
                I = np.array([idx_map[k] for k in keys], int)
                I = np.sort(I)
                groups.append(I)
                xs.append(_decimal_year_from_ym(int(yy), int(end_month[sname])))
                labels.append(f"{int(yy):04d}-{sname}")

        n = len(groups)
        draws_s = np.full((S, n), np.nan, float)
        obs_s = np.full((n,), np.nan, float)
        fmask = np.zeros((n,), bool)

        last_obs_idx = -1
        for j, I in enumerate(groups):
            draws_s[:, j] = np.mean(y_full[:, I], axis=1)
            if np.all(I < T_obs):
                obs_s[j] = float(draws_s[0, j])
                last_obs_idx = j
            else:
                fmask[j] = True

            all_entries.append((xs[j], labels[j], I))

        split_x = float(xs[last_obs_idx]) if last_obs_idx >= 0 and len(xs) else float(xs[0]) if xs else float("nan")
        per_season[sname] = (np.array(xs, float), obs_s, draws_s, fmask, split_x)

    # Stack all seasons in chronological order
    all_entries.sort(key=lambda z: z[0])
    x_all = np.array([z[0] for z in all_entries], float)
    labels_all = [z[1] for z in all_entries]
    groups_all = [z[2] for z in all_entries]

    nA = len(groups_all)
    draws_all = np.full((S, nA), np.nan, float)
    obs_all = np.full((nA,), np.nan, float)
    fmask_all = np.zeros((nA,), bool)

    last_obs_idx = -1
    for j, I in enumerate(groups_all):
        draws_all[:, j] = np.mean(y_full[:, I], axis=1)
        if np.all(I < T_obs):
            obs_all[j] = float(draws_all[0, j])
            last_obs_idx = j
        else:
            fmask_all[j] = True

    split_x_all = float(x_all[last_obs_idx]) if last_obs_idx >= 0 else float(x_all[0]) if nA else float("nan")
    seasonal_all = (x_all, obs_all, draws_all, fmask_all, split_x_all, labels_all)
    return per_season, seasonal_all


# =============================================================================
# Plotting
# =============================================================================
def plot_forecast(
    *,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_fore: np.ndarray,
    fore_draws: np.ndarray,  # (S, len(x_fore))
    level: float,
    split_x: float,
    title: str,
    ylabel: str,
    save_path: str,
    show: bool,
) -> None:
    med, lo, hi = _summarize_ribbon(fore_draws, level=level)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
    ax.plot(x_obs, y_obs, lw=1.2, label="observed")
    ax.plot(x_fore, med, lw=1.6, label="forecast median")
    ax.fill_between(x_fore, lo, hi, alpha=0.25, label=f"forecast {int(round(level * 100))}% band")

    ax.axvline(float(split_x), lw=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"[save] {save_path}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Forecasting script for the Gaussian DLM posterior.\n"
            "- Loads latest posterior by default (like dlm_plotter.py).\n"
            "- Fine-scale posterior predictive via state propagation + observation simulation.\n"
            "- Coarse-graining is done by aggregating each simulated fine-scale predictive path.\n"
            "- Optionally print selected forecast summaries (median + interval) to stdout.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to a run directory or directly to posterior.npz. If omitted, uses latest under --root.",
    )
    p.add_argument("--root", type=str, default="results/simulations/DLM", help="Search root when --target is omitted.")

    p.add_argument("--horizon", type=int, default=72, help="Forecast horizon in *fine-scale* steps.")
    p.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for predictive simulation.")

    p.add_argument(
        "--start-date",
        type=str,
        default="2000-01-01",
        help=(
            "Optional start date for x-axis (YYYY / YYYY-MM / YYYY-MM-DD). "
            "If omitted, tries meta['start_date'], else uses numeric index."
        ),
    )
    p.add_argument("--out", type=str, default=None, help="Output directory. Default: <run>/forecast")
    p.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")

    # plot windows (last N points of the *observed* series; forecast always shown)
    p.add_argument("--window-months", type=int, default=240, help="Fine-scale plot: last N observed points.")
    p.add_argument("--window-years", type=int, default=60, help="Annual plot: last N observed years.")
    p.add_argument("--window-seasons", type=int, default=120, help="Seasonal plot: last N observed seasons.")

    # ---- Printing selected forecasts ----
    p.add_argument("--print-forecast", action="store_true", default=True, help="Print selected forecast summaries.")
    p.add_argument(
        "--print-scales",
        type=str,
        default="seasonal_all",
        help="Comma-separated: fine,annual,seasonal_all,DJF,MAM,JJA,SON",
    )
    p.add_argument(
        "--print-horizon",
        type=int,
        default=50,
        help="(fine) If no --print-dates/--print-start/--print-end are given, print first N forecast steps ahead.",
    )
    p.add_argument(
        "--print-dates",
        type=str,
        default=None,
        help="Comma-separated dates (YYYY or YYYY-MM or YYYY-MM-DD). Used to select points (fine: months; annual/seasonal: years).",
    )
    p.add_argument("--print-start", type=str, default=None, help="Window start (YYYY or YYYY-MM or YYYY-MM-DD).")
    p.add_argument("--print-end", type=str, default=None, help="Window end (YYYY or YYYY-MM or YYYY-MM-DD).")
    p.add_argument(
        "--print-include-observed",
        action="store_true",
        default=False,
        help="If set, allow printing points that fall in the observed segment too.",
    )
    p.add_argument(
        "--print-max-lines",
        type=int,
        default=80,
        help="Maximum lines printed per scale (truncate beyond this).",
    )

    args = p.parse_args()

    # resolve posterior
    run_path = args.target
    if run_path is None:
        print(f"[info] --target not provided; searching for the latest posterior under --root={args.root!r} ...")
        run_path = find_latest_run(root=args.root)
        if run_path is None:
            raise SystemExit(f"[error] No 'posterior.npz' found under {args.root!r}. Provide --target or change --root.")
        print(f"[info] Using latest run: {run_path}")

    bundle = load_posterior(run_path)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "forecast")
    _ensure_dir(out_dir)
    print(f"[info] saving outputs to: {out_dir}")

    # start date discovery
    sd = _parse_date(args.start_date)
    if sd is None:
        sd = _parse_date(meta.get("start_date")) if isinstance(meta, dict) else None

    # simulate fine-scale posterior predictive
    fr = simulate_dlm_forecast(
        draws=draws,
        meta=meta,
        horizon=int(args.horizon),
        seed=int(args.seed),
        start_date=sd,
    )

    y_obs = fr.y_obs
    y_future = fr.y_future
    y_full = fr.y_full
    x_full = fr.x_axis_full
    T = int(y_obs.size)
    H = int(args.horizon)
    L_full = T + H

    # -----------------------------
    # 1) Fine-scale plot
    # -----------------------------
    wM = max(1, int(args.window_months))
    i0 = max(0, T - wM)
    x_obs_m = x_full[i0:T]
    y_obs_m = y_obs[i0:T]

    x_fore_m = x_full[T:L_full]
    fore_draws_m = y_future  # (S, H)

    plot_forecast(
        x_obs=x_obs_m,
        y_obs=y_obs_m,
        x_fore=x_fore_m,
        fore_draws=fore_draws_m,
        level=float(args.level),
        split_x=float(fr.split_x),
        title=f"DLM fine-scale forecast (h={H})",
        ylabel="y (fine-scale)",
        save_path=os.path.join(out_dir, "dlm_forecast_fine.png"),
        show=bool(args.show),
    )

    np.savez_compressed(
        os.path.join(out_dir, "dlm_forecast_fine.npz"),
        y_obs=y_obs,
        y_future=y_future,
        x_full=x_full,
        T=T,
        H=H,
        level=float(args.level),
    )

    # -----------------------------
    # 2) Annual averages (always)
    # -----------------------------
    period = int(meta.get("period", 12))
    xY, yY_obs, drawsY, maskY, splitY = coarse_grain_annual_average(
        y_full=y_full,
        T_obs=T,
        period=period,
        start_year=fr.start_year,
        start_month=fr.start_month,
    )

    obs_idx = np.isfinite(yY_obs)
    x_obs_y = xY[obs_idx]
    y_obs_y = yY_obs[obs_idx]

    x_fore_y = xY[maskY]
    fore_draws_y = drawsY[:, maskY] if x_fore_y.size else np.zeros((drawsY.shape[0], 0))

    wY = max(1, int(args.window_years))
    if x_obs_y.size > wY:
        x_obs_y = x_obs_y[-wY:]
        y_obs_y = y_obs_y[-wY:]

    plot_forecast(
        x_obs=x_obs_y,
        y_obs=y_obs_y,
        x_fore=x_fore_y,
        fore_draws=fore_draws_y,
        level=float(args.level),
        split_x=float(splitY),
        title="DLM annual-average forecast",
        ylabel="annual average",
        save_path=os.path.join(out_dir, "dlm_forecast_annual_avg.png"),
        show=bool(args.show),
    )

    # -----------------------------
    # 3) Meteorological seasons (period=12 only)
    # -----------------------------
    per_season: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]] = None
    seasonal_all: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, List[str]]] = None

    if period != 12 or fr.start_year is None or fr.start_month is None:
        print("[warn] meteorological seasons (DJF/MAM/JJA/SON) require period=12 and a known start-date. Skipping.")
    else:
        per_season, seasonal_all = coarse_grain_meteo_seasons(
            y_full=y_full,
            T_obs=T,
            start_year=int(fr.start_year),
            start_month=int(fr.start_month),
        )

        # seasonal_all (all seasons stacked)
        xA, obsA, drawsA, maskA, splitA, labelsA = seasonal_all

        obs_idxA = np.isfinite(obsA)
        x_obs_A = xA[obs_idxA]
        y_obs_A = obsA[obs_idxA]

        x_fore_A = xA[maskA]
        fore_draws_A = drawsA[:, maskA] if x_fore_A.size else np.zeros((drawsA.shape[0], 0))

        wS = max(1, int(args.window_seasons))
        if x_obs_A.size > wS:
            x_obs_A = x_obs_A[-wS:]
            y_obs_A = y_obs_A[-wS:]

        plot_forecast(
            x_obs=x_obs_A,
            y_obs=y_obs_A,
            x_fore=x_fore_A,
            fore_draws=fore_draws_A,
            level=float(args.level),
            split_x=float(splitA),
            title="DLM seasonal-average forecast (all seasons in sequence)",
            ylabel="seasonal average",
            save_path=os.path.join(out_dir, "dlm_forecast_seasonal_all_avg.png"),
            show=bool(args.show),
        )

        # separate season plots
        for sname in ("DJF", "MAM", "JJA", "SON"):
            xS, obsS, drawsS, maskS, splitS = per_season[sname]

            obs_idxS = np.isfinite(obsS)
            x_obs_S = xS[obs_idxS]
            y_obs_S = obsS[obs_idxS]

            x_fore_S = xS[maskS]
            fore_draws_S = drawsS[:, maskS] if x_fore_S.size else np.zeros((drawsS.shape[0], 0))

            if x_obs_S.size > wY:
                x_obs_S = x_obs_S[-wY:]
                y_obs_S = y_obs_S[-wY:]

            plot_forecast(
                x_obs=x_obs_S,
                y_obs=y_obs_S,
                x_fore=x_fore_S,
                fore_draws=fore_draws_S,
                level=float(args.level),
                split_x=float(splitS),
                title=f"DLM seasonal-average forecast ({sname})",
                ylabel="seasonal average",
                save_path=os.path.join(out_dir, f"dlm_forecast_seasonal_{sname}_avg.png"),
                show=bool(args.show),
            )

    # -----------------------------
    # 4) Print selected forecast summaries (stdout)
    # -----------------------------
    if bool(args.print_forecast):
        scales = [s.strip() for s in str(args.print_scales).split(",") if s.strip()]
        dates = _parse_csv(args.print_dates)
        include_obs = bool(args.print_include_observed)
        max_lines = int(args.print_max_lines)

        fine_labels = _calendar_labels(
            L_full=L_full, start_year=fr.start_year, start_month=fr.start_month, period=period
        )

        # ---- fine ----
        if "fine" in scales:
            idxs = _select_fine_indices(
                L_full=L_full,
                T_obs=T,
                period=period,
                start_year=fr.start_year,
                start_month=fr.start_month,
                labels=fine_labels,
                dates=dates,
                start=args.print_start,
                end=args.print_end,
                include_observed=include_obs,
                print_horizon=args.print_horizon,
                H=H,
            )
            labs = [fine_labels[i] if fine_labels is not None else str(i) for i in idxs]
            xx = x_full[idxs] if idxs else np.array([], float)
            DD = y_full[:, idxs] if idxs else np.zeros((y_full.shape[0], 0), float)
            isF = [bool(i >= T) for i in idxs]
            _print_block(
                name="fine",
                labels=labs,
                x=xx,
                draws=DD,
                is_forecast=isF,
                level=float(args.level),
                max_lines=max_lines,
            )

        # ---- annual ----
        if "annual" in scales:
            idxsY = _select_by_years(
                x=xY,
                forecast_mask=maskY,
                dates=dates,
                start=args.print_start,
                end=args.print_end,
                include_observed=include_obs,
            )
            labs = [str(int(np.floor(xY[i]))) for i in idxsY]
            xx = xY[idxsY] if idxsY else np.array([], float)
            DD = drawsY[:, idxsY] if idxsY else np.zeros((drawsY.shape[0], 0), float)
            isF = [bool(maskY[i]) for i in idxsY]
            _print_block(
                name="annual",
                labels=labs,
                x=xx,
                draws=DD,
                is_forecast=isF,
                level=float(args.level),
                max_lines=max_lines,
            )

        # ---- seasonal_all ----
        if "seasonal_all" in scales:
            if seasonal_all is None:
                print("\n[print] seasonal_all: not available (need period=12 and known start-date)\n")
            else:
                xA, obsA, drawsA, maskA, splitA, labelsA = seasonal_all
                idxsA = _select_by_years(
                    x=xA,
                    forecast_mask=maskA,
                    dates=dates,
                    start=args.print_start,
                    end=args.print_end,
                    include_observed=include_obs,
                )
                labs = [labelsA[i] for i in idxsA]
                xx = xA[idxsA] if idxsA else np.array([], float)
                DD = drawsA[:, idxsA] if idxsA else np.zeros((drawsA.shape[0], 0), float)
                isF = [bool(maskA[i]) for i in idxsA]
                _print_block(
                    name="seasonal_all",
                    labels=labs,
                    x=xx,
                    draws=DD,
                    is_forecast=isF,
                    level=float(args.level),
                    max_lines=max_lines,
                )

        # ---- individual seasons ----
        for sname in ("DJF", "MAM", "JJA", "SON"):
            if sname not in scales:
                continue
            if per_season is None or sname not in per_season:
                print(f"\n[print] {sname}: not available (need period=12 and known start-date)\n")
                continue

            xS, obsS, drawsS, maskS, splitS = per_season[sname]
            idxsS = _select_by_years(
                x=xS,
                forecast_mask=maskS,
                dates=dates,
                start=args.print_start,
                end=args.print_end,
                include_observed=include_obs,
            )
            labs = [f"{int(np.floor(xS[i])):04d}-{sname}" for i in idxsS]
            xx = xS[idxsS] if idxsS else np.array([], float)
            DD = drawsS[:, idxsS] if idxsS else np.zeros((drawsS.shape[0], 0), float)
            isF = [bool(maskS[i]) for i in idxsS]
            _print_block(
                name=sname,
                labels=labs,
                x=xx,
                draws=DD,
                is_forecast=isF,
                level=float(args.level),
                max_lines=max_lines,
            )

    print("[done] fine + annual + seasonal forecasts written.")
