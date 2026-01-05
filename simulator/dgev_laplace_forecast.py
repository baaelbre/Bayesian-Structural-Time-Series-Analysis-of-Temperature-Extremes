# simulator/dgev_forecast.py
from __future__ import annotations

"""
DGEV posterior predictive forecasting (fine + annual + meteorological seasons).

Key design choices (mirrors your manuscript + Laplace posterior format):
- Fine forecast: simulate future latent state paths under the structural evolution,
  then draw y_{T+1:T+h} from the GEV observation model.
- Annual / seasonal block extremes: conditional on the future (mu_t, sigma, xi),
  use the product-CDF max distribution (eq. 31) via 1D inversion; optionally
  combine with already-observed months in the same block via max(m0, Z_future).

Plot policy (as requested):
- Only plot observed training data + forecast median + forecast band.
- No posterior fit ribbon on training period.

Seasonal aggregation policy (as requested):
- Meteorological seasons (DJF/MAM/JJA/SON) only when period==12 AND dates available.
- Annual summaries always work (calendar-year if dates available; else blocks of length=period).

Latest-run discovery (as requested):
- If --target omitted, uses find_latest_run(root=--root) exactly like the plotter.
"""

import os
import sys
import math
import json
import argparse
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make optimization package visible (mirrors dgev_laplace_plotter.py)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------
# I/O helpers: posterior loader (EXACTLY like the plotter pattern)
# ---------------------------------------------------------------------
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


# =============================================================================
# Small utils
# =============================================================================
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


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
    """
    Same logic as in your plotter: check meta flags, transform hints, series naming.
    """
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


def _infer_period(meta: Dict[str, Any], draws: Dict[str, np.ndarray]) -> int:
    if isinstance(meta, dict) and "period" in meta:
        return int(meta["period"])
    if "period" in draws:
        return int(np.asarray(draws["period"]).item())
    if "gamma0" in draws:
        return int(np.asarray(draws["gamma0"]).shape[1] + 1)
    if "x" in draws:
        # x: (S,T,dim) with dim = 2 + (p-1) => dim = p+1 => p = dim-1
        dim = int(np.asarray(draws["x"]).shape[2])
        return int(dim - 1)
    raise ValueError("Could not infer period from meta/draws.")


def _tail_quantiles(alpha: float) -> Tuple[float, float]:
    a = float(alpha)
    a = min(max(a, 1e-6), 0.49)
    return (a, 1.0 - a)


# =============================================================================
# GEV helpers
# =============================================================================
def gev_cdf(z: np.ndarray, mu: np.ndarray, sigma: float, xi: float) -> np.ndarray:
    z = np.asarray(z, float)
    mu = np.asarray(mu, float)
    sig = float(sigma)
    x = float(xi)

    if sig <= 0.0 or not np.isfinite(sig):
        return np.full_like(z, np.nan)

    t = (z - mu) / sig
    if abs(x) < 1e-12:
        return np.exp(-np.exp(-t))

    a = 1.0 + x * t
    out = np.zeros_like(a)
    ok = a > 0
    out[~ok] = 0.0
    out[ok] = np.exp(-(a[ok]) ** (-1.0 / x))
    return out


def gev_ppf(u: np.ndarray, mu: np.ndarray, sigma: float, xi: float) -> np.ndarray:
    u = np.asarray(u, float)
    mu = np.asarray(mu, float)
    sig = float(sigma)
    x = float(xi)

    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    if abs(x) < 1e-12:
        return mu - sig * np.log(-np.log(u))
    return mu + (sig / x) * ((-np.log(u)) ** (-x) - 1.0)


# =============================================================================
# Structural seasonal rotation (dummy seasonal block)
# =============================================================================
def _season_rotate(g_prev: np.ndarray) -> np.ndarray:
    """
    Seasonal state for dummy encoding with sum-to-zero:
      g_new[0] = -sum(g_prev)
      g_new[1:] = g_prev[:-1]
    Innovation noise is added only to g_new[0].
    """
    g_prev = np.asarray(g_prev, float)
    if g_prev.size == 0:
        return g_prev.copy()
    out = np.empty_like(g_prev)
    out[0] = -float(np.sum(g_prev))
    if g_prev.size > 1:
        out[1:] = g_prev[:-1]
    return out


def _baseline_effect(season_idx: int, gamma0: np.ndarray) -> float:
    """
    Fixed seasonal baseline (sum-to-zero dummy coding) with length K=p-1:
    months 0..K-1 use gamma0[k], the last season uses -sum(gamma0).
    """
    gamma0 = np.asarray(gamma0, float)
    K = int(gamma0.size)
    if season_idx < K:
        return float(gamma0[season_idx])
    return -float(np.sum(gamma0))


# =============================================================================
# Dates (optional, if meta has start_date)
# =============================================================================
def _try_build_dates(meta: Dict[str, Any], T: int, h: int, period: int):
    if pd is None:
        return None
    start = meta.get("start_date") or meta.get("start-date") or meta.get("startDate")
    if start is None:
        return None
    try:
        start_dt = pd.to_datetime(start)
    except Exception:
        return None

    # monthly
    if period == 12:
        try:
            return pd.date_range(start=start_dt, periods=T + h, freq="MS")
        except Exception:
            return pd.date_range(start=start_dt, periods=T + h, freq=pd.DateOffset(months=1))

    # if period divides 12, interpret as regular sub-annual blocks (e.g. 4=quarters, 6=bimonthly)
    if 12 % period == 0:
        step_m = 12 // period
        return pd.date_range(start=start_dt, periods=T + h, freq=pd.DateOffset(months=step_m))

    return None


def _year_float_from_dates(dates) -> np.ndarray:
    years = dates.year.astype(float)
    # represent within-year position using month
    years = years + (dates.month.astype(float) - 1.0) / 12.0
    return np.asarray(years, float)


# =============================================================================
# Product-CDF inversion sampler for block maxima (eq. 31)
# =============================================================================
def _max_cdf(z: float, mus: np.ndarray, sigma: float, xi: float) -> float:
    """
    CDF of the maximum of independent GEV draws with varying mu_t (shared sigma, xi):
      P(max <= z) = prod_t G(z; mu_t, sigma, xi).
    """
    mus = np.asarray(mus, float)
    G = gev_cdf(np.full_like(mus, float(z)), mus, float(sigma), float(xi))
    G = np.clip(G, 0.0, 1.0)
    # compute in log space to reduce underflow
    with np.errstate(divide="ignore"):
        lg = np.log(np.maximum(G, 1e-300))
    return float(np.exp(np.sum(lg)))


def _sample_max_product_cdf(
    rng: np.random.Generator,
    mus: np.ndarray,
    sigma: float,
    xi: float,
    *,
    max_iter: int = 80,
) -> float:
    """
    Sample Z = max(Y_1,...,Y_m) where Y_j ~ GEV(mu_j, sigma, xi) independent,
    using 1D inversion by bisection on F_Z(z) = prod_j G(z; mu_j, sigma, xi).

    Falls back to direct simulation if bracketing fails.
    """
    mus = np.asarray(mus, float)
    if mus.size == 0:
        return float("nan")

    u = float(rng.uniform(1e-12, 1.0 - 1e-12))
    sig = float(sigma)
    x = float(xi)

    if sig <= 0 or not np.isfinite(sig) or not np.isfinite(x):
        # degrade gracefully: draw directly using an arbitrary positive sigma
        sig = 1.0 if (not np.isfinite(sig) or sig <= 0) else sig
        x = 0.0 if (not np.isfinite(x)) else x

    # choose a reasonable bracket [lo, hi]
    lo = float(np.min(mus) - 10.0 * sig)

    if x < -1e-12:
        # finite upper endpoint for each margin: mu - sigma/xi ; max endpoint is min over endpoints
        endpoints = mus - sig / x
        hi = float(np.min(endpoints) - 1e-9 * max(1.0, abs(np.min(endpoints))))
    else:
        hi = float(np.max(mus) + 10.0 * sig)

    # widen lo until F(lo) <= u
    f_lo = _max_cdf(lo, mus, sig, x)
    step = 10.0 * sig if sig > 0 else 10.0
    k = 0
    while (np.isfinite(f_lo) and f_lo > u) and (k < 50):
        lo -= step
        step *= 1.5
        f_lo = _max_cdf(lo, mus, sig, x)
        k += 1

    # widen hi until F(hi) >= u (only if xi>=0)
    f_hi = _max_cdf(hi, mus, sig, x)
    step = 10.0 * sig if sig > 0 else 10.0
    k = 0
    while (np.isfinite(f_hi) and f_hi < u) and (k < 60):
        if x < -1e-12:
            break
        hi += step
        step *= 1.5
        f_hi = _max_cdf(hi, mus, sig, x)
        k += 1

    # fallback if bracketing fails
    if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)) or (not (f_lo <= u <= f_hi)):
        yy = gev_ppf(rng.uniform(size=mus.size), mus, sig, x)
        return float(np.max(yy))

    # bisection
    a, b = lo, hi
    for _ in range(int(max_iter)):
        m = 0.5 * (a + b)
        fm = _max_cdf(m, mus, sig, x)
        if not np.isfinite(fm):
            # if numerics go weird, shrink interval conservatively
            b = m
            continue
        if fm < u:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


# =============================================================================
# Aggregation helpers (annual + meteorological seasons)
# =============================================================================
def _season_label(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _season_year(year: int, month: int) -> int:
    # DJF is grouped with "season-year" = year+1 for December
    return year + 1 if month == 12 else year


def _sorted_unique(items: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


# =============================================================================
# Forecast core
# =============================================================================
def forecast_from_bundle(
    draws: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    *,
    h: int,
    rng: np.random.Generator,
    n_draws: Optional[int],
    rep_per_draw: int,
    rep_per_draw_max: int,
    alpha: float,
    minima: bool,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - fine forecast: y_obs (training), x_train, x_fore, y_fore_samples (S*rep_per_draw, h)
      - annual: years, obs, med, lo, hi (aligned arrays with NaNs where unavailable)
      - seasons (only if period==12 and dates): season_x, season_year, season_name, obs, med, lo, hi
    """
    period = _infer_period(meta, draws)

    # stored model-scale series (may be negated for minima-series)
    y_model = np.asarray(draws["y"], float)
    T = int(y_model.size)
    if T < 1:
        raise ValueError("Posterior bundle has empty 'y'.")

    # core posterior draws
    sigma_all = np.asarray(draws["sigma"], float)
    xi_all = np.asarray(draws["xi"], float)
    x_all = np.asarray(draws["x"], float)  # (S,T,dim) with dim = 2 + (p-1)

    S_all = int(sigma_all.size)
    if x_all.ndim != 3 or x_all.shape[0] != S_all or x_all.shape[1] != T:
        raise ValueError("Expected draws['x'] to have shape (S,T,dim) consistent with sigma and y.")

    # baseline seasonal means gamma0 (S, p-1) in your Laplace code
    if "gamma0" in draws:
        gamma0_all = np.asarray(draws["gamma0"], float)
        if gamma0_all.ndim != 2 or gamma0_all.shape[0] != S_all:
            raise ValueError("draws['gamma0'] must be (S,p-1).")
    else:
        gamma0_all = np.zeros((S_all, period - 1), float)

    # process variances (prefer Q_*; else use s_*^2)
    if all(k in draws for k in ("Q_alpha", "Q_beta", "Q_gamma")):
        Q_alpha_all = np.asarray(draws["Q_alpha"], float)
        Q_beta_all = np.asarray(draws["Q_beta"], float)
        Q_gamma_all = np.asarray(draws["Q_gamma"], float)
    else:
        s_alpha = np.asarray(draws.get("s_alpha"), float)
        s_beta = np.asarray(draws.get("s_beta"), float)
        s_gamma = np.asarray(draws.get("s_gamma"), float)
        if s_alpha.size != S_all or s_beta.size != S_all or s_gamma.size != S_all:
            raise ValueError("Need either Q_alpha/Q_beta/Q_gamma or s_alpha/s_beta/s_gamma in posterior.")
        Q_alpha_all = s_alpha * s_alpha
        Q_beta_all = s_beta * s_beta
        Q_gamma_all = s_gamma * s_gamma

    # optional subsample
    if n_draws is not None and 1 <= int(n_draws) < S_all:
        idx = rng.choice(S_all, size=int(n_draws), replace=False)
        idx.sort()
    else:
        idx = np.arange(S_all)

    sigma = sigma_all[idx]
    xi = xi_all[idx]
    x = x_all[idx, :, :]
    gamma0 = gamma0_all[idx, :]
    Q_alpha = Q_alpha_all[idx]
    Q_beta = Q_beta_all[idx]
    Q_gamma = Q_gamma_all[idx]
    S = int(idx.size)

    # build dates if possible
    dates_full = _try_build_dates(meta, T=T, h=int(h), period=period)
    if dates_full is not None:
        x_full = _year_float_from_dates(dates_full)
        x_train = x_full[:T]
        x_fore = x_full[T:]

        if period == 12:
            season_idx_fore = (dates_full[T:].month.values - 1).astype(int)  # 0..11
        else:
            season_idx_fore = (np.arange(T, T + h) % period).astype(int)

        years_full = dates_full.year.values.astype(int)   # length T+h
        months_full = dates_full.month.values.astype(int) # length T+h
    else:
        x_train = np.arange(T, dtype=float)
        x_fore = np.arange(T, T + h, dtype=float)
        season_idx_fore = (np.arange(T, T + h) % period).astype(int)
        years_full = None
        months_full = None

    # state layout: [alpha, beta, g1..g_{p-1}]
    idx_alpha = 0
    idx_beta = 1
    idx_g_start = 2
    K = int(period - 1)

    # --- simulate future mu_t on model scale, and fine predictive y ---
    mu_fore = np.zeros((S, h), float)
    y_fore_samples_model = np.zeros((S * rep_per_draw, h), float)

    for s in range(S):
        alpha_t = float(x[s, -1, idx_alpha])
        beta_t = float(x[s, -1, idx_beta])
        gvec = np.asarray(x[s, -1, idx_g_start:idx_g_start + K], float).copy()

        sd_a = math.sqrt(max(float(Q_alpha[s]), 0.0))
        sd_b = math.sqrt(max(float(Q_beta[s]), 0.0))
        sd_g = math.sqrt(max(float(Q_gamma[s]), 0.0))

        # one latent path per posterior draw (keeps forecast dependence structure simple)
        for ell in range(h):
            alpha_t = alpha_t + beta_t + rng.normal(0.0, sd_a)
            beta_t = beta_t + rng.normal(0.0, sd_b)

            if K > 0:
                gnew = _season_rotate(gvec)
                gnew[0] = gnew[0] + rng.normal(0.0, sd_g)  # innovation only in first seasonal component
                gvec = gnew
                g1 = float(gvec[0])
                base = _baseline_effect(int(season_idx_fore[ell]), gamma0[s, :])
            else:
                g1 = 0.0
                base = 0.0

            mu_fore[s, ell] = alpha_t + g1 + base

        # draw y from observation model
        for r in range(int(rep_per_draw)):
            uu = rng.uniform(size=h)
            y_fore_samples_model[s * rep_per_draw + r, :] = gev_ppf(
                uu, mu_fore[s, :], float(sigma[s]), float(xi[s])
            )

    # backtransform for plotting/original scale
    if minima:
        y_obs = -y_model
        y_fore_samples = -y_fore_samples_model
    else:
        y_obs = y_model
        y_fore_samples = y_fore_samples_model

    # observed in model scale (for combining with future max blocks via max(m0, Z_future))
    y_obs_model = y_model

    lo_q, hi_q = _tail_quantiles(alpha)

    # =============================================================================
    # Annual aggregation (ALWAYS produced)
    # =============================================================================
    if dates_full is not None and years_full is not None:
        # calendar-year blocks
        year_labels = years_full  # length T+h
        uniq_years = sorted(set(int(v) for v in year_labels))

        yrs, obs_list, med_list, lo_list, hi_list = [], [], [], [], []
        for yy in uniq_years:
            idx_all = np.where(year_labels == yy)[0]
            idx_train = idx_all[idx_all < T]
            idx_fore = idx_all[idx_all >= T]

            if idx_train.size == 0 and idx_fore.size == 0:
                continue

            # observed annual block value (training only)
            obs_val = float(np.max(y_obs[idx_train])) if idx_train.size > 0 else float("nan")

            # aligned padding when no forecast in that year
            if idx_fore.size == 0:
                yrs.append(int(yy))
                obs_list.append(obs_val)
                med_list.append(float("nan"))
                lo_list.append(float("nan"))
                hi_list.append(float("nan"))
                continue

            # combine already-observed part of the block (if any) with future part
            m0 = float(np.max(y_obs_model[idx_train])) if idx_train.size > 0 else -float("inf")
            pos = (idx_fore - T).astype(int)

            draws_block = np.zeros((S * rep_per_draw_max,), float)
            for s in range(S):
                mus = mu_fore[s, pos]
                for r in range(int(rep_per_draw_max)):
                    zf = _sample_max_product_cdf(rng, mus, float(sigma[s]), float(xi[s]))
                    draws_block[s * rep_per_draw_max + r] = max(m0, float(zf))

            draws_plot = -draws_block if minima else draws_block

            yrs.append(int(yy))
            obs_list.append(obs_val)
            med_list.append(float(np.quantile(draws_plot, 0.5)))
            lo_list.append(float(np.quantile(draws_plot, lo_q)))
            hi_list.append(float(np.quantile(draws_plot, hi_q)))

        annual = {
            "years": np.asarray(yrs, int),
            "obs": np.asarray(obs_list, float),
            "med": np.asarray(med_list, float),
            "lo": np.asarray(lo_list, float),
            "hi": np.asarray(hi_list, float),
        }
    else:
        # fallback: define "years" as blocks of length=period
        n_total = T + h
        year_id = np.arange(n_total, dtype=int) // int(period)
        uniq_years = sorted(set(int(v) for v in year_id))

        yrs, obs_list, med_list, lo_list, hi_list = [], [], [], [], []
        for yy in uniq_years:
            idx_all = np.where(year_id == yy)[0]
            idx_train = idx_all[idx_all < T]
            idx_fore = idx_all[idx_all >= T]

            if idx_train.size == 0 and idx_fore.size == 0:
                continue

            obs_val = float(np.max(y_obs[idx_train])) if idx_train.size > 0 else float("nan")

            if idx_fore.size == 0:
                yrs.append(int(yy))
                obs_list.append(obs_val)
                med_list.append(float("nan"))
                lo_list.append(float("nan"))
                hi_list.append(float("nan"))
                continue

            m0 = float(np.max(y_obs_model[idx_train])) if idx_train.size > 0 else -float("inf")
            pos = (idx_fore - T).astype(int)

            draws_block = np.zeros((S * rep_per_draw_max,), float)
            for s in range(S):
                mus = mu_fore[s, pos]
                for r in range(int(rep_per_draw_max)):
                    zf = _sample_max_product_cdf(rng, mus, float(sigma[s]), float(xi[s]))
                    draws_block[s * rep_per_draw_max + r] = max(m0, float(zf))

            draws_plot = -draws_block if minima else draws_block

            yrs.append(int(yy))
            obs_list.append(obs_val)
            med_list.append(float(np.quantile(draws_plot, 0.5)))
            lo_list.append(float(np.quantile(draws_plot, lo_q)))
            hi_list.append(float(np.quantile(draws_plot, hi_q)))

        annual = {
            "years": np.asarray(yrs, int),
            "obs": np.asarray(obs_list, float),
            "med": np.asarray(med_list, float),
            "lo": np.asarray(lo_list, float),
            "hi": np.asarray(hi_list, float),
        }

    # =============================================================================
    # Meteorological seasons (ONLY for monthly data with dates)
    # =============================================================================
    seasons: Dict[str, Any] = {}
    if period == 12 and dates_full is not None and years_full is not None and months_full is not None:
        labels: List[Tuple[int, str]] = []
        for dt in dates_full:
            sy = _season_year(int(dt.year), int(dt.month))
            sl = _season_label(int(dt.month))
            labels.append((sy, sl))

        # sort season blocks chronologically
        order = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
        uniq = _sorted_unique(labels)
        uniq = sorted(uniq, key=lambda x: (int(x[0]) * 4 + order[str(x[1])]))

        season_years: List[int] = []
        season_names: List[str] = []
        season_x: List[float] = []
        obs_vals: List[float] = []
        med_vals: List[float] = []
        lo_vals: List[float] = []
        hi_vals: List[float] = []

        for (sy, sl) in uniq:
            idx_all = np.array([i for i, lab in enumerate(labels) if lab == (sy, sl)], dtype=int)
            idx_train = idx_all[idx_all < T]
            idx_fore = idx_all[idx_all >= T]

            if idx_train.size == 0 and idx_fore.size == 0:
                continue

            obs_val = float(np.max(y_obs[idx_train])) if idx_train.size > 0 else float("nan")

            # aligned padding when no forecast in that season-block
            if idx_fore.size == 0:
                season_years.append(int(sy))
                season_names.append(str(sl))
                season_x.append(float(sy) + float(order[str(sl)]) / 4.0)

                obs_vals.append(obs_val)
                med_vals.append(float("nan"))
                lo_vals.append(float("nan"))
                hi_vals.append(float("nan"))
                continue

            m0 = float(np.max(y_obs_model[idx_train])) if idx_train.size > 0 else -float("inf")
            pos = (idx_fore - T).astype(int)

            draws_block = np.zeros((S * rep_per_draw_max,), float)
            for s in range(S):
                mus = mu_fore[s, pos]
                for r in range(int(rep_per_draw_max)):
                    zf = _sample_max_product_cdf(rng, mus, float(sigma[s]), float(xi[s]))
                    draws_block[s * rep_per_draw_max + r] = max(m0, float(zf))

            draws_plot = -draws_block if minima else draws_block

            season_years.append(int(sy))
            season_names.append(str(sl))
            season_x.append(float(sy) + float(order[str(sl)]) / 4.0)

            obs_vals.append(obs_val)
            med_vals.append(float(np.quantile(draws_plot, 0.5)))
            lo_vals.append(float(np.quantile(draws_plot, lo_q)))
            hi_vals.append(float(np.quantile(draws_plot, hi_q)))

        seasons = {
            "season_x": np.asarray(season_x, float),         # sequential axis for seasonal_all plot
            "season_year": np.asarray(season_years, int),
            "season_name": np.asarray(season_names, object),
            "obs": np.asarray(obs_vals, float),
            "med": np.asarray(med_vals, float),
            "lo": np.asarray(lo_vals, float),
            "hi": np.asarray(hi_vals, float),
        }

    return {
        "period": int(period),
        "T": int(T),
        "h": int(h),
        "minima": bool(minima),
        "dates_full": dates_full,  # may be None (not npz-serializable; kept for interactive use)
        "x_train": np.asarray(x_train, float),
        "x_fore": np.asarray(x_fore, float),
        "y_obs": np.asarray(y_obs, float),
        "y_fore_samples": np.asarray(y_fore_samples, float),
        "mu_fore": np.asarray(mu_fore, float),  # model scale
        "annual": annual,
        "seasons": seasons,
    }


# =============================================================================
# Plotting (only observed + forecast median/band)
# =============================================================================
def _plot_fine(fc: Dict[str, Any], out_path: str, *, title: str, alpha: float) -> None:
    y_obs = np.asarray(fc["y_obs"], float)
    x_train = np.asarray(fc["x_train"], float)
    x_fore = np.asarray(fc["x_fore"], float)
    y_fore = np.asarray(fc["y_fore_samples"], float)

    lo, hi = _tail_quantiles(alpha)
    med = np.quantile(y_fore, 0.5, axis=0)
    qlo = np.quantile(y_fore, lo, axis=0)
    qhi = np.quantile(y_fore, hi, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x_train, y_obs, lw=1.2, label="observed")

    ax.fill_between(x_fore, qlo, qhi, alpha=0.2, label="forecast band")
    ax.plot(x_fore, med, lw=2.0, label="forecast median")

    if x_fore.size:
        ax.axvline(x_fore[0], lw=1.0, alpha=0.7)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_annual(annual: Dict[str, np.ndarray], out_path: str, *, title: str) -> None:
    years = np.asarray(annual["years"], int)
    obs = np.asarray(annual["obs"], float)
    med = np.asarray(annual["med"], float)
    lo = np.asarray(annual["lo"], float)
    hi = np.asarray(annual["hi"], float)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    mask_obs = np.isfinite(obs)
    ax.plot(years[mask_obs], obs[mask_obs], lw=1.5, label="observed")

    mask_fc = np.isfinite(med)
    if np.any(mask_fc):
        ax.fill_between(years[mask_fc], lo[mask_fc], hi[mask_fc], alpha=0.2, label="forecast band")
        ax.plot(years[mask_fc], med[mask_fc], lw=2.0, label="forecast median")
        ax.axvline(years[mask_fc][0], lw=1.0, alpha=0.7)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_seasonal_all(seasons: Dict[str, Any], out_path: str, *, title: str) -> None:
    sx = np.asarray(seasons["season_x"], float)
    obs = np.asarray(seasons["obs"], float)
    med = np.asarray(seasons["med"], float)
    lo = np.asarray(seasons["lo"], float)
    hi = np.asarray(seasons["hi"], float)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    mask_obs = np.isfinite(obs)
    ax.plot(sx[mask_obs], obs[mask_obs], lw=1.3, label="observed")

    mask_fc = np.isfinite(med)
    if np.any(mask_fc):
        ax.fill_between(sx[mask_fc], lo[mask_fc], hi[mask_fc], alpha=0.2, label="forecast band")
        ax.plot(sx[mask_fc], med[mask_fc], lw=2.0, label="forecast median")
        ax.axvline(sx[mask_fc][0], lw=1.0, alpha=0.7)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_season_by_name(seasons: Dict[str, Any], season_name: str, out_path: str, *, title: str) -> None:
    sy = np.asarray(seasons["season_year"], int)
    sn = np.asarray(seasons["season_name"], object)
    obs = np.asarray(seasons["obs"], float)
    med = np.asarray(seasons["med"], float)
    lo = np.asarray(seasons["lo"], float)
    hi = np.asarray(seasons["hi"], float)

    mask_name = np.array([str(x) == season_name for x in sn], dtype=bool)
    if not np.any(mask_name):
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    mask_obs = mask_name & np.isfinite(obs)
    ax.plot(sy[mask_obs], obs[mask_obs], lw=1.3, label="observed")

    mask_fc = mask_name & np.isfinite(med)
    if np.any(mask_fc):
        ax.fill_between(sy[mask_fc], lo[mask_fc], hi[mask_fc], alpha=0.2, label="forecast band")
        ax.plot(sy[mask_fc], med[mask_fc], lw=2.0, label="forecast median")
        ax.axvline(sy[mask_fc][0], lw=1.0, alpha=0.7)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# =============================================================================
# CLI (mirrors plotter: --target optional, else find_latest_run(--root))
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "DGEV posterior predictive forecasting (fine + annual + meteorological seasons).\n"
            "Latest posterior discovery mirrors dgev_laplace_plotter.py:\n"
            "  - if --target omitted, uses find_latest_run(root=--root)\n"
            "  - load_posterior(run_path) provides draws/meta/npz_path\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--target", type=str, default=None,
                        help="Path to a run directory or directly to posterior.npz. If omitted, searches under --root.")
    parser.add_argument("--root", type=str, default="results/simulations/DGEV_NCP_LASSO",
                        help="Search root if --target is omitted.")

    parser.add_argument("--h", type=int, default=60, help="Forecast horizon in native block units.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Band tail prob (alpha=0.1 -> 80% band).")
    parser.add_argument("--n-draws", type=int, default=None, help="Subsample this many posterior draws (default all).")
    parser.add_argument("--rep-per-draw", type=int, default=1, help="Fine predictive replicates per posterior draw.")
    parser.add_argument("--rep-per-draw-max", type=int, default=1, help="Block-max replicates per posterior draw.")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")

    parser.add_argument("--out", type=str, default=None,
                        help="Output dir. Default: <run>/forecasts")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--minima", action="store_true", help="Force minima=True (back-transform stored negated series).")
    g.add_argument("--maxima", action="store_true", help="Force minima=False (no back-transform).")

    parser.add_argument("--save-npz", action="store_true", default=True, help="Save forecast_summary.npz")
    args = parser.parse_args()

    run_path = args.target
    if run_path is None:
        print(f"[info] --target not provided; searching for the latest posterior under --root={args.root!r} ...")
        run_path = find_latest_run(root=args.root)
        if run_path is None:
            print(f"[error] No 'posterior.npz' found under {args.root!r}. Provide --target or change --root.")
            sys.exit(1)
        print(f"[info] Using latest run: {run_path}")

    bundle = load_posterior(run_path)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "forecasts")
    _ensure_dir(out_dir)
    print(f"[info] saving forecasts to: {out_dir}")

    # minima override (like plotter)
    if args.minima:
        minima = True
    elif args.maxima:
        minima = False
    else:
        minima = _detect_minima_from_meta(meta)
        if minima:
            print("[info] minima=True detected from meta → back-transforming forecasts for plotting.")

    rng = np.random.default_rng(int(args.seed))

    fc = forecast_from_bundle(
        draws=draws,
        meta=meta,
        h=int(args.h),
        rng=rng,
        n_draws=args.n_draws,
        rep_per_draw=int(args.rep_per_draw),
        rep_per_draw_max=int(args.rep_per_draw_max),
        alpha=float(args.alpha),
        minima=bool(minima),
    )

    ext_word = "min" if fc["minima"] else "max"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # fine forecast plot
    _plot_fine(
        fc,
        os.path.join(out_dir, f"forecast_fine_{stamp}.png"),
        title=f"DGEV fine forecast (h={int(args.h)})",
        alpha=float(args.alpha),
    )

    # annual plot (always available)
    _plot_annual(
        fc["annual"],
        os.path.join(out_dir, f"forecast_annual_{ext_word}_{stamp}.png"),
        title=f"DGEV annual-{ext_word} forecast",
    )

    # seasonal plots only if computed
    seasons = fc.get("seasons", {})
    if seasons:
        _plot_seasonal_all(
            seasons,
            os.path.join(out_dir, f"forecast_seasonal_all_{ext_word}_{stamp}.png"),
            title=f"DGEV seasonal-{ext_word} forecast (all seasons sequential)",
        )
        for s in ["DJF", "MAM", "JJA", "SON"]:
            _plot_season_by_name(
                seasons,
                s,
                os.path.join(out_dir, f"forecast_seasonal_{s}_{ext_word}_{stamp}.png"),
                title=f"DGEV seasonal-{ext_word} forecast ({s})",
            )

    # save NPZ summary (dates_full is not saved; everything else is)
    if args.save_npz:
        npz_out = os.path.join(out_dir, f"forecast_summary_{stamp}.npz")
        payload: Dict[str, Any] = {
            "period": np.asarray([fc["period"]], int),
            "T": np.asarray([fc["T"]], int),
            "h": np.asarray([fc["h"]], int),
            "minima": np.asarray([fc["minima"]], bool),
            "x_train": np.asarray(fc["x_train"], float),
            "x_fore": np.asarray(fc["x_fore"], float),
            "y_obs": np.asarray(fc["y_obs"], float),
            "y_fore_samples": np.asarray(fc["y_fore_samples"], float),
            "mu_fore": np.asarray(fc["mu_fore"], float),
        }
        for k, v in fc["annual"].items():
            payload[f"annual_{k}"] = np.asarray(v)
        if seasons:
            for k, v in seasons.items():
                payload[f"seasons_{k}"] = np.asarray(v)

        np.savez_compressed(npz_out, **payload)

        meta_out = {
            "source_run_path": str(run_path),
            "npz_path": str(npz_path),
            "created": stamp,
            "alpha": float(args.alpha),
            "seed": int(args.seed),
            "n_draws": None if args.n_draws is None else int(args.n_draws),
            "rep_per_draw": int(args.rep_per_draw),
            "rep_per_draw_max": int(args.rep_per_draw_max),
            "minima_used": bool(minima),
            "notes": {
                "seasonal_grouping": "meteorological seasons only when period==12 and dates are available",
                "annual_grouping": "calendar-year if dates available else blocks of length=period",
                "plot_policy": "observed + forecast median + forecast band only (no fit ribbons)",
            },
        }
        with open(npz_out.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)

        print(f"[save] {npz_out}")

    print("[done] forecasts written.")
