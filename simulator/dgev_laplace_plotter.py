# %% simulator/dgev_laplace_plotter.py
from __future__ import annotations

import os
import sys
import re
from typing import Optional, Tuple, Dict, Any, List, Sequence, Union, Literal

import numpy as np
import matplotlib.pyplot as plt

# Make optimization package visible (mirrors dlm_plotter.py)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------
# I/O helpers: posterior loader
# ---------------------------------------------------------------------
try:
    from optimization.posterior_bundle import load_posterior, find_latest_run  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import optimization.posterior_bundle.load_posterior/find_latest_run.\n"
        "Make sure optimization/posterior_bundle.py is on PYTHONPATH."
    ) from e

# ---------------------------------------------------------------------
# Local helper utilities
# ---------------------------------------------------------------------
try:
    from simulator.utils import (  # type: ignore
        _ensure_dir,
        _maybe,
        _mad,
        _robust_sd_from_mad,
        _acf,
        _ess,
        _geweke_z,
        _normalize_center,
        _center_label,
        _parse_date_ymd,
        _monthly_time_axis_from_meta,
        _format_time_axis,
        find_latest_posterior_npz,
        _parse_kv_list,
    )
except Exception as e:
    raise ImportError(
        "Could not import simulator.utils.\n"
        "Make sure simulator/utils.py is on PYTHONPATH."
    ) from e

# =============================================================================
# DGEV-specific helpers: interval policy + minima detection/back-transform
# =============================================================================
Interval = Literal["eti", "hpd"]


def _normalize_interval(interval: str) -> Interval:
    s = str(interval).strip().lower()
    if s in ("eti", "equal", "equal-tailed", "equaltail", "equaltails", "quantile", "qt"):
        return "eti"
    if s in ("hpd", "hdr", "hd", "highest", "highest-density", "highestdensity"):
        return "hpd"
    raise ValueError("interval must be one of {'eti','hpd'} (aliases: equal-tailed/quantile, hdr).")


def _hpd_1d(x: np.ndarray, mass: float) -> Tuple[float, float]:
    """Shortest contiguous interval containing `mass` probability, approximated from samples."""
    v = np.asarray(x, float).ravel()
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(v[0]), float(v[0])

    mass = float(mass)
    mass = min(max(mass, 0.0), 1.0)
    if mass <= 0.0:
        m = float(np.median(v))
        return m, m
    if mass >= 1.0:
        return float(np.min(v)), float(np.max(v))

    xs = np.sort(v)
    m = int(np.floor(mass * n))
    m = max(1, min(m, n - 1))
    widths = xs[m:] - xs[: n - m]
    j = int(np.argmin(widths))
    return float(xs[j]), float(xs[j + m])


def _interval_1d(x: np.ndarray, *, mass: float, interval: Interval) -> Tuple[float, float]:
    v = np.asarray(x, float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")

    interval = _normalize_interval(interval)
    mass = float(mass)

    if interval == "eti":
        lo_q = (1.0 - mass) / 2.0
        hi_q = 1.0 - lo_q
        return float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q))
    return _hpd_1d(v, mass=mass)


def _interval_2d(arr_2d: np.ndarray, *, mass: float, interval: Interval) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(arr_2d, float)
    if A.ndim != 2:
        raise ValueError("arr_2d must be 2D (S,T).")
    _, T = A.shape
    interval = _normalize_interval(interval)
    mass = float(mass)

    if interval == "eti":
        lo_q = (1.0 - mass) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(A, lo_q, axis=0)
        hi = np.quantile(A, hi_q, axis=0)
        return lo, hi

    lo = np.empty(T, dtype=float)
    hi = np.empty(T, dtype=float)
    for t in range(T):
        lo[t], hi[t] = _hpd_1d(A[:, t], mass=mass)
    return lo, hi


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


def _apply_sign_backtransform(draws: Dict[str, Any], *, minima: bool) -> Dict[str, Any]:
    """
    If minima=True we assume location-related objects were stored on a negated scale.
    We flip them back for plotting. We do NOT flip GEV sigma/xi or Q.
    """
    if not minima:
        return draws

    out: Dict[str, Any] = dict(draws)

    loc_keys = {
        "y",
        "mu",
        "x",
        "alpha0",
        "beta0",
        "gamma0",
        "m0_alpha",
        "m0_beta",
        "m0_gamma",
        "true_mu_t",
        "true_alpha_t",
        "true_beta_t",
        "true_gamma_t",
    }
    for k in list(loc_keys):
        if k in out and out[k] is not None:
            out[k] = -np.asarray(out[k])

    # (optional) signed SDs for structural components
    for k in ("s_alpha", "s_beta", "s_gamma"):
        if k in out and out[k] is not None:
            out[k] = -np.asarray(out[k])

    return out


# =============================================================================
# Plotter
# =============================================================================
TimeLike = Union[int, str]
SeasonWhich = Literal["total", "dynamic", "baseline"]


class DGEVPlotter:
    """
    Plotter for posterior bundles from Laplace-based structural GEV samplers.

    Seasonal diagnostics are aligned with DLMPlotter:
      - dynamic seasonal state: x[..., idx_g0]
      - baseline seasonal dummies: gamma0 mapped to time by month-of-year
      - total seasonal contribution: dynamic + baseline

    Seasonal figures (all support which=total|dynamic|baseline):
      - figure_seasonal_patterns(...)
      - figure_seasonal_variance(...)
      - figure_seasonal_dummies_heatmap(...)
      - figure_seasonal_dummies_by_month(...)
    """

    def __init__(
        self,
        draws: Dict[str, np.ndarray],
        meta: Dict[str, Any],
        level: float = 0.90,
        *,
        minima: Optional[bool] = None,
        interval: str = "eti",
    ):
        # keep copies: don't mutate callers
        self.meta: Dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}

        minima_detected = _detect_minima_from_meta(self.meta)
        self.minima = bool(minima_detected) if minima is None else bool(minima)

        self.interval: Interval = _normalize_interval(interval)

        # backtransform (if needed) for plotting
        self.draws: Dict[str, Any] = _apply_sign_backtransform(dict(draws), minima=self.minima)

        disp = self.meta.get("display", {})
        if not isinstance(disp, dict):
            disp = {}
        disp.update(
            {
                "minima": self.minima,
                "backtransform_applied": self.minima,
                "interval": self.interval,
            }
        )
        self.meta["display"] = disp

        self.level = float(level)
        if not (0.0 < self.level < 1.0):
            raise ValueError("level must be in (0,1)")

        if "mu" not in self.draws:
            raise ValueError("draws must contain 'mu' of shape (S, T).")
        self.mu = np.asarray(self.draws["mu"], float)
        if self.mu.ndim != 2:
            raise ValueError("'mu' must be a 2D array (S, T).")

        self.S, self.T = self.mu.shape
        self.period = int(self.meta.get("period", 12))

        # time axis (EXACTLY like DLMPlotter)
        t, is_time = _monthly_time_axis_from_meta(self.meta, self.T)
        if t is None:
            self.t = np.arange(self.T)
            self.is_time = False
        else:
            self.t = t
            self.is_time = bool(is_time)

        # optional data & truth overlays (already backtransformed if minima=True)
        self.y = _maybe(self.draws, "y")
        self.true_mu = _maybe(self.draws, "true_mu_t")
        self.true_alpha = _maybe(self.draws, "true_alpha_t")
        self.true_beta = _maybe(self.draws, "true_beta_t")
        self.true_gamma = _maybe(self.draws, "true_gamma_t")

        # GEV sigma/xi (NOT sign-flipped)
        self.sigma: Optional[np.ndarray] = None
        if (
            "sigma" in self.draws
            and np.asarray(self.draws["sigma"]).ndim == 1
            and np.asarray(self.draws["sigma"]).shape[0] == self.S
        ):
            self.sigma = np.asarray(self.draws["sigma"], float)
        elif (
            "sigma2" in self.draws
            and np.asarray(self.draws["sigma2"]).ndim == 1
            and np.asarray(self.draws["sigma2"]).shape[0] == self.S
        ):
            self.sigma = np.sqrt(np.clip(np.asarray(self.draws["sigma2"], float), 0.0, None))

        self.xi: Optional[np.ndarray] = None
        if (
            "xi" in self.draws
            and np.asarray(self.draws["xi"]).ndim == 1
            and np.asarray(self.draws["xi"]).shape[0] == self.S
        ):
            self.xi = np.asarray(self.draws["xi"], float)

        # baselines (already backtransformed if minima=True)
        self.alpha0 = (
            np.asarray(self.draws["alpha0"], float) if "alpha0" in self.draws else _maybe(self.draws, "m0_alpha")
        )
        self.beta0 = (
            np.asarray(self.draws["beta0"], float) if "beta0" in self.draws else _maybe(self.draws, "m0_beta")
        )
        self.gamma0 = (
            np.asarray(self.draws["gamma0"], float) if "gamma0" in self.draws else _maybe(self.draws, "m0_gamma")
        )

        # scalar & vector params (excluding time series shaped (S,T))
        self.scalar_params: Dict[str, np.ndarray] = {}
        self.vector_params: Dict[str, np.ndarray] = {}
        self._collect_params()

        # signed SDs (if present)
        self.s_alpha = self.scalar_params.get("s_alpha", None)
        self.s_beta = self.scalar_params.get("s_beta", None)
        self.s_gamma = self.scalar_params.get("s_gamma", None)

        # process variances Q (+ names)
        self.Q: Optional[np.ndarray] = None
        self.Q_names: List[str] = []
        self._init_Q()

        # state draws x
        self.has_x = ("x" in self.draws) and (np.asarray(self.draws["x"]).ndim == 3)
        self.idx_alpha: Optional[int] = None
        self.idx_beta: Optional[int] = None
        self.idx_g0: Optional[int] = None
        self._init_state_indices()

        # default band label (include interval type; can be overridden by user)
        self.band_label_default = rf"{int(round(self.level * 100))}% {self.interval.upper()}"

        if self.minima:
            print("[info] minima=True detected → back-transforming (negated) location-related draws for plotting.")
        print(f"[info] credible interval type: {self.interval.upper()}")

    # ----------------------------- init helpers ----------------------------- #
    def _collect_params(self) -> None:
        skip = {"y", "mu", "x", "true_mu_t", "true_alpha_t", "true_beta_t", "true_gamma_t"}
        for k, v in self.draws.items():
            if k in skip:
                continue
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.shape[0] == self.S:
                self.scalar_params[k] = arr.astype(float)
            elif arr.ndim == 2 and arr.shape[0] == self.S and arr.shape[1] != self.T:
                self.vector_params[k] = arr.astype(float)

    def _init_Q(self) -> None:
        draws, meta = self.draws, self.meta

        # 1) direct Q matrix
        if "Q" in draws:
            Qmat = np.asarray(draws["Q"], float)
            if Qmat.ndim == 2 and Qmat.shape[0] == self.S:
                self.Q = Qmat
                layout = meta.get("layout")
                if isinstance(layout, (list, tuple)) and len(layout) == Qmat.shape[1]:
                    self.Q_names = [rf"$Q_{{{nm}}}$" for nm in layout]
                else:
                    self.Q_names = [rf"$Q_{{{j}}}$" for j in range(Qmat.shape[1])]
                return

        # 2) separate scalar Q_alpha/Q_beta/Q_gamma
        cols, names = [], []
        for nm, lab in (("Q_alpha", r"$Q_\alpha$"), ("Q_beta", r"$Q_\beta$"), ("Q_gamma", r"$Q_\gamma$")):
            if nm in self.scalar_params:
                cols.append(self.scalar_params[nm].reshape(self.S, 1))
                names.append(lab)
        if cols:
            self.Q = np.concatenate(cols, axis=1)
            self.Q_names = names
            return

        # 3) reconstruct from signed SDs if present (Q = s^2)
        cols, names = [], []
        if self.s_alpha is not None:
            cols.append((np.asarray(self.s_alpha) ** 2).reshape(self.S, 1))
            names.append(r"$Q_\alpha$")
        if self.s_beta is not None:
            cols.append((np.asarray(self.s_beta) ** 2).reshape(self.S, 1))
            names.append(r"$Q_\beta$")
        if self.s_gamma is not None:
            cols.append((np.asarray(self.s_gamma) ** 2).reshape(self.S, 1))
            names.append(r"$Q_\gamma$")
        if cols:
            self.Q = np.concatenate(cols, axis=1)
            self.Q_names = names

    def _init_state_indices(self) -> None:
        if not self.has_x:
            return
        x = np.asarray(self.draws["x"])
        dim = int(x.shape[2])

        layout = self.meta.get("layout")
        layout_list = list(layout) if isinstance(layout, (list, tuple)) else None

        if layout_list and len(layout_list) == dim:
            if "alpha" in layout_list:
                self.idx_alpha = layout_list.index("alpha")
            if "beta" in layout_list:
                self.idx_beta = layout_list.index("beta")

            # first seasonal component: "gamma" if present, else first name starting with "g"
            if "gamma" in layout_list:
                self.idx_g0 = layout_list.index("gamma")
            else:
                g_indices = [i for i, nm in enumerate(layout_list) if str(nm).startswith("g")]
                if g_indices:
                    self.idx_g0 = g_indices[0]
            return

        # fallback assumption: (alpha, beta, gamma0, ...)
        self.idx_alpha = 0 if dim >= 1 else None
        self.idx_beta = 1 if dim >= 2 else None
        self.idx_g0 = 2 if dim >= 3 else None

    # ----------------------------- core helpers ----------------------------- #
    def _summarize_ribbon(
        self,
        arr_2d: np.ndarray,
        *,
        center: str = "median",
        level: Optional[float] = None,
        interval: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = np.asarray(arr_2d, float)
        lev = float(self.level if level is None else level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")
        c = _normalize_center(center)
        itv = self.interval if interval is None else _normalize_interval(interval)

        lo, hi = _interval_2d(A, mass=lev, interval=itv)
        ctr = np.mean(A, axis=0) if c == "mean" else np.quantile(A, 0.5, axis=0)
        return ctr, lo, hi

    def _component_draws(self, which: str) -> Optional[np.ndarray]:
        if not self.has_x:
            return None
        x = np.asarray(self.draws["x"])
        if which == "alpha" and self.idx_alpha is not None:
            return x[:, :, self.idx_alpha]
        if which == "beta" and self.idx_beta is not None:
            return x[:, :, self.idx_beta]
        if which in ("gamma", "seasonal") and self.idx_g0 is not None:
            return x[:, :, self.idx_g0]
        return None

    # ----------------------------- seasonal indexing helpers (same as DLMPlotter) ----------------------------- #
    def _year_month_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return arrays (year[t], month[t]) with month in 1..period."""
        if self.is_time and hasattr(self.t, "__len__"):
            years = np.array([dt.year for dt in self.t], dtype=int)
            months = np.array([dt.month for dt in self.t], dtype=int)
            return years, months

        start_date = self.meta.get("start_date", None) or self.meta.get("start", None) or self.meta.get("t0", None)
        if start_date is None:
            raise ValueError(
                "Cannot build year/month indexing: meta['start_date'] (or start/t0) is missing. "
                "Add start_date='YYYY-MM-DD' to meta or pass --start-date."
            )

        dt0 = _parse_date_ymd(str(start_date))
        period = int(self.period)

        tt = np.arange(self.T, dtype=int)
        months0 = dt0.year * period + (dt0.month - 1)
        months = months0 + tt
        years = months // period
        mon = (months % period) + 1
        return years.astype(int), mon.astype(int)

    def _year_slices(self) -> Tuple[List[int], Dict[int, np.ndarray]]:
        years, _months = self._year_month_arrays()
        uniq = np.unique(years)
        idx_by_year: Dict[int, np.ndarray] = {int(y): np.where(years == y)[0] for y in uniq}
        return [int(y) for y in uniq], idx_by_year

    def _full_years(self) -> Tuple[List[int], Dict[int, np.ndarray], np.ndarray]:
        """Full year means exactly `period` months present in that calendar year."""
        uniq_years, idx_by_year = self._year_slices()
        _, months_arr = self._year_month_arrays()
        p = int(self.period)
        full = [int(y) for y in uniq_years if idx_by_year[int(y)].size == p]
        full = sorted(full)
        return full, idx_by_year, months_arr

    # ----------------------------- seasonal composition (DLMPlotter-style) ----------------------------- #
    def _gamma0_full(self) -> Optional[np.ndarray]:
        """
        Ensure baseline seasonal dummy vector has length period (includes implied last month).
        Accepts gamma0 shape:
          - (S, period-1): append last = -sum
          - (S, period):   already full
          - (period-1,) or (period,): treated as fixed across draws (broadcast to (S, ...))
        """
        if self.gamma0 is None:
            return None

        p = int(self.period)
        G0 = np.asarray(self.gamma0, float)

        # allow fixed gamma0 vector (no draw dimension)
        if G0.ndim == 1 and G0.size in (p - 1, p):
            G0 = np.broadcast_to(G0.reshape(1, -1), (self.S, G0.size)).copy()

        if G0.ndim != 2 or G0.shape[0] != self.S:
            return None

        if G0.shape[1] == p:
            return G0

        if G0.shape[1] == p - 1:
            last = -np.sum(G0, axis=1, keepdims=True)
            return np.concatenate([G0, last], axis=1)

        return None

    def _baseline_season_series(self) -> Optional[np.ndarray]:
        """
        Baseline seasonal dummy contribution per draw and time:
          base_t = gamma0_full[m_t], with m_t in {0..period-1}
        Returns (S, T) or None if gamma0 missing.
        """
        G0 = self._gamma0_full()
        if G0 is None:
            return None
        p = int(self.period)
        m = (np.arange(self.T, dtype=int) % p)
        return G0[:, m]  # (S, T)

    def seasonal_dynamic_series(self) -> Optional[np.ndarray]:
        """Dynamic seasonal state series (S, T) from x[..., idx_g0]."""
        return self._component_draws("gamma")

    def seasonal_total_series(self) -> Optional[np.ndarray]:
        """
        Total seasonal contribution (S, T):
          season_t = gamma_dyn_t + baseline_dummy_t
        If gamma0 missing, returns dynamic series.
        """
        dyn = self.seasonal_dynamic_series()
        if dyn is None:
            return None
        base = self._baseline_season_series()
        if base is None:
            return dyn
        return dyn + base

    def seasonal_matrix_year_month(
        self,
        *,
        which: str = "total",  # total|dynamic|baseline
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """
        Returns:
          years_full: list of calendar years with full period months
          months:     array [1..period]
          M:          (S, Ny, period) with M[:, j, m-1] as seasonal contribution for that year & month
        """
        which = str(which).lower().strip()
        if which not in ("total", "dynamic", "baseline"):
            raise ValueError("which must be one of: 'total', 'dynamic', 'baseline'")

        if which == "dynamic":
            season = self.seasonal_dynamic_series()
        elif which == "baseline":
            season = self._baseline_season_series()
        else:
            season = self.seasonal_total_series()

        if season is None:
            raise RuntimeError("No seasonal series available (need draws['x'] with seasonal index, and/or gamma0).")

        full_years, idx_by_year, months_arr = self._full_years()
        p = int(self.period)
        if not full_years:
            raise RuntimeError("No full calendar years found (need exactly 'period' months per year).")

        Ny = len(full_years)
        M = np.empty((self.S, Ny, p), float)

        for j, y in enumerate(full_years):
            idx = idx_by_year[int(y)]
            order = np.argsort(months_arr[idx])
            idx = idx[order]  # month order 1..p
            M[:, j, :] = season[:, idx]

        months = np.arange(1, p + 1, dtype=int)
        return full_years, months, M

    # ----------------------------- time indexing (same as DLMPlotter) ----------------------------- #
    def _time_to_index(self, t: TimeLike) -> Tuple[int, str]:
        T = self.T
        if isinstance(t, int):
            i = int(t)
            if not (0 <= i <= T - 1):
                raise ValueError(f"Index {i} out of range [0, {T - 1}].")
            return i, f"t={i}"

        s = str(t).strip().lower()
        if s in ("start", "begin", "beginning"):
            return 0, "start"
        if s in ("mid", "middle", "center", "centre"):
            return T // 2, "mid"
        if s in ("end", "last"):
            return T - 1, "end"

        start_date = self.meta.get("start_date", None) or self.meta.get("start", None) or self.meta.get("t0", None)
        period = int(self.meta.get("period", 12) or 12)
        if start_date is None:
            raise ValueError(
                f"Got date-like time {t!r} but meta['start_date'] (or start/t0) is missing. "
                "Provide integer indices or set start_date in meta."
            )

        dt0 = _parse_date_ymd(str(start_date))
        dt = _parse_date_ymd(str(t))

        months0 = dt0.year * period + (dt0.month - 1)
        months = dt.year * period + (dt.month - 1)
        i = months - months0
        if not (0 <= i <= T - 1):
            raise ValueError(f"Date {t!r} maps to index {i}, out of range [0, {T - 1}].")
        return int(i), str(t)

    # ----------------------------- printing: level/slope ----------------------------- #
    def print_level_slope_at(
        self,
        times: Sequence[TimeLike] = ("start", "mid", "end"),
        *,
        slope_scale: float = 1.0,
        level: Optional[float] = None,
        digits: int = 4,
        use_mean: bool = False,
        interval: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Print posterior summaries for level alpha_t and slope beta_t at selected times."""
        if not self.has_x or self.idx_alpha is None or self.idx_beta is None:
            raise RuntimeError("Cannot print level/slope: draws['x'] missing or layout lacks alpha/beta.")

        X = np.asarray(self.draws["x"])
        lev = float(self.level if level is None else level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")
        itv = self.interval if interval is None else _normalize_interval(interval)

        rows: List[Dict[str, Any]] = []
        for t in times:
            idx, label = self._time_to_index(t)
            a = X[:, idx, self.idx_alpha]
            b = X[:, idx, self.idx_beta] * float(slope_scale)

            a_ctr = float(np.mean(a)) if use_mean else float(np.quantile(a, 0.5))
            b_ctr = float(np.mean(b)) if use_mean else float(np.quantile(b, 0.5))

            a_lo, a_hi = _interval_1d(a, mass=lev, interval=itv)
            b_lo, b_hi = _interval_1d(b, mass=lev, interval=itv)

            rows.append(
                {
                    "time": label,
                    "idx": int(idx),
                    "alpha_ctr": a_ctr,
                    "alpha_lo": float(a_lo),
                    "alpha_hi": float(a_hi),
                    "beta_ctr": b_ctr,
                    "beta_lo": float(b_lo),
                    "beta_hi": float(b_hi),
                }
            )

        fmt = f"{{:.{int(digits)}f}}".format
        ctr_name = "mean" if use_mean else "median"
        print(
            f"\n[level/slope summaries] S={X.shape[0]}, T={self.T}, "
            f"CI={int(round(100*lev))}% {itv.upper()} (center={ctr_name})"
        )
        print(" time        idx |   alpha (ctr [lo, hi])          |   beta (ctr [lo, hi])")
        print("-" * 86)
        for r in rows:
            print(
                f" {r['time']:<10} {r['idx']:>4d} | "
                f"{fmt(r['alpha_ctr'])} [{fmt(r['alpha_lo'])}, {fmt(r['alpha_hi'])}] | "
                f"{fmt(r['beta_ctr'])} [{fmt(r['beta_lo'])}, {fmt(r['beta_hi'])}]"
            )
        print()
        return rows

    # ----------------------------- static param printing ----------------------------- #
    @staticmethod
    def _clean_name(s: str) -> str:
        x = str(s)
        x = x.replace("$", "").replace("\\", "")
        x = x.replace("{", "").replace("}", "")
        return x

    def _summarize_1d(
        self,
        x: np.ndarray,
        *,
        level: float,
        center: str,
        interval: Interval,
        drop_nonfinite: bool,
        max_lag: int,
        diagnostics: bool,
    ) -> Dict[str, Any]:
        v = np.asarray(x, float).ravel()
        if drop_nonfinite:
            v = v[np.isfinite(v)]
        out: Dict[str, Any] = {"n": int(v.size)}
        if v.size == 0:
            out.update({"ctr": np.nan, "lo": np.nan, "hi": np.nan, "ess": np.nan, "z": np.nan})
            return out

        c = _normalize_center(center)
        ctr = float(np.mean(v)) if c == "mean" else float(np.quantile(v, 0.5))
        lo, hi = _interval_1d(v, mass=float(level), interval=interval)

        out.update({"ctr": ctr, "lo": float(lo), "hi": float(hi)})

        if diagnostics:
            out["ess"] = float(_ess(v, max_lag=max_lag))
            out["z"] = float(_geweke_z(v))
        else:
            out["ess"] = np.nan
            out["z"] = np.nan
        return out

    def print_static_params(
        self,
        *,
        level: Optional[float] = None,
        center: str = "median",
        digits: int = 4,
        include_vectors: bool = True,
        max_vector_cols: Optional[int] = None,
        include_derived_Q: bool = True,
        include_log_process: bool = True,
        log_eps: float = 1e-20,
        include_diagnostics: bool = True,
        max_lag: int = 200,
        drop_nonfinite: bool = True,
        sort_names: bool = True,
        interval: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        lev = float(self.level if level is None else level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")
        c = _normalize_center(center)
        ctr_name = "mean" if c == "mean" else "median"
        ci_pct = int(round(100 * lev))
        fmt = f"{{:.{int(digits)}f}}".format
        itv = self.interval if interval is None else _normalize_interval(interval)

        rows: List[Dict[str, Any]] = []
        used: set[str] = set()

        def add_scalar(name: str, arr: Optional[np.ndarray]) -> None:
            if arr is None:
                return
            nm = str(name)
            if nm in used:
                return
            used.add(nm)
            s = self._summarize_1d(
                arr,
                level=lev,
                center=center,
                interval=itv,
                drop_nonfinite=drop_nonfinite,
                max_lag=max_lag,
                diagnostics=include_diagnostics,
            )
            s["name"] = nm
            s["kind"] = "scalar"
            rows.append(s)

        def add_vector(name: str, mat: Optional[np.ndarray], col_labels: Optional[List[str]] = None) -> None:
            if mat is None:
                return
            M = np.asarray(mat, float)
            if M.ndim != 2 or M.shape[0] != self.S:
                return
            K = int(M.shape[1])
            lim = K if (max_vector_cols is None) else min(K, int(max_vector_cols))

            for j in range(lim):
                if col_labels is not None and j < len(col_labels):
                    nm = f"{name}[{j}] {col_labels[j]}"
                else:
                    nm = f"{name}[{j}]"
                if nm in used:
                    continue
                used.add(nm)
                s = self._summarize_1d(
                    M[:, j],
                    level=lev,
                    center=center,
                    interval=itv,
                    drop_nonfinite=drop_nonfinite,
                    max_lag=max_lag,
                    diagnostics=include_diagnostics,
                )
                s["name"] = nm
                s["kind"] = "vector"
                rows.append(s)

            if lim < K:
                rows.append(
                    {
                        "name": f"{name}[...]",
                        "kind": "vector",
                        "n": int(self.S),
                        "ctr": np.nan,
                        "lo": np.nan,
                        "hi": np.nan,
                        "ess": np.nan,
                        "z": np.nan,
                        "note": f"omitted {K - lim} columns (set --static-max-cols to print all)",
                    }
                )

        # canonical: sigma/xi
        if self.sigma is not None:
            add_scalar("sigma", self.sigma)
        if self.xi is not None:
            add_scalar("xi", self.xi)

        # canonical baselines
        if self.alpha0 is not None:
            A0 = np.asarray(self.alpha0)
            add_scalar("alpha0", A0) if A0.ndim == 1 else add_vector("alpha0", A0)
        if self.beta0 is not None:
            B0 = np.asarray(self.beta0)
            add_scalar("beta0", B0) if B0.ndim == 1 else add_vector("beta0", B0)
        if self.gamma0 is not None:
            G0 = np.asarray(self.gamma0)
            add_scalar("gamma0", G0) if G0.ndim == 1 else add_vector("gamma0", G0)

        if include_derived_Q and (self.Q is not None) and np.size(self.Q):
            col = [self._clean_name(x) for x in (self.Q_names or [])]
            add_vector("Q", np.asarray(self.Q), col_labels=col if col else None)

        if include_log_process:
            eps = float(log_eps)
            if self.s_alpha is not None:
                add_scalar("log10|s_alpha|", np.log10(np.clip(np.abs(self.s_alpha), eps, None)))
            if self.s_beta is not None:
                add_scalar("log10|s_beta|", np.log10(np.clip(np.abs(self.s_beta), eps, None)))
            if self.s_gamma is not None:
                add_scalar("log10|s_gamma|", np.log10(np.clip(np.abs(self.s_gamma), eps, None)))

            if (self.Q is not None) and np.size(self.Q):
                col = [self._clean_name(x) for x in (self.Q_names or [])]
                logQ = np.log10(np.clip(np.asarray(self.Q, float), eps, None))
                add_vector("log10Q", logQ, col_labels=col if col else None)

        skip_keys = {
            "mu",
            "x",
            "y",
            "true_mu_t",
            "true_alpha_t",
            "true_beta_t",
            "true_gamma_t",
            "sigma",
            "sigma2",
            "xi",
            "alpha0",
            "beta0",
            "gamma0",
            "m0_alpha",
            "m0_beta",
            "m0_gamma",
            "Q",
        }
        for k, arr in self.scalar_params.items():
            if k in skip_keys:
                continue
            add_scalar(k, arr)

        if include_vectors:
            for k, mat in self.vector_params.items():
                if k in skip_keys:
                    continue
                add_vector(k, mat)

        if sort_names:
            main = [r for r in rows if not str(r.get("name", "")).endswith("[...]")]
            tail = [r for r in rows if str(r.get("name", "")).endswith("[...]")]
            main = sorted(main, key=lambda r: str(r.get("name", "")))
            rows = main + tail

        print(f"\n[static parameter summaries] S={self.S}, CI={ci_pct}% {itv.upper()} (center={ctr_name})")
        if include_diagnostics:
            print(" name                           n |       ctr [        lo,        hi] |    ESS   z")
            print("-" * 94)
        else:
            print(" name                           n |       ctr [        lo,        hi]")
            print("-" * 74)

        for r in rows:
            nm = self._clean_name(r.get("name", ""))
            n = int(r.get("n", 0))
            ctr = r.get("ctr", np.nan)
            lo = r.get("lo", np.nan)
            hi = r.get("hi", np.nan)

            if include_diagnostics:
                ess = r.get("ess", np.nan)
                z = r.get("z", np.nan)
                print(
                    f" {nm:<30} {n:>5d} | "
                    f"{fmt(ctr):>10} [{fmt(lo):>10}, {fmt(hi):>10}] | "
                    f"{(f'{ess:.0f}' if np.isfinite(ess) else 'nan'):>6} "
                    f"{(f'{z:.2f}' if np.isfinite(z) else 'nan'):>5}"
                )
            else:
                print(f" {nm:<30} {n:>5d} | " f"{fmt(ctr):>10} [{fmt(lo):>10}, {fmt(hi):>10}]")

            if "note" in r:
                print(f"   -> {r['note']}")

        print()
        return rows

    # ----------------------------- trace/hist/acf panel ----------------------------- #
    def _trace_hist_acf_panel(
        self,
        series: np.ndarray,
        name: str,
        *,
        title_trace: Optional[str] = None,
        title_hist: Optional[str] = None,
        title_acf: Optional[str] = None,
        xlabel_trace: str = "iteration",
        xlabel_acf: str = "lag",
        max_lag: int = 200,
        save_dir: Optional[str] = None,
        fname: Optional[str] = None,
        show: bool = True,
        trace_ylim: Optional[Tuple[float, float]] = None,
        hist_xlim: Optional[Tuple[float, float]] = None,
        plot_policy: str = "none",  # "none" | "clip" | "drop"
        diag_policy: str = "clean",  # "raw" | "clipped" | "clean"
        drop_nonfinite: bool = True,
        clip_q: Optional[Tuple[float, float]] = None,
        clip_nmad: Optional[float] = None,
        max_abs: Optional[float] = None,
        hist_bins: int = 40,
        auto_zoom_if_clipped: bool = True,
    ) -> None:
        s_raw = np.asarray(series, float).ravel()
        finite_mask = np.isfinite(s_raw)
        n_nonfinite = int(np.sum(~finite_mask))
        s_finite = s_raw[finite_mask] if drop_nonfinite else s_raw.copy()

        if s_finite.size == 0:
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            for ax in axs:
                ax.axis("off")
            fig.suptitle(rf"{name}: no finite samples")
            plt.tight_layout()
            if save_dir and fname:
                _ensure_dir(save_dir)
                out = os.path.join(save_dir, fname)
                fig.savefig(out, dpi=200, bbox_inches="tight")
                print(f"[save] {out}")
            if show:
                plt.show()
            else:
                plt.close(fig)
            return

        plot_policy = str(plot_policy).lower()
        diag_policy = str(diag_policy).lower()
        if plot_policy not in ("none", "clip", "drop"):
            raise ValueError("plot_policy must be one of: 'none', 'clip', 'drop'")
        if diag_policy not in ("raw", "clipped", "clean"):
            raise ValueError("diag_policy must be one of: 'raw', 'clipped', 'clean'")

        lo_bound, hi_bound = -np.inf, np.inf

        if max_abs is not None:
            a = float(abs(max_abs))
            lo_bound = max(lo_bound, -a)
            hi_bound = min(hi_bound, +a)

        if clip_q is not None:
            ql, qh = float(clip_q[0]), float(clip_q[1])
            ql = max(0.0, min(1.0, ql))
            qh = max(0.0, min(1.0, qh))
            if qh <= ql:
                raise ValueError(f"clip_q must satisfy q_high > q_low, got {clip_q}")
            qlo = float(np.quantile(s_finite, ql))
            qhi = float(np.quantile(s_finite, qh))
            lo_bound = max(lo_bound, qlo)
            hi_bound = min(hi_bound, qhi)

        if clip_nmad is not None:
            k = float(clip_nmad)
            med = float(np.median(s_finite))
            mad = _mad(s_finite)
            rsd = _robust_sd_from_mad(mad)
            if rsd > 0:
                lo_bound = max(lo_bound, med - k * rsd)
                hi_bound = min(hi_bound, med + k * rsd)

        use_bounds = np.isfinite(lo_bound) or np.isfinite(hi_bound)
        if not use_bounds:
            lo_bound, hi_bound = -np.inf, np.inf

        out_mask_finite = ((s_finite < lo_bound) | (s_finite > hi_bound)) if use_bounds else np.zeros_like(
            s_finite, bool
        )
        n_out = int(np.sum(out_mask_finite))

        # plotting series (keep length = original draws; NaN out nonfinite/outliers if requested)
        s_plot = s_raw.copy()
        s_plot[~finite_mask] = np.nan

        if use_bounds and plot_policy == "clip":
            s_plot = np.clip(s_plot, lo_bound, hi_bound)
        elif use_bounds and plot_policy == "drop":
            out_mask_raw = np.zeros_like(s_raw, dtype=bool)
            out_mask_raw[finite_mask] = out_mask_finite
            s_plot[out_mask_raw] = np.nan

        # histogram series
        s_hist = s_finite.copy()
        if use_bounds and plot_policy == "clip":
            s_hist = np.clip(s_hist, lo_bound, hi_bound)
        elif use_bounds and plot_policy == "drop":
            s_hist = s_hist[~out_mask_finite]

        # diagnostics series
        if diag_policy == "raw":
            s_diag = s_finite.copy()
        elif diag_policy == "clipped":
            s_diag = np.clip(s_finite, lo_bound, hi_bound) if use_bounds else s_finite.copy()
        else:
            s_diag = s_finite[~out_mask_finite] if use_bounds else s_finite.copy()

        ac = _acf(s_diag, max_lag=max_lag)
        ess = _ess(s_diag, max_lag=max_lag)
        gz = _geweke_z(s_diag)

        extra = []
        if drop_nonfinite and n_nonfinite > 0:
            extra.append(f"nonfinite={n_nonfinite}")
        if use_bounds and n_out > 0:
            extra.append(f"outliers={n_out}")
        extra_txt = f" ({', '.join(extra)})" if extra else ""

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        axs[0].plot(s_plot, lw=1)
        axs[0].set_title(title_trace or rf"trace: {name}{extra_txt}")
        axs[0].set_xlabel(xlabel_trace)
        if trace_ylim is not None:
            axs[0].set_ylim(*trace_ylim)
        elif (
            auto_zoom_if_clipped
            and use_bounds
            and plot_policy in ("clip", "drop")
            and np.isfinite(lo_bound)
            and np.isfinite(hi_bound)
        ):
            axs[0].set_ylim(lo_bound, hi_bound)

        hist_range = None
        if hist_xlim is not None:
            hist_range = (float(hist_xlim[0]), float(hist_xlim[1]))
        elif (
            auto_zoom_if_clipped
            and use_bounds
            and plot_policy in ("clip", "drop")
            and np.isfinite(lo_bound)
            and np.isfinite(hi_bound)
        ):
            hist_range = (float(lo_bound), float(hi_bound))

        axs[1].hist(s_hist, bins=int(hist_bins), density=True, range=hist_range)
        axs[1].set_title(title_hist or rf"hist: {name}{extra_txt}")
        if hist_xlim is not None:
            axs[1].set_xlim(*hist_xlim)

        axs[2].bar(np.arange(ac.size), ac, width=0.9)
        axs[2].set_xlim(-0.5, ac.size - 0.5)
        axs[2].set_title(title_acf or rf"ACF: {name} (ESS$\approx${ess:.0f}, z$\approx${gz:.2f})")
        axs[2].set_xlabel(xlabel_acf)

        plt.tight_layout()
        if save_dir and fname:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ----------------------------- figures ----------------------------- #
    def figure_overview(
        self,
        *,
        save_dir: Optional[str] = None,
        fname: str = "overview.png",
        show: bool = True,
        color: str = "C0",
        band_alpha: float = 0.25,
        band_label: Optional[str] = None,
        center: str = "median",
        interval: Optional[str] = None,
        title_mu: str = r"Posterior $\mu_t$",
        title_sigma_trace: str = r"trace: $\sigma$",
        title_sigma_hist: Optional[str] = None,
        title_Q: str = r"Process variances (log$_{10}$ scale)",
        title_baselines: str = r"Baselines",
        title_rmse: str = r"running RMSE($\mu$) vs truth",
        xlabel_time: str = r"$t$",
        ylabel_mu: str = r"$\mu_t$",
        ylims_mu: Optional[Tuple[float, float]] = None,
        yscale_mu: Optional[str] = None,
        show_legend_mu: bool = True,
        include_xi_in_baselines: bool = True,
    ) -> None:
        band_label = self.band_label_default if band_label is None else band_label
        c_lab = _center_label(center)

        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        axs = axs.ravel()

        t = self.t
        ctr, lo, hi = self._summarize_ribbon(self.mu, center=center, level=self.level, interval=interval)

        # mu
        axs[0].plot(t, ctr, lw=1.6, color=color, label=(c_lab if show_legend_mu else "_nolegend_"))
        axs[0].fill_between(
            t,
            lo,
            hi,
            alpha=band_alpha,
            color=color,
            label=(band_label if show_legend_mu else "_nolegend_"),
        )
        if self.y is not None and len(self.y) == self.T:
            axs[0].plot(t, self.y, lw=1.0, alpha=0.6, label=(r"$y_t$" if show_legend_mu else "_nolegend_"))
        if self.true_mu is not None and len(self.true_mu) == self.T:
            axs[0].plot(
                t,
                self.true_mu,
                lw=1.2,
                ls="--",
                color="k",
                alpha=0.8,
                label=("truth" if show_legend_mu else "_nolegend_"),
            )

        axs[0].set_title(title_mu)
        axs[0].set_xlabel(xlabel_time)
        axs[0].set_ylabel(ylabel_mu)
        if yscale_mu is not None:
            axs[0].set_yscale(yscale_mu)
        if ylims_mu is not None:
            axs[0].set_ylim(*ylims_mu)
        if show_legend_mu:
            axs[0].legend(loc="upper left")
        if self.is_time:
            _format_time_axis(axs[0])

        # sigma trace/hist
        if self.sigma is not None:
            axs[1].plot(self.sigma, lw=1)
            axs[1].set_title(title_sigma_trace)
            axs[1].set_xlabel("kept draw")

            axs[2].hist(self.sigma, bins=40, density=True)
            if title_sigma_hist is None:
                es = _ess(self.sigma)
                gz = _geweke_z(self.sigma)
                axs[2].set_title(rf"posterior: $\sigma$  (ESS$\approx${es:.0f}, z$\approx${gz:.2f})")
            else:
                axs[2].set_title(title_sigma_hist)
        else:
            axs[1].axis("off")
            axs[2].axis("off")

        # Q hist(s) on log10 scale
        if self.Q is not None and np.size(self.Q):
            Q = np.asarray(self.Q, float)
            logQ = np.log10(np.clip(Q, 1e-20, None))
            ax = axs[3]
            labels = self.Q_names if self.Q_names else [rf"$Q_{{{j}}}$" for j in range(logQ.shape[1])]
            for j in range(logQ.shape[1]):
                ax.hist(logQ[:, j], bins=40, density=True, alpha=0.55, label=rf"$\log_{{10}}$ {labels[j]}")
            ax.set_title(title_Q)
            ax.legend(loc="best")
        else:
            axs[3].axis("off")

        # baseline hist(s) (+ optional xi)
        any_baseline = (self.alpha0 is not None) or (self.beta0 is not None) or (self.gamma0 is not None) or (
            include_xi_in_baselines and (self.xi is not None)
        )
        if any_baseline:
            ax = axs[4]
            if self.alpha0 is not None:
                ax.hist(np.asarray(self.alpha0).ravel(), bins=40, density=True, alpha=0.6, label=r"$\alpha_0$")
            if self.beta0 is not None:
                ax.hist(np.asarray(self.beta0).ravel(), bins=40, density=True, alpha=0.6, label=r"$\beta_0$")
            if self.gamma0 is not None:
                G0 = np.asarray(self.gamma0)
                if G0.ndim == 1:
                    ax.hist(G0, bins=40, density=True, alpha=0.6, label=r"$\gamma_0$")
                elif G0.ndim == 2 and G0.shape[1] > 0:
                    ax.hist(G0[:, 0], bins=40, density=True, alpha=0.6, label=r"$\gamma_0[0]$")
            if include_xi_in_baselines and (self.xi is not None):
                ax.hist(np.asarray(self.xi).ravel(), bins=40, density=True, alpha=0.6, label=r"$\xi$")
            ax.set_title(title_baselines)
            ax.legend(loc="best")
        else:
            axs[4].axis("off")

        # running RMSE (if truth exists)
        if self.true_mu is not None and len(self.true_mu) == self.T:
            err = np.mean((self.mu - self.true_mu.reshape(1, -1)) ** 2, axis=1) ** 0.5
            running = np.cumsum(err) / np.arange(1, err.size + 1)
            axs[5].plot(running, lw=1.2)
            axs[5].set_title(title_rmse)
            axs[5].set_xlabel("kept draw")
        else:
            axs[5].axis("off")

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def figure_trace_acf_core(
        self,
        *,
        save_dir: Optional[str] = None,
        show: bool = True,
        max_lag: int = 200,
        plot_other_scalars: bool = True,
        trace_ylim: Optional[Tuple[float, float]] = None,
        hist_xlim: Optional[Tuple[float, float]] = None,
        plot_policy: str = "none",
        diag_policy: str = "clean",
        drop_nonfinite: bool = True,
        clip_q: Optional[Tuple[float, float]] = None,
        clip_nmad: Optional[float] = None,
        max_abs: Optional[float] = None,
        hist_bins: int = 40,
        auto_zoom_if_clipped: bool = True,
    ) -> None:
        def _panel(series: np.ndarray, nm: str, out_name: str) -> None:
            self._trace_hist_acf_panel(
                series,
                nm,
                max_lag=max_lag,
                save_dir=save_dir,
                fname=out_name,
                show=show,
                trace_ylim=trace_ylim,
                hist_xlim=hist_xlim,
                plot_policy=plot_policy,
                diag_policy=diag_policy,
                drop_nonfinite=drop_nonfinite,
                clip_q=clip_q,
                clip_nmad=clip_nmad,
                max_abs=max_abs,
                hist_bins=hist_bins,
                auto_zoom_if_clipped=auto_zoom_if_clipped,
            )

        if self.sigma is not None:
            _panel(self.sigma, r"$\sigma$", "trace_hist_acf_sigma.png")
        if self.xi is not None:
            _panel(self.xi, r"$\xi$", "trace_hist_acf_xi.png")

        if self.s_alpha is not None:
            _panel(self.s_alpha, r"$s_\alpha$", "trace_hist_acf_s_alpha.png")
        if self.s_beta is not None:
            _panel(self.s_beta, r"$s_\beta$", "trace_hist_acf_s_beta.png")
        if self.s_gamma is not None:
            _panel(self.s_gamma, r"$s_\gamma$", "trace_hist_acf_s_gamma.png")

        if self.Q is not None and np.size(self.Q):
            Q = np.asarray(self.Q, float)
            labels = self.Q_names if self.Q_names else [rf"$Q_{{{j}}}$" for j in range(Q.shape[1])]
            for j in range(Q.shape[1]):
                series = np.log10(np.clip(Q[:, j], 1e-20, None))
                _panel(series, rf"$\log_{{10}}({labels[j]})$", f"trace_hist_acf_log10Q_{j}.png")

        if plot_other_scalars:
            skip = {
                "sigma",
                "sigma2",
                "xi",
                "s_alpha",
                "s_beta",
                "s_gamma",
                "Q_alpha",
                "Q_beta",
                "Q_gamma",
                "lambda2",
                "tau_alpha",
                "tau_beta",
                "tau_gamma",
            }
            for k, arr in sorted(self.scalar_params.items()):
                if k in skip or str(k).startswith("tau_"):
                    continue
                _panel(arr, str(k), f"trace_hist_acf_{k}.png")

    def figure_states_separate(
        self,
        *,
        save_dir: Optional[str] = None,
        fname_prefix: str = "state",
        show: bool = True,
        color: str = "C0",
        band_alpha: float = 0.25,
        band_label: Optional[str] = None,
        center: str = "median",
        interval: Optional[str] = None,
        xlabel_time: str = r"$t$",
        title_level: str = "",
        ylabel_level: str = r"$\alpha_t$",
        title_slope: str = "",
        ylabel_slope: str = r"$\beta_t$",
        title_seasonality: str = "",  # keep empty (no title)
        ylabel_seasonality: str = r"$\gamma_t$",
        slope_scale: float = 1.0,
        ylims: Optional[Dict[str, Tuple[float, float]]] = None,
        yscales: Optional[Dict[str, str]] = None,
        zero_line_slope: bool = True,
        zero_line_seasonality: bool = True,
        show_legend: bool = True,
        seasonality_which: str = "dynamic",  # dynamic|baseline|total
    ) -> None:
        if not self.has_x:
            print("[states] no centred state draws 'x' found; skipping.")
            return

        band_label = self.band_label_default if band_label is None else band_label
        c_lab = _center_label(center)

        t = self.t
        ylims = ylims or {}
        yscales = yscales or {}

        def _plot_component(
            arr2d: np.ndarray,
            *,
            out_name: str,
            title: str,
            ylabel: str,
            truth: Optional[np.ndarray],
            zero_line: bool,
            ylim: Optional[Tuple[float, float]],
            yscale: Optional[str],
        ) -> None:
            ctr, lo, hi = self._summarize_ribbon(arr2d, center=center, level=self.level, interval=interval)
            fig, ax = plt.subplots(1, 1, figsize=(12, 3.4))

            lab_ctr = c_lab if show_legend else "_nolegend_"
            lab_band = band_label if show_legend else "_nolegend_"

            ax.plot(t, ctr, lw=1.6, color=color, label=lab_ctr)
            ax.fill_between(t, lo, hi, alpha=band_alpha, color=color, label=lab_band)

            if truth is not None and len(truth) == self.T:
                ax.plot(
                    t,
                    truth,
                    lw=1.2,
                    ls="--",
                    color="k",
                    alpha=0.8,
                    label=("truth" if show_legend else "_nolegend_"),
                )

            if zero_line:
                ax.axhline(0.0, lw=0.8, color="k", alpha=0.25)

            if title:
                ax.set_title(title)
            ax.set_xlabel(xlabel_time)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            if show_legend:
                ax.legend(loc="best")
            if self.is_time:
                _format_time_axis(ax)

            if yscale is not None:
                ax.set_yscale(yscale)
            if ylim is not None:
                ax.set_ylim(*ylim)

            plt.tight_layout()
            if save_dir:
                _ensure_dir(save_dir)
                out = os.path.join(save_dir, out_name)
                fig.savefig(out, dpi=200, bbox_inches="tight")
                print(f"[save] {out}")
            if show:
                plt.show()
            else:
                plt.close(fig)

        A = self._component_draws("alpha")
        if A is not None:
            _plot_component(
                A,
                out_name=f"{fname_prefix}_level.png",
                title=title_level,
                ylabel=ylabel_level,
                truth=self.true_alpha,
                zero_line=False,
                ylim=ylims.get("level"),
                yscale=yscales.get("level"),
            )

        B = self._component_draws("beta")
        if B is not None:
            sc = float(slope_scale)
            Bp = B * sc
            tb = self.true_beta * sc if (self.true_beta is not None and len(self.true_beta) == self.T) else None
            _plot_component(
                Bp,
                out_name=f"{fname_prefix}_slope.png",
                title=title_slope,
                ylabel=ylabel_slope,
                truth=tb,
                zero_line=zero_line_slope,
                ylim=ylims.get("slope"),
                yscale=yscales.get("slope"),
            )

        which = str(seasonality_which).lower().strip()
        if which not in ("dynamic", "baseline", "total"):
            raise ValueError("seasonality_which must be one of: dynamic|baseline|total")

        if which == "dynamic":
            G = self.seasonal_dynamic_series()
            truthG = self.true_gamma
        elif which == "baseline":
            G = self._baseline_season_series()
            truthG = None
        else:
            G = self.seasonal_total_series()
            truthG = None

        if G is not None:
            _plot_component(
                G,
                out_name=f"{fname_prefix}_seasonality_{which}.png",
                title=title_seasonality,  # default "" => no title
                ylabel=ylabel_seasonality,
                truth=truthG,
                zero_line=zero_line_seasonality,
                ylim=ylims.get("seasonality"),
                yscale=yscales.get("seasonality"),
            )

    def quick_report(
        self,
        *,
        save_dir: Optional[str] = None,
        fname: str = "quick_report.png",
        show: bool = True,
        color: str = "C0",
        band_alpha: float = 0.25,
        band_label: Optional[str] = None,
        center: str = "median",
        interval: Optional[str] = None,
        title_mu: str = r"$\mu_t$",
        title_sigma: str = r"$\sigma \mid y$",
        title_scale: str = r"$\xi \mid y$",
        xlabel_time: str = r"$t$",
        ylabel_mu: str = r"$\mu_t$",
        ylabel_scale: Optional[str] = None,
        show_legend_mu: bool = True,
    ) -> None:
        """
        Mirrors DLMPlotter.quick_report signature; for DGEV, the 3rd panel defaults to xi.
        If xi is missing, it falls back to process scale (s_alpha) or log10(Q[:,0]).
        """
        band_label = self.band_label_default if band_label is None else band_label
        c_lab = _center_label(center)

        ctr, lo, hi = self._summarize_ribbon(self.mu, center=center, level=self.level, interval=interval)
        t = self.t

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        axs[0].plot(t, ctr, lw=1.6, color=color, label=(c_lab if show_legend_mu else "_nolegend_"))
        axs[0].fill_between(
            t,
            lo,
            hi,
            alpha=band_alpha,
            color=color,
            label=(band_label if show_legend_mu else "_nolegend_"),
        )
        if self.true_mu is not None and len(self.true_mu) == self.T:
            axs[0].plot(
                t,
                self.true_mu,
                lw=1.2,
                ls="--",
                color="k",
                alpha=0.8,
                label=("truth" if show_legend_mu else "_nolegend_"),
            )
        axs[0].set_title(title_mu)
        axs[0].set_xlabel(xlabel_time)
        axs[0].set_ylabel(ylabel_mu)
        if show_legend_mu:
            axs[0].legend(loc="best")
        if self.is_time:
            _format_time_axis(axs[0])

        if self.sigma is not None:
            axs[1].hist(self.sigma, bins=40, density=True)
            axs[1].set_title(title_sigma)
        else:
            axs[1].axis("off")

        if self.xi is not None:
            axs[2].hist(self.xi, bins=40, density=True)
            axs[2].set_title(title_scale)
            if ylabel_scale is not None:
                axs[2].set_xlabel(ylabel_scale)
        elif self.s_alpha is not None:
            axs[2].hist(self.s_alpha, bins=40, density=True)
            axs[2].set_title("process scale")
            if ylabel_scale is not None:
                axs[2].set_xlabel(ylabel_scale)
        elif self.Q is not None and np.size(self.Q):
            logQ = np.log10(np.clip(np.asarray(self.Q)[:, 0], 1e-20, None))
            axs[2].hist(logQ, bins=40, density=True)
            axs[2].set_title("process scale")
            if ylabel_scale is not None:
                axs[2].set_xlabel(ylabel_scale)
        else:
            axs[2].axis("off")

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def figure_process_variances_hist(
        self,
        *,
        save_dir: Optional[str] = None,
        fname: str = "process_variances_log_hist.png",
        show: bool = True,
        bins: int = 40,
        alpha: float = 0.55,
        xlabel: str = r"$\log_{10}(Q)$",
        xlim: Optional[Tuple[float, float]] = None,
        show_legend: bool = True,
    ) -> None:
        """
        Separate histogram of process variances on log10 scale.

        Requirement: NO title.
        """
        if self.Q is None or (not np.size(self.Q)):
            print("[qhist] no process variances found; skipping.")
            return

        Q = np.asarray(self.Q, float)
        logQ = np.log10(np.clip(Q, 1e-20, None))
        labels = self.Q_names if self.Q_names else [rf"$Q_{{{j}}}$" for j in range(logQ.shape[1])]

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))
        for j in range(logQ.shape[1]):
            ax.hist(logQ[:, j], bins=int(bins), density=True, alpha=float(alpha), label=str(labels[j]))

        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        if show_legend and logQ.shape[1] > 1:
            ax.legend(loc="best")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ----------------------------- NEW seasonal figures (DLMPlotter-style) ----------------------------- #
    def figure_seasonal_patterns(
        self,
        *,
        which: str = "total",  # total|dynamic|baseline
        years: Union[str, Sequence[int]] = "auto",
        save_dir: Optional[str] = None,
        fname: str = "seasonal_patterns.png",
        show: bool = True,
        center: str = "median",
        level: Optional[float] = None,
        interval: Optional[str] = None,
        show_band: bool = True,
        band_alpha: float = 0.15,
        title: str = "",  # requirement: NO title by default
        ylabel: str = r"$\gamma_t$",
        xlabel: str = "month",
        show_legend: bool = True,
        legend_title: str = "",
    ) -> None:
        """
        Seasonal pattern curves for selected calendar years.
        Uses full years only and plots month-of-year (1..period) on x-axis.
        """
        lev = float(self.level if level is None else level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")
        itv = self.interval if interval is None else _normalize_interval(interval)
        c = _normalize_center(center)

        try:
            full_years, months, M = self.seasonal_matrix_year_month(which=which)
        except Exception as e:
            print(f"[seasonal_patterns] {e}; skipping.")
            return

        if isinstance(years, str) and years.strip().lower() == "auto":
            y0 = full_years[0]
            y1 = full_years[len(full_years) // 2]
            y2 = full_years[-1]
            years_list = list(dict.fromkeys([y0, y1, y2]))
        else:
            years_list = [int(y) for y in years]  # type: ignore[arg-type]

        # map year->index in M
        year_to_j = {int(y): j for j, y in enumerate(full_years)}
        years_list = [y for y in years_list if y in year_to_j]
        if not years_list:
            print("[seasonal_patterns] requested years not available as full years; skipping.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        lo_q = (1.0 - lev) / 2.0
        hi_q = 1.0 - lo_q

        for y in years_list:
            j = year_to_j[int(y)]
            Gy = M[:, j, :]  # (S, p)
            ctr = np.mean(Gy, axis=0) if c == "mean" else np.quantile(Gy, 0.5, axis=0)
            ax.plot(months, ctr, lw=1.6, label=str(y) if show_legend else "_nolegend_")

            if show_band:
                if itv == "eti":
                    lo = np.quantile(Gy, lo_q, axis=0)
                    hi = np.quantile(Gy, hi_q, axis=0)
                else:
                    lo, hi = _interval_2d(Gy, mass=lev, interval=itv)
                ax.fill_between(months, lo, hi, alpha=band_alpha)

        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(months)
        ax.grid(True, alpha=0.25)
        if show_legend:
            leg = ax.legend(loc="best", title=(legend_title if legend_title else None))
            if leg is not None and legend_title == "":
                try:
                    leg.set_title("")
                except Exception:
                    pass

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def figure_seasonal_variance(
        self,
        *,
        which: str = "total",  # total|dynamic|baseline
        save_dir: Optional[str] = None,
        fname: str = "seasonal_variance.png",
        show: bool = True,
        center: str = "median",
        level: Optional[float] = None,
        interval: Optional[str] = None,
        title: str = "",  # requirement: NO title by default
        ylabel: str = r"$A_\gamma$",
        xlabel: str = "year",
        band_alpha: float = 0.25,
    ) -> None:
        """
        A_gamma(year) = Var_m(season_{y,m}) computed draw-by-draw, then ribbon over years.
        """
        lev = float(self.level if level is None else level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")
        itv = self.interval if interval is None else _normalize_interval(interval)
        c = _normalize_center(center)

        try:
            years_full, _months, M = self.seasonal_matrix_year_month(which=which)  # (S, Ny, p)
        except Exception as e:
            print(f"[seasonal_variance] {e}; skipping.")
            return

        A = np.var(M, axis=2, ddof=0)  # (S, Ny)

        ctr = np.mean(A, axis=0) if c == "mean" else np.quantile(A, 0.5, axis=0)
        if itv == "eti":
            lo_q = (1.0 - lev) / 2.0
            hi_q = 1.0 - lo_q
            lo = np.quantile(A, lo_q, axis=0)
            hi = np.quantile(A, hi_q, axis=0)
        else:
            lo = np.empty(A.shape[1], dtype=float)
            hi = np.empty(A.shape[1], dtype=float)
            for j in range(A.shape[1]):
                lo[j], hi[j] = _hpd_1d(A[:, j], mass=lev)

        fig, ax = plt.subplots(1, 1, figsize=(10.5, 3.6))
        ax.plot(years_full, ctr, lw=1.6)
        ax.fill_between(years_full, lo, hi, alpha=band_alpha)

        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def figure_seasonal_dummies_heatmap(
        self,
        *,
        which: str = "baseline",  # total|dynamic|baseline
        save_dir: Optional[str] = None,
        fname: str = "seasonal_heatmap.png",
        show: bool = True,
        center: str = "median",
        title: str = "",  # default no title
        xlabel: str = "month",
        ylabel: str = "year",
        show_colorbar: bool = True,
        aspect: str = "auto",
        interpolation: str = "nearest",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """
        Heatmap of seasonal contribution by year (rows) and month (cols).
        Plots the posterior center (mean/median) of season_{y,m}.
        """
        c = _normalize_center(center)
        try:
            years_full, months, M = self.seasonal_matrix_year_month(which=which)  # (S, Ny, p)
        except Exception as e:
            print(f"[seasonal_heatmap] {e}; skipping.")
            return

        H = np.mean(M, axis=0) if c == "mean" else np.quantile(M, 0.5, axis=0)  # (Ny, p)

        fig, ax = plt.subplots(1, 1, figsize=(11, 4.2))
        im = ax.imshow(
            H,
            aspect=aspect,
            interpolation=interpolation,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )

        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # ticks
        ax.set_xticks(np.arange(months.size))
        ax.set_xticklabels([str(int(m)) for m in months])
        # y ticks: keep readable (every k years)
        Ny = len(years_full)
        if Ny <= 15:
            yt = np.arange(Ny)
        else:
            k = max(1, Ny // 10)
            yt = np.arange(0, Ny, k)
        ax.set_yticks(yt)
        ax.set_yticklabels([str(years_full[i]) for i in yt])

        ax.grid(False)
        if show_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def figure_seasonal_dummies_by_month(
        self,
        *,
        which: str = "baseline",  # total|dynamic|baseline
        save_dir: Optional[str] = None,
        fname: str = "seasonal_by_month.png",
        show: bool = True,
        center: str = "median",
        level: Optional[float] = None,
        interval: Optional[str] = None,
        title: str = "",  # default no title
        xlabel: str = "year",
        ylabel: str = r"$\gamma$",
        band_alpha: float = 0.20,
        show_band: bool = False,  # off by default (can get busy)
        legend: bool = True,
        max_months_in_legend: int = 12,
    ) -> None:
        """
        For each month-of-year m, plot the (center) seasonal contribution vs year.
        Optionally add a per-year band (ETI/HPD) for each month.
        """
        lev = float(self.level if level is None else level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")
        itv = self.interval if interval is None else _normalize_interval(interval)
        c = _normalize_center(center)

        try:
            years_full, months, M = self.seasonal_matrix_year_month(which=which)  # (S, Ny, p)
        except Exception as e:
            print(f"[seasonal_by_month] {e}; skipping.")
            return

        Ny = len(years_full)
        p = months.size

        # center across draws for each (year, month)
        C = np.mean(M, axis=0) if c == "mean" else np.quantile(M, 0.5, axis=0)  # (Ny, p)

        # optional bands per (year, month) across draws -> (Ny, p)
        if show_band:
            if itv == "eti":
                lo_q = (1.0 - lev) / 2.0
                hi_q = 1.0 - lo_q
                Lo = np.quantile(M, lo_q, axis=0)
                Hi = np.quantile(M, hi_q, axis=0)
            else:
                Lo = np.empty((Ny, p), float)
                Hi = np.empty((Ny, p), float)
                for j in range(Ny):
                    for m in range(p):
                        Lo[j, m], Hi[j, m] = _hpd_1d(M[:, j, m], mass=lev)
        else:
            Lo = Hi = None  # type: ignore[assignment]

        fig, ax = plt.subplots(1, 1, figsize=(11, 4.2))
        for m_idx, m in enumerate(months):
            ax.plot(years_full, C[:, m_idx], lw=1.2, label=f"m={int(m)}" if (legend and p <= max_months_in_legend) else "_nolegend_")
            if show_band and Lo is not None and Hi is not None:
                ax.fill_between(years_full, Lo[:, m_idx], Hi[:, m_idx], alpha=band_alpha)

        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if legend and p <= max_months_in_legend:
            ax.legend(loc="best", ncol=4)

        plt.tight_layout()
        if save_dir:
            _ensure_dir(save_dir)
            out = os.path.join(save_dir, fname)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[save] {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# =============================================================================
# CLI (mirrors dlm_plotter.py)
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "DGEV Laplace plotter (mirrors the Gaussian DLMPlotter CLI).\n"
            "Produces overview, scalar trace/hist/ACF, separate state plots, quick report,\n"
            "a separate log10(Q) histogram, and seasonal diagnostics.\n"
            "Auto-detects minima=True from meta and back-transforms for plotting.\n"
            "Use --<section>-kw K=V (repeatable) to override kwargs.\n"
            "Nested dicts: use dot notation, e.g. ylims.slope=(-1,1).\n"
            "Also supports printing level/slope and static parameter summaries.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to a run directory or directly to posterior.npz. If omitted, searches under --root.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results/simulations/DGEV_NCP_LASSO",
        help="Search root if --target is omitted.",
    )
    parser.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    parser.add_argument(
        "--interval",
        type=str,
        default="eti",
        choices=["eti", "hpd"],
        help="Credible interval type (ETI or HPD).",
    )
    parser.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    parser.add_argument("--out", type=str, default=None, help="Directory to save figures. Default: <run>/figures")

    parser.add_argument(
        "--start-date",
        type=str,
        default="1997-01-01",
        help="Override meta start_date (YYYY-MM-DD) to build a monthly datetime axis.",
    )

    gmin = parser.add_mutually_exclusive_group()
    gmin.add_argument("--minima", action="store_true", help="Force minima=True (treat stored series as negated; back-transform).")
    gmin.add_argument("--maxima", action="store_true", help="Force minima=False (no back-transform).")

    parser.add_argument("--skip-overview", action="store_true", help="Skip overview figure.")
    parser.add_argument("--skip-traceacf", action="store_true", help="Skip trace+hist+ACF panels.")
    parser.add_argument("--skip-states", action="store_true", help="Skip separate state plots.")
    parser.add_argument("--skip-quick", action="store_true", help="Skip quick report.")
    parser.add_argument("--skip-qhist", action="store_true", help="Skip separate log10(Q) histogram.")

    # NEW seasonal toggles
    parser.add_argument("--skip-seasonal-patterns", action="store_true", help="Skip seasonal patterns figure.")
    parser.add_argument("--skip-seasonal-variance", action="store_true", help="Skip seasonal variance-by-year figure.")
    parser.add_argument("--skip-seasonal-heatmap", action="store_true", help="Skip seasonal heatmap figure.")
    parser.add_argument("--skip-seasonal-bymonth", action="store_true", help="Skip seasonal by-month trends figure.")

    parser.add_argument(
        "--seasonal-years",
        type=str,
        default="auto",
        help="Comma-separated calendar years for seasonal patterns, or 'auto'. Example: 1950,1980,2020",
    )
    parser.add_argument(
        "--seasonal-which",
        type=str,
        default="total",
        choices=["total", "dynamic", "baseline"],
        help="Which seasonal contribution to plot (applies to seasonal diagnostics unless overridden by section kwargs).",
    )

    parser.add_argument(
        "--overview-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_overview(...). Repeatable. Supports nested keys via dots.",
    )
    parser.add_argument(
        "--traceacf-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_trace_acf_core(...). Repeatable. Supports nested keys via dots.",
    )
    parser.add_argument(
        "--states-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_states_separate(...). Repeatable. Supports nested keys via dots.",
    )
    parser.add_argument(
        "--quick-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.quick_report(...). Repeatable. Supports nested keys via dots.",
    )
    parser.add_argument(
        "--qhist-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_process_variances_hist(...). Repeatable.",
    )

    # seasonal kwargs
    parser.add_argument(
        "--seasonal-patterns-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_seasonal_patterns(...). Repeatable.",
    )
    parser.add_argument(
        "--seasonal-variance-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_seasonal_variance(...). Repeatable.",
    )
    parser.add_argument(
        "--seasonal-heatmap-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_seasonal_dummies_heatmap(...). Repeatable.",
    )
    parser.add_argument(
        "--seasonal-bymonth-kw",
        action="append",
        default=[],
        metavar="K=V",
        help="Override kwargs for plotter.figure_seasonal_dummies_by_month(...). Repeatable.",
    )

    # printing toggles (same pattern as dlm_plotter.py)
    gp = parser.add_mutually_exclusive_group()
    gp.add_argument(
        "--print-level-slope",
        dest="print_level_slope",
        action="store_true",
        default=True,
        help="Print level/slope summaries at times given by --times.",
    )
    gp.add_argument(
        "--no-print-level-slope",
        dest="print_level_slope",
        action="store_false",
        help="Disable printing of level/slope summaries.",
    )
    parser.add_argument(
        "--times",
        type=str,
        default="start,mid,end",
        help="Comma-separated times for level/slope printing. Each can be index or start/mid/end or YYYY-MM.",
    )
    parser.add_argument(
        "--slope-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to beta_t when printing (e.g. 120 for per-decade if beta is per-month).",
    )

    parser.add_argument(
        "--print-static",
        action="store_true",
        default=True,
        help="Print summaries (center + CI) for all static parameters (scalars and vectors).",
    )
    parser.add_argument("--static-level", type=float, default=None, help="Credible level for static params (defaults to --level).")
    parser.add_argument("--static-center", type=str, default="median", help="Center for static summaries: median or mean.")
    parser.add_argument("--static-digits", type=int, default=4, help="Digits for static summary printing.")
    parser.add_argument("--static-max-cols", type=int, default=None, help="Max columns to print per vector parameter (None = all).")
    parser.add_argument("--static-no-diag", action="store_true", default=False, help="Disable ESS/Geweke diagnostics in static summary.")
    parser.add_argument("--static-interval", type=str, default=None, choices=["eti", "hpd"], help="Override interval type for static printing.")

    args = parser.parse_args()

    # --- resolve posterior (exactly like dlm_plotter.py) ---
    run_path = args.target
    if run_path is None:
        print(f"[info] --target not provided; searching for the latest posterior under --root={args.root!r} ...")
        run_path = find_latest_run(root=args.root)
        if run_path is None:
            npz_path = find_latest_posterior_npz(args.root)
            if npz_path is None:
                print(f"[error] No posterior runs found under {args.root!r}. Provide --target or change --root.")
                sys.exit(1)
            run_path = npz_path
            print(f"[info] find_latest_run found nothing; using latest npz: {run_path}")
        else:
            print(f"[info] Using latest run: {run_path}")

    bundle = load_posterior(run_path)
    draws, meta, npz_path = bundle.draws, bundle.meta, bundle.npz_path

    if args.start_date:
        meta = dict(meta)
        meta["start_date"] = str(args.start_date)

    out_dir = args.out or os.path.join(os.path.dirname(npz_path), "figures")
    _ensure_dir(out_dir)
    print(f"[info] using posterior: {npz_path}")
    print(f"[info] saving figures to: {out_dir}")

    minima_override: Optional[bool]
    if args.minima:
        minima_override = True
    elif args.maxima:
        minima_override = False
    else:
        minima_override = None

    plotter = DGEVPlotter(
        draws=draws,
        meta=meta,
        level=float(args.level),
        minima=minima_override,
        interval=str(args.interval),
    )

    overview_kw = _parse_kv_list(args.overview_kw)
    traceacf_kw = _parse_kv_list(args.traceacf_kw)
    states_kw = _parse_kv_list(args.states_kw)
    quick_kw = _parse_kv_list(args.quick_kw)
    qhist_kw = _parse_kv_list(args.qhist_kw)

    seasonal_patterns_kw = _parse_kv_list(args.seasonal_patterns_kw)
    seasonal_variance_kw = _parse_kv_list(args.seasonal_variance_kw)
    seasonal_heatmap_kw = _parse_kv_list(args.seasonal_heatmap_kw)
    seasonal_bymonth_kw = _parse_kv_list(args.seasonal_bymonth_kw)

    # parse seasonal years (auto or list)
    sy = str(args.seasonal_years).strip()
    if sy.lower() == "auto" or sy == "":
        seasonal_years: Union[str, List[int]] = "auto"
    else:
        seasonal_years = [int(z) for z in sy.split(",") if z.strip() != ""]

    seasonal_which = str(args.seasonal_which).strip().lower()

    # --- optional printing (same structure as dlm_plotter.py) ---
    if args.print_level_slope:
        raw_times = [s.strip() for s in str(args.times).split(",") if s.strip() != ""]
        times: List[TimeLike] = []
        for rt in raw_times:
            if re.fullmatch(r"\d+", rt):
                times.append(int(rt))
            else:
                times.append(rt)
        plotter.print_level_slope_at(times=times, slope_scale=float(args.slope_scale))

    if args.print_static:
        plotter.print_static_params(
            level=args.static_level,
            center=args.static_center,
            digits=int(args.static_digits),
            max_vector_cols=args.static_max_cols,
            include_diagnostics=(not args.static_no_diag),
            interval=args.static_interval,
        )

    # --- figures ---
    if not args.skip_overview:
        plotter.figure_overview(save_dir=out_dir, show=args.show, **overview_kw)
    if not args.skip_traceacf:
        plotter.figure_trace_acf_core(save_dir=out_dir, show=args.show, **traceacf_kw)
    if not args.skip_states:
        plotter.figure_states_separate(save_dir=out_dir, show=args.show, **states_kw)
    if not args.skip_quick:
        plotter.quick_report(save_dir=out_dir, show=args.show, **quick_kw)
    if not args.skip_qhist:
        plotter.figure_process_variances_hist(save_dir=out_dir, show=args.show, **qhist_kw)

    # --- seasonal diagnostics (new) ---
    if not args.skip_seasonal_patterns:
        if "which" not in seasonal_patterns_kw:
            seasonal_patterns_kw["which"] = seasonal_which
        plotter.figure_seasonal_patterns(
            years=seasonal_years,
            save_dir=out_dir,
            show=args.show,
            **seasonal_patterns_kw,
        )

    if not args.skip_seasonal_variance:
        if "which" not in seasonal_variance_kw:
            seasonal_variance_kw["which"] = seasonal_which
        plotter.figure_seasonal_variance(
            save_dir=out_dir,
            show=args.show,
            **seasonal_variance_kw,
        )

    if not args.skip_seasonal_heatmap:
        if "which" not in seasonal_heatmap_kw:
            seasonal_heatmap_kw["which"] = seasonal_which
        plotter.figure_seasonal_dummies_heatmap(
            save_dir=out_dir,
            show=args.show,
            **seasonal_heatmap_kw,
        )

    if not args.skip_seasonal_bymonth:
        if "which" not in seasonal_bymonth_kw:
            seasonal_bymonth_kw["which"] = seasonal_which
        plotter.figure_seasonal_dummies_by_month(
            save_dir=out_dir,
            show=args.show,
            **seasonal_bymonth_kw,
        )

    print("[done] plots written.")