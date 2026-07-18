# %% simulator/dlm_plotter.py
from __future__ import annotations
"""
DLM Plotter (Gaussian structural models)
======================================

A self-contained plotter for posterior bundles produced by the Gaussian DLM samplers.

Conventions / expectations
--------------------------
Required draws:
  - mu: (S, T)

Optional draws:
  - y: (T,)
  - x: (S, T, dim)  (centred state draws saved in sampler)
  - sigma or sigma2: (S,)
  - alpha0, beta0, gamma0: (S, ...) or m0_alpha, m0_beta, m0_gamma
  - s_alpha, s_beta, s_gamma: (S,)
  - Q: (S, K) OR separate Q_alpha/Q_beta/Q_gamma: (S,)

Meta:
  - period (default 12)
  - layout (names for state vector in x and/or Q)
  - start_date (YYYY-MM-DD) recommended for year/month grouping and datetime x-axis

Notes
-----
- Bands are quantile-based; center line is median (default) or mean.
- If meta has a monthly start date, x-axis is datetime and gets a concise formatter.
- Seasonal diagnostics:
    * "dynamic seasonal state" is x[..., idx_g0]  (often the rotating dummy seasonal state)
    * "baseline seasonal dummies" are S[t,:] @ gamma0  (static)
    * "total seasonal contribution" is  season_t = dynamic + baseline

This plotter provides both:
  - the old seasonal plots (dynamic-only), and
  - the requested seasonal-dummies-over-time views (total seasonal contribution),
    as year×month matrices and plots (heatmap / month-specific trajectories).
"""

import os
import sys
import re
from typing import Optional, Tuple, Dict, Any, List, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

# Make optimization/ visible
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
# Plotter
# =============================================================================
TimeLike = Union[int, str]


class DLMPlotter:
    """
    Plotter for posterior bundles from Gaussian DLM samplers.
    """

    # ----------------------------- construction ----------------------------- #
    def __init__(self, draws: Dict[str, np.ndarray], meta: Dict[str, Any], level: float = 0.90):
        self.draws = draws
        self.meta = meta
        self.level = float(level)
        if not (0.0 < self.level < 1.0):
            raise ValueError("level must be in (0,1)")

        if "mu" not in draws:
            raise ValueError("draws must contain 'mu' of shape (S, T).")
        self.mu = np.asarray(draws["mu"], float)
        if self.mu.ndim != 2:
            raise ValueError("'mu' must be a 2D array (S, T).")

        self.S, self.T = self.mu.shape
        self.period = int(meta.get("period", 12))

        # time axis
        t, is_time = _monthly_time_axis_from_meta(meta, self.T)
        if t is None:
            self.t = np.arange(self.T)
            self.is_time = False
        else:
            self.t = t
            self.is_time = bool(is_time)

        # optional data & truth overlays
        self.y = _maybe(draws, "y")
        self.true_mu = _maybe(draws, "true_mu_t")
        self.true_alpha = _maybe(draws, "true_alpha_t")
        self.true_beta = _maybe(draws, "true_beta_t")
        self.true_gamma = _maybe(draws, "true_gamma_t")

        # sigma on SD scale (prefer sigma; fallback sqrt(sigma2))
        self.sigma: Optional[np.ndarray] = None
        if "sigma" in draws and np.asarray(draws["sigma"]).ndim == 1 and np.asarray(draws["sigma"]).shape[0] == self.S:
            self.sigma = np.asarray(draws["sigma"], float)
        elif "sigma2" in draws and np.asarray(draws["sigma2"]).ndim == 1 and np.asarray(draws["sigma2"]).shape[0] == self.S:
            self.sigma = np.sqrt(np.clip(np.asarray(draws["sigma2"], float), 0.0, None))

        # baselines (may be saved under alpha0/beta0/gamma0 or m0_alpha/... )
        self.alpha0 = np.asarray(draws["alpha0"], float) if "alpha0" in draws else _maybe(draws, "m0_alpha")
        self.beta0 = np.asarray(draws["beta0"], float) if "beta0" in draws else _maybe(draws, "m0_beta")
        self.gamma0 = np.asarray(draws["gamma0"], float) if "gamma0" in draws else _maybe(draws, "m0_gamma")

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
        self.has_x = ("x" in draws) and (np.asarray(draws["x"]).ndim == 3)
        self.idx_alpha: Optional[int] = None
        self.idx_beta: Optional[int] = None
        self.idx_g0: Optional[int] = None
        self._init_state_indices()

        self.band_label_default = rf"{int(round(self.level * 100))}% band"

    # ----------------------------- param collection ----------------------------- #
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

    # ----------------------------- init helpers ----------------------------- #
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

            # seasonal state: try "gamma" then "g1"/"g0" style
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = np.asarray(arr_2d, float)
        lev = float(self.level if level is None else level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")
        c = _normalize_center(center)

        lo_q = (1.0 - lev) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(A, lo_q, axis=0)
        hi = np.quantile(A, hi_q, axis=0)
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

    # ----------------------------- time indexing ----------------------------- #
    def _time_to_index(self, t: TimeLike) -> Tuple[int, str]:
        """
        Accepts:
          - int index (0..T-1)
          - 'start'/'mid'/'end'
          - date string 'YYYY'/'YYYY-MM'/'YYYY-MM-DD' (monthly mapping if meta has start_date & period)
        """
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

    # ----------------------------- seasonal indexing ----------------------------- #
    def _year_month_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return arrays (year[t], month[t]) for t=0..T-1, where month is 1..period.
        Uses datetime axis if available; otherwise constructs from meta['start_date'].
        """
        if self.is_time and hasattr(self.t, "__len__"):
            years = np.array([dt.year for dt in self.t], dtype=int)
            months = np.array([dt.month for dt in self.t], dtype=int)  # 1..12
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
        """
        Returns (full_years_sorted, idx_by_year, months_arr).
        Full year means exactly `period` months present in that calendar year.
        """
        uniq_years, idx_by_year = self._year_slices()
        _, months_arr = self._year_month_arrays()
        p = int(self.period)
        full = [int(y) for y in uniq_years if idx_by_year[int(y)].size == p]
        full = sorted(full)
        return full, idx_by_year, months_arr

    # ----------------------------- seasonal composition ----------------------------- #
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
          base_t = gamma0_full[m_t], with m_t = (t mod period) in {0..period-1}
        Returns (S, T) or None if gamma0 missing.
        """
        G0 = self._gamma0_full()
        if G0 is None:
            return None

        p = int(self.period)
        m = (np.arange(self.T, dtype=int) % p)  # 0..p-1
        # pick month-specific baseline per time
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
        which: str = "total",  # "total" | "dynamic" | "baseline"
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """
        Returns:
          years_full: list of calendar years with full period months
          months:     array [1..period]
          M:          (S, Ny, period) with M[:, j, m-1] as seasonal contribution for that year & month

        This is the object you described: year1: gamma_1..gamma_p, year2: gamma_{p+1}..gamma_{2p}, etc.
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
            raise RuntimeError("No seasonal series available (need draws['x'] with seasonal index).")

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

    # ----------------------------- printing: level/slope ----------------------------- #
    def print_level_slope_at(
        self,
        times: Sequence[TimeLike] = ("start", "mid", "end"),
        *,
        slope_scale: float = 1.0,
        level: Optional[float] = None,
        digits: int = 4,
        use_mean: bool = False,
    ) -> List[Dict[str, Any]]:
        if not self.has_x or self.idx_alpha is None or self.idx_beta is None:
            raise RuntimeError("Cannot print level/slope: draws['x'] missing or layout lacks alpha/beta.")

        X = np.asarray(self.draws["x"])
        lev = float(self.level if level is None else level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")
        lo_q = (1.0 - lev) / 2.0
        hi_q = 1.0 - lo_q

        rows: List[Dict[str, Any]] = []
        for t in times:
            idx, label = self._time_to_index(t)
            a = X[:, idx, self.idx_alpha]
            b = X[:, idx, self.idx_beta] * float(slope_scale)

            a_ctr = float(np.mean(a)) if use_mean else float(np.quantile(a, 0.5))
            b_ctr = float(np.mean(b)) if use_mean else float(np.quantile(b, 0.5))

            rows.append(
                {
                    "time": label,
                    "idx": int(idx),
                    "alpha_ctr": a_ctr,
                    "alpha_lo": float(np.quantile(a, lo_q)),
                    "alpha_hi": float(np.quantile(a, hi_q)),
                    "beta_ctr": b_ctr,
                    "beta_lo": float(np.quantile(b, lo_q)),
                    "beta_hi": float(np.quantile(b, hi_q)),
                }
            )

        fmt = f"{{:.{int(digits)}f}}".format
        ctr_name = "mean" if use_mean else "median"
        print(f"\n[level/slope summaries] S={X.shape[0]}, T={self.T}, CI={int(round(100*lev))}% (center={ctr_name})")
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

    # ----------------------------- printing: static params ----------------------------- #
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

        lo_q = (1.0 - float(level)) / 2.0
        hi_q = 1.0 - lo_q
        c = _normalize_center(center)

        ctr = float(np.mean(v)) if c == "mean" else float(np.quantile(v, 0.5))
        lo = float(np.quantile(v, lo_q))
        hi = float(np.quantile(v, hi_q))

        out.update({"ctr": ctr, "lo": lo, "hi": hi})

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
    ) -> List[Dict[str, Any]]:
        lev = float(self.level if level is None else level)
        if not (0.0 < lev < 1.0):
            raise ValueError("level must be in (0,1)")
        c = _normalize_center(center)
        ctr_name = "mean" if c == "mean" else "median"
        ci_pct = int(round(100 * lev))
        fmt = f"{{:.{int(digits)}f}}".format

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

        if self.sigma is not None:
            add_scalar("sigma", self.sigma)

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
            "mu", "x", "y",
            "true_mu_t", "true_alpha_t", "true_beta_t", "true_gamma_t",
            "sigma", "sigma2",
            "alpha0", "beta0", "gamma0",
            "m0_alpha", "m0_beta", "m0_gamma",
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

        print(f"\n[static parameter summaries] S={self.S}, CI={ci_pct}% (center={ctr_name})")
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
                print(
                    f" {nm:<30} {n:>5d} | "
                    f"{fmt(ctr):>10} [{fmt(lo):>10}, {fmt(hi):>10}]"
                )

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
        plot_policy: str = "none",    # "none" | "clip" | "drop"
        diag_policy: str = "clean",   # "raw" | "clipped" | "clean"
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

        out_mask_finite = ((s_finite < lo_bound) | (s_finite > hi_bound)) if use_bounds else np.zeros_like(s_finite, bool)
        n_out = int(np.sum(out_mask_finite))

        s_plot = s_raw.copy()
        s_plot[~finite_mask] = np.nan

        if use_bounds and plot_policy == "clip":
            s_plot = np.clip(s_plot, lo_bound, hi_bound)
        elif use_bounds and plot_policy == "drop":
            out_mask_raw = np.zeros_like(s_raw, dtype=bool)
            out_mask_raw[finite_mask] = out_mask_finite
            s_plot[out_mask_raw] = np.nan

        s_hist = s_finite.copy()
        if use_bounds and plot_policy == "clip":
            s_hist = np.clip(s_hist, lo_bound, hi_bound)
        elif use_bounds and plot_policy == "drop":
            s_hist = s_hist[~out_mask_finite]

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
            auto_zoom_if_clipped and use_bounds and plot_policy in ("clip", "drop")
            and np.isfinite(lo_bound) and np.isfinite(hi_bound)
        ):
            axs[0].set_ylim(lo_bound, hi_bound)

        hist_range = None
        if hist_xlim is not None:
            hist_range = (float(hist_xlim[0]), float(hist_xlim[1]))
        elif (
            auto_zoom_if_clipped and use_bounds and plot_policy in ("clip", "drop")
            and np.isfinite(lo_bound) and np.isfinite(hi_bound)
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
    ) -> None:
        band_label = self.band_label_default if band_label is None else band_label
        c_lab = _center_label(center)

        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        axs = axs.ravel()

        t = self.t
        ctr, lo, hi = self._summarize_ribbon(self.mu, center=center, level=self.level)

        axs[0].plot(t, ctr, lw=1.6, color=color, label=(c_lab if show_legend_mu else "_nolegend_"))
        axs[0].fill_between(
            t, lo, hi, alpha=band_alpha, color=color,
            label=(band_label if show_legend_mu else "_nolegend_"),
        )
        if self.y is not None and len(self.y) == self.T:
            axs[0].plot(t, self.y, lw=1.0, alpha=0.6, label=(r"$y_t$" if show_legend_mu else "_nolegend_"))
        if self.true_mu is not None and len(self.true_mu) == self.T:
            axs[0].plot(
                t, self.true_mu, lw=1.2, ls="--", color="k", alpha=0.8,
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

        any_baseline = (self.alpha0 is not None) or (self.beta0 is not None) or (self.gamma0 is not None)
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
            ax.set_title(title_baselines)
            ax.legend(loc="best")
        else:
            axs[4].axis("off")

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
                "sigma", "sigma2",
                "s_alpha", "s_beta", "s_gamma",
                "Q_alpha", "Q_beta", "Q_gamma",
                "lambda2", "tau_alpha", "tau_beta", "tau_gamma",
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
        xlabel_time: str = r"$t$",
        title_level: str = "",
        ylabel_level: str = r"$\alpha_t$",
        title_slope: str = "",
        ylabel_slope: str = r"$\beta_t$",
        title_seasonality: str = "",
        ylabel_seasonality: str = r"$\gamma_t$",
        slope_scale: float = 1.0,
        ylims: Optional[Dict[str, Tuple[float, float]]] = None,
        yscales: Optional[Dict[str, str]] = None,
        zero_line_slope: bool = True,
        zero_line_seasonality: bool = True,
        show_legend: bool = True,
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
            ctr, lo, hi = self._summarize_ribbon(arr2d, center=center, level=self.level)
            fig, ax = plt.subplots(1, 1, figsize=(12, 3.4))

            lab_ctr = c_lab if show_legend else "_nolegend_"
            lab_band = band_label if show_legend else "_nolegend_"

            ax.plot(t, ctr, lw=1.6, color=color, label=lab_ctr)
            ax.fill_between(t, lo, hi, alpha=band_alpha, color=color, label=lab_band)

            if truth is not None and len(truth) == self.T:
                ax.plot(
                    t, truth, lw=1.2, ls="--", color="k", alpha=0.8,
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

        G = self._component_draws("gamma")
        if G is not None:
            _plot_component(
                G,
                out_name=f"{fname_prefix}_seasonality.png",
                title=title_seasonality,
                ylabel=ylabel_seasonality,
                truth=self.true_gamma,
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
        title_mu: str = r"$\mu_t$",
        title_sigma: str = r"$\sigma \mid y$",
        title_scale: str = r"process scale",
        xlabel_time: str = r"$t$",
        ylabel_mu: str = r"$\mu_t$",
        ylabel_scale: Optional[str] = None,
        show_legend_mu: bool = True,
    ) -> None:
        band_label = self.band_label_default if band_label is None else band_label
        c_lab = _center_label(center)

        ctr, lo, hi = self._summarize_ribbon(self.mu, center=center, level=self.level)
        t = self.t

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        axs[0].plot(t, ctr, lw=1.6, color=color, label=(c_lab if show_legend_mu else "_nolegend_"))
        axs[0].fill_between(
            t, lo, hi, alpha=band_alpha, color=color,
            label=(band_label if show_legend_mu else "_nolegend_"),
        )
        if self.true_mu is not None and len(self.true_mu) == self.T:
            axs[0].plot(
                t, self.true_mu, lw=1.2, ls="--", color="k", alpha=0.8,
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

        if self.s_alpha is not None:
            axs[2].hist(self.s_alpha, bins=40, density=True)
            axs[2].set_title(title_scale)
            if ylabel_scale is not None:
                axs[2].set_xlabel(ylabel_scale)
        elif self.Q is not None and np.size(self.Q):
            logQ = np.log10(np.clip(np.asarray(self.Q)[:, 0], 1e-20, None))
            axs[2].hist(logQ, bins=40, density=True)
            axs[2].set_title(title_scale)
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

    # ----------------------------- seasonal diagnostics ----------------------------- #
    def figure_seasonal_patterns(
        self,
        *,
        years: Union[str, Sequence[int]] = "auto",
        which: str = "total",            # total|dynamic|baseline
        save_dir: Optional[str] = None,
        fname: str = "seasonal_patterns.png",
        show: bool = True,
        center: str = "median",
        level: Optional[float] = None,
        show_band: bool = True,
        band_alpha: float = 0.15,
        title: str = "",                 # NO title by default
        ylabel: str = r"$\gamma_{y,m}$",  # year-month seasonal effect
        xlabel: str = "month",
    ) -> None:
        """
        Plot posterior seasonal pattern (year y, months 1..period) for selected years.

        IMPORTANT:
          - If which="total" (default), this uses total seasonal contribution:
                dynamic seasonal state + baseline dummy contribution (S@gamma0).
          - If you want the old behaviour, use which="dynamic".
        """
        which = str(which).lower().strip()

        lev = float(self.level if level is None else level)
        lo_q = (1.0 - lev) / 2.0
        hi_q = 1.0 - lo_q
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

        # keep only full years
        yset = set(full_years)
        years_list = [y for y in years_list if y in yset]
        if not years_list:
            print("[seasonal_patterns] requested years not available as full years; skipping.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        for y in years_list:
            j = full_years.index(int(y))
            Gy = M[:, j, :]  # (S, p)
            ctr = np.mean(Gy, axis=0) if c == "mean" else np.quantile(Gy, 0.5, axis=0)
            ax.plot(months, ctr, lw=1.6, label=str(y))

            if show_band:
                lo = np.quantile(Gy, lo_q, axis=0)
                hi = np.quantile(Gy, hi_q, axis=0)
                ax.fill_between(months, lo, hi, alpha=band_alpha)

        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(months)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", title="year")

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
        which: str = "total",            # total|dynamic|baseline
        save_dir: Optional[str] = None,
        fname: str = "seasonal_variance.png",
        show: bool = True,
        center: str = "median",
        level: Optional[float] = None,
        title: str = "",                # NO title by default
        ylabel: str = r"$A_\gamma$",     # Var_m(gamma_{y,m})
        xlabel: str = "year",
        band_alpha: float = 0.25,
    ) -> None:
        """
        Compute annual seasonal variance A_y = Var_m(gamma_{y,m}) draw-by-draw and plot ribbon.
        Uses year×month seasonal matrix based on `which`.
        """
        which = str(which).lower().strip()
        lev = float(self.level if level is None else level)
        c = _normalize_center(center)

        try:
            full_years, _months, M = self.seasonal_matrix_year_month(which=which)
        except Exception as e:
            print(f"[seasonal_variance] {e}; skipping.")
            return

        # per draw, per year: variance across months
        A = np.var(M, axis=2, ddof=0)  # (S, Ny)

        ctr = np.mean(A, axis=0) if c == "mean" else np.quantile(A, 0.5, axis=0)
        lo_q = (1.0 - lev) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(A, lo_q, axis=0)
        hi = np.quantile(A, hi_q, axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(10.5, 3.6))
        ax.plot(full_years, ctr, lw=1.6)
        ax.fill_between(full_years, lo, hi, alpha=band_alpha)
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

    # ---- NEW: seasonal dummies over time ----
    def figure_seasonal_dummies_heatmap(
        self,
        *,
        which: str = "total",  # total|dynamic|baseline
        save_dir: Optional[str] = None,
        fname: str = "seasonal_dummies_heatmap.png",
        show: bool = True,
        center: str = "median",
        title: str = "",       # keep default no title
        xlabel: str = "month",
        ylabel: str = "year",
        cmap: str = "viridis",
    ) -> None:
        """
        Heatmap: rows=years, cols=months, values=center of seasonal contribution gamma_{y,m}.
        This is the cleanest visual for "gamma_1..gamma_12, gamma_13..gamma_24, ...".
        """
        which = str(which).lower().strip()
        c = _normalize_center(center)

        try:
            years, months, M = self.seasonal_matrix_year_month(which=which)
        except Exception as e:
            print(f"[seasonal_heatmap] {e}; skipping.")
            return

        Z = np.mean(M, axis=0) if c == "mean" else np.quantile(M, 0.5, axis=0)  # (Ny, p)

        fig, ax = plt.subplots(1, 1, figsize=(12, 4.6))
        im = ax.imshow(
            Z,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[months[0] - 0.5, months[-1] + 0.5, years[0] - 0.5, years[-1] + 0.5],
            cmap=cmap,
        )
        ax.set_xticks(months)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("seasonal contribution")

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
        which: str = "total",  # total|dynamic|baseline
        save_dir: Optional[str] = None,
        fname: str = "seasonal_dummies_by_month.png",
        show: bool = True,
        center: str = "median",
        level: Optional[float] = None,
        band_alpha: float = 0.20,
        title: str = "",
        xlabel: str = "year",
        ylabel: str = "seasonal contribution",
        legend_ncol: int = 6,
    ) -> None:
        """
        12 lines: for each month m, plot gamma_{year,m} over years with CI band.
        """
        which = str(which).lower().strip()
        lev = float(self.level if level is None else level)
        lo_q = (1.0 - lev) / 2.0
        hi_q = 1.0 - lo_q
        c = _normalize_center(center)

        try:
            years, months, M = self.seasonal_matrix_year_month(which=which)
        except Exception as e:
            print(f"[seasonal_by_month] {e}; skipping.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 4.2))

        for mi, m in enumerate(months):
            Xm = M[:, :, mi]  # (S, Ny)
            ctr = np.mean(Xm, axis=0) if c == "mean" else np.quantile(Xm, 0.5, axis=0)
            lo = np.quantile(Xm, lo_q, axis=0)
            hi = np.quantile(Xm, hi_q, axis=0)

            ax.plot(years, ctr, lw=1.2, label=str(int(m)))
            ax.fill_between(years, lo, hi, alpha=float(band_alpha))

        ax.axhline(0.0, lw=0.8, color="k", alpha=0.25)
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(title="month", ncol=int(legend_ncol), fontsize=9)

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
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "DLM plotter for Gaussian structural models.\n"
            "Produces overview, scalar trace/hist/ACF, separate state plots, quick report,\n"
            "and a separate log10(Q) histogram.\n"
            "Seasonal diagnostics:\n"
            "  - seasonal patterns (year slices)\n"
            "  - seasonal variance per year\n"
            "  - NEW: seasonal dummies over time (heatmap, month trajectories)\n"
            "Use --<section>-kw K=V (repeatable) to override kwargs.\n"
            "Nested dicts: use dot notation, e.g. ylims.slope=(-1,1).\n"
            "Also supports printing level/slope and static parameter summaries.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--target", type=str, default=None,
        help="Path to a run directory or directly to posterior.npz. If omitted, searches under --root."
    )
    parser.add_argument(
        "--root", type=str, default="results/simulations/DLM",
        help="Search root if --target is omitted."
    )
    parser.add_argument("--level", type=float, default=0.90, help="Credible band level.")
    parser.add_argument("--show", action="store_true", default=False, help="Show figures interactively.")
    parser.add_argument("--out", type=str, default=None, help="Directory to save figures. Default: <run>/figures")

    parser.add_argument(
        "--start-date", type=str, default="1997-08-13",
        help="Override meta start_date (YYYY-MM-DD) to build a monthly datetime axis."
    )

    parser.add_argument("--skip-overview", action="store_true", help="Skip overview figure.")
    parser.add_argument("--skip-traceacf", action="store_true", help="Skip trace+hist+ACF panels.")
    parser.add_argument("--skip-states", action="store_true", help="Skip separate state plots.")
    parser.add_argument("--skip-quick", action="store_true", help="Skip quick report.")
    parser.add_argument("--skip-qhist", action="store_true", help="Skip separate log10(Q) histogram.")

    # seasonal toggles
    parser.add_argument("--skip-seasonal-patterns", action="store_true", help="Skip seasonal pattern figure.")
    parser.add_argument("--skip-seasonal-variance", action="store_true", help="Skip seasonal variance-by-year figure.")
    parser.add_argument("--skip-seasonal-heatmap", action="store_true", help="Skip seasonal dummies heatmap.")
    parser.add_argument("--skip-seasonal-by-month", action="store_true", help="Skip seasonal dummies-by-month plot.")
    parser.add_argument(
        "--seasonal-years", type=str, default="auto",
        help="Comma-separated calendar years for seasonal pattern plot, or 'auto'. Example: 1950,1980,2020"
    )
    parser.add_argument(
        "--seasonal-which", type=str, default="total",
        help="Which seasonal contribution to plot: total | dynamic | baseline."
    )

    parser.add_argument(
        "--overview-kw", action="append", default=[], metavar="K=V",
        help="Override kwargs for plotter.figure_overview(...). Repeatable. Supports nested keys via dots."
    )
    parser.add_argument(
        "--traceacf-kw", action="append", default=[], metavar="K=V",
        help="Override kwargs for plotter.figure_trace_acf_core(...). Repeatable. Supports nested keys via dots."
    )
    parser.add_argument(
        "--states-kw", action="append", default=[], metavar="K=V",
        help="Override kwargs for plotter.figure_states_separate(...). Repeatable. Supports nested keys via dots."
    )
    parser.add_argument(
        "--quick-kw", action="append", default=[], metavar="K=V",
        help="Override kwargs for plotter.quick_report(...). Repeatable. Supports nested keys via dots."
    )
    parser.add_argument(
        "--qhist-kw", action="append", default=[], metavar="K=V",
        help="Override kwargs for plotter.figure_process_variances_hist(...). Repeatable."
    )

    parser.add_argument(
        "--seasonal-patterns-kw", action="append", default=[], metavar="K=V",
        help="Override kwargs for plotter.figure_seasonal_patterns(...). Repeatable."
    )
    parser.add_argument(
        "--seasonal-variance-kw", action="append", default=[], metavar="K=V",
        help="Override kwargs for plotter.figure_seasonal_variance(...). Repeatable."
    )
    parser.add_argument(
        "--seasonal-heatmap-kw", action="append", default=[], metavar="K=V",
        help="Override kwargs for plotter.figure_seasonal_dummies_heatmap(...). Repeatable."
    )
    parser.add_argument(
        "--seasonal-by-month-kw", action="append", default=[], metavar="K=V",
        help="Override kwargs for plotter.figure_seasonal_dummies_by_month(...). Repeatable."
    )

    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--print-level-slope", dest="print_level_slope", action="store_true", default=True,
        help="Print level/slope summaries at times given by --times."
    )
    g.add_argument(
        "--no-print-level-slope", dest="print_level_slope", action="store_false",
        help="Disable printing of level/slope summaries."
    )

    parser.add_argument(
        "--times", type=str, default="start,mid,end",
        help="Comma-separated times for level/slope printing. Each can be index or start/mid/end or YYYY-MM."
    )

    parser.add_argument(
        "--print-static", action="store_true", default=True,
        help="Print summaries (center + CI) for all static parameters (scalars and vectors)."
    )
    parser.add_argument(
        "--static-level", type=float, default=None,
        help="Credible level for static params (defaults to --level)."
    )
    parser.add_argument(
        "--static-center", type=str, default="median",
        help="Center for static summaries: median or mean."
    )
    parser.add_argument(
        "--static-digits", type=int, default=4,
        help="Digits for static summary printing."
    )
    parser.add_argument(
        "--static-max-cols", type=int, default=None,
        help="Max columns to print per vector parameter (None = all)."
    )
    parser.add_argument(
        "--static-no-diag", action="store_true", default=False,
        help="Disable ESS/Geweke diagnostics in static summary."
    )

    args = parser.parse_args()

    # --- resolve posterior ---
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

    plotter = DLMPlotter(draws=draws, meta=meta, level=float(args.level))

    overview_kw = _parse_kv_list(args.overview_kw)
    traceacf_kw = _parse_kv_list(args.traceacf_kw)
    states_kw = _parse_kv_list(args.states_kw)
    quick_kw = _parse_kv_list(args.quick_kw)
    qhist_kw = _parse_kv_list(args.qhist_kw)
    seasonal_patterns_kw = _parse_kv_list(args.seasonal_patterns_kw)
    seasonal_variance_kw = _parse_kv_list(args.seasonal_variance_kw)
    seasonal_heatmap_kw = _parse_kv_list(args.seasonal_heatmap_kw)
    seasonal_by_month_kw = _parse_kv_list(args.seasonal_by_month_kw)

    # seasonal-years parsing
    sy = str(args.seasonal_years).strip()
    if sy.lower() == "auto" or sy == "":
        seasonal_years: Union[str, List[int]] = "auto"
    else:
        seasonal_years = [int(z) for z in sy.split(",") if z.strip() != ""]

    seasonal_which = str(args.seasonal_which).strip().lower()

    # --- optional printing ---
    if args.print_level_slope:
        raw_times = [s.strip() for s in str(args.times).split(",") if s.strip() != ""]
        times: List[TimeLike] = []
        for rt in raw_times:
            if re.fullmatch(r"\d+", rt):
                times.append(int(rt))
            else:
                times.append(rt)
        plotter.print_level_slope_at(times=times)

    if args.print_static:
        plotter.print_static_params(
            level=args.static_level,
            center=args.static_center,
            digits=int(args.static_digits),
            max_vector_cols=args.static_max_cols,
            include_diagnostics=(not args.static_no_diag),
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

    # seasonal diagnostics
    if not args.skip_seasonal_patterns:
        plotter.figure_seasonal_patterns(
            years=seasonal_years,
            which=seasonal_which,
            save_dir=out_dir,
            show=args.show,
            **seasonal_patterns_kw,
        )
    if not args.skip_seasonal_variance:
        plotter.figure_seasonal_variance(
            which=seasonal_which,
            save_dir=out_dir,
            show=args.show,
            **seasonal_variance_kw,
        )
    if not args.skip_seasonal_heatmap:
        plotter.figure_seasonal_dummies_heatmap(
            which=seasonal_which,
            save_dir=out_dir,
            show=args.show,
            **seasonal_heatmap_kw,
        )
    if not args.skip_seasonal_by_month:
        plotter.figure_seasonal_dummies_by_month(
            which=seasonal_which,
            save_dir=out_dir,
            show=args.show,
            **seasonal_by_month_kw,
        )

    print("[done] plots written.")