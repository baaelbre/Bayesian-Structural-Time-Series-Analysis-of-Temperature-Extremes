from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict, Any, Sequence, Tuple
from scipy.stats import norm

# =============================================================================
# Gaussian structural time series with harmonic (Fourier) seasonality — FFT version
# =============================================================================
# Modes
#   level_mode   : {"dynamic","deterministic"}
#   trend_mode   : {"dynamic","deterministic","none"}
#   seasonal_mode: {"dynamic","deterministic","none"}
#
# Seasonality input:
#   (A) Harmonics: m0_cos, m0_sin, (optional) m0_nyq  → overrides dummies
#   (B) Dummies  : season_dummies (length = period) → ALWAYS centered + FFT-projected
#
# Always:
#   • We center provided dummies once (print mean & sum); no extra centering later.
#   • “Seasonal dummies” exposed by the class are the mean-zero period profile.
#   • One global seasonal process variance q_season across all seasonal states.
# =============================================================================

# ------------------------------- config --------------------------------------

@dataclass
class SeasonalSpec:
    period: int
    harmonics: Optional[int] = None     # None -> full K=floor((s-1)/2)
    use_nyquist: Optional[bool] = None  # None -> auto if even s and K>=s/2-1

# ------------------------------- helpers -------------------------------------

def _center_and_report_dummies(dummies: Optional[Sequence[float]], s: int, tol: float = 1e-12) -> np.ndarray:
    """Center to zero mean. Print original mean and sum. Return mean-zero vector (length s)."""
    if dummies is None:
        x = np.zeros(s, float)
        print("[season] No dummies provided → using zeros (mean=0, sum=0).")
        return x
    x = np.asarray(dummies, float)
    if x.size != s:
        raise ValueError(f"season_dummies must have length = period = {s} (got {x.size})")
    m = float(x.mean()); sm = float(x.sum())
    print(f"[season] Provided dummies: mean={m:.6g}, sum={sm:.6g}")
    if abs(sm) > tol:
        print(f"[season][warn] Dummies do not sum to 0 (|sum|>{tol:g}). They will be centered.")
    return x - m

def _dummies_to_harmonics_fft(xc: np.ndarray, K: int, use_nyquist: bool) -> tuple[np.ndarray, np.ndarray, float|None]:
    """
    xc must be mean-zero (already centered).
    Convert to real Fourier coefficients via rfft and map to (cos, sin, nyquist).
    """
    s = int(xc.size)
    even = (s % 2) == 0

    # Real FFT: length s//2 + 1 bins: k=0..s//2 (if even), else 0..floor(s/2)
    F = np.fft.rfft(xc)

    # Decide how many harmonics to keep in 1..K and whether to keep Nyquist (k=s/2)
    kmax = min(K, (s // 2) - 1) if even else min(K, (s - 1) // 2)

    # Zero-out all bins beyond truncation; keep DC=0 for sanity
    keep = np.zeros_like(F, dtype=bool)
    keep[0] = True
    if kmax >= 1:
        keep[1:kmax+1] = True

    nyq_val = None
    if even and use_nyquist:
        keep[s//2] = True

    F_trunc = np.where(keep, F, 0.0)

    # Map rfft bins → (cos, sin) by the identity:
    #   For k>=1 not Nyquist: F[k] = (a_k/2) - i (b_k/2)  ⇒ a_k = 2*Re(F[k]),  b_k = -2*Im(F[k])
    a = np.zeros(K, float)
    b = np.zeros(K, float)
    for k in range(1, K+1):
        if k <= kmax:
            a[k-1] =  2.0 * F_trunc[k].real
            b[k-1] = -2.0 * F_trunc[k].imag
        else:
            a[k-1] = 0.0
            b[k-1] = 0.0

    # Nyquist bin (pure cosine with frequency pi): F[s/2] = nyq/2  (purely real)
    if even and use_nyquist:
        nyq_val = 2.0 * F_trunc[s//2].real

    return a, b, nyq_val

def _harmonics_to_dummies_fft(s: int, c: np.ndarray, sines: np.ndarray, nyq: float | None) -> np.ndarray:
    """Rebuild mean-zero seasonal profile from (cos, sin, nyquist) using irfft (no extra centering)."""
    even = (s % 2) == 0
    F = np.zeros(s//2 + 1, dtype=np.complex128)
    F[0] = 0.0  # DC should be zero for a pure seasonal profile (already centered)

    K = int(len(c))
    kmax = min(K, (s // 2) - 1) if even else min(K, (s - 1) // 2)
    for k in range(1, kmax+1):
        # Inverse mapping: F[k] = (a_k/2) - i (b_k/2)
        F[k] = (c[k-1] / 2.0) - 1j * (sines[k-1] / 2.0)

    if even and nyq is not None:
        F[s//2] = nyq / 2.0

    return np.fft.irfft(F, n=s)

def _round_list(a: Sequence[float], ndigits: int = 2) -> list[float]:
    a = np.asarray(a, float)
    return [float(np.round(x, ndigits)) for x in a]

# --------------------------- main simulator ----------------------------------

class Mean_Time_Series:
    def __init__(
        self,
        # observation noise
        sigma: float = 1.0,
        # component modes
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # seasonal structure
        period: int = 12,
        season_spec: Optional[SeasonalSpec] = None,
        season_harmonics: Optional[int] = None,     # CLI alias
        season_use_nyquist: Optional[bool] = None,  # CLI alias
        # innovations (dynamic components) — ONE global seasonal noise:
        q_level: float = 0.05,
        q_trend: float = 0.01,
        q_season: float = 0.10,   # single scalar variance used for all seasonal states
        # level/trend priors (or fixed values in deterministic modes)
        m0_level: float = 0.0,
        v0_level: float = 1.0,
        m0_trend: float = 0.0,
        v0_trend: float = 1.0,
        # (A) HARMONICS input (preferred)
        m0_cos: Optional[Sequence[float]] = None,
        m0_sin: Optional[Sequence[float]] = None,
        m0_nyq: float = 0.0,
        v0_cos: Optional[Sequence[float]] = None,   # dynamic only
        v0_sin: Optional[Sequence[float]] = None,   # dynamic only
        v0_nyq: float = 1.0,                    # dynamic only, if used
        # (B) DUMMIES input (alternative) — used ONLY if harmonics not provided
        season_dummies: Optional[Sequence[float]] = None,  # length = period
        # calendar / RNG
        start_date: Optional[datetime] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        # ---------- validate & store modes ----------
        assert level_mode in {"dynamic", "deterministic"}
        assert trend_mode in {"dynamic", "deterministic", "none"}
        assert seasonal_mode in {"dynamic", "deterministic", "none"}
        self.level_mode = level_mode
        self.trend_mode = trend_mode
        self.seasonal_mode = seasonal_mode

        self.sigma = float(sigma)
        self.s = int(period)
        if self.s < 2:
            raise ValueError("period must be >= 2")

        # ---------- resolve seasonal spec (K, Nyquist) ----------
        if season_spec is None:
            season_spec = SeasonalSpec(period=self.s,
                                       harmonics=season_harmonics,
                                       use_nyquist=season_use_nyquist)
        else:
            if season_harmonics is not None:
                season_spec.harmonics = season_harmonics
            if season_use_nyquist is not None:
                season_spec.use_nyquist = season_use_nyquist
        self._resolve_season_spec(season_spec)

        self.rng = rng if rng is not None else np.random.default_rng()

        # ---------- level/trend ----------
        self.q_level = float(q_level)
        self.q_trend = float(q_trend)
        self.q_season = float(q_season)  # single scalar

        self.m0_level = float(m0_level); self.v0_level = float(v0_level)
        self.m0_trend = float(m0_trend); self.v0_trend = float(v0_trend)

        # auto-degenerate dynamic → deterministic
        if self.level_mode == "dynamic" and (self.q_level == 0.0 or self.v0_level == 0.0):
            self.level_mode = "deterministic"
        if self.trend_mode == "dynamic" and (self.q_trend == 0.0 or self.v0_trend == 0.0):
            self.trend_mode = "none" if np.isclose(self.m0_trend, 0.0) else "deterministic"

        # ---------- harmonics vs dummies (via FFT) ----------
        K = self.K
        have_harmonics = (m0_cos is not None) and (m0_sin is not None)

        if have_harmonics:
            if len(m0_cos) != K or len(m0_sin) != K:
                raise ValueError(f"m0_cos and m0_sin must have length K={K}")
            self._m0_cos = np.asarray(m0_cos, float)
            self._m0_sin = np.asarray(m0_sin, float)
            self._m0_nyq = float(m0_nyq) if self.use_nyquist else None
            source = "harmonics"
            # For printing: reconstruct via FFT mapping
            self.season_dummies = _harmonics_to_dummies_fft(self.s, self._m0_cos, self._m0_sin, self._m0_nyq)
        else:
            # Use dummies → ALWAYS center, then FFT-map to harmonics
            xc = _center_and_report_dummies(season_dummies, self.s, tol=1e-12)
            c, sines, nyq_val = _dummies_to_harmonics_fft(xc, K=K, use_nyquist=self.use_nyquist)
            self._m0_cos = c
            self._m0_sin = sines
            self._m0_nyq = float(nyq_val) if (nyq_val is not None and self.use_nyquist) else None
            self.season_dummies = _harmonics_to_dummies_fft(self.s, self._m0_cos, self._m0_sin, self._m0_nyq)
            # Sanity check: exact when full K(+Nyquist)
            source = "dummies"

        self._season_dummies_rounded = np.array(_round_list(self.season_dummies, 2), float)

        # rotation caches for dynamic seasonal evolution (pairwise rotations)
        self._omegas = 2.0 * np.pi * (np.arange(1, K + 1, dtype=float)) / float(self.s)
        self._cos = np.cos(self._omegas)
        self._sin = np.sin(self._omegas)

        # ---------- state layout ----------
        layout: list[str] = []
        if self.level_mode == "dynamic":
            layout.append("alpha")
        if self.trend_mode == "dynamic":
            layout.append("beta")
        if self.seasonal_mode == "dynamic":
            for k in range(1, K + 1):
                layout += [f"c{k}", f"s{k}"]
            if self.use_nyquist:
                layout.append("nyq")
        self._layout = layout
        self.n_latent = len(layout)

        # ---------- initial state ----------
        x0_mean, x0_var = [], []
        if self.level_mode == "dynamic":
            x0_mean += [self.m0_level]; x0_var += [self.v0_level]
        if self.trend_mode == "dynamic":
            x0_mean += [self.m0_trend]; x0_var += [self.v0_trend]

        if self.seasonal_mode == "dynamic" and K > 0:
            v0_cos = np.ones(K, float) if v0_cos is None else np.asarray(v0_cos, float)
            v0_sin = np.ones(K, float) if v0_sin is None else np.asarray(v0_sin, float)
            if len(v0_cos) != K or len(v0_sin) != K:
                raise ValueError(f"v0_cos and v0_sin must have length K={K} in dynamic mode")
            self._v0_cos = v0_cos
            self._v0_sin = v0_sin
            self._v0_nyq = float(v0_nyq)

            for k in range(K):
                x0_mean += [self._m0_cos[k], self._m0_sin[k]]
                x0_var  += [self._v0_cos[k],  self._v0_sin[k]]
            if self.use_nyquist:
                x0_mean.append(float(0.0 if self._m0_nyq is None else self._m0_nyq))
                x0_var.append(float(self._v0_nyq))

        self.x_t = np.array([], float) if not x0_mean else \
            self.rng.normal(loc=np.array(x0_mean), scale=np.sqrt(np.array(x0_var)))

        # deterministic proxies for level/trend
        self.fixed_level = self.m0_level if self.level_mode == "deterministic" else None
        if self.trend_mode == "deterministic":
            self.fixed_trend = float(self.m0_trend)
        elif self.trend_mode == "none":
            self.fixed_trend = 0.0
        else:
            self.fixed_trend = None

        # time & paths
        self.t = 0
        self.current_date = start_date if start_date else datetime.now()
        self.index: list[datetime] = []
        self.all_measurements: list[float] = []
        self.mu_path: list[float] = []
        self.alpha_path: list[float] = []
        self.beta_path: list[float] = []
        self.gamma_path: list[float] = []

        # record t=0
        self._record_truth()

        print(f"[season] period={self.s}  K={self.K}  nyquist={self.use_nyquist}  source={source}")
        print(f"[season] dummies = {self._season_dummies_rounded.tolist()}")

        # if dynamic season but q_season ~ 0 → switch to deterministic (no latent seasonal evolution)
        if self.seasonal_mode == "dynamic" and np.isclose(self.q_season, 0.0):
            self.seasonal_mode = "deterministic"

    # ---------------- seasonal spec ----------------
    def _resolve_season_spec(self, spec: SeasonalSpec) -> None:
        s = spec.period
        if spec.harmonics is None:
            K = (s - 1) // 2
        else:
            if not (0 <= spec.harmonics <= (s - 1) // 2):
                raise ValueError(f"harmonics must be in 0..{(s-1)//2}")
            K = int(spec.harmonics)
        even = (s % 2) == 0
        if spec.use_nyquist is None:
            use_nyq = bool(even and K >= (s // 2 - 1))
        else:
            use_nyq = bool(spec.use_nyquist and even)
        self.K = K
        self.use_nyquist = use_nyq

    # ---------------- indices ----------------
    def _idx_alpha(self) -> Optional[int]:
        return self._layout.index("alpha") if "alpha" in self._layout else None

    def _idx_beta(self) -> Optional[int]:
        return self._layout.index("beta") if "beta" in self._layout else None

    def _idx_pair(self, k: int) -> int:
        pos = 0
        if self.level_mode == "dynamic": pos += 1
        if self.trend_mode == "dynamic": pos += 1
        pos += 2 * (k - 1)
        return pos

    def _idx_nyq(self) -> Optional[int]:
        if not self.use_nyquist or self.seasonal_mode != "dynamic":
            return None
        pos = 0
        if self.level_mode == "dynamic": pos += 1
        if self.trend_mode == "dynamic": pos += 1
        pos += 2 * self.K
        return pos

    # ---------------- contributions ----------------
    def _alpha_contribution(self) -> float:
        if self.level_mode == "dynamic":
            i = self._idx_alpha()
            return 0.0 if i is None else float(self.x_t[i])
        base = float(self.fixed_level)
        if self.trend_mode == "dynamic":
            j = self._idx_beta()
            beta_val = 0.0 if j is None else float(self.x_t[j])
            return base + beta_val * self.t
        elif self.trend_mode == "deterministic":
            return base + float(self.fixed_trend) * self.t
        else:
            return base

    def _beta_value_for_path(self) -> float:
        if self.trend_mode == "dynamic":
            j = self._idx_beta()
            return 0.0 if j is None else float(self.x_t[j])
        elif self.trend_mode == "deterministic":
            return float(self.fixed_trend)
        else:
            return 0.0

    def _seasonal_contribution(self) -> float:
        if self.seasonal_mode == "none":
            return 0.0
        if self.seasonal_mode == "deterministic":
            val = 0.0
            for k in range(self.K):
                w = self._omegas[k]
                val += self._m0_cos[k] * np.cos(w * self.t) + self._m0_sin[k] * np.sin(w * self.t)
            if self._m0_nyq is not None:
                val += self._m0_nyq * ((-1.0) ** self.t)
            return float(val)
        # dynamic: observation loads the cosine entry of each pair (+ Nyquist)
        val = 0.0
        for k in range(1, self.K + 1):
            i = self._idx_pair(k)
            val += float(self.x_t[i])
        nyq_idx = self._idx_nyq()
        if nyq_idx is not None:
            val += float(self.x_t[nyq_idx])
        return float(val)

    # ---------------- step & measure ----------------
    def move(self) -> None:
        """Advance time and evolve dynamic states."""
        if self.n_latent:
            new_x = np.array(self.x_t, copy=True)
            # level
            i_alpha = self._idx_alpha()
            if i_alpha is not None:
                drift = 0.0
                if self.trend_mode == "dynamic":
                    j_beta = self._idx_beta()
                    drift = 0.0 if j_beta is None else float(self.x_t[j_beta])
                elif self.trend_mode == "deterministic":
                    drift = float(self.fixed_trend)
                new_x[i_alpha] = self.x_t[i_alpha] + drift + self.rng.normal(0.0, np.sqrt(self.q_level))
            # trend
            j_beta = self._idx_beta()
            if j_beta is not None:
                new_x[j_beta] = self.x_t[j_beta] + self.rng.normal(0.0, np.sqrt(self.q_trend))
            # season
            if self.seasonal_mode == "dynamic":
                q = float(self.q_season)  # ONE scalar
                for k in range(1, self.K + 1):
                    idx = self._idx_pair(k)
                    c, s = float(self.x_t[idx]), float(self.x_t[idx + 1])
                    co, si = float(self._cos[k - 1]), float(self._sin[k - 1])
                    # rotation + isotropic noise (same variance for c and s)
                    c_next =  c * co + s * si + self.rng.normal(0.0, np.sqrt(q))
                    s_next = -c * si + s * co + self.rng.normal(0.0, np.sqrt(q))
                    new_x[idx], new_x[idx + 1] = c_next, s_next
                nyq_idx = self._idx_nyq()
                if nyq_idx is not None:
                    new_x[nyq_idx] = -float(self.x_t[nyq_idx]) + self.rng.normal(0.0, np.sqrt(q))
            self.x_t = new_x
        self.t += 1
        self._record_truth()

    def measure(self) -> float:
        mu = self.mu_path[-1]
        y = norm.rvs(loc=mu, scale=self.sigma, random_state=self.rng)
        self.all_measurements.append(y)
        self.index.append(self._advance_and_get_time())
        return float(y)

    # ---------------- internals ----------------
    def _advance_and_get_time(self) -> datetime:
        dt = self.current_date
        if self.s == 12:
            self.current_date += relativedelta(months=+1)
        elif self.s == 4:
            self.current_date += relativedelta(months=+3)
        else:
            self.current_date += relativedelta(years=+1)
        return dt

    def _record_truth(self) -> None:
        a = self._alpha_contribution()
        b = self._beta_value_for_path()
        g = self._seasonal_contribution()
        self.alpha_path.append(a)
        self.beta_path.append(b)
        self.gamma_path.append(g)
        self.mu_path.append(float(a + g))

    # ---------------- getters ----------------
    def get_truth_paths(self, as_numpy: bool = True) -> Dict[str, Any]:
        to_arr = np.asarray if as_numpy else (lambda x: list(x))
        return {
            "alpha_t": to_arr(self.alpha_path),
            "beta_t":  to_arr(self.beta_path),
            "gamma_t": to_arr(self.gamma_path),
            "mu_t":    to_arr(self.mu_path),
            "index":   list(self.index),
            "season_dummies": np.array(self.season_dummies, float),  # full precision
            "season_dummies_rounded": np.array(self._season_dummies_rounded, float),
            "m0_cos": np.array(self._m0_cos, float),
            "m0_sin": np.array(self._m0_sin, float),
            "m0_nyq": (0.0 if self._m0_nyq is None else float(self._m0_nyq)),
        }

# ---------------------------------------------------------------------------
# CLI (optional demo)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt

    def _parse_date(s: Optional[str]):
        if not s:
            return None
        parts = [int(p) for p in s.split("-")]
        if len(parts) == 1: return datetime(parts[0], 1, 1)
        if len(parts) == 2: return datetime(parts[0], parts[1], 1)
        if len(parts) == 3: return datetime(parts[0], parts[1], parts[2])
        raise ValueError("start-date must be YYYY, YYYY-MM, or YYYY-MM-DD")

    p = argparse.ArgumentParser(description="Gaussian TS with harmonic seasonality (one global seasonal noise).")
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--period", type=int, default=4)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--start-date", type=str, default="2000-01-01")

    p.add_argument("--level-mode", choices=["dynamic", "deterministic"], default="dynamic")
    p.add_argument("--trend-mode", choices=["dynamic", "deterministic", "none"], default="dynamic")
    p.add_argument("--seasonal-mode", choices=["dynamic", "deterministic", "none"], default="deterministic")

    p.add_argument("--q-level", type=float, default=0.05)
    p.add_argument("--q-trend", type=float, default=0.002)
    p.add_argument("--q-season", type=float, default=0.10, help="ONE scalar seasonal process variance")

    p.add_argument("--m0-level", type=float, default=5.0)
    p.add_argument("--v0-level", type=float, default=0.25)
    p.add_argument("--m0-trend", type=float, default=0.015)
    p.add_argument("--v0-trend", type=float, default=0.05)

    p.add_argument("--season-harmonics", type=int, default=None, help="K (pairs), None=full")
    p.add_argument("--season-use-nyquist", type=int, default=None, help="1/0 force (default auto if even period)")

    # A) harmonics (preferred)
    p.add_argument("--m0-cos", type=str, default=None, help="CSV length K (if provided, overrides dummies)")
    p.add_argument("--m0-sin", type=str, default=None, help="CSV length K (if provided, overrides dummies)")
    p.add_argument("--m0-nyquist", type=float, default=0.0)
    p.add_argument("--v0-cos", type=str, default=None, help="CSV length K (dynamic only) or empty for ones")
    p.add_argument("--v0-sin", type=str, default=None, help="CSV length K (dynamic only) or empty for ones")
    p.add_argument("--v0-nyquist", type=float, default=1.0)

    # B) dummies (alternative)
    p.add_argument("--season-dummies", type=str, default="1,1,1,-3", help="CSV of length=period; used if harmonics not provided")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", type=int, default=1)
    p.add_argument("--save-csv", type=str, default="")
    p.add_argument("--print-summary", type=int, default=1)

    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    K_default = (args.season_harmonics if args.season_harmonics is not None else (args.period - 1) // 2)

    # Parse lists (or None) from CSV
    if args.m0_cos is not None:
        args.m0_cos = [float(z) for z in args.m0_cos.split(",")]
    if args.m0_sin is not None:
        args.m0_sin = [float(z) for z in args.m0_sin.split(",")]
    if args.v0_cos is not None:
        args.v0_cos = [float(z) for z in args.v0_cos.split(",")]
    if args.v0_sin is not None:
        args.v0_sin = [float(z) for z in args.v0_sin.split(",")]
    if args.season_dummies is not None:
        args.season_dummies = [float(z) for z in args.season_dummies.split(",")]

    if args.m0_cos is not None and len(args.m0_cos) != K_default:
        raise ValueError(f"--m0-cos must have length K={K_default}")
    if args.m0_sin is not None and len(args.m0_sin) != K_default:
        raise ValueError(f"--m0-sin must have length K={K_default}")
    if args.v0_cos is not None and len(args.v0_cos) != K_default:
        raise ValueError(f"--v0-cos must have length K={K_default}")
    if args.v0_sin is not None and len(args.v0_sin) != K_default:
        raise ValueError(f"--v0-sin must have length K={K_default}")

    mts = Mean_Time_Series(
        sigma=args.sigma,
        level_mode=args.level_mode,
        trend_mode=args.trend_mode,
        seasonal_mode=args.seasonal_mode,
        period=args.period,
        season_harmonics=args.season_harmonics,
        season_use_nyquist=(None if args.season_use_nyquist is None else bool(int(args.season_use_nyquist))),
        q_level=args.q_level,
        q_trend=args.q_trend,
        q_season=args.q_season,
        m0_level=args.m0_level,
        v0_level=args.v0_level,
        m0_trend=(0.0 if args.trend_mode == "none" else args.m0_trend),
        v0_trend=args.v0_trend,
        # A) harmonics (if present)
        m0_cos=args.m0_cos,
        m0_sin=args.m0_sin,
        m0_nyq=args.m0_nyq,
        v0_cos=args.v0_cos,
        v0_sin=args.v0_sin,
        v0_nyq=args.v0_nyq,
        # B) dummies (only used if harmonics are not given)
        season_dummies=args.season_dummies,
        start_date=_parse_date(args.start_date),
        rng=rng,
    )

    # simulate
    y = []
    for _ in range(args.T):
        mts.move()
        y.append(mts.measure())

    truth = mts.get_truth_paths(as_numpy=True)
    y_arr     = np.asarray(y, float)
    mu_arr    = truth["mu_t"][1:1 + args.T]
    alpha_arr = truth["alpha_t"][1:1 + args.T]
    beta_arr  = truth["beta_t"][1:1 + args.T]
    gamma_arr = truth["gamma_t"][1:1 + args.T]
    dates_T   = truth["index"][:args.T]
    season_vec_round = truth["season_dummies_rounded"]

    df = pd.DataFrame({
        "date": dates_T,
        "y_t": y_arr,
        "mu_t": mu_arr,
        "alpha_t": alpha_arr,
        "beta_t": beta_arr,
        "gamma_t": gamma_arr,
    })

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"Saved {len(df)} rows to {args.save_csv}")

    if args.print_summary:
        print("\n--- Summary ---")
        print(f"level={args.level_mode}, trend={args.trend_mode}, season={args.seasonal_mode}")
        print(f"sigma={args.sigma}, q_level={args.q_level}, q_trend={args.q_trend}, q_season={args.q_season}")
        print(f"period={args.period}, K={mts.K}, nyquist={mts.use_nyquist}")
        print(f"seasonal dummies (length {args.period}) = {season_vec_round.tolist()}")
        print(f"date range: {dates_T[0]} .. {dates_T[-1]}")
        print(f"y mean={y_arr.mean():.3f}, sd={y_arr.std(ddof=1):.3f}")

    if args.plot:
        plt.figure(figsize=(10, 4))
        plt.plot(dates_T, y_arr, label=r"$y_t$", linewidth=1.0)
        plt.plot(dates_T, mu_arr, "--", label=r"$\mu_t$", linewidth=1.0)
        plt.title(f"level={args.level_mode}, trend={args.trend_mode}, season={args.seasonal_mode} "
                  f"(K={mts.K}, nyq={mts.use_nyquist})")
        plt.grid(True); plt.legend(); plt.tight_layout()

        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax2[0].plot(dates_T, alpha_arr, linewidth=1.0); ax2[0].set_ylabel(r"$\alpha_t$"); ax2[0].grid(True)
        ax2[1].plot(dates_T, beta_arr,  linewidth=1.0); ax2[1].set_ylabel(r"$\beta_t$");  ax2[1].grid(True)
        ax2[2].plot(dates_T, gamma_arr, linewidth=1.0); ax2[2].set_ylabel(r"$\gamma_t$"); ax2[2].grid(True)
        fig2.suptitle("Truth paths")
        fig2.tight_layout()
        plt.show()
