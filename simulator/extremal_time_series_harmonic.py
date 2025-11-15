from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Sequence, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme
from dateutil.relativedelta import relativedelta

@dataclass
class SeasonalSpec:
    period: int
    harmonics: Optional[int] = None     # None -> full K=floor((s-1)/2)
    use_nyquist: Optional[bool] = None  # None -> auto if even s and K>=s/2-1
    
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

class Extremal_Time_Series:
    """
    Structural DGEV simulator with harmonic (Fourier) seasonality.

    Modes
    -----
      level_mode      ∈ {"dynamic","deterministic"}        (level cannot be 'none')
      trend_mode      ∈ {"dynamic","deterministic","none"}
      seasonal_mode   ∈ {"dynamic","deterministic","none"}

    Observation
    -----------
      y_t ~ GEV(mu_t, σ, ξ), using SciPy's genextreme with shape c = -ξ.

    Location decomposition
    ----------------------
      mu_t = (level + linear trend contribution) + (seasonal contribution)

    Seasonality input (harmonic FFT version)
    ----------------------------------------
      (A) Harmonics: m0_cos, m0_sin, (optional) m0_nyq  → overrides dummies
      (B) Dummies  : season_dummies (length = period) → ALWAYS centered + FFT-projected

    Always:
      • We center provided dummies once; no extra centering later.
      • “season_dummies” exposed by the class are the mean-zero period profile.
      • One global seasonal process variance q_season across all seasonal states.
    """

    def __init__(
        self,
        # GEV scale and shape
        parameters: Tuple[float, float] = (1.0, 0.1),
        # component modes
        level_mode: str = "dynamic",
        trend_mode: str = "dynamic",
        seasonal_mode: str = "dynamic",
        # seasonal structure
        period: int = 12,
        season_spec: Optional[SeasonalSpec] = None,
        season_harmonics: Optional[int] = None,     # alias
        season_use_nyquist: Optional[bool] = None,  # alias
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
        v0_nyq: float = 1.0,                        # dynamic only, if used
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

        self.sigma, self.xi = float(parameters[0]), float(parameters[1])

        self.s = int(period)
        if self.s < 2:
            raise ValueError("period must be >= 2")

        # ---------- resolve seasonal spec (K, Nyquist) ----------
        if season_spec is None:
            season_spec = SeasonalSpec(
                period=self.s,
                harmonics=season_harmonics,
                use_nyquist=season_use_nyquist,
            )
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

        # auto-degenerate dynamic → deterministic / none
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
            self.season_dummies = _harmonics_to_dummies_fft(
                self.s, self._m0_cos, self._m0_sin, self._m0_nyq
            )
        else:
            # Use dummies → ALWAYS center, then FFT-map to harmonics
            xc = _center_and_report_dummies(season_dummies, self.s, tol=1e-12)
            c, sines, nyq_val = _dummies_to_harmonics_fft(
                xc, K=K, use_nyquist=self.use_nyquist
            )
            self._m0_cos = c
            self._m0_sin = sines
            self._m0_nyq = float(nyq_val) if (nyq_val is not None and self.use_nyquist) else None
            self.season_dummies = _harmonics_to_dummies_fft(
                self.s, self._m0_cos, self._m0_sin, self._m0_nyq
            )
            source = "dummies"

        self._season_dummies_rounded = np.array(
            _round_list(self.season_dummies, 2), float
        )

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

        print(f"[extreme-season] period={self.s}  K={self.K}  nyquist={self.use_nyquist}  source={source}")
        print(f"[extreme-season] dummies = {self._season_dummies_rounded.tolist()}")

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
        """Level + linear trend contribution to μ_t."""
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
            # Deterministic harmonic seasonal curve via fixed m0_cos/m0_sin/m0_nyq
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
        """Draw y_t ~ GEV(mu_t, sigma, xi) using last recorded mu_t."""
        mu = self.mu_path[-1]
        # SciPy genextreme uses shape c = -xi
        y = genextreme.rvs(-self.xi, loc=mu, scale=self.sigma, random_state=self.rng)
        self.all_measurements.append(float(y))
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
        """
        Returns:
            alpha:   path of alpha contribution to mu (float array)
            beta:    path of beta value (float array)
            gamma:   path of seasonal contribution
            mu:      path of mu_t
            index:   list of datetimes
            (plus seasonal metadata similar to Mean_Time_Series)
        """
        to_arr = np.asarray if as_numpy else (lambda x: list(x))
        out = {
            "alpha": to_arr(self.alpha_path),
            "beta":  to_arr(self.beta_path),
            "gamma": to_arr(self.gamma_path),
            "mu":    to_arr(self.mu_path),
            "index": list(self.index),
        }
        # optional extras, harmless for old code
        out.update({
            "season_dummies": np.array(self.season_dummies, float),
            "season_dummies_rounded": np.array(self._season_dummies_rounded, float),
            "m0_cos": np.array(self._m0_cos, float),
            "m0_sin": np.array(self._m0_sin, float),
            "m0_nyq": (0.0 if self._m0_nyq is None else float(self._m0_nyq)),
        })
        return out


# -------------------------------
# Generate and save all 18 combos
# -------------------------------
if __name__ == "__main__":
    # Reproducibility
    rng = np.random.default_rng(123)

    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    outdir = os.path.join(base_dir, "simulated_extremal_series_harmonic")
    os.makedirs(outdir, exist_ok=True)

    T = 240
    SIGMA_XI = (8.0, 0.1)
    period = 12

    # Full seasonal shape for convenience (length = period)
    seas_vec_full = [
        np.cos(2 * np.pi * k / period) + 0.25 * np.cos(4 * np.pi * k / period)
        for k in range(period)
    ]

    K_default = (period - 1) // 2

    level_grid    = ["dynamic", "deterministic"]               # 2
    trend_grid    = ["dynamic", "deterministic", "none"]       # 3
    seasonal_grid = ["dynamic", "deterministic", "none"]       # 3  -> 18

    run_id = 0
    for lev in level_grid:
        for tr in trend_grid:
            for seas in seasonal_grid:
                run_id += 1

                # Seasonal priors per mode (only needed in dynamic mode)
                if seas == "dynamic":
                    v0_cos = [0.5] * K_default
                    v0_sin = [0.5] * K_default
                    q_season = 0.15
                else:
                    v0_cos = None
                    v0_sin = None
                    q_season = 0.15  # ignored for det/none

                ets = Extremal_Time_Series(
                    parameters=SIGMA_XI,
                    level_mode=lev,
                    trend_mode=tr,
                    seasonal_mode=seas,
                    period=period,
                    season_harmonics=None,             # full set by default
                    season_use_nyquist=None,           # auto when even period
                    q_level=0.05,
                    q_trend=0.02,
                    q_season=q_season,
                    m0_level=5.5,
                    v0_level=0.25,
                    m0_trend=(0.012 if tr != "none" else 0.0),
                    v0_trend=0.05,
                    # Dummies → FFT → harmonics
                    season_dummies=seas_vec_full,
                    v0_cos=v0_cos,
                    v0_sin=v0_sin,
                    v0_nyq=0.5,
                    start_date=datetime(2000, 1, 1),
                    rng=rng,
                )

                # Simulate exactly T observations
                y = []
                for _ in range(T):
                    ets.move()
                    y.append(ets.measure())

                truths = ets.get_truth_paths(as_numpy=False)
                # Shift by one to align with measurements
                mu_T    = np.array(truths["mu"][1:1 + T], dtype=float)
                alpha_T = np.array(truths["alpha"][1:1 + T], dtype=float)
                beta_T  = np.array(truths["beta"][1:1 + T], dtype=float)
                gamma_T = np.array(truths["gamma"][1:1 + T], dtype=float)
                dates_T = truths["index"][:T]
                y_T     = np.array(y, dtype=float)

                tag = f"case_{run_id:02d}_level-{lev}_trend-{tr}_season-{seas}"
                csv_path = os.path.join(outdir, f"{tag}.csv")
                png_path = os.path.join(outdir, f"{tag}.png")

                df = pd.DataFrame({
                    "date": [d.isoformat() for d in dates_T],
                    "y": y_T,
                    "mu": mu_T,
                    "alpha": alpha_T,
                    "beta": beta_T,
                    "gamma": gamma_T,
                })
                df.to_csv(csv_path, index=False)

                # Quick plot: y vs mu
                plt.figure(figsize=(11, 3.8))
                plt.plot(y_T, label="y_t", alpha=0.75)
                plt.plot(mu_T, "--", label="mu_t")
                plt.title(f"{tag} (harmonic seasonality)")
                plt.xlabel("t")
                plt.ylabel("value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(png_path, dpi=150)
                plt.close()

                # Component plots
                png_comp_path = os.path.join(outdir, f"{tag}_components.png")
                fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
                ax[0].plot(alpha_T); ax[0].set_title("alpha (intercept + trend contribution)")
                ax[1].plot(beta_T);  ax[1].set_title("beta (slope)")
                ax[2].plot(gamma_T); ax[2].set_title("gamma (seasonal contribution)")
                for a in ax: a.grid(True)
                plt.tight_layout()
                plt.savefig(png_comp_path, dpi=150)
                plt.close()

    print(f"Saved 18 simulated harmonic extremal series (CSV + PNG + components) to: {outdir}")
