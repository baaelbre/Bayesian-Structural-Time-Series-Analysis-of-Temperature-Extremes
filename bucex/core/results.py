from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

Array = np.ndarray


@dataclass
class FilterResult:
    """
    Output of a Gaussian Kalman filter.

    Time indexing convention
    ------------------------
    Arrays indexed by state time have length T+1:
      - index 0 stores the prior/filter at x_0
      - index t stores objects for x_t, t=1,...,T

    Arrays indexed by observations have length T:
      - row t-1 corresponds to observation y_t
    """
    y: Array                       # (T, p)
    m0: Array                      # (m,)
    P0: Array                      # (m, m)

    m_pred: Array                  # (T+1, m), predicted means; m_pred[t] = E[x_t | y_1:t-1]
    P_pred: Array                  # (T+1, m, m)

    m_filt: Array                  # (T+1, m), filtered means;  m_filt[t] = E[x_t | y_1:t]
    P_filt: Array                  # (T+1, m, m)

    loglik: float

    innovations: Optional[Array] = None       # (T, p)
    innovation_cov: Optional[Array] = None    # (T, p, p)
    kalman_gain: Optional[Array] = None       # (T, m, p)
    missing: Optional[Array] = None           # (T,), True if update skipped because y_t had NaN

    # Stored system/design sequence, useful for smoothing / FFBS
    T_seq: Optional[Array] = None             # (T+1, m, m), slot 0 unused
    c_seq: Optional[Array] = None             # (T+1, m)
    Z_seq: Optional[Array] = None             # (T+1, p, m), slot 0 unused
    d_seq: Optional[Array] = None             # (T+1, p)
    H_seq: Optional[Array] = None             # (T+1, p, p), slot 0 unused

    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_time(self) -> int:
        return int(self.y.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.m_filt.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.y.shape[1])


@dataclass
class SmootherResult:
    """
    Output of a Rauch-Tung-Striebel (RTS) smoother.
    """
    m_smooth: Array                # (T+1, m)
    P_smooth: Array                # (T+1, m, m)
    smoother_gain: Optional[Array] = None     # (T, m, m), gain from x_t to x_{t+1}
    lag_cov: Optional[Array] = None           # optional Cov(x_t, x_{t+1} | y_1:T)
    filter_result: Optional[FilterResult] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_time(self) -> int:
        return int(self.m_smooth.shape[0] - 1)

    @property
    def state_dim(self) -> int:
        return int(self.m_smooth.shape[1])


@dataclass
class StateSample:
    """
    One sampled latent trajectory, typically from FFBS.
    """
    x: Array                       # (T+1, m)
    filter_result: Optional[FilterResult] = None
    smoother_result: Optional[SmootherResult] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_time(self) -> int:
        return int(self.x.shape[0] - 1)

    @property
    def state_dim(self) -> int:
        return int(self.x.shape[1])


@dataclass
class PosteriorBundle:
    """Posterior draws plus the information needed for analysis and plotting.

    v0.2 deliberately keeps the fitted model, observations, time index, state
    names and data transformation with the draws. This turns a fit into a useful
    analysis object rather than a bare array container.
    """

    draws_static: Dict[str, Array] = field(default_factory=dict)
    draws_states: Optional[Array] = None      # (M, T+1, m)
    logpost: Optional[Array] = None           # (M,)
    acceptance: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    y: Optional[Array] = None
    dates: Optional[Array] = None
    model: Any = None
    state_names: Optional[tuple[str, ...]] = None
    series_name: Optional[str] = None
    transform_sign: float = 1.0

    @property
    def n_draws(self) -> int:
        if self.logpost is not None:
            return int(len(self.logpost))
        if self.draws_states is not None:
            return int(self.draws_states.shape[0])
        if self.draws_static:
            first = next(iter(self.draws_static.values()))
            return int(len(first))
        return 0

    @property
    def n_time(self) -> Optional[int]:
        if self.draws_states is not None:
            return int(self.draws_states.shape[1] - 1)
        if self.y is not None:
            return int(np.asarray(self.y).size)
        return None

    @property
    def state_dim(self) -> Optional[int]:
        if self.draws_states is None:
            return None
        return int(self.draws_states.shape[2])

    @property
    def obs_name(self) -> Optional[str]:
        spec = getattr(getattr(self.model, "obs", None), "spec", None)
        return getattr(spec, "name", None)

    @property
    def time(self) -> Array:
        if self.dates is not None:
            return np.asarray(self.dates)
        n = self.n_time or 0
        return np.arange(1, n + 1)

    def _resolved_state_names(self) -> tuple[str, ...]:
        if self.state_names is not None:
            return tuple(self.state_names)
        names = getattr(self.model, "state_names", None)
        if names is not None:
            return tuple(names)
        if self.state_dim is None:
            return tuple()
        return tuple(f"state_{i}" for i in range(self.state_dim))

    def state_draws(self, name: str, *, original_scale: bool = True) -> Array:
        """Return posterior draws of one centred structural state for times 1:T."""
        if self.draws_states is None:
            raise ValueError("This posterior does not contain state draws.")
        names = self._resolved_state_names()
        if name not in names:
            raise KeyError(f"Unknown state '{name}'. Available states: {names}")
        out = np.asarray(self.draws_states[:, 1:, names.index(name)], dtype=float)
        if original_scale and name in {"alpha", "beta"} | {n for n in names if n.startswith("g")}:
            out = float(self.transform_sign) * out
        return out

    def mu_draws(self, *, original_scale: bool = False) -> Array:
        """Posterior draws of the observation-location trajectory."""
        if self.draws_states is None:
            raise ValueError("This posterior does not contain state draws.")
        names = self._resolved_state_names()
        if "alpha" not in names:
            raise ValueError("The current high-level mu extraction requires an 'alpha' state.")
        mu = np.asarray(self.draws_states[:, 1:, names.index("alpha")], dtype=float).copy()
        if "g1" in names:
            mu += np.asarray(self.draws_states[:, 1:, names.index("g1")], dtype=float)
        if original_scale:
            mu *= float(self.transform_sign)
        return mu

    def posterior_summary(
        self,
        values: Array,
        *,
        credible_interval: float = 0.90,
        axis: int = 0,
    ) -> dict[str, Array]:
        alpha = 1.0 - float(credible_interval)
        q = [alpha / 2.0, 0.5, 1.0 - alpha / 2.0]
        low, median, high = np.quantile(np.asarray(values, dtype=float), q, axis=axis)
        return {"lower": low, "median": median, "upper": high}

    def _monthly_event_probability_draws(self, threshold: float) -> Array:
        """Probability of the relevant tail event on the original data scale.

        For ordinary maxima this is ``P(Y_t > threshold)``. For a minima series
        fitted through ``Z_t=-Y_t`` this is ``P(Y_t < threshold)``.
        """
        if self.obs_name != "gev":
            raise ValueError("Tail-risk calculations currently require a GEV fit.")
        from scipy.stats import genextreme

        mu_model = self.mu_draws(original_scale=False)
        sigma = np.asarray(self.draws_static["sigma"], dtype=float)[:, None]
        xi = np.asarray(self.draws_static["xi"], dtype=float)[:, None]
        threshold_model = float(self.transform_sign) * float(threshold)
        cdf = genextreme.cdf(threshold_model, c=-xi, loc=mu_model, scale=sigma)
        return np.clip(1.0 - cdf, 0.0, 1.0)

    def _annual_groups(self) -> tuple[list[Array], Array]:
        n = self.n_time or 0
        if n == 0:
            return [], np.asarray([])
        if self.dates is not None:
            try:
                import pandas as pd

                dt = pd.to_datetime(np.asarray(self.dates))
                years = np.asarray(dt.year)
                unique = np.unique(years)
                return [np.flatnonzero(years == year) for year in unique], unique
            except Exception:
                pass
        period = int(self.meta.get("period", 12))
        groups = [np.arange(i, min(i + period, n)) for i in range(0, n, period)]
        return groups, np.arange(1, len(groups) + 1)

    def exceedance_probability_draws(
        self,
        threshold: float,
        *,
        annual: bool = False,
    ) -> tuple[Array, Array]:
        """Return posterior risk trajectories and their time labels."""
        p = self._monthly_event_probability_draws(threshold)
        if not annual:
            return p, self.time
        groups, labels = self._annual_groups()
        annual_p = np.column_stack(
            [1.0 - np.prod(1.0 - p[:, idx], axis=1) for idx in groups]
        )
        return np.clip(annual_p, 0.0, 1.0), labels

    def return_period_draws(
        self,
        threshold: float,
        *,
        annual: bool = True,
        min_probability: float = 1e-12,
    ) -> tuple[Array, Array]:
        p, labels = self.exceedance_probability_draws(threshold, annual=annual)
        return 1.0 / np.maximum(p, min_probability), labels

    def endpoint_draws(self, *, original_scale: bool = True) -> Array:
        """Posterior GEV endpoint trajectories.

        For maxima, the finite endpoint is an upper endpoint. For negated minima,
        transforming back gives the corresponding lower endpoint.
        """
        if self.obs_name != "gev":
            raise ValueError("Endpoint trajectories require a GEV fit.")
        mu = self.mu_draws(original_scale=False)
        sigma = np.asarray(self.draws_static["sigma"], dtype=float)[:, None]
        xi = np.asarray(self.draws_static["xi"], dtype=float)[:, None]
        endpoint = np.where(xi < 0.0, mu - sigma / xi, np.inf)
        if original_scale:
            endpoint = float(self.transform_sign) * endpoint
        return endpoint

    def event_label(self, threshold: float) -> str:
        if float(self.transform_sign) < 0:
            return f"P({self.series_name or 'Y'} < {threshold:g})"
        return f"P({self.series_name or 'Y'} > {threshold:g})"

    def level_rate_draws(
        self,
        start_year: int,
        end_year: int,
        *,
        scale: str | float = "decade",
    ) -> Array:
        """Posterior finite-change rate from the latent level trajectory.

        The latent level is averaged within the first and last calendar years
        and the difference is divided by elapsed time. Unlike the instantaneous
        slope state, this summary includes all changes in the fitted level and
        is therefore robust to the allocation between level and slope
        innovations.
        """

        if self.dates is None:
            raise ValueError("level_rate_draws requires calendar dates.")
        if int(end_year) <= int(start_year):
            raise ValueError("Require end_year > start_year.")
        import pandas as pd

        years = np.asarray(pd.to_datetime(np.asarray(self.dates)).year, dtype=int)
        start_idx = np.flatnonzero(years == int(start_year))
        end_idx = np.flatnonzero(years == int(end_year))
        if start_idx.size == 0 or end_idx.size == 0:
            available = (int(years.min()), int(years.max()))
            raise ValueError(
                f"Requested years are unavailable; fitted range is {available[0]}-{available[1]}."
            )

        level = self.state_draws("alpha", original_scale=True)
        start_level = np.mean(level[:, start_idx], axis=1)
        end_level = np.mean(level[:, end_idx], axis=1)
        annual_rate = (end_level - start_level) / float(end_year - start_year)

        if isinstance(scale, (int, float)):
            multiplier = float(scale)
        else:
            key = str(scale).lower()
            if key in {"year", "annual"}:
                multiplier = 1.0
            elif key in {"decade", "decadal"}:
                multiplier = 10.0
            elif key in {"century", "centennial"}:
                multiplier = 100.0
            else:
                raise ValueError(
                    "scale must be 'year', 'decade', 'century', or a numeric multiplier."
                )
        return multiplier * annual_rate

    def period_rate_draws(
        self,
        periods: Dict[str, tuple[int, int]],
        *,
        scale: str | float = "decade",
    ) -> Dict[str, Array]:
        """Posterior level-change rates for several named periods."""

        return {
            str(label): self.level_rate_draws(start, end, scale=scale)
            for label, (start, end) in periods.items()
        }

    def period_rate_summary(
        self,
        periods: Dict[str, tuple[int, int]],
        *,
        scale: str | float = "decade",
        credible_interval: float = 0.90,
    ):
        """Summarize finite-period rates and positive-change probabilities."""

        import pandas as pd

        rows = []
        for label, values in self.period_rate_draws(periods, scale=scale).items():
            summary = self.posterior_summary(
                values, credible_interval=credible_interval, axis=0
            )
            rows.append(
                {
                    "period": label,
                    "lower": float(summary["lower"]),
                    "median": float(summary["median"]),
                    "upper": float(summary["upper"]),
                    "probability_positive": float(np.mean(values > 0.0)),
                }
            )
        return pd.DataFrame(rows).set_index("period")

    def rate_contrast_draws(
        self,
        recent: tuple[int, int],
        reference: tuple[int, int],
        *,
        scale: str | float = "decade",
    ) -> Array:
        """Posterior difference between two finite-period level-change rates."""

        recent_draws = self.level_rate_draws(*recent, scale=scale)
        reference_draws = self.level_rate_draws(*reference, scale=scale)
        return recent_draws - reference_draws

    def rate_contrast_summary(
        self,
        recent: tuple[int, int],
        reference: tuple[int, int],
        *,
        scale: str | float = "decade",
        credible_interval: float = 0.90,
    ) -> Dict[str, float]:
        """Summarize acceleration as a contrast of finite-period rates."""

        values = self.rate_contrast_draws(recent, reference, scale=scale)
        summary = self.posterior_summary(
            values, credible_interval=credible_interval, axis=0
        )
        return {
            "lower": float(summary["lower"]),
            "median": float(summary["median"]),
            "upper": float(summary["upper"]),
            "probability_positive": float(np.mean(values > 0.0)),
        }

    def component_probabilities(self):
        """Posterior zero/fixed/dynamic probabilities for SSVS components."""
        import pandas as pd

        labels = {0: "zero", 1: "fixed", 2: "dynamic"}
        rows = []
        for component in ("level", "trend", "season"):
            key = f"state_{component}"
            if key not in self.draws_static:
                continue
            values = np.asarray(self.draws_static[key], dtype=int).reshape(-1)
            row = {"component": component}
            for code, label in labels.items():
                row[label] = float(np.mean(values == code))
            rows.append(row)
        if not rows:
            raise ValueError("This fit does not contain structural SSVS draws.")
        return pd.DataFrame(rows).set_index("component")

    def component_transition_summary(self):
        """Transition counts and switching rates for SSVS component states."""

        import pandas as pd

        labels = {0: "zero", 1: "fixed", 2: "dynamic"}
        rows = []
        for component in ("level", "trend", "season"):
            key = f"state_{component}"
            if key not in self.draws_static:
                continue
            values = np.asarray(self.draws_static[key], dtype=int).reshape(-1)
            changes = int(np.sum(values[1:] != values[:-1])) if values.size > 1 else 0
            rows.append(
                {
                    "component": component,
                    "n_draws": int(values.size),
                    "n_switches": changes,
                    "switch_rate": float(changes / max(values.size - 1, 1)),
                    "first_state": labels[int(values[0])],
                    "last_state": labels[int(values[-1])],
                }
            )
        if not rows:
            raise ValueError("This fit does not contain structural SSVS draws.")
        return pd.DataFrame(rows).set_index("component")

    def structural_model_probabilities(self):
        """Posterior probabilities of the joint structural specifications."""
        import pandas as pd

        keys = ["state_level", "state_trend", "state_season"]
        if not all(key in self.draws_static for key in keys):
            raise ValueError("This fit does not contain structural SSVS draws.")
        values = np.column_stack(
            [np.asarray(self.draws_static[key], dtype=int).reshape(-1) for key in keys]
        )
        unique, counts = np.unique(values, axis=0, return_counts=True)
        labels = {0: "zero", 1: "fixed", 2: "dynamic"}
        rows = []
        for state, count in zip(unique, counts):
            rows.append(
                {
                    "level": labels[int(state[0])],
                    "trend": labels[int(state[1])],
                    "season": labels[int(state[2])],
                    "probability": float(count / values.shape[0]),
                }
            )
        return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)

    def most_probable_structure(self) -> Dict[str, Any]:
        """Return the maximum-posterior joint SSVS structure."""
        table = self.structural_model_probabilities()
        row = table.iloc[0]
        return {
            "level": str(row["level"]),
            "trend": str(row["trend"]),
            "season": str(row["season"]),
            "probability": float(row["probability"]),
        }

    def summary_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "series": self.series_name,
            "n_draws": self.n_draws,
            "n_time": self.n_time,
            "state_dim": self.state_dim,
            "state_names": self._resolved_state_names(),
            "observation_model": self.obs_name,
            "transform_sign": self.transform_sign,
            "static_keys": list(self.draws_static.keys()),
            "acceptance": dict(self.acceptance),
        }
        out.update({k: v for k, v in self.meta.items() if k != "draws_states_ncp"})
        return out

    def static_summary(self, credible_interval: float = 0.90) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for key, values in self.draws_static.items():
            arr = np.asarray(values)
            if arr.ndim != 1:
                continue
            s = self.posterior_summary(arr, credible_interval=credible_interval)
            out[key] = {name: float(value) for name, value in s.items()}
        return out

    def plot(self, type: str = "level", **kwargs):
        """High-level plotting method; see :func:`bucex.plot`."""
        from ..plotting import plot

        return plot(self, type=type, **kwargs)

    def save(self, path: str) -> None:
        """Save the complete fitted object with Python pickle."""
        import pickle
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "PosteriorBundle":
        import pickle
        from pathlib import Path

        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError("The file does not contain a PosteriorBundle.")
        return obj

