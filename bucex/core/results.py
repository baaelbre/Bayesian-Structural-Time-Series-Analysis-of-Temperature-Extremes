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
    """
    Generic posterior storage object for full Bayesian fitting.

    This is intentionally lightweight; file IO can remain in io/.
    """
    draws_static: Dict[str, Array] = field(default_factory=dict)
    draws_states: Optional[Array] = None      # (M, T+1, m)
    logpost: Optional[Array] = None           # (M,)
    acceptance: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

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
        if self.draws_states is None:
            return None
        return int(self.draws_states.shape[1] - 1)

    @property
    def state_dim(self) -> Optional[int]:
        if self.draws_states is None:
            return None
        return int(self.draws_states.shape[2])

    def summary_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n_draws": self.n_draws,
            "has_states": self.draws_states is not None,
            "static_keys": list(self.draws_static.keys()),
            "acceptance": dict(self.acceptance),
        }
        if self.draws_states is not None:
            out["n_time"] = self.n_time
            out["state_dim"] = self.state_dim
        out.update(self.meta)
        return out