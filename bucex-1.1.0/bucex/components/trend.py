from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .base import Array, ComponentSpec, ParamDict


@dataclass
class LocalLinearTrend:
    """Local-linear-trend component.

    Supported modes
    -----------------------
    level_mode:
        - "dynamic": latent level state ``alpha``
        - "static" : deterministic fixed level contribution

    trend_mode:
        - "dynamic": latent slope state ``beta``
        - "static" : deterministic linear trend with coefficient ``fixed_trend``
        - "off"    : no trend contribution
    """

    level_mode: str = "dynamic"
    trend_mode: str = "dynamic"
    initial_level: Optional[float] = None
    initial_slope: Optional[float] = None

    def __post_init__(self) -> None:
        if self.level_mode not in {"dynamic", "static"}:
            raise ValueError("level_mode must be 'dynamic' or 'static'.")
        if self.trend_mode not in {"dynamic", "static", "off"}:
            raise ValueError("trend_mode must be 'dynamic', 'static', or 'off'.")

        names = []
        if self.level_mode == "dynamic":
            names.append("alpha")
        if self.trend_mode == "dynamic":
            names.append("beta")

        mode = "dynamic" if names else "static"
        self.spec = ComponentSpec(
            name="llt",
            mode=mode,
            state_dim=len(names),
            noise_dim=len(names),
            state_names=tuple(names),
        )

    def initial_mean_var(self, params: ParamDict) -> Tuple[Array, Array]:
        if self.spec.state_dim == 0:
            return np.zeros(0), np.zeros(0)

        m0, v0 = [], []
        if self.level_mode == "dynamic":
            m0.append(float(params.get("m0_level", 0.0)))
            v0.append(float(params.get("v0_level", 1.0)))
        if self.trend_mode == "dynamic":
            m0.append(float(params.get("m0_trend", 0.0)))
            v0.append(float(params.get("v0_trend", 1.0)))
        v0 = np.asarray(v0, dtype=float)
        if np.any(v0 < 0.0):
            raise ValueError("Initial variances must be >= 0.")
        return np.asarray(m0, dtype=float), v0

    def system_matrices(self, t: int, params: ParamDict):
        m = self.spec.state_dim
        if m == 0:
            return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0,))

        T = np.eye(m, dtype=float)
        R = np.eye(m, dtype=float)
        c = np.zeros(m, dtype=float)

        q = []
        if self.level_mode == "dynamic":
            q.append(float(params.get("q_level", 0.0)))
        if self.trend_mode == "dynamic":
            q.append(float(params.get("q_trend", 0.0)))
        q = np.asarray(q, dtype=float)
        if np.any(q < 0.0):
            raise ValueError("q_level and q_trend must be >= 0.")
        Q = np.diag(q)

        if self.level_mode == "dynamic":
            ia = 0
            if self.trend_mode == "dynamic":
                ib = 1
                T[ia, ib] = 1.0
            elif self.trend_mode == "static":
                c[ia] = float(params.get("fixed_trend", 0.0))

        return T, R, Q, c

    def design_matrices(self, t: int, params: ParamDict, exog_t: Optional[Array] = None):
        m = self.spec.state_dim
        Z = np.zeros((1, m), dtype=float)
        d = np.zeros((1,), dtype=float)

        if self.level_mode == "dynamic":
            Z[0, 0] = 1.0
        else:
            d[0] += float(params.get("fixed_level", 0.0))

        if self.level_mode != "dynamic" and self.trend_mode == "static":
            d[0] += float(params.get("fixed_trend", 0.0)) * float(t)

        return Z, d
