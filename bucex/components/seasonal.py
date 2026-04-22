from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .base import Array, ComponentSpec, ParamDict


@dataclass
class DummySeasonal:
    """Dummy seasonal component with sum-to-zero closure.

    Dynamic mode uses a state of length ``period - 1`` ordered newest-first.
    Only the first seasonal coordinate receives an innovation.

    Static mode contributes a deterministic seasonal vector through the design
    offset ``season_vec[t]``. The expected length is ``period``.
    """

    period: int = 12
    mode: str = "dynamic"

    def __post_init__(self) -> None:
        if self.period < 2:
            raise ValueError("period must be >= 2.")
        if self.mode not in {"dynamic", "static", "off"}:
            raise ValueError("mode must be 'dynamic', 'static', or 'off'.")

        if self.mode == "dynamic":
            names = tuple(f"g{k}" for k in range(1, self.period))
            self.spec = ComponentSpec(
                name="seasonal",
                mode="dynamic",
                state_dim=self.period - 1,
                noise_dim=1,
                state_names=names,
            )
        else:
            self.spec = ComponentSpec(name="seasonal", mode=self.mode, state_dim=0, noise_dim=0)

    def initial_mean_var(self, params: ParamDict) -> Tuple[Array, Array]:
        if self.mode != "dynamic":
            return np.zeros(0), np.zeros(0)
        m0 = np.asarray(params.get("m0_season", np.zeros(self.period - 1)), dtype=float).reshape(-1)
        v0 = np.asarray(params.get("v0_season", np.ones(self.period - 1)), dtype=float).reshape(-1)
        if m0.size != self.period - 1 or v0.size != self.period - 1:
            raise ValueError(f"m0_season and v0_season must have length {self.period - 1}.")
        if np.any(v0 < 0.0):
            raise ValueError("v0_season must be >= 0.")
        return m0, v0

    def system_matrices(self, t: int, params: ParamDict):
        if self.mode != "dynamic":
            return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0,))

        m = self.period - 1
        q = float(params.get("q_season", 0.0))
        if q < 0.0:
            raise ValueError("q_season must be >= 0.")

        T = np.zeros((m, m), dtype=float)
        T[0, :] = -1.0
        if m > 1:
            T[1:, :-1] = np.eye(m - 1, dtype=float)

        R = np.zeros((m, 1), dtype=float)
        R[0, 0] = 1.0
        Q = np.asarray([[q]], dtype=float)
        c = np.zeros(m, dtype=float)
        return T, R, Q, c

    def design_matrices(self, t: int, params: ParamDict, exog_t: Optional[Array] = None):
        if self.mode == "dynamic":
            Z = np.zeros((1, self.period - 1), dtype=float)
            Z[0, 0] = 1.0
            return Z, np.asarray([0.0], dtype=float)

        if self.mode == "static":
            vec = np.asarray(params.get("season_vec", np.zeros(self.period)), dtype=float).reshape(-1)
            if vec.size != self.period:
                raise ValueError(f"season_vec must have length {self.period}.")
            season = (t - 1) % self.period
            return np.zeros((1, 0), dtype=float), np.asarray([float(vec[season])], dtype=float)

        return np.zeros((1, 0), dtype=float), np.asarray([0.0], dtype=float)
