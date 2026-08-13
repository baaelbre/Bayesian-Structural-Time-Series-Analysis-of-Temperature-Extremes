from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .base import Array, ComponentSpec, ParamDict


def _as_1d_exog(exog_t: Optional[Array], n_features: int) -> Array:
    if exog_t is None:
        raise ValueError("RegressionComponent requires exog_t at design time.")
    x = np.asarray(exog_t, dtype=float).reshape(-1)
    if x.size != n_features:
        raise ValueError(f"Expected exog_t of length {n_features}, got {x.size}.")
    return x


@dataclass
class RegressionComponent:
    """Regression component for time-varying exogenous covariates.

    The current implementation supports:
      - mode="static"  : fixed coefficients in ``beta_reg``
      - mode="dynamic" : random-walk coefficients stored in the state
      - mode="off"     : no contribution

    Parameters expected in ``params``
    --------------------------------
    static mode:
      beta_reg : array-like of length n_features

    dynamic mode:
      m0_reg   : array-like of length n_features
      v0_reg   : array-like of length n_features
      q_reg    : scalar or array-like of length n_features
    """

    n_features: int
    mode: str = "static"
    name: str = "regression"
    coefficient_prefix: str = "beta_reg"

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1.")
        if self.mode not in {"dynamic", "static", "off"}:
            raise ValueError("mode must be 'dynamic', 'static', or 'off'.")

        if self.mode == "dynamic":
            names = tuple(f"{self.coefficient_prefix}_{j+1}" for j in range(self.n_features))
            self.spec = ComponentSpec(
                name=self.name,
                mode="dynamic",
                state_dim=self.n_features,
                noise_dim=self.n_features,
                state_names=names,
            )
        else:
            self.spec = ComponentSpec(name=self.name, mode=self.mode, state_dim=0, noise_dim=0)

    def initial_mean_var(self, params: ParamDict) -> Tuple[Array, Array]:
        if self.mode != "dynamic":
            return np.zeros(0), np.zeros(0)
        m0 = np.asarray(params.get("m0_reg", np.zeros(self.n_features)), dtype=float).reshape(-1)
        v0 = np.asarray(params.get("v0_reg", np.ones(self.n_features)), dtype=float).reshape(-1)
        if m0.size != self.n_features or v0.size != self.n_features:
            raise ValueError(f"m0_reg and v0_reg must have length {self.n_features}.")
        if np.any(v0 < 0.0):
            raise ValueError("v0_reg must be >= 0.")
        return m0, v0

    def system_matrices(self, t: int, params: ParamDict):
        if self.mode != "dynamic":
            return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0,))
        q = np.asarray(params.get("q_reg", np.zeros(self.n_features)), dtype=float).reshape(-1)
        if q.size == 1:
            q = np.repeat(q, self.n_features)
        if q.size != self.n_features:
            raise ValueError(f"q_reg must be scalar or length {self.n_features}.")
        if np.any(q < 0.0):
            raise ValueError("q_reg must be >= 0.")
        T = np.eye(self.n_features, dtype=float)
        R = np.eye(self.n_features, dtype=float)
        Q = np.diag(q)
        c = np.zeros(self.n_features, dtype=float)
        return T, R, Q, c

    def design_matrices(self, t: int, params: ParamDict, exog_t: Optional[Array] = None):
        x = _as_1d_exog(exog_t, self.n_features)
        if self.mode == "dynamic":
            return x.reshape(1, self.n_features), np.asarray([0.0], dtype=float)
        if self.mode == "static":
            beta = np.asarray(params.get("beta_reg", np.zeros(self.n_features)), dtype=float).reshape(-1)
            if beta.size != self.n_features:
                raise ValueError(f"beta_reg must have length {self.n_features}.")
            return np.zeros((1, 0), dtype=float), np.asarray([float(x @ beta)], dtype=float)
        return np.zeros((1, 0), dtype=float), np.asarray([0.0], dtype=float)
