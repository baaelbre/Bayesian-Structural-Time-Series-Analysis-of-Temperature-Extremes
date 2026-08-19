"""Static and dynamic regression components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from .base import Array, ComponentSpec, ParamDict


@dataclass
class Regression:
    """Regression coefficients represented as static or random-walk states."""

    n_features: int
    dynamic: bool = False
    name: str = "regression"
    feature_names: tuple[str, ...] | None = None
    initial_mean: tuple[float, ...] | None = None
    initial_sd: float | None = None
    mode: str | None = None
    spec: ComponentSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.n_features) < 1:
            raise ValueError("n_features must be at least one.")
        if self.mode is not None:
            if self.mode not in {"dynamic", "static", "off"}:
                raise ValueError("mode must be 'dynamic', 'static', or 'off'.")
            self.dynamic = self.mode == "dynamic"
        else:
            self.mode = "dynamic" if self.dynamic else "static"
        if self.feature_names is not None:
            self.feature_names = tuple(str(value) for value in self.feature_names)
            if len(self.feature_names) != int(self.n_features):
                raise ValueError("feature_names must have length n_features.")
        if self.initial_mean is not None:
            self.initial_mean = tuple(float(value) for value in self.initial_mean)
            if len(self.initial_mean) != int(self.n_features):
                raise ValueError("initial_mean must have length n_features.")
        dim = 0 if self.mode == "off" else int(self.n_features)
        labels = self.feature_names or tuple(str(j + 1) for j in range(int(self.n_features)))
        names = tuple(f"{self.name}[{label}]" for label in labels) if dim else ()
        self.spec = ComponentSpec(
            name=self.name,
            mode=self.mode,
            state_dim=dim,
            noise_dim=dim if self.dynamic else 0,
            state_names=names,
            noise_names=names if self.dynamic else (),
        )

    @property
    def state_dim(self) -> int:
        return self.spec.state_dim

    @property
    def noise_dim(self) -> int:
        return self.spec.noise_dim

    def _exog(self, exog_t: Optional[Array]) -> Array:
        if exog_t is None:
            raise ValueError(f"Regression '{self.name}' requires exogenous values.")
        values = np.asarray(exog_t, dtype=float).reshape(-1)
        if values.size != int(self.n_features):
            raise ValueError(f"Expected {self.n_features} exogenous values, got {values.size}.")
        return values

    def initial_mean_var(self, params: ParamDict) -> Tuple[Array, Array]:
        if self.mode == "off":
            return np.zeros(0), np.zeros(0)
        default_mean = np.zeros(self.state_dim) if self.initial_mean is None else self.initial_mean
        mean = np.asarray(params.get(f"m0_{self.name}", params.get("m0_reg", default_mean)), dtype=float).reshape(-1)
        default_var = 1.0 if self.initial_sd is None else float(self.initial_sd) ** 2
        variance = np.asarray(
            params.get(f"v0_{self.name}", params.get("v0_reg", np.full(self.state_dim, default_var))),
            dtype=float,
        ).reshape(-1)
        if variance.size == 1:
            variance = np.repeat(variance, self.state_dim)
        if mean.size != self.state_dim or variance.size != self.state_dim:
            raise ValueError("Regression initial values have incompatible dimensions.")
        return mean, variance

    def system_matrices(self, t: int, params: ParamDict):
        if self.mode == "off":
            return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
        transition = np.eye(self.state_dim)
        if not self.dynamic:
            return transition, np.zeros((self.state_dim, 0)), np.zeros((0, 0)), np.zeros(self.state_dim)
        variances = []
        for index, name in enumerate(self.spec.noise_names):
            if f"sd.{name}" in params:
                variances.append(float(params[f"sd.{name}"]) ** 2)
            else:
                raw = np.asarray(params.get("q_reg", np.zeros(self.state_dim)), dtype=float).reshape(-1)
                if raw.size == 1:
                    raw = np.repeat(raw, self.state_dim)
                variances.append(float(raw[index]))
        return transition, np.eye(self.state_dim), np.diag(variances), np.zeros(self.state_dim)

    def design_matrices(self, t: int, params: ParamDict, exog_t=None):
        if self.mode == "off":
            return np.zeros((1, 0)), np.zeros(1)
        return self._exog(exog_t).reshape(1, -1), np.zeros(1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "regression",
            "n_features": int(self.n_features),
            "dynamic": bool(self.dynamic),
            "name": self.name,
            "feature_names": None if self.feature_names is None else list(self.feature_names),
            "initial_mean": None if self.initial_mean is None else list(self.initial_mean),
            "initial_sd": self.initial_sd,
            "mode": self.mode,
        }


RegressionComponent = Regression
