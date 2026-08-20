"""Dummy seasonal components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from .base import Array, ComponentSpec, ParamDict, process_variance


@dataclass
class DummySeasonal:
    """Sum-to-zero dummy seasonal component with one optional innovation."""

    period: int = 12
    mode: str = "dynamic"
    name: str = "seasonal"
    initial_mean: Optional[Tuple[float, ...]] = None
    initial_sd: Optional[float] = None
    spec: ComponentSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.period) < 2:
            raise ValueError("period must be at least 2.")
        if self.mode not in {"dynamic", "static", "off"}:
            raise ValueError("mode must be 'dynamic', 'static', or 'off'.")
        if self.initial_mean is not None:
            values = tuple(float(value) for value in self.initial_mean)
            if len(values) != int(self.period) - 1 or not np.all(np.isfinite(values)):
                raise ValueError("initial_mean must contain period - 1 finite values.")
            self.initial_mean = values
        dim = 0 if self.mode == "off" else int(self.period) - 1
        names = tuple(f"{self.name}[{j + 1}]" for j in range(dim))
        self.spec = ComponentSpec(
            name="seasonal",
            mode=self.mode,
            state_dim=dim,
            noise_dim=int(self.mode == "dynamic"),
            state_names=names,
            noise_names=(self.name,) if self.mode == "dynamic" else (),
        )

    @property
    def state_dim(self) -> int:
        return self.spec.state_dim

    @property
    def noise_dim(self) -> int:
        return self.spec.noise_dim

    def initial_mean_var(self, params: ParamDict) -> Tuple[Array, Array]:
        if self.mode == "off":
            return np.zeros(0), np.zeros(0)
        default_mean = np.zeros(self.state_dim) if self.initial_mean is None else self.initial_mean
        mean = np.asarray(params.get("m0_season", default_mean), dtype=float).reshape(-1)
        default_var = 1.0 if self.initial_sd is None else float(self.initial_sd) ** 2
        variance = np.asarray(
            params.get("v0_season", np.full(self.state_dim, default_var)),
            dtype=float,
        ).reshape(-1)
        if mean.size != self.state_dim or variance.size != self.state_dim:
            raise ValueError("Seasonal initial values must have period - 1 entries.")
        if np.any(variance < 0.0):
            raise ValueError("Seasonal initial variances must be non-negative.")
        return mean, variance

    def system_matrices(self, t: int, params: ParamDict):
        if self.mode == "off":
            return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
        dim = self.state_dim
        transition = np.zeros((dim, dim))
        transition[0, :] = -1.0
        if dim > 1:
            transition[1:, :-1] = np.eye(dim - 1)
        if self.mode == "dynamic":
            loading = np.zeros((dim, 1))
            loading[0, 0] = 1.0
            q = process_variance(
                params,
                noise_name=self.name,
                legacy_key="q_season",
            )
            covariance = np.asarray([[q]])
        else:
            loading = np.zeros((dim, 0))
            covariance = np.zeros((0, 0))
        return transition, loading, covariance, np.zeros(dim)

    def design_matrices(self, t: int, params: ParamDict, exog_t=None):
        if self.mode == "off":
            return np.zeros((1, 0)), np.zeros(1)
        design = np.zeros((1, self.state_dim))
        design[0, 0] = 1.0
        return design, np.zeros(1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "dummy_seasonal",
            "period": int(self.period),
            "mode": self.mode,
            "name": self.name,
            "initial_mean": None if self.initial_mean is None else list(self.initial_mean),
            "initial_sd": self.initial_sd,
        }
