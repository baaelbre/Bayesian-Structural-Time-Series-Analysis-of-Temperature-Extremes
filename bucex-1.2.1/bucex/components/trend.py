"""Level and trend components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from .base import Array, ComponentSpec, ParamDict, process_variance


@dataclass
class LocalLevel:
    """Random-walk level; ``mode='static'`` fixes its process SD at zero."""

    name: str = "level"
    mode: str = "dynamic"
    initial_mean: float | None = None
    initial_sd: float | None = None
    spec: ComponentSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"dynamic", "static"}:
            raise ValueError("LocalLevel.mode must be 'dynamic' or 'static'.")
        self.spec = ComponentSpec(
            name="level",
            mode=self.mode,
            state_dim=1,
            noise_dim=int(self.mode == "dynamic"),
            state_names=(self.name,),
            noise_names=(self.name,) if self.mode == "dynamic" else (),
        )

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def noise_dim(self) -> int:
        return self.spec.noise_dim

    def initial_mean_var(self, params: ParamDict) -> Tuple[Array, Array]:
        mean = float(
            params.get(
                "m0_level",
                0.0 if self.initial_mean is None else self.initial_mean,
            )
        )
        default_var = 1.0 if self.initial_sd is None else float(self.initial_sd) ** 2
        variance = float(params.get("v0_level", default_var))
        if variance < 0.0:
            raise ValueError("Initial level variance must be non-negative.")
        return np.asarray([mean]), np.asarray([variance])

    def system_matrices(self, t: int, params: ParamDict):
        if self.mode == "dynamic":
            q = process_variance(
                params,
                noise_name=self.name,
                legacy_key="q_level",
            )
            return np.ones((1, 1)), np.ones((1, 1)), np.asarray([[q]]), np.zeros(1)
        return np.ones((1, 1)), np.zeros((1, 0)), np.zeros((0, 0)), np.zeros(1)

    def design_matrices(self, t: int, params: ParamDict, exog_t=None):
        return np.ones((1, 1)), np.zeros(1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "local_level",
            "name": self.name,
            "mode": self.mode,
            "initial_mean": self.initial_mean,
            "initial_sd": self.initial_sd,
        }


@dataclass
class LocalLinearTrend:
    """Local level plus optional local slope.

    The default semantic state names are ``level`` and ``slope`` for every
    parameterization.  The FS implementation still uses its traditional
    ``alpha0``, ``beta0`` and signed ``s_*`` regression coefficients
    internally; those names no longer leak into the state API.
    """

    level_mode: str = "dynamic"
    trend_mode: str = "dynamic"
    level_name: str = "level"
    slope_name: str = "slope"
    initial_level: Optional[float] = None
    initial_slope: float = 0.0
    initial_level_sd: Optional[float] = None
    initial_slope_sd: Optional[float] = None
    spec: ComponentSpec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.level_mode not in {"dynamic", "static"}:
            raise ValueError("level_mode must be 'dynamic' or 'static'.")
        if self.trend_mode not in {"dynamic", "static", "off"}:
            raise ValueError("trend_mode must be 'dynamic', 'static', or 'off'.")
        state_names = [self.level_name]
        if self.trend_mode != "off":
            state_names.append(self.slope_name)
        noise_names = []
        if self.level_mode == "dynamic":
            noise_names.append(self.level_name)
        if self.trend_mode == "dynamic":
            noise_names.append(self.slope_name)
        self.spec = ComponentSpec(
            name="trend",
            mode="dynamic" if noise_names else "static",
            state_dim=len(state_names),
            noise_dim=len(noise_names),
            state_names=tuple(state_names),
            noise_names=tuple(noise_names),
        )

    @property
    def state_dim(self) -> int:
        return self.spec.state_dim

    @property
    def noise_dim(self) -> int:
        return self.spec.noise_dim

    def initial_mean_var(self, params: ParamDict) -> Tuple[Array, Array]:
        means = [
            float(
                params.get(
                    "m0_level",
                    params.get("alpha0", 0.0 if self.initial_level is None else self.initial_level),
                )
            )
        ]
        level_var = float(
            params.get(
                "v0_level",
                1.0 if self.initial_level_sd is None else self.initial_level_sd**2,
            )
        )
        variances = [level_var]
        if self.trend_mode != "off":
            means.append(float(params.get("m0_trend", params.get("beta0", self.initial_slope))))
            variances.append(
                float(
                    params.get(
                        "v0_trend",
                        1.0 if self.initial_slope_sd is None else self.initial_slope_sd**2,
                    )
                )
            )
        if np.any(np.asarray(variances) < 0.0):
            raise ValueError("Initial trend variances must be non-negative.")
        return np.asarray(means), np.asarray(variances)

    def system_matrices(self, t: int, params: ParamDict):
        dim = self.state_dim
        transition = np.eye(dim)
        if dim == 2:
            transition[0, 1] = 1.0
        columns = []
        variances = []
        if self.level_mode == "dynamic":
            column = np.zeros(dim)
            column[0] = 1.0
            columns.append(column)
            variances.append(
                process_variance(
                    params,
                    noise_name=self.level_name,
                    legacy_key="q_level",
                )
            )
        if self.trend_mode == "dynamic":
            column = np.zeros(dim)
            column[1] = 1.0
            columns.append(column)
            variances.append(
                process_variance(
                    params,
                    noise_name=self.slope_name,
                    legacy_key="q_trend",
                )
            )
        loading = np.column_stack(columns) if columns else np.zeros((dim, 0))
        covariance = np.diag(variances) if variances else np.zeros((0, 0))
        return transition, loading, covariance, np.zeros(dim)

    def design_matrices(self, t: int, params: ParamDict, exog_t=None):
        design = np.zeros((1, self.state_dim))
        design[0, 0] = 1.0
        return design, np.zeros(1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "local_linear_trend",
            "level_mode": self.level_mode,
            "trend_mode": self.trend_mode,
            "level_name": self.level_name,
            "slope_name": self.slope_name,
            "initial_level": self.initial_level,
            "initial_slope": self.initial_slope,
            "initial_level_sd": self.initial_level_sd,
            "initial_slope_sd": self.initial_slope_sd,
        }
