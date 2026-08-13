"""Declarative model grammar for first-release bucex models."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .observations import GEV, Gaussian, Observation, observation_from_dict


@dataclass(frozen=True)
class LocalLevel:
    """Random-walk level component.

    Set its process prior to :class:`bucex.FixedSD` for a constant level.
    """

    name: str = "level"
    initial_mean: float | None = None
    initial_sd: float | None = None

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def noise_dim(self) -> int:
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "local_level",
            "name": self.name,
            "initial_mean": self.initial_mean,
            "initial_sd": self.initial_sd,
        }


@dataclass(frozen=True)
class LocalLinearTrend:
    """Local level plus local slope.

    The level and slope have separate named process standard deviations. A
    fixed slope is obtained with ``FixedSD(0)`` on ``slope_name``.
    """

    level_name: str = "level"
    slope_name: str = "slope"
    initial_level: float | None = None
    initial_slope: float = 0.0
    initial_level_sd: float | None = None
    initial_slope_sd: float | None = None

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def noise_dim(self) -> int:
        return 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "local_linear_trend",
            "level_name": self.level_name,
            "slope_name": self.slope_name,
            "initial_level": self.initial_level,
            "initial_slope": self.initial_slope,
            "initial_level_sd": self.initial_level_sd,
            "initial_slope_sd": self.initial_slope_sd,
        }


@dataclass(frozen=True)
class DummySeasonal:
    """Sum-to-zero dummy seasonal component with one innovation per time step."""

    period: int = 12
    name: str = "seasonal"
    initial_mean: tuple[float, ...] | None = None
    initial_sd: float | None = None

    def __post_init__(self) -> None:
        if int(self.period) < 2:
            raise ValueError("DummySeasonal.period must be at least 2.")
        if self.initial_mean is not None:
            values = tuple(float(value) for value in self.initial_mean)
            if len(values) != self.state_dim or not all(np.isfinite(values)):
                raise ValueError("DummySeasonal.initial_mean must contain period - 1 finite values.")
            object.__setattr__(self, "initial_mean", values)

    @property
    def state_dim(self) -> int:
        return int(self.period) - 1

    @property
    def noise_dim(self) -> int:
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "dummy_seasonal",
            "period": int(self.period),
            "name": self.name,
            "initial_mean": None if self.initial_mean is None else list(self.initial_mean),
            "initial_sd": self.initial_sd,
        }


@dataclass(frozen=True)
class Regression:
    """Static or random-walk regression coefficients.

    Regressions consume consecutive columns from the ``exog`` matrix in model
    order. ``feature_names`` are metadata and, when a pandas DataFrame is
    supplied, are also used to select and order columns.
    """

    n_features: int
    dynamic: bool = False
    name: str = "regression"
    feature_names: tuple[str, ...] | None = None
    initial_mean: tuple[float, ...] | None = None
    initial_sd: float | None = None

    def __post_init__(self) -> None:
        if int(self.n_features) < 1:
            raise ValueError("Regression.n_features must be at least 1.")
        if self.feature_names is not None and len(self.feature_names) != int(self.n_features):
            raise ValueError("feature_names must have length n_features.")
        if self.initial_mean is not None:
            values = tuple(float(value) for value in self.initial_mean)
            if len(values) != int(self.n_features) or not all(np.isfinite(values)):
                raise ValueError("Regression.initial_mean must contain n_features finite values.")
            object.__setattr__(self, "initial_mean", values)

    @property
    def state_dim(self) -> int:
        return int(self.n_features)

    @property
    def noise_dim(self) -> int:
        return int(self.n_features) if self.dynamic else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "regression",
            "n_features": int(self.n_features),
            "dynamic": bool(self.dynamic),
            "name": self.name,
            "feature_names": None if self.feature_names is None else list(self.feature_names),
            "initial_mean": None if self.initial_mean is None else list(self.initial_mean),
            "initial_sd": self.initial_sd,
        }


Component = LocalLevel | LocalLinearTrend | DummySeasonal | Regression


def component_from_dict(value: dict[str, Any]) -> Component:
    kind = str(value["type"])
    args = {key: val for key, val in value.items() if key != "type"}
    for name in ("feature_names", "initial_mean"):
        if name in args and args[name] is not None:
            args[name] = tuple(args[name])
    if kind == "local_level":
        return LocalLevel(**args)
    if kind == "local_linear_trend":
        return LocalLinearTrend(**args)
    if kind == "dummy_seasonal":
        return DummySeasonal(**args)
    if kind == "regression":
        return Regression(**args)
    raise ValueError(f"Unknown component type '{kind}'.")


@dataclass(frozen=True)
class Model:
    """A declarative structural model specification."""

    observation: Observation
    components: Sequence[Component]
    name: str | None = None

    def __post_init__(self) -> None:
        components = tuple(self.components)
        object.__setattr__(self, "components", components)
        if not isinstance(self.observation, (Gaussian, GEV)):
            raise TypeError("observation must be Gaussian() or GEV().")
        if not components:
            raise ValueError("A model requires at least one structural component.")
        trend_like = sum(isinstance(c, (LocalLevel, LocalLinearTrend)) for c in components)
        if trend_like != 1:
            raise ValueError("Specify exactly one LocalLevel or LocalLinearTrend component.")
        if sum(isinstance(c, DummySeasonal) for c in components) > 1:
            raise ValueError("At most one DummySeasonal component is supported.")
        regression_names = [c.name for c in components if isinstance(c, Regression)]
        if len(regression_names) != len(set(regression_names)):
            raise ValueError("Regression component names must be unique.")

    @property
    def family(self) -> str:
        return self.observation.name

    @property
    def period(self) -> int | None:
        for component in self.components:
            if isinstance(component, DummySeasonal):
                return int(component.period)
        return None

    @property
    def n_exog(self) -> int:
        return sum(c.n_features for c in self.components if isinstance(c, Regression))

    def with_observation(self, observation: Observation) -> "Model":
        return replace(self, observation=observation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "components": [component.to_dict() for component in self.components],
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Model":
        return cls(
            observation=observation_from_dict(value["observation"]),
            components=[component_from_dict(item) for item in value["components"]],
            name=value.get("name"),
        )


def structural_model(
    family: str = "gaussian",
    *,
    trend: str = "local_linear",
    period: int | None = None,
    xi_bounds: tuple[float, float] = (-0.5, 0.5),
) -> Model:
    """Convenience constructor for the common level/trend/seasonal grammar."""

    family_key = str(family).lower()
    if family_key == "gaussian":
        observation: Observation = Gaussian()
    elif family_key == "gev":
        observation = GEV(xi_bounds=xi_bounds)
    else:
        raise ValueError("family must be 'gaussian' or 'gev'.")
    if trend == "local_level":
        components: list[Component] = [LocalLevel()]
    elif trend == "local_linear":
        components = [LocalLinearTrend()]
    else:
        raise ValueError("trend must be 'local_level' or 'local_linear'.")
    if period is not None:
        components.append(DummySeasonal(period=int(period)))
    return Model(observation=observation, components=components)


StructuralModel = Model
Seasonal = DummySeasonal
