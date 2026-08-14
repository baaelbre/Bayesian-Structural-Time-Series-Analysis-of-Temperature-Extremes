"""Declarative multichannel models with shared dynamic factors."""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from ..components import (
    DummySeasonal,
    LocalLevel,
    LocalLinearTrend,
    Regression,
    component_from_dict,
)
from ..components.base import Component
from ..observation import GEV, Gaussian, Observation, observation_from_dict


def _validate_scope_components(
    components: Sequence[Component],
    *,
    label: str,
    require_trend: bool,
) -> tuple[Component, ...]:
    resolved = tuple(components)
    supported = (LocalLevel, LocalLinearTrend, DummySeasonal, Regression)
    unsupported = [type(component).__name__ for component in resolved if not isinstance(component, supported)]
    if unsupported:
        raise TypeError(f"Unsupported components in {label}: {unsupported}.")
    trends = sum(isinstance(component, (LocalLevel, LocalLinearTrend)) for component in resolved)
    if trends > 1 or (require_trend and trends != 1):
        requirement = "exactly one" if require_trend else "at most one"
        raise ValueError(f"{label} requires {requirement} LocalLevel or LocalLinearTrend.")
    if sum(isinstance(component, DummySeasonal) for component in resolved) > 1:
        raise ValueError(f"{label} supports at most one DummySeasonal component.")
    regression_names = [
        component.name for component in resolved if isinstance(component, Regression)
    ]
    if len(regression_names) != len(set(regression_names)):
        raise ValueError(f"Regression component names must be unique in {label}.")
    state_names = [name for component in resolved for name in component.spec.state_names]
    if len(state_names) != len(set(state_names)):
        raise ValueError(f"State names must be unique within {label}.")
    return resolved


@dataclass(frozen=True)
class Loading:
    """A fixed or estimated factor loading.

    Plain numbers in :class:`Factor.loadings` are interpreted as fixed
    loadings.  ``Loading(value)`` is estimated with a normal prior.  At least
    one non-zero fixed loading is required whenever a factor contains an
    estimated loading; this anchors its scale and sign.
    """

    value: float = 0.5
    fixed: bool = False
    prior_mean: float = 0.0
    prior_sd: float = 1.0

    def __post_init__(self) -> None:
        values = (self.value, self.prior_mean, self.prior_sd)
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("Loading values and prior hyperparameters must be finite.")
        if float(self.prior_sd) <= 0.0:
            raise ValueError("Loading.prior_sd must be positive.")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "prior_mean", float(self.prior_mean))
        object.__setattr__(self, "prior_sd", float(self.prior_sd))

    @classmethod
    def estimated(
        cls,
        initial: float = 0.5,
        *,
        mean: float = 0.0,
        sd: float = 1.0,
    ) -> "Loading":
        return cls(value=initial, fixed=False, prior_mean=mean, prior_sd=sd)

    @classmethod
    def constant(cls, value: float) -> "Loading":
        return cls(value=value, fixed=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": float(self.value),
            "fixed": bool(self.fixed),
            "prior_mean": float(self.prior_mean),
            "prior_sd": float(self.prior_sd),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Loading":
        return cls(**dict(value))


def _as_loading(value: Loading | Real | None) -> Loading:
    if isinstance(value, Loading):
        return value
    if value is None:
        return Loading.estimated()
    if isinstance(value, Real):
        return Loading.constant(float(value))
    raise TypeError("Factor loadings must be numbers, Loading objects, or None.")


@dataclass(frozen=True)
class Channel:
    """One named observation channel and its individual state components."""

    name: str
    observation: Observation
    components: tuple[Component, ...] = ()
    tail: str | None = None
    eta_name: str = "mu"
    description: str | None = None

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name or "." in name:
            raise ValueError("Channel names must be non-empty and may not contain '.'.")
        if not isinstance(self.observation, (Gaussian, GEV)):
            raise TypeError("Channel observation must be Gaussian() or GEV().")
        components = _validate_scope_components(
            self.components,
            label=f"channel '{name}'",
            require_trend=bool(self.components),
        )
        tail = None if self.tail is None else str(self.tail).lower()
        aliases = {"max": "upper", "maximum": "upper", "min": "lower", "minimum": "lower"}
        tail = aliases.get(tail, tail)
        if isinstance(self.observation, Gaussian):
            if tail not in {None, "upper"}:
                raise ValueError("A Gaussian channel does not use a lower-tail orientation.")
            tail = None
        elif tail is None:
            tail = "upper"
        elif tail not in {"upper", "lower"}:
            raise ValueError("GEV channel tail must be 'upper'/'max' or 'lower'/'min'.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "tail", tail)
        object.__setattr__(self, "eta_name", str(self.eta_name))

    @property
    def family(self) -> str:
        return self.observation.name

    @property
    def transform_sign(self) -> float:
        return -1.0 if self.tail == "lower" else 1.0

    @property
    def period(self) -> int | None:
        for component in self.components:
            if isinstance(component, DummySeasonal) and component.mode != "off":
                return int(component.period)
        return None

    @property
    def n_exog(self) -> int:
        return sum(
            int(component.n_features)
            for component in self.components
            if isinstance(component, Regression) and component.mode != "off"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "observation": self.observation.to_dict(),
            "components": [component.to_dict() for component in self.components],
            "tail": self.tail,
            "eta_name": self.eta_name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Channel":
        return cls(
            name=value["name"],
            observation=observation_from_dict(value["observation"]),
            components=tuple(component_from_dict(item) for item in value.get("components", ())),
            tail=value.get("tail"),
            eta_name=value.get("eta_name", "mu"),
            description=value.get("description"),
        )


@dataclass(frozen=True)
class Factor:
    """A reusable structural state block shared across named channels."""

    name: str
    components: tuple[Component, ...]
    loadings: Mapping[str, Loading | Real | None]
    description: str | None = None
    _resolved_loadings: Mapping[str, Loading] = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name or "." in name:
            raise ValueError("Factor names must be non-empty and may not contain '.'.")
        components = _validate_scope_components(
            self.components,
            label=f"factor '{name}'",
            require_trend=True,
        )
        if any(isinstance(component, Regression) for component in components):
            raise ValueError(
                "Regression belongs to an observation channel; shared factors currently "
                "accept trend and seasonal components only."
            )
        resolved = {str(channel): _as_loading(value) for channel, value in self.loadings.items()}
        if not resolved:
            raise ValueError(f"Factor '{name}' requires at least one channel loading.")
        if not any(abs(spec.value) > 0.0 for spec in resolved.values()):
            raise ValueError(f"Factor '{name}' cannot have all-zero initial loadings.")
        estimated = any(not spec.fixed for spec in resolved.values())
        anchored = any(spec.fixed and abs(spec.value) > 0.0 for spec in resolved.values())
        if estimated and not anchored:
            raise ValueError(
                f"Factor '{name}' has estimated loadings but no fixed non-zero anchor."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "loadings", dict(resolved))
        object.__setattr__(self, "_resolved_loadings", dict(resolved))

    def loading_for(self, channel: str) -> Loading:
        return self._resolved_loadings.get(str(channel), Loading.constant(0.0))

    @property
    def estimated_channels(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self._resolved_loadings.items() if not spec.fixed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "components": [component.to_dict() for component in self.components],
            "loadings": {
                channel: spec.to_dict() for channel, spec in self._resolved_loadings.items()
            },
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Factor":
        return cls(
            name=value["name"],
            components=tuple(component_from_dict(item) for item in value["components"]),
            loadings={
                name: Loading.from_dict(spec) for name, spec in value["loadings"].items()
            },
            description=value.get("description"),
        )


@dataclass(frozen=True)
class FactorModel:
    """Mixed-family structural model with shared and individual state blocks.

    Conditional on the latent path, channel likelihoods factorize.  Marginally,
    channels are dependent because their predictors contain the same factor
    states.  Residual/coplanar dependence beyond those shared states is not
    introduced by this class.
    """

    channels: tuple[Channel, ...]
    factors: tuple[Factor, ...]
    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        factors = tuple(self.factors)
        if len(channels) < 2:
            raise ValueError("A FactorModel requires at least two channels.")
        if not factors:
            raise ValueError("A FactorModel requires at least one shared factor.")
        channel_names = [channel.name for channel in channels]
        factor_names = [factor.name for factor in factors]
        if len(channel_names) != len(set(channel_names)):
            raise ValueError("FactorModel channel names must be unique.")
        if len(factor_names) != len(set(factor_names)):
            raise ValueError("FactorModel factor names must be unique.")
        known = set(channel_names)
        for factor in factors:
            unknown = sorted(set(factor._resolved_loadings) - known)
            if unknown:
                raise ValueError(
                    f"Factor '{factor.name}' references unknown channels: {unknown}."
                )
        if len(factors) > len(channels):
            raise ValueError("A FactorModel cannot have more factors than channels.")
        fixed_anchor_matrix = np.zeros((len(channels), len(factors)), dtype=float)
        for row, channel_name in enumerate(channel_names):
            for column, factor in enumerate(factors):
                specification = factor.loading_for(channel_name)
                if specification.fixed:
                    fixed_anchor_matrix[row, column] = specification.value
        fixed_rank = int(np.linalg.matrix_rank(fixed_anchor_matrix))
        if fixed_rank < len(factors):
            raise ValueError(
                "The fixed factor-loading anchors must have full column rank; "
                f"rank is {fixed_rank} for {len(factors)} factors. Use distinct "
                "fixed anchors/zeros or fixed contrast loadings."
            )
        if not any(channel.components for channel in channels) and not factors:
            raise ValueError("The model contains no latent state components.")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "factors", factors)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(factor.name for factor in self.factors)

    @property
    def eta_dim(self) -> int:
        return len(self.channels)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(channel.family for channel in self.channels)

    @property
    def all_gaussian(self) -> bool:
        return all(family == "gaussian" for family in self.families)

    @property
    def supports_fs_parameterization(self) -> bool:
        """Whether the one-factor FS decomposition is mathematically defined.

        The v2.1 factor NCP deliberately targets the manuscript model: one
        anchored shared local-linear trend, one dynamic idiosyncratic local
        level per channel, and an optional dynamic dummy seasonal per channel.
        The factor's initial level is fixed to zero to separate its location
        from the channel intercepts.
        """

        if len(self.factors) != 1:
            return False
        factor = self.factors[0]
        trends = [
            component
            for component in factor.components
            if isinstance(component, LocalLinearTrend)
        ]
        if len(trends) != 1 or len(factor.components) != 1:
            return False
        trend = trends[0]
        if (
            trend.level_mode != "dynamic"
            or trend.trend_mode != "dynamic"
            or trend.initial_level is None
            or not np.isclose(float(trend.initial_level), 0.0)
            or trend.initial_level_sd is None
            or not np.isclose(float(trend.initial_level_sd), 0.0)
        ):
            return False
        for channel in self.channels:
            levels = [
                component
                for component in channel.components
                if isinstance(component, LocalLevel)
            ]
            seasons = [
                component
                for component in channel.components
                if isinstance(component, DummySeasonal)
            ]
            if len(levels) != 1 or levels[0].mode != "dynamic":
                return False
            if len(seasons) > 1 or any(
                component.mode not in {"dynamic", "off"} for component in seasons
            ):
                return False
            if len(channel.components) != len(levels) + len(seasons):
                return False
        return True

    @property
    def family(self) -> str:
        if self.all_gaussian:
            return "multivariate_gaussian"
        if all(family == "gev" for family in self.families):
            return "multivariate_gev"
        return "mixed"

    @property
    def observations(self) -> Mapping[str, Observation]:
        return {channel.name: channel.observation for channel in self.channels}

    @property
    def obs(self) -> tuple[Observation, ...]:
        return tuple(channel.observation for channel in self.channels)

    @property
    def period(self) -> int | None:
        periods = {
            period
            for period in (
                *(channel.period for channel in self.channels),
                *(
                    int(component.period)
                    for factor in self.factors
                    for component in factor.components
                    if isinstance(component, DummySeasonal) and component.mode != "off"
                ),
            )
            if period is not None
        }
        return periods.pop() if len(periods) == 1 else None

    @property
    def transform_signs(self) -> np.ndarray:
        return np.asarray([channel.transform_sign for channel in self.channels], dtype=float)

    @property
    def estimated_loading_names(self) -> tuple[str, ...]:
        return tuple(
            f"loading.{factor.name}.{channel}"
            for factor in self.factors
            for channel in factor.estimated_channels
        )

    def channel(self, name: str) -> Channel:
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(f"Unknown channel '{name}'. Available: {self.channel_names}")

    def factor(self, name: str) -> Factor:
        for factor in self.factors:
            if factor.name == name:
                return factor
        raise KeyError(f"Unknown factor '{name}'. Available: {self.factor_names}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "factor",
            "channels": [channel.to_dict() for channel in self.channels],
            "factors": [factor.to_dict() for factor in self.factors],
            "name": self.name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FactorModel":
        return cls(
            channels=tuple(Channel.from_dict(item) for item in value["channels"]),
            factors=tuple(Factor.from_dict(item) for item in value["factors"]),
            name=value.get("name"),
            description=value.get("description"),
        )


DynamicFactorModel = FactorModel


__all__ = [
    "Channel",
    "Loading",
    "Factor",
    "FactorModel",
    "DynamicFactorModel",
]
