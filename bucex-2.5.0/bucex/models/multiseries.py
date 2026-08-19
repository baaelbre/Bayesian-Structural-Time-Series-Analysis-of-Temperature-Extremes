"""Declarative models for related time series.

The channels keep separate latent structural paths. Dependence is introduced
through a :class:`~bucex.priors.HierarchicalPrior`, which can pool structural
selection probabilities, innovation-slab scales, or both. This borrows
strength without imposing an identical time-varying signal on physically
different temperature summaries.
"""
from __future__ import annotations

from dataclasses import dataclass
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


def _validate_channel_components(
    components: Sequence[Component], *, label: str
) -> tuple[Component, ...]:
    resolved = tuple(components)
    supported = (LocalLevel, LocalLinearTrend, DummySeasonal, Regression)
    unsupported = [
        type(component).__name__
        for component in resolved
        if not isinstance(component, supported)
    ]
    if unsupported:
        raise TypeError(f"Unsupported components in {label}: {unsupported}.")
    trends = sum(
        isinstance(component, (LocalLevel, LocalLinearTrend))
        for component in resolved
    )
    if trends != 1:
        raise ValueError(f"{label} requires exactly one LocalLevel or LocalLinearTrend.")
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
class Channel:
    """One named observation series and its structural components."""

    name: str
    observation: Observation
    components: tuple[Component, ...]
    tail: str | None = None
    eta_name: str = "mu"
    description: str | None = None

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name or "." in name:
            raise ValueError("Channel names must be non-empty and may not contain '.'.")
        if not isinstance(self.observation, (Gaussian, GEV)):
            raise TypeError("Channel observation must be Gaussian() or GEV().")
        components = _validate_channel_components(
            self.components, label=f"channel '{name}'"
        )
        tail = None if self.tail is None else str(self.tail).lower()
        aliases = {
            "max": "upper",
            "maximum": "upper",
            "min": "lower",
            "minimum": "lower",
        }
        tail = aliases.get(tail, tail)
        if isinstance(self.observation, Gaussian):
            if tail not in {None, "upper"}:
                raise ValueError("A Gaussian channel does not use lower-tail orientation.")
            tail = None
        elif tail is None:
            tail = "upper"
        elif tail not in {"upper", "lower"}:
            raise ValueError("GEV channel tail must be upper/max or lower/min.")
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
            components=tuple(
                component_from_dict(item) for item in value.get("components", ())
            ),
            tail=value.get("tail"),
            eta_name=value.get("eta_name", "mu"),
            description=value.get("description"),
        )


@dataclass(frozen=True)
class MultiSeriesModel:
    """Several named series fitted jointly through a shared prior hierarchy."""

    channels: tuple[Channel, ...]
    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        if len(channels) < 2:
            raise ValueError("A MultiSeriesModel requires at least two channels.")
        names = [channel.name for channel in channels]
        if len(names) != len(set(names)):
            raise ValueError("MultiSeriesModel channel names must be unique.")
        object.__setattr__(self, "channels", channels)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

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
    def transform_signs(self) -> np.ndarray:
        return np.asarray(
            [channel.transform_sign for channel in self.channels], dtype=float
        )

    @property
    def period(self) -> int | None:
        periods = {
            channel.period for channel in self.channels if channel.period is not None
        }
        return periods.pop() if len(periods) == 1 else None

    @property
    def supports_fs_parameterization(self) -> bool:
        """Whether every channel has the structural hierarchy layout."""

        for channel in self.channels:
            trends = [
                component
                for component in channel.components
                if isinstance(component, LocalLinearTrend)
            ]
            seasons = [
                component
                for component in channel.components
                if isinstance(component, DummySeasonal)
            ]
            regressions = [
                component
                for component in channel.components
                if isinstance(component, Regression)
            ]
            if (
                len(trends) != 1
                or trends[0].level_mode != "dynamic"
                or trends[0].trend_mode not in {"dynamic", "off"}
                or len(seasons) > 1
                or any(component.mode not in {"dynamic", "off"} for component in seasons)
                or regressions
                or len(channel.components) != len(trends) + len(seasons)
            ):
                return False
        return True

    def channel(self, name: str) -> Channel:
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(f"Unknown channel '{name}'. Available: {self.channel_names}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "multiseries",
            "channels": [channel.to_dict() for channel in self.channels],
            "name": self.name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MultiSeriesModel":
        return cls(
            channels=tuple(Channel.from_dict(item) for item in value["channels"]),
            name=value.get("name"),
            description=value.get("description"),
        )


__all__ = ["Channel", "MultiSeriesModel"]
