"""Declarative collections of related structural time series.

``MultiSeriesModel`` deliberately has no shared latent time path. Its channels
are conditionally independent given parameters, but a hierarchical prior may
pool structural-selection probabilities and dynamic slab scales. Use
:class:`~bucex.models.factor.FactorModel` when the scientific question instead
calls for one or more common dynamic factors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..components import DummySeasonal, LocalLinearTrend, Regression
from ..observation import Observation
from .factor import Channel


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
        if not all(channel.components for channel in channels):
            raise ValueError(
                "Every MultiSeriesModel channel needs structural components."
            )
        object.__setattr__(self, "channels", channels)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    @property
    def factor_names(self) -> tuple[str, ...]:
        return ()

    @property
    def factors(self) -> tuple:
        return ()

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
    def estimated_loading_names(self) -> tuple[str, ...]:
        return ()

    @property
    def supports_fs_parameterization(self) -> bool:
        """Whether every channel has the exact structural SSVS FS layout."""

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

    def factor(self, name: str):
        raise KeyError(
            f"MultiSeriesModel has no shared factors (requested {name!r}); "
            "use FactorModel for a common dynamic path."
        )

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


PanelModel = MultiSeriesModel


__all__ = ["MultiSeriesModel", "PanelModel"]
