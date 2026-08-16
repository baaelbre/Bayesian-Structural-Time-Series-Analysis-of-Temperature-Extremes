"""Hierarchical structural SSVS priors for related time series."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..models.factor_compiler import CompiledFactorModel
from ..models.multiseries import MultiSeriesModel
from .structural import (
    FSGaussianPriors,
    FSGEVPriors,
    SSVSPrior,
    ssvs_gaussian_priors,
    ssvs_gev_priors,
)


def _positive_mapping(
    values: Mapping[str, float], *, label: str
) -> dict[str, float]:
    required = {"level", "trend", "season"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"{label} is missing {sorted(missing)}.")
    output = {name: float(values[name]) for name in required}
    if any(not np.isfinite(value) or value <= 0.0 for value in output.values()):
        raise ValueError(f"Every {label} value must be finite and positive.")
    return output


@dataclass(frozen=True)
class HierarchicalSSVSPrior:
    """Shared model probabilities and dynamic-slab scales.

    For each structural component, series-specific states are categorical and
    their probability vector receives a Dirichlet prior.  Conditional on the
    dynamic state, the signed FS innovation scale is

    ``s[i, k] ~ Normal(0, coefficient_scale[k]^2 * slab_scale[k]^2)``.

    A half-Student-t prior on each learned ``slab_scale`` supplies robust
    partial pooling.  Structural zeros remain exact point masses; this is not
    a continuous approximation to SSVS.
    """

    level_concentration: Sequence[float] = (1.0, 1.0)
    trend_concentration: Sequence[float] = (1.0, 1.0, 1.0)
    season_concentration: Sequence[float] = (1.0, 1.0, 1.0)
    coefficient_scale: Mapping[str, float] = field(
        default_factory=lambda: {
            "level": 0.03,
            "trend": 0.0002,
            "season": 0.03,
        }
    )
    learn_slab_scale: bool = True
    slab_df: float = 4.0
    slab_prior_scale: Mapping[str, float] = field(
        default_factory=lambda: {"level": 1.0, "trend": 1.0, "season": 1.0}
    )
    initial_slab_scale: Mapping[str, float] = field(
        default_factory=lambda: {"level": 1.0, "trend": 1.0, "season": 1.0}
    )

    def __post_init__(self) -> None:
        for name, values, expected in (
            ("level_concentration", self.level_concentration, 2),
            ("trend_concentration", self.trend_concentration, 3),
            ("season_concentration", self.season_concentration, 3),
        ):
            array = np.asarray(values, dtype=float)
            if array.shape != (expected,) or np.any(~np.isfinite(array)) or np.any(array <= 0.0):
                raise ValueError(f"{name} must contain {expected} positive values.")
        if not np.isfinite(float(self.slab_df)) or float(self.slab_df) <= 0.0:
            raise ValueError("slab_df must be finite and positive.")
        _positive_mapping(self.coefficient_scale, label="coefficient_scale")
        _positive_mapping(self.slab_prior_scale, label="slab_prior_scale")
        _positive_mapping(self.initial_slab_scale, label="initial_slab_scale")

    def concentration(self, component: str) -> np.ndarray:
        values = {
            "level": self.level_concentration,
            "trend": self.trend_concentration,
            "season": self.season_concentration,
        }[component]
        return np.asarray(values, dtype=float)

    def component_ssvs(
        self,
        probabilities: Mapping[str, np.ndarray],
        slab_scale: Mapping[str, float],
    ) -> SSVSPrior:
        level = np.asarray(probabilities["level"], dtype=float)
        trend = np.asarray(probabilities["trend"], dtype=float)
        season = np.asarray(probabilities["season"], dtype=float)
        return SSVSPrior(
            innovation_slab_sd={
                name: float(self.coefficient_scale[name]) * float(slab_scale[name])
                for name in ("level", "trend", "season")
            },
            level_dynamic_probability=float(level[1]),
            trend_probabilities=tuple(trend),
            season_probabilities=tuple(season),
        )


@dataclass(frozen=True)
class HierarchicalSSVSPriors:
    """Resolved hierarchy plus channel-specific nuisance-parameter priors."""

    hierarchy: HierarchicalSSVSPrior = field(default_factory=HierarchicalSSVSPrior)
    channels: Mapping[str, FSGaussianPriors | FSGEVPriors] = field(
        default_factory=dict
    )

    @property
    def profile(self) -> str:
        return "hierarchical_ssvs"


def resolve_hierarchical_ssvs_priors(
    compiled: CompiledFactorModel,
    priors: Any,
) -> HierarchicalSSVSPriors:
    """Resolve one transparent prior object for a ``MultiSeriesModel``."""

    if not isinstance(compiled.model, MultiSeriesModel):
        raise TypeError("Hierarchical SSVS is defined for MultiSeriesModel.")
    if isinstance(priors, HierarchicalSSVSPriors):
        hierarchy = priors.hierarchy
        supplied = dict(priors.channels)
    elif isinstance(priors, HierarchicalSSVSPrior):
        hierarchy = priors
        supplied = {}
    elif priors is None or (
        isinstance(priors, str)
        and str(priors).lower().replace("-", "_")
        in {"ssvs", "hierarchical_ssvs", "hierarchical"}
    ):
        hierarchy = HierarchicalSSVSPrior()
        supplied = {}
    else:
        raise TypeError(
            "MultiSeriesModel currently requires priors='hierarchical_ssvs', "
            "HierarchicalSSVSPrior(...), or HierarchicalSSVSPriors(...)."
        )

    unknown = sorted(set(supplied) - set(compiled.channel_names))
    if unknown:
        raise ValueError(f"Priors were supplied for unknown channels: {unknown}.")
    resolved: dict[str, FSGaussianPriors | FSGEVPriors] = {}
    for channel in compiled.model.channels:
        prior = supplied.get(channel.name)
        expected = FSGaussianPriors if channel.family == "gaussian" else FSGEVPriors
        if prior is None:
            period = int(channel.period or 1)
            alpha_mean = float(compiled.channel_location[channel.name])
            builder = (
                ssvs_gaussian_priors
                if channel.family == "gaussian"
                else ssvs_gev_priors
            )
            prior = builder(period=period, alpha_mean=alpha_mean)
        if not isinstance(prior, expected):
            raise TypeError(
                f"Channel '{channel.name}' requires {expected.__name__}; "
                f"got {type(prior).__name__}."
            )
        if prior.ssvs is None:
            raise ValueError(
                f"Channel '{channel.name}' prior must contain an SSVSPrior."
            )
        resolved[channel.name] = prior
    return HierarchicalSSVSPriors(hierarchy=hierarchy, channels=resolved)


__all__ = [
    "HierarchicalSSVSPrior",
    "HierarchicalSSVSPriors",
    "resolve_hierarchical_ssvs_priors",
]
