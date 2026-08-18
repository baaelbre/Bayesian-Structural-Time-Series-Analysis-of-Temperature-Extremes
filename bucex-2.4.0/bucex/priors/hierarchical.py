"""Hierarchical innovation priors for related structural time series."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..models.multiseries import MultiSeriesModel
from ..models.multiseries_compiler import CompiledMultiSeriesModel
from .structural import (
    FSGaussianPriors,
    FSGEVPriors,
    SSVSPrior,
    ssvs_gaussian_priors,
    ssvs_gev_priors,
)


_LABELS = {
    "level": ("fixed", "dynamic"),
    "trend": ("zero", "fixed", "dynamic"),
    "season": ("zero", "fixed", "dynamic"),
}


def _positive_mapping(values: Mapping[str, float], *, label: str) -> dict[str, float]:
    required = {"level", "trend", "season"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"{label} is missing {sorted(missing)}.")
    output = {name: float(values[name]) for name in required}
    if any(not np.isfinite(value) or value <= 0.0 for value in output.values()):
        raise ValueError(f"Every {label} value must be finite and positive.")
    return output


def _states(values: Sequence[str], component: str) -> tuple[str, ...]:
    resolved = tuple(str(value).lower() for value in values)
    allowed = _LABELS[component]
    if not resolved or len(resolved) != len(set(resolved)):
        raise ValueError(f"{component}_states must contain unique state names.")
    unknown = sorted(set(resolved) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {component} states: {unknown}; choose from {allowed}.")
    if component == "level" and "zero" in resolved:
        raise ValueError("The level is always present and cannot use state 'zero'.")
    return resolved


@dataclass(frozen=True)
class HierarchicalPrior:
    """Pool structural selection, normal-slab scales, or both.

    ``pool='selection'`` learns shared SSVS probabilities while keeping the
    normal slab scales fixed. ``pool='slab'`` keeps every available component
    dynamic and learns a shared scale for its signed normal coefficients.
    ``pool='both'`` combines both mechanisms.

    The learned scale has a half-Student-t hyperprior.  Thus the default model
    is deliberately close to the signed-normal non-centred formulation: the
    methodological extension is transparent partial pooling, not another
    opaque shrinkage family.

    Seasonality is present by default. Its allocation is fixed versus dynamic,
    not absent versus present, because a monthly temperature cycle is known
    physically before seeing these data.
    """

    pool: str = "both"
    slab: str = "normal"
    level_states: Sequence[str] = ("fixed", "dynamic")
    trend_states: Sequence[str] = ("zero", "fixed", "dynamic")
    season_states: Sequence[str] = ("fixed", "dynamic")
    level_concentration: Sequence[float] = (1.0, 1.0)
    trend_concentration: Sequence[float] = (1.0, 1.0, 1.0)
    season_concentration: Sequence[float] = (1.0, 1.0)
    coefficient_scale: Mapping[str, float] = field(
        default_factory=lambda: {
            "level": 0.03,
            "trend": 0.0002,
            "season": 0.03,
        }
    )
    slab_df: float = 4.0
    slab_prior_scale: Mapping[str, float] = field(
        default_factory=lambda: {"level": 1.0, "trend": 1.0, "season": 1.0}
    )
    initial_slab_scale: Mapping[str, float] = field(
        default_factory=lambda: {"level": 1.0, "trend": 1.0, "season": 1.0}
    )

    def __post_init__(self) -> None:
        pool = str(self.pool).lower().replace("-", "_")
        aliases = {
            "ssvs": "selection",
            "scale": "slab",
            "ssvs_and_slab": "both",
            "selection_and_slab": "both",
        }
        pool = aliases.get(pool, pool)
        if pool not in {"selection", "slab", "both"}:
            raise ValueError("pool must be 'selection', 'slab', or 'both'.")
        if str(self.slab).lower() != "normal":
            raise ValueError("The hierarchical release currently uses a normal slab.")
        object.__setattr__(self, "pool", pool)
        object.__setattr__(self, "slab", "normal")
        for component in ("level", "trend", "season"):
            object.__setattr__(
                self,
                f"{component}_states",
                _states(getattr(self, f"{component}_states"), component),
            )
            concentration = np.asarray(
                getattr(self, f"{component}_concentration"), dtype=float
            )
            expected = len(getattr(self, f"{component}_states"))
            if (
                concentration.shape != (expected,)
                or np.any(~np.isfinite(concentration))
                or np.any(concentration <= 0.0)
            ):
                raise ValueError(
                    f"{component}_concentration must contain {expected} positive values."
                )
        if not np.isfinite(float(self.slab_df)) or float(self.slab_df) <= 0.0:
            raise ValueError("slab_df must be finite and positive.")
        _positive_mapping(self.coefficient_scale, label="coefficient_scale")
        _positive_mapping(self.slab_prior_scale, label="slab_prior_scale")
        _positive_mapping(self.initial_slab_scale, label="initial_slab_scale")

    @property
    def pools_selection(self) -> bool:
        return self.pool in {"selection", "both"}

    @property
    def pools_slab(self) -> bool:
        return self.pool in {"slab", "both"}

    def allowed_indices(self, component: str) -> np.ndarray:
        labels = _LABELS[component]
        states = getattr(self, f"{component}_states")
        return np.asarray([labels.index(state) for state in states], dtype=int)

    def concentration(self, component: str) -> np.ndarray:
        return np.asarray(getattr(self, f"{component}_concentration"), dtype=float)

    def initial_probabilities(self, component: str) -> np.ndarray:
        size = len(_LABELS[component])
        output = np.zeros(size, dtype=float)
        if self.pools_selection:
            concentration = self.concentration(component)
            output[self.allowed_indices(component)] = concentration / concentration.sum()
        else:
            output[_LABELS[component].index("dynamic")] = 1.0
        return output

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
class HierarchicalPriors:
    """Resolved hierarchy plus channel-specific observation priors."""

    hierarchy: HierarchicalPrior = field(default_factory=HierarchicalPrior)
    channels: Mapping[str, FSGaussianPriors | FSGEVPriors] = field(
        default_factory=dict
    )

    @property
    def profile(self) -> str:
        return f"hierarchical_{self.hierarchy.pool}"


def resolve_hierarchical_priors(
    compiled: CompiledMultiSeriesModel, priors: Any
) -> HierarchicalPriors:
    """Resolve one transparent hierarchy for a ``MultiSeriesModel``."""

    if not isinstance(compiled.model, MultiSeriesModel):
        raise TypeError("Hierarchical priors are defined for MultiSeriesModel.")
    if isinstance(priors, HierarchicalPriors):
        hierarchy = priors.hierarchy
        supplied = dict(priors.channels)
    elif isinstance(priors, HierarchicalPrior):
        hierarchy = priors
        supplied = {}
    elif priors is None:
        hierarchy = HierarchicalPrior()
        supplied = {}
    elif isinstance(priors, str):
        key = str(priors).lower().replace("-", "_")
        profiles = {
            "hierarchical": "both",
            "ssvs": "selection",
            "pooled_selection": "selection",
            "pooled_ssvs": "selection",
            "hierarchical_slab": "slab",
            "pooled_slab": "slab",
            "hierarchical_both": "both",
            "pooled_both": "both",
        }
        if key not in profiles:
            raise ValueError(
                "MultiSeriesModel priors must be pooled_selection, pooled_slab, "
                "or pooled_both."
            )
        hierarchy = HierarchicalPrior(pool=profiles[key])
        supplied = {}
    else:
        raise TypeError(
            "Use a hierarchical profile, HierarchicalPrior, or HierarchicalPriors."
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
                f"Channel '{channel.name}' must use the signed normal SSVS slab."
            )
        resolved[channel.name] = prior
    return HierarchicalPriors(hierarchy=hierarchy, channels=resolved)


__all__ = [
    "HierarchicalPrior",
    "HierarchicalPriors",
    "resolve_hierarchical_priors",
]
