"""Priors for shared-factor and multichannel observation parameters."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from ..components import DummySeasonal, LocalLevel, LocalLinearTrend
from ..models.factor_compiler import CompiledFactorModel
from .process import (
    FixedSD,
    HalfNormalSD,
    HalfStudentTSD,
    PCSD,
    SDPrior,
    ShapePrior,
    TruncatedNormalPrior,
    sd_prior_from_dict,
    shape_prior_from_dict,
)
from .structural import DiagonalNormalPrior, NormalPrior, RegularizedHorseshoePrior


@dataclass(frozen=True)
class FactorPriors:
    """Explicit priors for one compiled multichannel model."""

    process: Mapping[str, SDPrior]
    observation_sd: Mapping[str, SDPrior]
    shape: Mapping[str, ShapePrior] = field(default_factory=dict)
    intercept: Mapping[str, NormalPrior] = field(default_factory=dict)
    factor_initial_slope: Mapping[str, NormalPrior] = field(default_factory=dict)
    seasonal_initial: Mapping[str, DiagonalNormalPrior] = field(default_factory=dict)
    horseshoe: RegularizedHorseshoePrior | None = None
    horseshoe_processes: tuple[str, ...] = ()
    profile: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "process": {name: prior.to_dict() for name, prior in self.process.items()},
            "observation_sd": {
                name: prior.to_dict() for name, prior in self.observation_sd.items()
            },
            "shape": {name: prior.to_dict() for name, prior in self.shape.items()},
            "intercept": {
                name: {"mean": float(prior.mean), "sd": float(prior.sd)}
                for name, prior in self.intercept.items()
            },
            "factor_initial_slope": {
                name: {"mean": float(prior.mean), "sd": float(prior.sd)}
                for name, prior in self.factor_initial_slope.items()
            },
            "seasonal_initial": {
                name: {
                    "mean": list(np.asarray(prior.mean, dtype=float)),
                    "sd": list(np.asarray(prior.sd, dtype=float)),
                }
                for name, prior in self.seasonal_initial.items()
            },
            "horseshoe": (
                None
                if self.horseshoe is None
                else {
                    "coefficient_scale": dict(self.horseshoe.coefficient_scale),
                    "global_scale": float(self.horseshoe.global_scale),
                    "slab_scale": float(self.horseshoe.slab_scale),
                    "slab_df": float(self.horseshoe.slab_df),
                    "initial_local": float(self.horseshoe.initial_local),
                    "initial_global": self.horseshoe.initial_global,
                    "initial_slab2": self.horseshoe.initial_slab2,
                }
            ),
            "horseshoe_processes": list(self.horseshoe_processes),
            "profile": self.profile,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FactorPriors":
        return cls(
            process={
                name: sd_prior_from_dict(prior)
                for name, prior in value["process"].items()
            },
            observation_sd={
                name: sd_prior_from_dict(prior)
                for name, prior in value["observation_sd"].items()
            },
            shape={
                name: shape_prior_from_dict(prior)
                for name, prior in value.get("shape", {}).items()
            },
            intercept={
                name: NormalPrior(**prior)
                for name, prior in value.get("intercept", {}).items()
            },
            factor_initial_slope={
                name: NormalPrior(**prior)
                for name, prior in value.get("factor_initial_slope", {}).items()
            },
            seasonal_initial={
                name: DiagonalNormalPrior(**prior)
                for name, prior in value.get("seasonal_initial", {}).items()
            },
            horseshoe=(
                None
                if value.get("horseshoe") is None
                else RegularizedHorseshoePrior(**value["horseshoe"])
            ),
            horseshoe_processes=tuple(value.get("horseshoe_processes", ())),
            profile=value.get("profile", "custom"),
            metadata=value.get("metadata", {}),
        )


def default_factor_priors(
    compiled: CompiledFactorModel,
    profile: str = "regularized_horseshoe",
) -> FactorPriors:
    """Data-scaled defaults that do not depend on record length."""

    key = str(profile).lower().replace("-", "_")
    key = {
        "normal": "half_normal",
        "pc": "regularized",
        "horseshoe": "regularized_horseshoe",
    }.get(key, key)
    if key not in {
        "regularized_horseshoe",
        "regularized",
        "half_normal",
        "weak",
        "strong",
    }:
        raise ValueError(
            "Factor prior profile must be regularized_horseshoe/horseshoe, "
            "regularized/pc, half_normal/normal, weak, or strong."
        )
    multiplier = {"weak": 2.0, "strong": 0.5}.get(key, 1.0)
    process: dict[str, SDPrior] = {}
    upper_by_process: dict[str, float] = {}
    for name in compiled.noise_names:
        reference = max(
            float(compiled.process_reference_scale[name]),
            float(compiled.y_scale) * 1e-8,
        )
        upper = 0.10 * multiplier * reference
        if name.endswith(".slope"):
            upper /= 10.0 * float(compiled.model.period or 1)
        upper = max(upper, float(compiled.y_scale) * 1e-10)
        upper_by_process[name] = upper
        process[name] = (
            HalfNormalSD(scale=upper / 1.96)
            if key == "half_normal"
            else PCSD(upper=upper, alpha=0.05)
        )

    idiosyncratic_processes: list[str] = []
    for channel in compiled.model.channels:
        level = next(
            (
                component
                for component in channel.components
                if isinstance(component, LocalLevel) and component.mode == "dynamic"
            ),
            None,
        )
        if level is not None:
            name = f"channel.{channel.name}.{level.name}"
            if name in process:
                idiosyncratic_processes.append(name)

    horseshoe = None
    if key == "regularized_horseshoe" and idiosyncratic_processes:
        horseshoe = RegularizedHorseshoePrior(
            coefficient_scale={
                name: max(upper_by_process[name], float(compiled.y_scale) * 1e-10)
                for name in idiosyncratic_processes
            },
            global_scale=0.25,
            slab_scale=2.0,
            slab_df=4.0,
        )

    observation_sd: dict[str, SDPrior] = {}
    shape: dict[str, ShapePrior] = {}
    for channel in compiled.model.channels:
        scale = max(
            float(compiled.channel_observation_scale[channel.name]),
            float(compiled.channel_y_scale[channel.name]) * 1e-8,
        )
        observation_sd[channel.name] = HalfStudentTSD(df=4.0, scale=scale)
        if channel.family == "gev":
            lower, upper = channel.observation.xi_bounds
            shape[channel.name] = TruncatedNormalPrior(
                mean=0.0,
                sd=0.2,
                lower=lower,
                upper=upper,
            )

    intercept: dict[str, NormalPrior] = {}
    seasonal_initial: dict[str, DiagonalNormalPrior] = {}
    for channel in compiled.model.channels:
        if channel.components:
            intercept[channel.name] = NormalPrior(
                mean=float(compiled.channel_location[channel.name]),
                sd=max(
                    5.0 * float(compiled.channel_observation_scale[channel.name]),
                    float(compiled.channel_y_scale[channel.name]),
                    1e-6,
                ),
            )
        seasonal = next(
            (
                component
                for component in channel.components
                if isinstance(component, DummySeasonal) and component.mode != "off"
            ),
            None,
        )
        if seasonal is not None:
            dimension = int(seasonal.period) - 1
            scale = max(float(compiled.channel_observation_scale[channel.name]), 1e-6)
            seasonal_initial[channel.name] = DiagonalNormalPrior(
                mean=tuple(np.zeros(dimension)),
                sd=tuple(np.full(dimension, 2.0 * scale)),
            )

    factor_initial_slope: dict[str, NormalPrior] = {}
    for factor in compiled.model.factors:
        trend = next(
            (
                component
                for component in factor.components
                if isinstance(component, LocalLinearTrend)
                and component.trend_mode != "off"
            ),
            None,
        )
        if trend is not None:
            process_name = f"factor.{factor.name}.{trend.slope_name}"
            scale = max(
                float(upper_by_process.get(process_name, compiled.difference_scale)),
                float(compiled.y_scale) * 1e-8,
            )
            factor_initial_slope[factor.name] = NormalPrior(0.0, 5.0 * scale)
    return FactorPriors(
        process=process,
        observation_sd=observation_sd,
        shape=shape,
        intercept=intercept,
        factor_initial_slope=factor_initial_slope,
        seasonal_initial=seasonal_initial,
        horseshoe=horseshoe,
        horseshoe_processes=tuple(idiosyncratic_processes if horseshoe is not None else ()),
        profile=key,
        metadata={
            "channel_observation_scale": dict(compiled.channel_observation_scale),
            "channel_difference_scale": dict(compiled.channel_difference_scale),
            "upper_by_process": upper_by_process,
            "n_time": int(compiled.n_time),
            "identification": (
                "Each estimated factor uses a fixed non-zero loading anchor; "
                "loading priors are stored in the Factor model."
            ),
            "calibration": (
                "Process priors use local-change scales and do not depend on record length."
            ),
            "horseshoe_scope": (
                "independent channel local-level innovations only"
                if horseshoe is not None
                else None
            ),
        },
    )


def resolve_factor_priors(
    compiled: CompiledFactorModel,
    priors: FactorPriors | str | None,
) -> FactorPriors:
    if priors is None:
        resolved = default_factor_priors(compiled)
    elif isinstance(priors, str):
        resolved = default_factor_priors(compiled, priors)
    elif isinstance(priors, FactorPriors):
        resolved = priors
    else:
        raise TypeError("Factor priors must be FactorPriors, a profile name, or None.")

    # FactorPriors predates the v2.1 FS decomposition.  Preserve custom v2.0
    # process/observation priors while supplying the newly explicit static
    # coefficient priors from the same data-scaled rules used by the defaults.
    # User-supplied entries always win, and unknown entries are rejected below.
    static_defaults = default_factor_priors(compiled, "regularized")
    resolved = replace(
        resolved,
        intercept={**static_defaults.intercept, **resolved.intercept},
        factor_initial_slope={
            **static_defaults.factor_initial_slope,
            **resolved.factor_initial_slope,
        },
        seasonal_initial={
            **static_defaults.seasonal_initial,
            **resolved.seasonal_initial,
        },
    )

    expected_process = set(compiled.noise_names)
    expected_channels = set(compiled.channel_names)
    expected_shape = {
        channel.name for channel in compiled.model.channels if channel.family == "gev"
    }
    checks = (
        ("process", expected_process, set(resolved.process)),
        ("observation_sd", expected_channels, set(resolved.observation_sd)),
        ("shape", expected_shape, set(resolved.shape)),
        ("intercept", set(static_defaults.intercept), set(resolved.intercept)),
        (
            "factor_initial_slope",
            set(static_defaults.factor_initial_slope),
            set(resolved.factor_initial_slope),
        ),
        (
            "seasonal_initial",
            set(static_defaults.seasonal_initial),
            set(resolved.seasonal_initial),
        ),
    )
    for label, expected, supplied in checks:
        if supplied != expected:
            raise ValueError(
                f"Factor {label} priors do not match the model; "
                f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}."
            )
    for name, prior in resolved.observation_sd.items():
        if isinstance(prior, FixedSD) and prior.value <= 0.0:
            raise ValueError(f"A fixed observation SD for '{name}' must be positive.")
    for channel in compiled.model.channels:
        if channel.family != "gev":
            continue
        prior = resolved.shape[channel.name]
        model_lower, model_upper = channel.observation.xi_bounds
        if prior.lower < model_lower or prior.upper > model_upper:
            raise ValueError(
                f"Shape prior for '{channel.name}' lies outside its GEV xi_bounds."
            )
    if resolved.horseshoe is None and resolved.horseshoe_processes:
        raise ValueError("horseshoe_processes requires a regularized-horseshoe prior.")
    if resolved.horseshoe is not None:
        selected = set(resolved.horseshoe_processes)
        if selected != set(resolved.horseshoe.coefficient_scale):
            raise ValueError(
                "horseshoe_processes must match horseshoe.coefficient_scale keys."
            )
        if not selected <= expected_process:
            raise ValueError(
                f"Unknown factor horseshoe processes: {sorted(selected - expected_process)}."
            )
    metadata = dict(resolved.metadata)
    metadata.update(
        resolved_initial_mean=compiled.initial_mean.tolist(),
        resolved_initial_sd=np.sqrt(np.diag(compiled.initial_cov)).tolist(),
    )
    return replace(resolved, metadata=metadata)


__all__ = ["FactorPriors", "default_factor_priors", "resolve_factor_priors"]
