"""Priors for shared-factor and multichannel observation parameters."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

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


@dataclass(frozen=True)
class FactorPriors:
    """Explicit priors for one compiled multichannel model."""

    process: Mapping[str, SDPrior]
    observation_sd: Mapping[str, SDPrior]
    shape: Mapping[str, ShapePrior] = field(default_factory=dict)
    profile: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "process": {name: prior.to_dict() for name, prior in self.process.items()},
            "observation_sd": {
                name: prior.to_dict() for name, prior in self.observation_sd.items()
            },
            "shape": {name: prior.to_dict() for name, prior in self.shape.items()},
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
            profile=value.get("profile", "custom"),
            metadata=value.get("metadata", {}),
        )


def default_factor_priors(
    compiled: CompiledFactorModel,
    profile: str = "regularized",
) -> FactorPriors:
    """Data-scaled defaults that do not depend on record length."""

    key = str(profile).lower().replace("-", "_")
    key = {"normal": "half_normal", "pc": "regularized"}.get(key, key)
    if key not in {"regularized", "half_normal", "weak", "strong"}:
        raise ValueError(
            "Factor prior profile must be regularized, half_normal/normal, weak, or strong."
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
    return FactorPriors(
        process=process,
        observation_sd=observation_sd,
        shape=shape,
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

    expected_process = set(compiled.noise_names)
    expected_channels = set(compiled.channel_names)
    expected_shape = {
        channel.name for channel in compiled.model.channels if channel.family == "gev"
    }
    checks = (
        ("process", expected_process, set(resolved.process)),
        ("observation_sd", expected_channels, set(resolved.observation_sd)),
        ("shape", expected_shape, set(resolved.shape)),
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
    metadata = dict(resolved.metadata)
    metadata.update(
        resolved_initial_mean=compiled.initial_mean.tolist(),
        resolved_initial_sd=np.sqrt(np.diag(compiled.initial_cov)).tolist(),
    )
    return replace(resolved, metadata=metadata)


__all__ = ["FactorPriors", "default_factor_priors", "resolve_factor_priors"]
