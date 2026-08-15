"""Priors for shared-factor and multichannel observation parameters."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

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
from .structural import (
    DiagonalNormalPrior,
    NormalPrior,
    RegularizedHorseshoePrior,
    TripleGammaPrior,
)


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
    triple_gamma: TripleGammaPrior | None = None
    triple_gamma_processes: tuple[str, ...] = ()
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
            "triple_gamma": (
                None
                if self.triple_gamma is None
                else {
                    "coefficient_scale": dict(self.triple_gamma.coefficient_scale),
                    "spike_shape": float(self.triple_gamma.spike_shape),
                    "tail_shape": float(self.triple_gamma.tail_shape),
                    "global_scale": float(self.triple_gamma.global_scale),
                    "learn_global": bool(self.triple_gamma.learn_global),
                    "learn_shapes": bool(self.triple_gamma.learn_shapes),
                    "spike_shape_prior": list(self.triple_gamma.spike_shape_prior),
                    "tail_shape_prior": list(self.triple_gamma.tail_shape_prior),
                    "regularized": bool(self.triple_gamma.regularized),
                    "slab_scale": float(self.triple_gamma.slab_scale),
                    "slab_df": float(self.triple_gamma.slab_df),
                    "initial_numerator": float(self.triple_gamma.initial_numerator),
                    "initial_denominator": float(self.triple_gamma.initial_denominator),
                    "initial_slab2": self.triple_gamma.initial_slab2,
                }
            ),
            "triple_gamma_processes": list(self.triple_gamma_processes),
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
            triple_gamma=(
                None
                if value.get("triple_gamma") is None
                else TripleGammaPrior(**value["triple_gamma"])
            ),
            triple_gamma_processes=tuple(
                value.get("triple_gamma_processes", ())
            ),
            profile=value.get("profile", "custom"),
            metadata=value.get("metadata", {}),
        )


def normalize_factor_prior_profile(profile: str | None) -> str:
    """Normalize documented aliases, whitespace and UK/US spellings."""

    raw = "regularized_horseshoe" if profile is None else str(profile)
    key = "_".join(raw.strip().lower().replace("-", "_").split())
    key = key.replace("regularised", "regularized")
    aliases = {
        "default": "regularized_horseshoe",
        "rh": "regularized_horseshoe",
        "horseshoe": "regularized_horseshoe",
        "tg": "triple_gamma",
        "triplegamma": "triple_gamma",
        "regularized_tg": "regularized_triple_gamma",
        "pc": "regularized",
        "normal": "half_normal",
    }
    return aliases.get(key, key)


def default_factor_priors(
    compiled: CompiledFactorModel,
    profile: str = "regularized_horseshoe",
    *,
    triple_gamma_options: Mapping[str, Any] | None = None,
) -> FactorPriors:
    """Data-scaled defaults that do not depend on record length.

    ``triple_gamma_options`` is forwarded to :class:`TripleGammaPrior` after
    the namespaced, data-scaled coefficient scales have been constructed. It
    provides a concise factor-model API for changing ``a``, ``c``,
    global/shape learning, or the optional slab without rebuilding the full
    :class:`FactorPriors` object.
    """

    key = normalize_factor_prior_profile(profile)
    if key not in {
        "regularized_horseshoe",
        "triple_gamma",
        "regularized_triple_gamma",
        "regularized",
        "half_normal",
        "weak",
        "strong",
    }:
        raise ValueError(
            f"Unknown factor prior profile {profile!r}. Choose "
            "regularized_horseshoe (or horseshoe), regularized (or pc), "
            "triple_gamma (or tg), regularized_triple_gamma, "
            "half_normal (or normal), weak, or strong. Univariate-only "
            "profiles such as manuscript_lasso are not factor priors."
        )
    if triple_gamma_options is not None and key not in {
        "triple_gamma", "regularized_triple_gamma"
    }:
        raise ValueError(
            "triple_gamma_options requires profile='triple_gamma' or "
            "'regularized_triple_gamma'."
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
    triple_gamma = None
    if key in {"triple_gamma", "regularized_triple_gamma"} and idiosyncratic_processes:
        options: dict[str, Any] = {
            "coefficient_scale": {
                name: max(upper_by_process[name], float(compiled.y_scale) * 1e-10)
                for name in idiosyncratic_processes
            },
            "spike_shape": 0.10,
            "tail_shape": 0.10,
            "global_scale": 1.0,
            "learn_global": True,
            "learn_shapes": False,
            "regularized": key == "regularized_triple_gamma",
            "slab_scale": 2.0,
            "slab_df": 4.0,
        }
        options.update({} if triple_gamma_options is None else triple_gamma_options)
        if bool(options["regularized"]) != (key == "regularized_triple_gamma"):
            raise ValueError(
                "The triple-gamma regularized option must agree with the "
                "selected factor prior profile."
            )
        triple_gamma = TripleGammaPrior(**options)

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
        triple_gamma=triple_gamma,
        triple_gamma_processes=tuple(
            idiosyncratic_processes if triple_gamma is not None else ()
        ),
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
            "triple_gamma_scope": (
                "independent channel local-level innovations only"
                if triple_gamma is not None
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
    if resolved.triple_gamma is None and resolved.triple_gamma_processes:
        raise ValueError(
            "triple_gamma_processes requires a triple-gamma prior."
        )
    if resolved.triple_gamma is not None:
        selected = set(resolved.triple_gamma_processes)
        if selected != set(resolved.triple_gamma.coefficient_scale):
            raise ValueError(
                "triple_gamma_processes must match "
                "triple_gamma.coefficient_scale keys."
            )
        if not selected <= expected_process:
            raise ValueError(
                "Unknown factor triple-gamma processes: "
                f"{sorted(selected - expected_process)}."
            )
    if resolved.horseshoe is not None and resolved.triple_gamma is not None:
        raise ValueError(
            "Choose either factor horseshoe or factor triple gamma, not both."
        )
    metadata = dict(resolved.metadata)
    metadata.update(
        resolved_initial_mean=compiled.initial_mean.tolist(),
        resolved_initial_sd=np.sqrt(np.diag(compiled.initial_cov)).tolist(),
    )
    return replace(resolved, metadata=metadata)


def identified_factor_priors(
    compiled: CompiledFactorModel,
    profile: str = "regularized_horseshoe",
    *,
    triple_gamma_options: Mapping[str, Any] | None = None,
    smooth_factor: bool = True,
    reference_channel: str | None = None,
    fixed_idiosyncratic: str | Iterable[str] = (),
) -> FactorPriors:
    """Build factor priors with explicit dynamic-identification constraints.

    ``smooth_factor=True`` fixes direct factor-level innovations at zero while
    retaining stochastic slope innovations. ``reference_channel`` fixes that
    channel's idiosyncratic local-level innovation at zero, making it a pure
    reference trajectory. Additional deviations can be fixed through
    ``fixed_idiosyncratic``; use ``"all"`` for a loading-only sensitivity fit.

    These constraints address different scientific questions explicitly. They
    do not pretend that an estimated loading and an unrestricted persistent
    idiosyncratic random walk are separately identified by the likelihood.
    """

    priors = default_factor_priors(
        compiled,
        profile,
        triple_gamma_options=triple_gamma_options,
    )
    known_channels = set(compiled.channel_names)
    if isinstance(fixed_idiosyncratic, str):
        fixed_channels = (
            set(known_channels)
            if fixed_idiosyncratic.strip().lower() == "all"
            else {fixed_idiosyncratic.strip()}
        )
    else:
        fixed_channels = {str(name) for name in fixed_idiosyncratic}
    if reference_channel is not None:
        fixed_channels.add(str(reference_channel))
    unknown = sorted(fixed_channels - known_channels)
    if unknown:
        raise KeyError(
            f"Unknown fixed-idiosyncratic channels {unknown}; "
            f"available={sorted(known_channels)}."
        )

    process = dict(priors.process)
    fixed_processes: set[str] = set()
    if smooth_factor:
        for factor in compiled.model.factors:
            for component in factor.components:
                if not isinstance(component, LocalLinearTrend):
                    continue
                name = f"factor.{factor.name}.{component.level_name}"
                if name in process:
                    process[name] = FixedSD(0.0)
                    fixed_processes.add(name)
    for channel in compiled.model.channels:
        if channel.name not in fixed_channels:
            continue
        for component in channel.components:
            if isinstance(component, LocalLevel) and component.mode == "dynamic":
                name = f"channel.{channel.name}.{component.name}"
                if name in process:
                    process[name] = FixedSD(0.0)
                    fixed_processes.add(name)

    selected = tuple(
        name
        for name in priors.horseshoe_processes
        if name not in fixed_processes
    )
    horseshoe = priors.horseshoe
    if horseshoe is not None:
        coefficient_scale = {
            name: scale
            for name, scale in horseshoe.coefficient_scale.items()
            if name in selected
        }
        horseshoe = (
            replace(horseshoe, coefficient_scale=coefficient_scale)
            if coefficient_scale
            else None
        )

    triple_gamma_selected = tuple(
        name
        for name in priors.triple_gamma_processes
        if name not in fixed_processes
    )
    triple_gamma = priors.triple_gamma
    if triple_gamma is not None:
        coefficient_scale = {
            name: scale
            for name, scale in triple_gamma.coefficient_scale.items()
            if name in triple_gamma_selected
        }
        triple_gamma = (
            replace(triple_gamma, coefficient_scale=coefficient_scale)
            if coefficient_scale
            else None
        )

    metadata = dict(priors.metadata)
    metadata.update(
        identification_strategy="explicit_constraints",
        smooth_factor=bool(smooth_factor),
        pure_reference_channel=reference_channel,
        fixed_idiosyncratic_channels=sorted(fixed_channels),
        fixed_processes=sorted(fixed_processes),
    )
    result = replace(
        priors,
        process=process,
        horseshoe=horseshoe,
        horseshoe_processes=selected,
        triple_gamma=triple_gamma,
        triple_gamma_processes=triple_gamma_selected,
        profile=f"identified_{priors.profile}",
        metadata=metadata,
    )
    return resolve_factor_priors(compiled, result)


__all__ = [
    "FactorPriors",
    "default_factor_priors",
    "identified_factor_priors",
    "normalize_factor_prior_profile",
    "resolve_factor_priors",
]
