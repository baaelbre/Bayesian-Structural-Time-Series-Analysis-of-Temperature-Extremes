"""Priors for process and observation standard deviations."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol

import numpy as np
from scipy.special import logsumexp
from scipy.stats import halfnorm, invgamma, t, truncnorm

from ..components import Regression
from ..models.compiler import CompiledModel
from .structural import UniformPrior


class SDPrior(Protocol):
    def logpdf(self, value: float, indicator: int | None = None) -> float: ...
    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...] | None = None,
        indicator: int | None = None,
    ): ...
    def initial(self) -> float: ...
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class HalfNormalSD:
    scale: float

    def __post_init__(self) -> None:
        if float(self.scale) <= 0.0:
            raise ValueError("HalfNormalSD.scale must be positive.")

    def logpdf(self, value: float, indicator: int | None = None) -> float:
        return float(halfnorm.logpdf(value, scale=self.scale)) if value >= 0.0 else -np.inf

    def sample(self, rng, size=None, indicator=None):
        return np.abs(rng.normal(scale=self.scale, size=size))

    def initial(self) -> float:
        return float(self.scale) * np.sqrt(2.0 / np.pi)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "half_normal", "scale": float(self.scale)}


@dataclass(frozen=True)
class HalfStudentTSD:
    df: float
    scale: float

    def __post_init__(self) -> None:
        if float(self.df) <= 0.0 or float(self.scale) <= 0.0:
            raise ValueError("HalfStudentTSD df and scale must be positive.")

    def logpdf(self, value: float, indicator: int | None = None) -> float:
        if value < 0.0:
            return -np.inf
        return float(np.log(2.0) + t.logpdf(value / self.scale, df=self.df) - np.log(self.scale))

    def sample(self, rng, size=None, indicator=None):
        return np.abs(self.scale * rng.standard_t(df=self.df, size=size))

    def initial(self) -> float:
        return float(self.scale) * 0.75

    def to_dict(self) -> dict[str, Any]:
        return {"type": "half_student_t", "df": float(self.df), "scale": float(self.scale)}


@dataclass(frozen=True)
class ExponentialSD:
    rate: float

    def __post_init__(self) -> None:
        if float(self.rate) <= 0.0:
            raise ValueError("ExponentialSD.rate must be positive.")

    def logpdf(self, value: float, indicator: int | None = None) -> float:
        return float(np.log(self.rate) - self.rate * value) if value >= 0.0 else -np.inf

    def sample(self, rng, size=None, indicator=None):
        return rng.exponential(scale=1.0 / self.rate, size=size)

    def initial(self) -> float:
        return 1.0 / float(self.rate)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "exponential", "rate": float(self.rate)}


@dataclass(frozen=True)
class PCSD:
    """PC/exponential prior calibrated by ``P(sd > upper) = alpha``."""

    upper: float
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if float(self.upper) <= 0.0:
            raise ValueError("PCSD.upper must be positive.")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("PCSD.alpha must lie in (0, 1).")

    @property
    def rate(self) -> float:
        return -float(np.log(self.alpha)) / float(self.upper)

    def logpdf(self, value: float, indicator: int | None = None) -> float:
        return ExponentialSD(self.rate).logpdf(value)

    def sample(self, rng, size=None, indicator=None):
        return rng.exponential(scale=1.0 / self.rate, size=size)

    def initial(self) -> float:
        return 1.0 / self.rate

    def to_dict(self) -> dict[str, Any]:
        return {"type": "pc", "upper": float(self.upper), "alpha": float(self.alpha)}


@dataclass(frozen=True)
class InverseGammaVariance:
    """Inverse-gamma prior on ``sd**2`` in shape/scale form."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        if float(self.shape) <= 0.0 or float(self.scale) <= 0.0:
            raise ValueError("InverseGammaVariance shape and scale must be positive.")

    def logpdf(self, value: float, indicator: int | None = None) -> float:
        if value <= 0.0:
            return -np.inf
        return float(invgamma.logpdf(value**2, a=self.shape, scale=self.scale) + np.log(2.0 * value))

    def sample(self, rng, size=None, indicator=None):
        variance = invgamma.rvs(a=self.shape, scale=self.scale, size=size, random_state=rng)
        return np.sqrt(variance)

    def initial(self) -> float:
        if self.shape > 1.0:
            return float(np.sqrt(self.scale / (self.shape - 1.0)))
        return float(np.sqrt(self.scale / (self.shape + 1.0)))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "inverse_gamma_variance", "shape": float(self.shape), "scale": float(self.scale)}


@dataclass(frozen=True)
class FixedSD:
    value: float = 0.0

    def __post_init__(self) -> None:
        if float(self.value) < 0.0:
            raise ValueError("FixedSD.value must be non-negative.")

    def logpdf(self, value: float, indicator: int | None = None) -> float:
        return 0.0 if np.isclose(value, self.value, atol=1e-14, rtol=0.0) else -np.inf

    def sample(self, rng, size=None, indicator=None):
        if size is None:
            return float(self.value)
        return np.full(size, float(self.value))

    def initial(self) -> float:
        return float(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "fixed", "value": float(self.value)}


@dataclass(frozen=True)
class SpikeSlabSD:
    """Continuous half-normal spike-and-slab prior for a process SD."""

    spike_scale: float
    slab_scale: float
    slab_probability: float = 0.5

    def __post_init__(self) -> None:
        if float(self.spike_scale) <= 0.0 or float(self.slab_scale) <= 0.0:
            raise ValueError("SpikeSlabSD scales must be positive.")
        if not float(self.spike_scale) < float(self.slab_scale):
            raise ValueError("spike_scale must be smaller than slab_scale.")
        if not 0.0 < float(self.slab_probability) < 1.0:
            raise ValueError("slab_probability must lie in (0, 1).")

    def component_logpdf(self, value: float, indicator: int) -> float:
        scale = self.slab_scale if int(indicator) == 1 else self.spike_scale
        return HalfNormalSD(scale).logpdf(value)

    def logpdf(self, value: float, indicator: int | None = None) -> float:
        if indicator is not None:
            return self.component_logpdf(value, int(indicator))
        terms = np.asarray(
            [
                np.log1p(-self.slab_probability) + self.component_logpdf(value, 0),
                np.log(self.slab_probability) + self.component_logpdf(value, 1),
            ]
        )
        return float(logsumexp(terms))

    def indicator_probability(self, value: float) -> float:
        terms = np.asarray(
            [
                np.log1p(-self.slab_probability) + self.component_logpdf(value, 0),
                np.log(self.slab_probability) + self.component_logpdf(value, 1),
            ]
        )
        return float(np.exp(terms[1] - logsumexp(terms)))

    def sample(self, rng, size=None, indicator=None):
        if indicator is None:
            indicator = rng.binomial(1, self.slab_probability, size=size)
        scale = np.where(np.asarray(indicator) == 1, self.slab_scale, self.spike_scale)
        return np.abs(rng.normal(scale=scale, size=size))

    def initial(self) -> float:
        return float(
            np.sqrt(2.0 / np.pi)
            * ((1.0 - self.slab_probability) * self.spike_scale + self.slab_probability * self.slab_scale)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "spike_slab",
            "spike_scale": float(self.spike_scale),
            "slab_scale": float(self.slab_scale),
            "slab_probability": float(self.slab_probability),
        }


@dataclass(frozen=True)
class TruncatedNormalPrior:
    mean: float
    sd: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if float(self.sd) <= 0.0 or not float(self.lower) < float(self.upper):
            raise ValueError("Invalid TruncatedNormalPrior parameters.")

    def _ab(self) -> tuple[float, float]:
        return ((self.lower - self.mean) / self.sd, (self.upper - self.mean) / self.sd)

    def logpdf(self, value: float) -> float:
        a, b = self._ab()
        return float(truncnorm.logpdf(value, a=a, b=b, loc=self.mean, scale=self.sd))

    def sample(self, rng, size=None):
        a, b = self._ab()
        return truncnorm.rvs(a=a, b=b, loc=self.mean, scale=self.sd, size=size, random_state=rng)

    def initial(self) -> float:
        return float(np.clip(self.mean, self.lower + 1e-8, self.upper - 1e-8))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "truncated_normal",
            "mean": float(self.mean),
            "sd": float(self.sd),
            "lower": float(self.lower),
            "upper": float(self.upper),
        }


ShapePrior = UniformPrior | TruncatedNormalPrior


@dataclass(frozen=True)
class Priors:
    process: Mapping[str, SDPrior]
    observation_sd: SDPrior
    shape: ShapePrior | None = None
    profile: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "process": {name: prior.to_dict() for name, prior in self.process.items()},
            "observation_sd": self.observation_sd.to_dict(),
            "shape": None if self.shape is None else self.shape.to_dict(),
            "profile": self.profile,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Priors":
        return cls(
            process={name: sd_prior_from_dict(item) for name, item in value["process"].items()},
            observation_sd=sd_prior_from_dict(value["observation_sd"]),
            shape=None if value.get("shape") is None else shape_prior_from_dict(value["shape"]),
            profile=value.get("profile", "custom"),
            metadata=value.get("metadata", {}),
        )


def sd_prior_from_dict(value: dict[str, Any]) -> SDPrior:
    kind = value["type"]
    args = {key: val for key, val in value.items() if key != "type"}
    classes = {
        "half_normal": HalfNormalSD,
        "half_student_t": HalfStudentTSD,
        "exponential": ExponentialSD,
        "pc": PCSD,
        "inverse_gamma_variance": InverseGammaVariance,
        "fixed": FixedSD,
        "spike_slab": SpikeSlabSD,
    }
    if kind not in classes:
        raise ValueError(f"Unknown SD prior type '{kind}'.")
    return classes[kind](**args)


def shape_prior_from_dict(value: dict[str, Any]) -> ShapePrior:
    kind = value["type"]
    args = {key: val for key, val in value.items() if key != "type"}
    if kind == "uniform":
        return UniformPrior(**args)
    if kind == "truncated_normal":
        return TruncatedNormalPrior(**args)
    raise ValueError(f"Unknown shape prior type '{kind}'.")


def _adaptive_upper(name: str, compiled: CompiledModel) -> float:
    # Calibrate against local response variation, never the number of
    # observations.  A longer record should sharpen the likelihood rather
    # than silently change the prior.
    scale = max(
        float(compiled.difference_scale),
        0.05 * float(compiled.observation_scale),
        float(compiled.y_scale) * 1e-8,
    )
    regularized_upper = 0.10 * scale
    if name == "slope":
        # A slope disturbance is integrated into the level.  Scale it over a
        # ten-cycle reference horizon (ten steps without seasonality).
        horizon = 10 * int(compiled.model.period or 1)
        return max(regularized_upper / horizon, compiled.y_scale * 1e-10)
    for component in compiled.model.components:
        if not isinstance(component, Regression) or not component.dynamic:
            continue
        feature_names = component.feature_names or tuple(
            str(index + 1) for index in range(component.n_features)
        )
        for offset, feature_name in enumerate(feature_names):
            if name != f"{component.name}[{feature_name}]":
                continue
            assert compiled.exog is not None
            column = compiled.regression_slices[component.name].start + offset
            rms = float(np.sqrt(np.mean(compiled.exog[:, column] ** 2)))
            return max(regularized_upper / max(rms, 1e-8), compiled.y_scale * 1e-10)
    return max(regularized_upper, compiled.y_scale * 1e-10)


def default_priors(compiled: CompiledModel, profile: str = "regularized") -> Priors:
    """Resolve an explicit, stored prior profile in the units of the response."""

    key = str(profile).lower()
    key = {
        "normal": "half_normal",
        "ssvs": "spike_slab",
        "manuscript": "legacy_ig",
        "lasso": "regularized",
    }.get(key, key)
    if key not in {"regularized", "weak", "strong", "half_normal", "spike_slab", "legacy_ig"}:
        raise ValueError(
            "profile must be regularized, weak, strong, half_normal, spike_slab, or legacy_ig."
        )
    multiplier = {"weak": 2.0, "strong": 0.5}.get(key, 1.0)
    uppers = {name: multiplier * _adaptive_upper(name, compiled) for name in compiled.noise_names}
    process: dict[str, SDPrior] = {}
    for name, upper in uppers.items():
        if key == "half_normal":
            process[name] = HalfNormalSD(scale=upper / 1.96)
        elif key == "spike_slab":
            process[name] = SpikeSlabSD(
                spike_scale=max(upper / 20.0, compiled.y_scale * 1e-10),
                slab_scale=upper,
                slab_probability=0.5,
            )
        elif key == "legacy_ig":
            process[name] = InverseGammaVariance(shape=2.0, scale=max(upper**2, 1e-16))
        else:
            process[name] = PCSD(upper=upper, alpha=0.05)
    observation_sd: SDPrior = HalfStudentTSD(
        df=4.0,
        scale=max(compiled.observation_scale, compiled.y_scale * 1e-8),
    )
    shape: ShapePrior | None = None
    if compiled.family == "gev":
        lo, hi = compiled.model.observation.xi_bounds
        shape = TruncatedNormalPrior(mean=0.0, sd=0.2, lower=lo, upper=hi)
    return Priors(
        process=process,
        observation_sd=observation_sd,
        shape=shape,
        profile=key,
        metadata={
            "response_scale": float(compiled.y_scale),
            "observation_reference_scale": float(compiled.observation_scale),
            "change_reference_scale": float(compiled.difference_scale),
            "n_time": int(compiled.n_time),
            "period": compiled.model.period,
            "calibration": (
                "PC profiles use P(sd > upper)=0.05; regularized process "
                "uppers are 10% of a robust local-change scale, with slope "
                "calibrated over ten seasonal cycles. Calibration does not "
                "depend on record length."
            ),
            "upper_by_process": uppers,
        },
    )


def resolve_priors(compiled: CompiledModel, priors: Priors | str | None) -> Priors:
    if priors is None:
        resolved = default_priors(compiled, "regularized")
    elif isinstance(priors, str):
        resolved = default_priors(compiled, priors)
    elif isinstance(priors, Priors):
        resolved = priors
    else:
        raise TypeError("priors must be a Priors object, profile name, or None.")
    expected = set(compiled.noise_names)
    supplied = set(resolved.process)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"Process priors do not match the model; missing={missing}, extra={extra}.")
    if compiled.family == "gev" and resolved.shape is None:
        raise ValueError("A GEV model requires a shape prior.")
    if compiled.family == "gaussian" and resolved.shape is not None:
        raise ValueError("A Gaussian model must not have a shape prior.")
    if isinstance(resolved.observation_sd, FixedSD) and resolved.observation_sd.value <= 0.0:
        raise ValueError("A fixed observation SD must be strictly positive.")
    if compiled.family == "gev":
        assert resolved.shape is not None
        model_lower, model_upper = compiled.model.observation.xi_bounds
        if resolved.shape.lower < model_lower or resolved.shape.upper > model_upper:
            raise ValueError(
                "The shape prior support must lie within model.observation.xi_bounds."
            )
    metadata = dict(resolved.metadata)
    metadata.update(
        resolved_initial_mean=compiled.initial_mean.tolist(),
        resolved_initial_sd=np.sqrt(np.diag(compiled.initial_cov)).tolist(),
        initial_prior_note=(
            "Component values explicitly supplied in the model are fixed hyperparameters; "
            "unspecified values use the recorded data-adaptive convenience defaults."
        ),
    )
    return replace(resolved, metadata=metadata)
