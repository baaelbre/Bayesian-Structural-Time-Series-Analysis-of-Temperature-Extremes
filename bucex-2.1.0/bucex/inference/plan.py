"""Resolve and validate model, engine and parameterization choices."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..components import DummySeasonal, LocalLinearTrend, Regression
from ..models.compiler import CompiledModel


@dataclass(frozen=True)
class InferencePlan:
    family: str
    engine: str
    parameterization: str
    asis: bool
    state_update: str
    targets_exact_posterior: bool
    approximation: str | None
    interweaves_with: str | None = None
    backend: str = "state_space"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InferencePlan":
        payload = dict(value)
        payload["warnings"] = tuple(payload.get("warnings", ()))
        payload.setdefault("interweaves_with", None)
        payload.setdefault("backend", "state_space")
        return cls(**payload)


def supports_fs(compiled: CompiledModel) -> bool:
    """Whether the exact FS augmented layout exists for this model."""

    if hasattr(compiled, "channel_names"):
        return bool(
            getattr(compiled.model, "supports_fs_parameterization", False)
        )
    if not isinstance(compiled, CompiledModel):
        return False

    trend = [component for component in compiled.model.components if isinstance(component, LocalLinearTrend)]
    seasonal = [component for component in compiled.model.components if isinstance(component, DummySeasonal)]
    regression = [component for component in compiled.model.components if isinstance(component, Regression)]
    return bool(
        len(trend) == 1
        and trend[0].level_mode == "dynamic"
        and trend[0].trend_mode in {"dynamic", "off"}
        and len(seasonal) <= 1
        and all(component.mode in {"dynamic", "off"} for component in seasonal)
        and not regression
    )


def normalize_parameterization(value: str, *, fs_supported: bool) -> str:
    key = str(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "centred": "centered",
        "fs": "fruehwirth_schnatter",
        "fruhwirth_schnatter": "fruehwirth_schnatter",
        "noncentered": "fruehwirth_schnatter",
        "noncentred": "fruehwirth_schnatter",
        "ncp": "fruehwirth_schnatter",
        "scaled_disturbances": "disturbance",
        "scaled_disturbance": "disturbance",
        "innovation": "disturbance",
    }
    if key == "auto":
        return "fruehwirth_schnatter" if fs_supported else "disturbance"
    resolved = aliases.get(key, key)
    if resolved not in {"centered", "fruehwirth_schnatter", "disturbance"}:
        raise ValueError(
            "parameterization must be centered, fruehwirth_schnatter, "
            "disturbance, or auto."
        )
    if resolved == "fruehwirth_schnatter" and not fs_supported:
        raise ValueError(
            "The FS augmented parameterization requires either one univariate "
            "local-linear trend with optional dummy seasonality, or the v2.1 "
            "single-factor layout with dynamic idiosyncratic local levels and "
            "series-specific dummy seasonality. Use "
            "parameterization='disturbance' for other model graphs."
        )
    return resolved


def inference_plan(
    compiled: Any,
    *,
    engine: str = "auto",
    parameterization: str = "auto",
    asis: bool = False,
) -> InferencePlan:
    family = compiled.family
    is_factor = hasattr(compiled, "channel_names")
    all_gaussian = bool(getattr(compiled, "all_gaussian", family == "gaussian"))
    requested_engine = str(engine).lower().replace("-", "_")
    if requested_engine == "particle":
        requested_engine = "pgas"
    resolved_engine = (
        "ffbs" if all_gaussian else ("pgas" if is_factor else "laplace")
    ) if requested_engine == "auto" else requested_engine
    allowed = {"ffbs"} if all_gaussian else {"laplace", "pgas"}
    if resolved_engine not in allowed:
        raise ValueError(
            f"engine='{resolved_engine}' is incompatible with family='{family}'; "
            f"choose {sorted(allowed)}."
        )

    resolved_parameterization = normalize_parameterization(
        parameterization,
        fs_supported=supports_fs(compiled),
    )
    exact = resolved_engine in {"ffbs", "pgas"}
    approximation = None if exact else "iterated_laplace"
    if asis:
        interweaves_with = (
            "centered"
            if resolved_parameterization in {"fruehwirth_schnatter", "disturbance"}
            else "disturbance"
        )
    else:
        interweaves_with = None

    warnings: list[str] = []
    covariance_rank = np.linalg.matrix_rank(compiled.loading @ compiled.loading.T)
    if resolved_engine == "pgas" and covariance_rank < compiled.state_dim:
        warnings.append(
            "The transition is singular; PGAS operates on its affine support. "
            "Inspect particle ESS and ancestor diversity."
        )
    if is_factor:
        warnings.append(
            "Channel likelihoods are conditionally independent given shared and "
            "individual states; residual/copula dependence is not modeled."
        )
    state_update = {
        "ffbs": "exact Gaussian FFBS",
        "pgas": "conditional SMC with ancestor sampling",
        "laplace": "iterated Laplace FFBS approximation",
    }[resolved_engine]
    return InferencePlan(
        family=family,
        engine=resolved_engine,
        parameterization=resolved_parameterization,
        asis=bool(asis),
        interweaves_with=interweaves_with,
        backend=(
            "factor_state_space"
            if is_factor
            else (
                "fruehwirth_schnatter"
                if resolved_parameterization == "fruehwirth_schnatter"
                else "state_space"
            )
        ),
        state_update=state_update,
        targets_exact_posterior=exact,
        approximation=approximation,
        warnings=tuple(warnings),
    )
