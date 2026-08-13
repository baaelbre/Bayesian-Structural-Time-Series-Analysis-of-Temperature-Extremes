"""Validation of model, parameterization, and inference-engine combinations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .compiler import CompiledModel


@dataclass(frozen=True)
class InferencePlan:
    family: str
    engine: str
    parameterization: str
    asis: bool
    state_update: str
    targets_exact_posterior: bool
    approximation: str | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InferencePlan":
        value = dict(value)
        value["warnings"] = tuple(value.get("warnings", ()))
        return cls(**value)


def inference_plan(
    compiled: CompiledModel,
    *,
    engine: str = "auto",
    parameterization: str = "auto",
    asis: bool = False,
) -> InferencePlan:
    family = compiled.family
    requested_engine = str(engine).lower()
    if requested_engine == "particle":
        requested_engine = "pgas"
    if requested_engine == "auto":
        resolved_engine = "ffbs" if family == "gaussian" else "laplace"
    else:
        resolved_engine = requested_engine
    allowed = {"ffbs", "pgas"} if family == "gaussian" else {"laplace", "pgas"}
    if resolved_engine not in allowed:
        raise ValueError(
            f"engine='{resolved_engine}' is incompatible with family='{family}'; choose {sorted(allowed)}."
        )

    requested_parameterization = str(parameterization).lower().replace("-", "")
    aliases = {"noncentred": "noncentered", "noncentered": "noncentered", "centered": "centered", "centred": "centered"}
    if requested_parameterization == "auto":
        resolved_parameterization = "noncentered"
    elif requested_parameterization in aliases:
        resolved_parameterization = aliases[requested_parameterization]
    elif requested_parameterization == "asis":
        resolved_parameterization = "noncentered"
        asis = True
    else:
        raise ValueError("parameterization must be auto, centered, noncentered, or asis.")

    exact = resolved_engine in {"ffbs", "pgas"}
    approximation = None if exact else "iterated Laplace approximation to the GEV state conditional"
    warnings: list[str] = []
    covariance_rank = np.linalg.matrix_rank(compiled.loading @ compiled.loading.T)
    if resolved_engine == "pgas" and covariance_rank < compiled.state_dim:
        warnings.append(
            "The transition is singular; PGAS remains valid on its affine support, "
            "but deterministic lag coordinates can reduce ancestor diversity."
        )
    return InferencePlan(
        family=family,
        engine=resolved_engine,
        parameterization=resolved_parameterization,
        asis=bool(asis),
        state_update=(
            "exact Gaussian FFBS"
            if resolved_engine == "ffbs"
            else (
                "conditional SMC with ancestor sampling"
                if resolved_engine == "pgas"
                else "iterated Laplace FFBS"
            )
        ),
        targets_exact_posterior=exact,
        approximation=approximation,
        warnings=tuple(warnings),
    )
