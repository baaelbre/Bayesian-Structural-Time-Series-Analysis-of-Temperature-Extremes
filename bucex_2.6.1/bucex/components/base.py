"""Shared structural-component contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

Array = np.ndarray
ParamDict = Dict[str, Any]


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    mode: str
    state_dim: int
    noise_dim: int
    state_names: Tuple[str, ...] = ()
    noise_names: Tuple[str, ...] = ()


class Component(Protocol):
    spec: ComponentSpec

    def initial_mean_var(self, params: ParamDict) -> Tuple[Array, Array]: ...
    def system_matrices(self, t: int, params: ParamDict) -> Tuple[Array, Array, Array, Array]: ...
    def design_matrices(
        self,
        t: int,
        params: ParamDict,
        exog_t: Optional[Array] = None,
    ) -> Tuple[Array, Array]: ...
    def to_dict(self) -> dict[str, Any]: ...


def process_variance(
    params: ParamDict,
    *,
    noise_name: str,
    legacy_key: str,
    default: float = 0.0,
) -> float:
    """Read a process variance from either canonical or 0.3 parameter names."""

    canonical = f"sd.{noise_name}"
    if canonical in params:
        value = float(params[canonical]) ** 2
    else:
        value = float(params.get(legacy_key, default))
    if value < 0.0 or not np.isfinite(value):
        raise ValueError(f"Process variance for '{noise_name}' must be finite and non-negative.")
    return value
