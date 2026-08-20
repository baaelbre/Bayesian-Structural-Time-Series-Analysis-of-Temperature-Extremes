"""Observation-family contracts used by every bucex inference strategy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ObsSpec:
    """Small compatibility descriptor retained from bucex 0.3."""

    name: str


@runtime_checkable
class ObservationModel(Protocol):
    """Common scalar/vector observation interface.

    Implementations accept either the old ``params={...}`` calling convention
    or explicit ``sigma=``/``xi=`` keywords.  This is the only observation
    contract used by the centred, FS and disturbance parameterizations.
    """

    name: str
    spec: ObsSpec

    def logpdf(self, y: Any, eta: Any, params: Mapping[str, float] | None = None, **kwargs): ...
    def sample(
        self,
        eta: Any,
        params: Mapping[str, float] | None = None,
        rng: np.random.Generator | None = None,
        **kwargs,
    ): ...
    def grad_eta(self, y: Any, eta: Any, params: Mapping[str, float] | None = None, **kwargs): ...
    def hess_eta(self, y: Any, eta: Any, params: Mapping[str, float] | None = None, **kwargs): ...
    def to_dict(self) -> dict[str, Any]: ...


def resolve_observation_parameters(
    params: Mapping[str, float] | float | None,
    *,
    sigma: Any | None,
    xi: Any | None,
) -> tuple[Any, Any | None]:
    """Resolve both supported observation call styles in one place."""

    if isinstance(params, Mapping):
        sigma = params["sigma"] if sigma is None else sigma
        if xi is None and "xi" in params:
            xi = params["xi"]
    elif params is not None and sigma is None:
        # ``Gaussian.logpdf(y, eta, sigma)`` remains a convenient positional
        # form; GEV callers should pass xi explicitly.
        sigma = params
    if sigma is None:
        raise TypeError("A positive observation scale 'sigma' is required.")
    sigma_array = np.asarray(sigma, dtype=float)
    xi_array = None if xi is None else np.asarray(xi, dtype=float)
    resolved_sigma = float(sigma_array) if sigma_array.ndim == 0 else sigma_array
    resolved_xi = (
        None
        if xi_array is None
        else (float(xi_array) if xi_array.ndim == 0 else xi_array)
    )
    return resolved_sigma, resolved_xi
