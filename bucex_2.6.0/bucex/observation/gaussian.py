"""Gaussian observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.stats import norm

from .base import ObsSpec, resolve_observation_parameters


@dataclass(frozen=True)
class Gaussian:
    """Gaussian observations with structural mean and static scale."""

    name: str = field(default="gaussian", init=False)
    spec: ObsSpec = field(default=ObsSpec("gaussian"), init=False, repr=False)

    def logpdf(
        self,
        y: Any,
        eta: Any,
        params: Mapping[str, float] | float | None = None,
        *,
        sigma: float | None = None,
        xi: float | None = None,
    ):
        sigma, _ = resolve_observation_parameters(params, sigma=sigma, xi=xi)
        if np.any(np.asarray(sigma) <= 0.0):
            return np.full(np.broadcast_shapes(np.shape(y), np.shape(eta)), -np.inf)
        value = norm.logpdf(y, loc=eta, scale=sigma)
        return float(value) if np.ndim(value) == 0 else value

    def cdf(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, _ = resolve_observation_parameters(params, sigma=sigma, xi=xi)
        return norm.cdf(y, loc=eta, scale=sigma)

    def ppf(self, probability: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, _ = resolve_observation_parameters(params, sigma=sigma, xi=xi)
        return norm.ppf(probability, loc=eta, scale=sigma)

    def sample(
        self,
        eta: Any,
        params: Mapping[str, float] | float | None = None,
        rng: np.random.Generator | None = None,
        *,
        sigma: float | None = None,
        xi: float | None = None,
    ):
        sigma, _ = resolve_observation_parameters(params, sigma=sigma, xi=xi)
        rng = np.random.default_rng() if rng is None else rng
        value = rng.normal(loc=eta, scale=sigma, size=np.shape(eta))
        return float(value) if np.ndim(value) == 0 else value

    def grad_eta(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, _ = resolve_observation_parameters(params, sigma=sigma, xi=xi)
        value = (np.asarray(y) - np.asarray(eta)) / sigma**2
        return float(value) if np.ndim(value) == 0 else value

    def hess_eta(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, _ = resolve_observation_parameters(params, sigma=sigma, xi=xi)
        value = np.full(np.broadcast_shapes(np.shape(y), np.shape(eta)), -1.0 / sigma**2)
        return float(value) if np.ndim(value) == 0 else value

    def support_ok(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None) -> bool:
        sigma, _ = resolve_observation_parameters(params, sigma=sigma, xi=xi)
        return bool(np.all(np.asarray(sigma) > 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.name}


GaussianObs = Gaussian
