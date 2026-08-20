"""Generalized-extreme-value observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.stats import genextreme

from .base import ObsSpec, resolve_observation_parameters


@dataclass(frozen=True)
class GEV:
    """GEV observations using the EVT shape convention ``xi``.

    SciPy uses ``c=-xi``.  Both legacy ``params={"sigma", "xi"}`` and
    explicit keyword calls are accepted.
    """

    xi_bounds: tuple[float, float] = (-0.5, 0.5)
    name: str = field(default="gev", init=False)
    spec: ObsSpec = field(default=ObsSpec("gev"), init=False, repr=False)

    def __post_init__(self) -> None:
        lo, hi = map(float, self.xi_bounds)
        if not lo < hi:
            raise ValueError("GEV xi_bounds must satisfy lower < upper.")
        object.__setattr__(self, "xi_bounds", (lo, hi))

    @staticmethod
    def _parameters(params, sigma, xi):
        sigma, xi = resolve_observation_parameters(params, sigma=sigma, xi=xi)
        if xi is None:
            raise TypeError("GEV observations require shape parameter 'xi'.")
        return sigma, xi

    def logpdf(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, xi = self._parameters(params, sigma, xi)
        if np.any(np.asarray(sigma) <= 0.0):
            return np.full(
                np.broadcast_shapes(np.shape(y), np.shape(eta), np.shape(xi)),
                -np.inf,
            )
        value = genextreme.logpdf(y, c=-np.asarray(xi), loc=eta, scale=sigma)
        return float(value) if np.ndim(value) == 0 else value

    def cdf(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, xi = self._parameters(params, sigma, xi)
        return genextreme.cdf(y, c=-np.asarray(xi), loc=eta, scale=sigma)

    def ppf(self, probability: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, xi = self._parameters(params, sigma, xi)
        return genextreme.ppf(probability, c=-np.asarray(xi), loc=eta, scale=sigma)

    def sample(
        self,
        eta: Any,
        params: Mapping[str, float] | None = None,
        rng: np.random.Generator | None = None,
        *,
        sigma: float | None = None,
        xi: float | None = None,
    ):
        sigma, xi = self._parameters(params, sigma, xi)
        rng = np.random.default_rng() if rng is None else rng
        value = genextreme.rvs(
            c=-xi,
            loc=eta,
            scale=sigma,
            size=np.shape(eta),
            random_state=rng,
        )
        return float(value) if np.ndim(value) == 0 else value

    def grad_eta(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, xi = self._parameters(params, sigma, xi)
        y_arr, eta_arr = np.broadcast_arrays(
            np.asarray(y, dtype=float), np.asarray(eta, dtype=float)
        )
        if np.any(np.asarray(sigma) <= 0.0):
            value = np.full(y_arr.shape, np.nan)
        else:
            z = (y_arr - eta_arr) / sigma
            if abs(xi) < 1e-7:
                value = (1.0 - np.exp(-z)) / sigma
            else:
                support = 1.0 + xi * z
                value = np.full(y_arr.shape, np.nan)
                ok = support > 0.0
                t = support[ok]
                value[ok] = ((1.0 + xi) / t - t ** (-1.0 / xi - 1.0)) / sigma
        return float(value) if np.ndim(value) == 0 else value

    def hess_eta(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None):
        sigma, xi = self._parameters(params, sigma, xi)
        y_arr, eta_arr = np.broadcast_arrays(
            np.asarray(y, dtype=float), np.asarray(eta, dtype=float)
        )
        if np.any(np.asarray(sigma) <= 0.0):
            value = np.full(y_arr.shape, np.nan)
        else:
            z = (y_arr - eta_arr) / sigma
            if abs(xi) < 1e-7:
                value = -np.exp(-z) / sigma**2
            else:
                support = 1.0 + xi * z
                value = np.full(y_arr.shape, np.nan)
                ok = support > 0.0
                t = support[ok]
                value[ok] = ((1.0 + xi) / sigma**2) * (
                    xi / t**2 - t ** (-1.0 / xi - 2.0)
                )
        return float(value) if np.ndim(value) == 0 else value

    def support_ok(self, y: Any, eta: Any, params=None, *, sigma=None, xi=None) -> bool:
        sigma, xi = self._parameters(params, sigma, xi)
        if np.any(np.asarray(sigma) <= 0.0):
            return False
        if abs(xi) < 1e-12:
            return True
        y_arr, eta_arr = np.broadcast_arrays(
            np.asarray(y, dtype=float), np.asarray(eta, dtype=float)
        )
        return bool(np.all(1.0 + xi * (y_arr - eta_arr) / sigma > 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.name, "xi_bounds": list(self.xi_bounds)}


GEVObs = GEV
