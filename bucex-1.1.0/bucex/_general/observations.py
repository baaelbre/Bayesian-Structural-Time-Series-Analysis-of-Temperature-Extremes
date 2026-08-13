"""Observation families used by the structural model grammar."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import genextreme, norm


Array = np.ndarray


@dataclass(frozen=True)
class Gaussian:
    """Gaussian observation model with a static standard deviation."""

    name: str = "gaussian"

    def logpdf(self, y: Any, eta: Any, sigma: float, xi: float | None = None):
        return norm.logpdf(y, loc=eta, scale=sigma)

    def cdf(self, y: Any, eta: Any, sigma: float, xi: float | None = None):
        return norm.cdf(y, loc=eta, scale=sigma)

    def ppf(self, probability: Any, eta: Any, sigma: float, xi: float | None = None):
        return norm.ppf(probability, loc=eta, scale=sigma)

    def sample(
        self,
        eta: Any,
        sigma: float,
        rng: np.random.Generator,
        xi: float | None = None,
    ):
        return rng.normal(loc=eta, scale=float(sigma), size=np.shape(eta))

    def grad_eta(self, y: Any, eta: Any, sigma: float, xi: float | None = None):
        sigma2 = float(sigma) ** 2
        return (np.asarray(y) - np.asarray(eta)) / sigma2

    def hess_eta(self, y: Any, eta: Any, sigma: float, xi: float | None = None):
        shape = np.broadcast_shapes(np.shape(y), np.shape(eta))
        return np.full(shape, -1.0 / float(sigma) ** 2)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.name}


@dataclass(frozen=True)
class GEV:
    """GEV observations with structural location and static scale and shape.

    The shape follows the EVT convention ``xi``. SciPy uses ``c=-xi``.
    Dynamic scale or shape is intentionally outside the first-release grammar.
    """

    xi_bounds: tuple[float, float] = (-0.5, 0.5)
    name: str = "gev"

    def __post_init__(self) -> None:
        lo, hi = map(float, self.xi_bounds)
        if not lo < hi:
            raise ValueError("GEV xi_bounds must satisfy lower < upper.")

    def logpdf(self, y: Any, eta: Any, sigma: float, xi: float):
        sigma_arr = np.asarray(sigma, dtype=float)
        if np.any(sigma_arr <= 0.0):
            shape = np.broadcast_shapes(np.shape(y), np.shape(eta), np.shape(sigma), np.shape(xi))
            return np.full(shape, -np.inf)
        return genextreme.logpdf(y, c=-np.asarray(xi), loc=eta, scale=sigma)

    def cdf(self, y: Any, eta: Any, sigma: float, xi: float):
        return genextreme.cdf(y, c=-np.asarray(xi), loc=eta, scale=sigma)

    def ppf(self, probability: Any, eta: Any, sigma: float, xi: float):
        return genextreme.ppf(probability, c=-np.asarray(xi), loc=eta, scale=sigma)

    def sample(self, eta: Any, sigma: float, xi: float, rng: np.random.Generator):
        return genextreme.rvs(
            c=-float(xi),
            loc=eta,
            scale=float(sigma),
            size=np.shape(eta),
            random_state=rng,
        )

    def grad_eta(self, y: Any, eta: Any, sigma: float, xi: float):
        y_arr, eta_arr = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(eta, dtype=float))
        sigma = float(sigma)
        xi = float(xi)
        if sigma <= 0.0:
            return np.full(y_arr.shape, np.nan)
        z = (y_arr - eta_arr) / sigma
        if abs(xi) < 1e-7:
            return (1.0 - np.exp(-z)) / sigma
        support = 1.0 + xi * z
        out = np.full(y_arr.shape, np.nan)
        ok = support > 0.0
        t = support[ok]
        out[ok] = ((1.0 + xi) / t - t ** (-1.0 / xi - 1.0)) / sigma
        return out

    def hess_eta(self, y: Any, eta: Any, sigma: float, xi: float):
        y_arr, eta_arr = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(eta, dtype=float))
        sigma = float(sigma)
        xi = float(xi)
        if sigma <= 0.0:
            return np.full(y_arr.shape, np.nan)
        z = (y_arr - eta_arr) / sigma
        if abs(xi) < 1e-7:
            return -np.exp(-z) / sigma**2
        support = 1.0 + xi * z
        out = np.full(y_arr.shape, np.nan)
        ok = support > 0.0
        t = support[ok]
        out[ok] = ((1.0 + xi) / sigma**2) * (
            xi / t**2 - t ** (-1.0 / xi - 2.0)
        )
        return out

    def support_ok(self, y: Any, eta: Any, sigma: float, xi: float) -> bool:
        if float(sigma) <= 0.0:
            return False
        y_arr, eta_arr = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(eta, dtype=float))
        if abs(float(xi)) < 1e-12:
            return True
        return bool(np.all(1.0 + float(xi) * (y_arr - eta_arr) / float(sigma) > 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.name, "xi_bounds": list(self.xi_bounds)}


Observation = Gaussian | GEV


def observation_from_dict(value: dict[str, Any]) -> Observation:
    family = str(value["family"]).lower()
    if family == "gaussian":
        return Gaussian()
    if family == "gev":
        return GEV(tuple(value.get("xi_bounds", (-0.5, 0.5))))
    raise ValueError(f"Unknown observation family '{family}'.")


# Compatibility names from the prototype.
GaussianObs = Gaussian
GEVObs = GEV
