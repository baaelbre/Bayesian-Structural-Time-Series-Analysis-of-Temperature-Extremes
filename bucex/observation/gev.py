from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import genextreme

from .base import ObsSpec, ObservationModel


@dataclass(frozen=True)
class GEVObs(ObservationModel):
    """
    y | eta ~ GEV(mu=eta, sigma, xi)

    SciPy's `genextreme` uses shape parameter c = -xi.

    Parameters expected in `params`
    -------------------------------
    sigma : float
        Positive scale parameter.
    xi : float
        Shape parameter.
    """
    spec: ObsSpec = ObsSpec("gev")

    def logpdf(self, y: float, eta: float, params: Dict[str, float]) -> float:
        sigma = float(params["sigma"])
        xi = float(params["xi"])
        return float(genextreme.logpdf(float(y), c=-xi, loc=float(eta), scale=sigma))

    def sample(self, eta: float, params: Dict[str, float], rng: np.random.Generator) -> float:
        sigma = float(params["sigma"])
        xi = float(params["xi"])
        return float(genextreme.rvs(c=-xi, loc=float(eta), scale=sigma, random_state=rng))

    def grad_eta(self, y: float, eta: float, params: Dict[str, float]) -> float:
        """
        Analytic derivative d/d eta log p(y | eta, sigma, xi).

        For xi != 0:
            t = 1 + xi * (y - eta) / sigma
            grad = ((1 + xi) / t - t^(-1/xi - 1)) / sigma

        In the Gumbel limit xi -> 0:
            z = (y - eta) / sigma
            grad = (1 - exp(-z)) / sigma
        """
        sigma = float(params["sigma"])
        xi = float(params["xi"])
        y = float(y)
        eta = float(eta)

        if sigma <= 0.0:
            raise ValueError("GEV scale parameter sigma must be > 0.")

        # Gumbel limit
        if abs(xi) < 1e-8:
            z = (y - eta) / sigma
            w = np.exp(-z)
            return float((1.0 - w) / sigma)

        t = 1.0 + xi * (y - eta) / sigma
        if t <= 0.0:
            raise ValueError(
                "GEV gradient undefined because the support condition "
                "1 + xi * (y - eta) / sigma > 0 is violated."
            )

        grad = ((1.0 + xi) / t - t ** (-1.0 / xi - 1.0)) / sigma
        return float(grad)

    def hess_eta(self, y: float, eta: float, params: Dict[str, float]) -> float:
        """
        Analytic second derivative d^2/d eta^2 log p(y | eta, sigma, xi).

        For xi != 0:
            t = 1 + xi * (y - eta) / sigma
            hess = ((1 + xi) / sigma^2) * (xi / t^2 - t^(-1/xi - 2))

        In the Gumbel limit xi -> 0:
            z = (y - eta) / sigma
            hess = -exp(-z) / sigma^2
        """
        sigma = float(params["sigma"])
        xi = float(params["xi"])
        y = float(y)
        eta = float(eta)

        if sigma <= 0.0:
            raise ValueError("GEV scale parameter sigma must be > 0.")

        # Gumbel limit
        if abs(xi) < 1e-8:
            z = (y - eta) / sigma
            w = np.exp(-z)
            return float(-w / (sigma * sigma))

        t = 1.0 + xi * (y - eta) / sigma
        if t <= 0.0:
            raise ValueError(
                "GEV Hessian undefined because the support condition "
                "1 + xi * (y - eta) / sigma > 0 is violated."
            )

        hess = ((1.0 + xi) / (sigma * sigma)) * (
            xi / (t * t) - t ** (-1.0 / xi - 2.0)
        )
        return float(hess)