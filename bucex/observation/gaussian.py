# src/sts_extremes/obs/gaussian.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
from scipy.stats import norm

from .base import ObsSpec, ObservationModel


@dataclass(frozen=True)
class GaussianObs(ObservationModel):
    """
    y | eta ~ Normal(mu=eta, sigma)
    params: {"sigma": ...}
    """
    spec: ObsSpec = ObsSpec("gaussian")

    def logpdf(self, y: float, eta: float, params: Dict[str, float]) -> float:
        sigma = float(params["sigma"])
        return float(norm.logpdf(y, loc=float(eta), scale=sigma))

    def sample(self, eta: float, params: Dict[str, float], rng: np.random.Generator) -> float:
        sigma = float(params["sigma"])
        return float(rng.normal(loc=float(eta), scale=sigma))

    def grad_eta(self, y: float, eta: float, params: Dict[str, float]) -> float:
        sigma = float(params["sigma"])
        return float((float(y) - float(eta)) / (sigma * sigma))

    def hess_eta(self, y: float, eta: float, params: Dict[str, float]) -> float:
        sigma = float(params["sigma"])
        return float(-1.0 / (sigma * sigma))