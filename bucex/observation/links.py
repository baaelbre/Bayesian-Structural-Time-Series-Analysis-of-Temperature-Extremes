# src/sts_extremes/obs/links.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Protocol
import numpy as np


class Link(Protocol):
    """Map unconstrained -> constrained parameter (and optionally inverse/jacobian later)."""
    def forward(self, x: float) -> float: ...


@dataclass(frozen=True)
class Identity:
    def forward(self, x: float) -> float:
        return float(x)


@dataclass(frozen=True)
class Exp:
    """Strictly positive (useful for sigma)."""
    def forward(self, x: float) -> float:
        return float(np.exp(x))


@dataclass(frozen=True)
class Softplus:
    """Positive, numerically stable, ~max(0,x) for large |x|."""
    beta: float = 1.0
    def forward(self, x: float) -> float:
        b = float(self.beta)
        z = b * float(x)
        # stable softplus
        return float((1.0 / b) * (np.log1p(np.exp(-abs(z))) + max(z, 0.0)))


@dataclass
class ParamMap:
    """
    Map eta_t and a params dict into distribution parameters.

    Typical usage:
      mu = eta
      sigma = link_sigma.forward(params["sigma_raw"])  OR params["sigma"]
      xi = params["xi"]

    Keep it simple for now: you can override __call__ in specific models.
    """
    def __call__(self, eta: float, params: Dict[str, float]) -> Dict[str, float]:
        return {"mu": float(eta), **params}