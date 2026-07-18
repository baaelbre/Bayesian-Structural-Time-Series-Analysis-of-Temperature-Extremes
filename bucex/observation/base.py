from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np


class ObservationModel(Protocol):
    def logpdf(self, y: float, eta: float, params: Dict[str, float]) -> float: ...
    def sample(self, eta: float, params: Dict[str, float], rng: np.random.Generator) -> float: ...
    def grad_eta(self, y: float, eta: float, params: Dict[str, float]) -> float: ...
    def hess_eta(self, y: float, eta: float, params: Dict[str, float]) -> float: ...


@dataclass(frozen=True)
class ObsSpec:
    name: str
