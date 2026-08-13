from __future__ import annotations

from typing import Sequence

from ..components.base import Component
from ..models.structural import StructuralSSM
from ..observation.gaussian import GaussianObs
from ..observation.gev import GEVObs


def make_gaussian_model(components: Sequence[Component]) -> StructuralSSM:
    return StructuralSSM(components=list(components), obs=GaussianObs(), eta_name="mu")


def make_gev_model(components: Sequence[Component]) -> StructuralSSM:
    return StructuralSSM(components=list(components), obs=GEVObs(), eta_name="mu")
