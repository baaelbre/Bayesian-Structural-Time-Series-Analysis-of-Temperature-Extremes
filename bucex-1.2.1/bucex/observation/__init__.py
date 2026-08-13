"""Canonical observation families."""
from .base import ObservationModel, ObsSpec
from .gaussian import Gaussian, GaussianObs
from .gev import GEV, GEVObs

Observation = Gaussian | GEV


def observation_from_dict(value):
    family = str(value["family"]).lower()
    if family == "gaussian":
        return Gaussian()
    if family == "gev":
        return GEV(tuple(value.get("xi_bounds", (-0.5, 0.5))))
    raise ValueError(f"Unknown observation family '{family}'.")


__all__ = [
    "ObservationModel",
    "ObsSpec",
    "Observation",
    "Gaussian",
    "GEV",
    "GaussianObs",
    "GEVObs",
    "observation_from_dict",
]
