"""Canonical observation-model namespace for bucex."""
from .base import ObservationModel, ObsSpec
from .gaussian import GaussianObs
from .gev import GEVObs
from .links import *

__all__ = ["ObservationModel", "ObsSpec", "GaussianObs", "GEVObs"]
