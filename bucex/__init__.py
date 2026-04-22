"""bucex: Bayesian structural time-series tools for extremes.

v0.1 provides a workable research package for structural Gaussian and GEV models
with interchangeable centered and specialized non-centred fitters.

The non-centred fitters are intentionally scoped to the local-linear-trend plus
optional dummy-seasonal model family.
"""
from __future__ import annotations

from .__about__ import __version__
from .models.structural import StructuralSSM, StructuralModel
from .components import LocalLinearTrend, DummySeasonal, RegressionComponent
from .observation.gaussian import GaussianObs
from .observation.gev import GEVObs
from .api.fit import fit_bayes, fit_gaussian_structural, fit_gev_structural
from .api.model_builders import make_gaussian_model, make_gev_model

__all__ = [
    "__version__",
    "StructuralModel",
    "StructuralSSM",
    "LocalLinearTrend",
    "DummySeasonal",
    "RegressionComponent",
    "GaussianObs",
    "GEVObs",
    "fit_bayes",
    "fit_gaussian_structural",
    "fit_gev_structural",
    "make_gaussian_model",
    "make_gev_model",
]
