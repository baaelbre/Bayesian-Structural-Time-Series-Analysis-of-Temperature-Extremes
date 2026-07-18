"""bucex: Bayesian unobserved components for extremes.

v0.2 adds manuscript-faithful non-centred Bayesian-lasso fitting, complete fit
objects, Uccle wrappers, and a high-level plotting/risk API.
"""
from __future__ import annotations

from .__about__ import __version__
from .api.fit import fit_bayes, fit_gaussian_structural, fit_gev_structural
from .api.model_builders import make_gaussian_model, make_gev_model
from .components import DummySeasonal, LocalLinearTrend, RegressionComponent
from .core.results import PosteriorBundle
from .datasets import (
    UccleFitCollection,
    fit_uccle_all,
    fit_uccle_series,
    load_uccle_series,
)
from .inference.fit import (
    BayesianLassoPrior,
    DiagonalNormalPrior,
    GammaPrior,
    GibbsConfig,
    InverseGammaPrior,
    NonCenteredGaussianPriors,
    NonCenteredGEVPriors,
    NormalPrior,
    UniformPrior,
    manuscript_gaussian_priors,
    manuscript_gev_priors,
)
from .models.structural import StructuralModel, StructuralSSM
from .observation.gaussian import GaussianObs
from .observation.gev import GEVObs
from .plotting import plot

__all__ = [
    "__version__",
    "StructuralModel",
    "StructuralSSM",
    "LocalLinearTrend",
    "DummySeasonal",
    "RegressionComponent",
    "GaussianObs",
    "GEVObs",
    "PosteriorBundle",
    "GibbsConfig",
    "InverseGammaPrior",
    "GammaPrior",
    "UniformPrior",
    "NormalPrior",
    "DiagonalNormalPrior",
    "BayesianLassoPrior",
    "NonCenteredGaussianPriors",
    "NonCenteredGEVPriors",
    "manuscript_gaussian_priors",
    "manuscript_gev_priors",
    "fit_bayes",
    "fit_gaussian_structural",
    "fit_gev_structural",
    "make_gaussian_model",
    "make_gev_model",
    "load_uccle_series",
    "fit_uccle_series",
    "fit_uccle_all",
    "UccleFitCollection",
    "plot",
]
