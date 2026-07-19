"""bucex: Bayesian unobserved components for extremes.

v0.3.3 adds transparent Laplace restoration diagnostics and
short multi-chain HPC workflows while preserving the v0.3.2 samplers.
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
    ComponentwiseBayesianLassoPrior,
    SSVSPrior,
    ComponentState,
    StructuralModelState,
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
    normal_gaussian_priors,
    normal_gev_priors,
    regularized_gaussian_priors,
    regularized_gev_priors,
    ssvs_gaussian_priors,
    ssvs_gev_priors,
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
    "ComponentwiseBayesianLassoPrior",
    "SSVSPrior",
    "ComponentState",
    "StructuralModelState",
    "NonCenteredGaussianPriors",
    "NonCenteredGEVPriors",
    "manuscript_gaussian_priors",
    "manuscript_gev_priors",
    "normal_gaussian_priors",
    "normal_gev_priors",
    "regularized_gaussian_priors",
    "regularized_gev_priors",
    "ssvs_gaussian_priors",
    "ssvs_gev_priors",
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
