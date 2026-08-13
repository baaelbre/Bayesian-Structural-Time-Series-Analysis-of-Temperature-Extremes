"""bucex: Bayesian unobserved components for bulk data and extremes.

Version 1.1 keeps the explicit 0.3 Fruehwirth--Schnatter implementation and
the flexible 1.0 compiled-model interface behind one small public API.
"""
from __future__ import annotations

from .__about__ import __version__
from .api import (
    fit,
    fit_bayes,
    fit_bulk_tail,
    combine_fs_fits,
    fit_gaussian_structural,
    fit_gev_structural,
    make_gaussian_model,
    make_gev_model,
    plan,
)

# Declarative 1.x model grammar.
from ._general.compiler import CompiledModel, NonCenteredPath, compile_model
from ._general.model import (
    DummySeasonal,
    LocalLevel,
    LocalLinearTrend,
    Model,
    Regression,
    Seasonal,
    structural_model,
)
from ._general.observations import GEV, Gaussian
from ._general.observations import GEVObs, GaussianObs
from ._general.particle import Particles, particle_filter, pgas
from ._general.kalman import ffbs, kalman_filter, kalman_smoother
from ._general.laplace import iterated_laplace
from ._general.plan import InferencePlan
from ._general.priors import (
    ExponentialSD,
    FixedSD,
    HalfNormalSD,
    HalfStudentTSD,
    InverseGammaVariance,
    PCSD,
    Priors,
    SpikeSlabSD,
    TruncatedNormalPrior,
    UniformPrior as SDUniformPrior,
    default_priors,
)
from ._general.results import BulkTailFit, FitResult, combine_fits
from ._general.forecast import Forecast
from ._general.sampler import Laplace, MCMC
from ._general.simulate import Simulation, simulate
from ._general.scores import (
    crps_ensemble,
    evaluate_ensemble,
    exceedance_brier_score,
    exceedance_log_score,
    quantile_score,
    threshold_weighted_crps,
)

# Explicit 0.3/FS implementation and prior profiles.
from .components import (
    DummySeasonal as LegacyDummySeasonal,
    LocalLinearTrend as LegacyLocalLinearTrend,
    RegressionComponent as LegacyRegressionComponent,
)
from .core.results import PosteriorBundle
from .inference.fit import (
    BayesianLassoPrior,
    ComponentState,
    ComponentwiseBayesianLassoPrior,
    DiagonalNormalPrior,
    GammaPrior,
    GibbsConfig,
    InverseGammaPrior,
    NonCenteredGaussianPriors,
    NonCenteredGEVPriors,
    NormalPrior,
    RegularizedHorseshoePrior,
    PCInnovationPrior,
    SSVSPrior,
    StructuralModelState,
    UniformPrior,
    manuscript_gaussian_priors,
    manuscript_gev_priors,
    normal_gaussian_priors,
    normal_gev_priors,
    regularized_gaussian_priors,
    regularized_gev_priors,
    regularized_horseshoe_gaussian_priors,
    regularized_horseshoe_gev_priors,
    pc_gaussian_priors,
    pc_gev_priors,
    ssvs_gaussian_priors,
    ssvs_gev_priors,
)
from .models.structural import StructuralModel, StructuralSSM
from .observation.gaussian import GaussianObs as LegacyGaussianObs
from .observation.gev import GEVObs as LegacyGEVObs

from .datasets import (
    UCCLE_INFO,
    UCCLE_SERIES,
    UccleFitCollection,
    derive_uccle_monthly,
    fit_uccle_all,
    fit_uccle_series,
    load_uccle_daily,
    load_uccle_series,
    validate_uccle_data,
)

RegressionComponent = Regression
fit_fs = fit_bayes

__all__ = [
    "__version__",
    # General model grammar and engine.
    "Model", "LocalLevel", "LocalLinearTrend", "DummySeasonal", "Seasonal",
    "Regression", "RegressionComponent", "Gaussian", "GEV", "GaussianObs",
    "GEVObs", "CompiledModel", "NonCenteredPath", "compile_model", "Priors",
    "HalfNormalSD", "HalfStudentTSD", "ExponentialSD", "PCSD",
    "InverseGammaVariance", "FixedSD", "SpikeSlabSD", "SDUniformPrior",
    "TruncatedNormalPrior", "default_priors", "MCMC", "Laplace", "Particles",
    "InferencePlan", "FitResult", "BulkTailFit", "combine_fits", "Forecast",
    "fit", "fit_bulk_tail", "structural_model", "plan", "particle_filter",
    "combine_fs_fits",
    "pgas", "kalman_filter", "kalman_smoother", "ffbs", "iterated_laplace",
    "Simulation", "simulate",
    # FS implementation.
    "StructuralModel", "StructuralSSM", "LegacyLocalLinearTrend",
    "LegacyDummySeasonal", "LegacyRegressionComponent", "LegacyGaussianObs",
    "LegacyGEVObs", "PosteriorBundle", "GibbsConfig", "InverseGammaPrior",
    "GammaPrior", "UniformPrior", "NormalPrior", "DiagonalNormalPrior",
    "BayesianLassoPrior", "ComponentwiseBayesianLassoPrior",
    "RegularizedHorseshoePrior", "SSVSPrior", "ComponentState",
    "PCInnovationPrior",
    "StructuralModelState", "NonCenteredGaussianPriors", "NonCenteredGEVPriors",
    "manuscript_gaussian_priors", "manuscript_gev_priors",
    "normal_gaussian_priors", "normal_gev_priors", "regularized_gaussian_priors",
    "regularized_gev_priors", "regularized_horseshoe_gaussian_priors",
    "regularized_horseshoe_gev_priors", "ssvs_gaussian_priors", "ssvs_gev_priors",
    "pc_gaussian_priors", "pc_gev_priors",
    "fit_bayes", "fit_gaussian_structural", "fit_gev_structural",
    "fit_fs",
    "make_gaussian_model", "make_gev_model",
    # Data and predictive summaries.
    "crps_ensemble", "threshold_weighted_crps", "quantile_score",
    "exceedance_brier_score", "exceedance_log_score", "evaluate_ensemble",
    "UCCLE_SERIES", "UCCLE_INFO", "UccleFitCollection", "derive_uccle_monthly",
    "load_uccle_series", "load_uccle_daily", "validate_uccle_data",
    "fit_uccle_series", "fit_uccle_all",
]


def plot(value, kind="state", **kwargs):
    """Plot a fit, forecast, or bulk/tail result."""
    if isinstance(value, Forecast):
        return value.plot(**kwargs)
    if isinstance(value, PosteriorBundle):
        return value.plot(type=kind, **kwargs)
    return value.plot(kind=kind, **kwargs)


def forecast(value, horizon, **kwargs):
    """Generate a posterior forecast from either fit representation."""
    if not hasattr(value, "forecast"):
        raise TypeError("forecast expects a bucex fit object.")
    return value.forecast(horizon, **kwargs)


def score(value, observed, **kwargs):
    """Score a forecast or a raw predictive ensemble."""
    if isinstance(value, Forecast):
        return value.score(observed, **kwargs)
    return evaluate_ensemble(value, observed, **kwargs)


__all__.extend(["plot", "forecast", "score"])
