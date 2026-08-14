"""Bayesian structural models for univariate data and shared dynamic factors.

Version 2.0 exposes one compiler contract, one fitter and one result type.  The
parameterization, state-update engine, prior profile and ASIS interweaving are
orthogonal choices validated by the inference planner.
"""
from __future__ import annotations

from .__about__ import __version__
from .api import (
    Forecast,
    combine_fits,
    combine_fs_fits,
    fit,
    fit_bayes,
    fit_bulk_tail,
    fit_gaussian_structural,
    fit_gev_structural,
    make_gaussian_model,
    make_gev_model,
    plan,
    posterior_predict,
)
from .components import (
    Component,
    DummySeasonal,
    LocalLevel,
    LocalLinearTrend,
    Regression,
    RegressionComponent,
    Seasonal,
)
from .core import BulkTailFit, FitResult, PosteriorBundle
from .datasets import (
    UCCLE_INFO,
    UCCLE_SERIES,
    UccleFitCollection,
    derive_uccle_monthly,
    fit_uccle_all,
    fit_uccle_factor,
    fit_uccle_series,
    load_uccle_daily,
    load_uccle_series,
    load_uccle_factor_data,
    make_uccle_factor_model,
    validate_uccle_data,
)
from .diagnostics import (
    crps_ensemble,
    ess_bulk,
    evaluate_ensemble,
    exceedance_brier_score,
    exceedance_log_score,
    fit_diagnostics,
    posterior_pit,
    quantile_score,
    rhat,
    threshold_weighted_crps,
)
from .inference import (
    GibbsConfig,
    InferencePlan,
    Laplace,
    MCMC,
    Particles,
    ffbs,
    iterated_laplace,
    kalman_filter,
    kalman_smoother,
    particle_filter,
    pgas,
)
from .inference.fit.model_space import ComponentState, StructuralModelState
from .models import (
    Channel,
    CompiledFactorModel,
    DynamicFactorModel,
    Factor,
    FactorModel,
    Loading,
    Model,
    StructuralModel,
    StructuralSSM,
    compile_factor_model,
    structural_model,
)
from .models.compiler import (
    CompiledModel,
    DisturbancePath,
    NonCenteredPath,
    compile_model,
)
from .observation import GEV, GEVObs, Gaussian, GaussianObs
from .priors import (
    BayesianLassoPrior,
    ComponentwiseBayesianLassoPrior,
    DiagonalNormalPrior,
    ExponentialSD,
    FixedSD,
    GammaPrior,
    HalfNormalSD,
    HalfStudentTSD,
    InverseGammaPrior,
    InverseGammaVariance,
    FSGaussianPriors,
    FSGEVPriors,
    FactorPriors,
    NormalPrior,
    PCInnovationPrior,
    PCSD,
    Priors,
    RegularizedHorseshoePrior,
    SDUniformPrior,
    SSVSPrior,
    SpikeSlabSD,
    TruncatedNormalPrior,
    UniformPrior,
    default_priors,
    default_factor_priors,
    manuscript_gaussian_priors,
    manuscript_gev_priors,
    normal_gaussian_priors,
    normal_gev_priors,
    pc_gaussian_priors,
    pc_gev_priors,
    regularized_gaussian_priors,
    regularized_gev_priors,
    regularized_horseshoe_gaussian_priors,
    regularized_horseshoe_gev_priors,
    ssvs_gaussian_priors,
    ssvs_gev_priors,
)
from .simulate import Simulation, simulate


def plot(value, kind: str = "state", **kwargs):
    """Plot a fit, forecast or bulk/tail result."""

    if isinstance(value, Forecast):
        return value.plot(**kwargs)
    return value.plot(kind=kind, **kwargs)


def forecast(value, horizon: int, **kwargs):
    """Generate a posterior forecast from a fitted model."""

    if not isinstance(value, FitResult):
        raise TypeError("forecast expects a bucex.FitResult.")
    return value.forecast(horizon, **kwargs)


def score(value, observed, **kwargs):
    """Score a forecast or a raw predictive ensemble."""

    if isinstance(value, Forecast):
        return value.score(observed, **kwargs)
    return evaluate_ensemble(value, observed, **kwargs)


__all__ = [
    "__version__",
    # Model grammar.
    "Model", "StructuralModel", "StructuralSSM", "Channel", "Loading",
    "Factor", "FactorModel", "DynamicFactorModel", "Component", "LocalLevel",
    "LocalLinearTrend", "DummySeasonal", "Seasonal", "Regression",
    "RegressionComponent", "Gaussian", "GEV", "GaussianObs", "GEVObs",
    "CompiledModel", "CompiledFactorModel", "DisturbancePath", "NonCenteredPath",
    "compile_model", "compile_factor_model",
    "structural_model",
    # Integrated fitting contract.
    "fit", "plan", "MCMC", "Laplace", "Particles", "GibbsConfig",
    "InferencePlan", "FitResult", "PosteriorBundle", "BulkTailFit",
    "combine_fits", "fit_bulk_tail", "fit_bayes", "combine_fs_fits",
    "fit_gaussian_structural", "fit_gev_structural", "make_gaussian_model",
    "make_gev_model",
    # Priors.
    "Priors", "FactorPriors", "default_factor_priors", "HalfNormalSD",
    "HalfStudentTSD", "ExponentialSD", "PCSD",
    "InverseGammaVariance", "FixedSD", "SpikeSlabSD", "SDUniformPrior",
    "TruncatedNormalPrior", "default_priors", "InverseGammaPrior", "GammaPrior",
    "UniformPrior", "NormalPrior", "DiagonalNormalPrior", "BayesianLassoPrior",
    "ComponentwiseBayesianLassoPrior", "RegularizedHorseshoePrior",
    "PCInnovationPrior", "SSVSPrior", "FSGaussianPriors",
    "FSGEVPriors", "ComponentState", "StructuralModelState",
    "manuscript_gaussian_priors",
    "manuscript_gev_priors", "normal_gaussian_priors", "normal_gev_priors",
    "regularized_gaussian_priors", "regularized_gev_priors",
    "regularized_horseshoe_gaussian_priors",
    "regularized_horseshoe_gev_priors", "pc_gaussian_priors", "pc_gev_priors",
    "ssvs_gaussian_priors", "ssvs_gev_priors",
    # Prediction, diagnostics and simulation.
    "Forecast", "posterior_predict", "forecast", "plot", "score",
    "kalman_filter", "kalman_smoother", "ffbs", "iterated_laplace",
    "particle_filter", "pgas", "Simulation", "simulate", "rhat", "ess_bulk",
    "posterior_pit", "fit_diagnostics", "crps_ensemble",
    "threshold_weighted_crps", "quantile_score", "exceedance_brier_score",
    "exceedance_log_score", "evaluate_ensemble",
    # Uccle data helpers.
    "UCCLE_SERIES", "UCCLE_INFO", "UccleFitCollection",
    "derive_uccle_monthly", "load_uccle_series", "load_uccle_daily",
    "validate_uccle_data", "fit_uccle_series", "fit_uccle_all",
    "load_uccle_factor_data", "make_uccle_factor_model", "fit_uccle_factor",
]
