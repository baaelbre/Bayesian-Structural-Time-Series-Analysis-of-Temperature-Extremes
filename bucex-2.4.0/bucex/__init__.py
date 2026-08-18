"""Bayesian structural models for one or several related time series.

The same :func:`fit` function handles a univariate :class:`Model` and a
:class:`MultiSeriesModel`. Multiple series retain separate latent paths while
sharing structural-selection probabilities, normal-slab scales, or both.
"""
from __future__ import annotations

from .__about__ import __version__
from .api import (
    Forecast, combine_fits, combine_fs_fits, fit, fit_bayes, fit_bulk_tail,
    fit_gaussian_structural, fit_gev_structural, make_gaussian_model,
    make_gev_model, plan, posterior_predict,
)
from .components import Component, DummySeasonal, LocalLevel, LocalLinearTrend, Regression, RegressionComponent, Seasonal
from .core import BulkTailFit, FitResult, PosteriorBundle
from .datasets import (
    UCCLE_INFO, UCCLE_SERIES, UccleFitCollection, derive_uccle_monthly,
    fit_uccle_all, fit_uccle_hierarchical, fit_uccle_series, load_uccle_daily,
    load_uccle_multiseries, load_uccle_series, make_uccle_hierarchical_model,
    validate_uccle_data,
)
from .diagnostics import (
    LFOResult, PITResult, crps_ensemble, ess_bulk, evaluate_ensemble,
    exceedance_brier_score, exceedance_log_score, fit_diagnostics,
    leave_future_out, log_predictive_score, pit_diagnostics, posterior_pit,
    quantile_score, rhat, threshold_weighted_crps,
)
from .inference import GibbsConfig, HierarchicalSampler, InferencePlan, Laplace, MCMC, Particles, ffbs, iterated_laplace, kalman_filter, kalman_smoother, particle_filter, pgas
from .inference.fit.model_space import (
    ComponentState,
    StructuralModelState,
    TrendModelClass,
)
from .models import Channel, CompiledMultiSeriesModel, Model, MultiSeriesModel, StructuralModel, StructuralSSM, compile_multiseries_model, structural_model
from .models.compiler import CompiledModel, DisturbancePath, NonCenteredPath, compile_model
from .observation import GEV, GEVObs, Gaussian, GaussianObs
from .priors import (
    BayesianLassoPrior, ComponentwiseBayesianLassoPrior, DiagonalNormalPrior,
    ExponentialSD, FixedSD, FSGaussianPriors, FSGEVPriors, GammaPrior,
    HalfNormalSD, HalfStudentTSD, HierarchicalPrior, HierarchicalPriors,
    InverseGammaPrior, InverseGammaVariance, NormalPrior, PCInnovationPrior,
    PCSD, Priors, RegularizedHorseshoePrior, SDUniformPrior, SSVSPrior,
    SpikeSlabSD, TripleGammaPrior, TruncatedNormalPrior, UniformPrior,
    default_priors, manuscript_gaussian_priors, manuscript_gev_priors,
    normal_gaussian_priors, normal_gev_priors, pc_gaussian_priors,
    pc_gev_priors, regularized_gaussian_priors, regularized_gev_priors,
    regularized_horseshoe_gaussian_priors, regularized_horseshoe_gev_priors,
    regularized_triple_gamma_gaussian_priors,
    regularized_triple_gamma_gev_priors, resolve_hierarchical_priors,
    calibrate_structural_scales, half_student_t_scale_for_median,
    structural_scale_implications,
    ssvs_gaussian_priors, ssvs_gev_priors, triple_gamma_gaussian_priors,
    triple_gamma_gev_priors,
)
from .simulate import Simulation, simulate


def plot(value, kind: str = "state", **kwargs):
    """Plot a fit, forecast, or bulk/tail result."""
    if isinstance(value, Forecast):
        return value.plot(**kwargs)
    return value.plot(kind=kind, **kwargs)


def forecast(value, horizon: int, **kwargs):
    """Generate a posterior forecast from a fitted model."""
    if not isinstance(value, FitResult):
        raise TypeError("forecast expects a bucex.FitResult.")
    return value.forecast(horizon, **kwargs)


def score(value, observed, **kwargs):
    """Score a forecast or raw predictive ensemble."""
    if isinstance(value, Forecast):
        return value.score(observed, **kwargs)
    return evaluate_ensemble(value, observed, **kwargs)


__all__ = [name for name in globals() if not name.startswith("_")]
