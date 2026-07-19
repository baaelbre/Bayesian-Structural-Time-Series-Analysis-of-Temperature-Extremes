from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from ..components import DummySeasonal, LocalLinearTrend
from ..core.results import PosteriorBundle
from ..inference.fit.base import GibbsConfig
from ..inference.fit.centered_gaussian import CenteredGaussianGibbs
from ..inference.fit.centered_gev import CenteredGEVGibbs
from ..inference.fit.noncentered_gaussian import NonCenteredGaussianGibbs
from ..inference.fit.noncentered_gev import NonCenteredGEVGibbs
from ..inference.fit.priors import (
    NonCenteredGaussianPriors,
    NonCenteredGEVPriors,
    manuscript_gaussian_priors,
    manuscript_gev_priors,
    normal_gaussian_priors,
    normal_gev_priors,
    regularized_gaussian_priors,
    regularized_gev_priors,
    ssvs_gaussian_priors,
    ssvs_gev_priors,
)
from ..models.base import StateSpaceModel
from ..models.structural import StructuralModel
from ..observation.gaussian import GaussianObs
from ..observation.gev import GEVObs


def _obs_name(model: StateSpaceModel) -> str | None:
    spec = getattr(getattr(model, "obs", None), "spec", None)
    return getattr(spec, "name", None) if spec is not None else None


def _default_model(family: str, period: int) -> StructuralModel:
    family = family.lower()
    if family not in {"gaussian", "gev"}:
        raise ValueError("family must be 'gaussian' or 'gev'.")
    obs = GaussianObs() if family == "gaussian" else GEVObs()
    return StructuralModel(
        components=[
            LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            DummySeasonal(period=period, mode="dynamic"),
        ],
        obs=obs,
    )


def _seasonal_initial(y: np.ndarray, period: int) -> np.ndarray:
    overall = float(np.nanmean(y))
    full = np.asarray(
        [np.nanmean(y[np.arange(y.size) % period == month]) - overall for month in range(period)],
        dtype=float,
    )
    full = np.nan_to_num(full, nan=0.0)
    full -= full.mean()
    return full[: period - 1]


def _default_initial_values(
    y_model: np.ndarray,
    model: StateSpaceModel,
    *,
    period: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = _obs_name(model)
    gamma0 = _seasonal_initial(y_model, period)
    month = np.arange(y_model.size) % period
    full = np.r_[gamma0, -gamma0.sum()]
    residual = y_model - float(np.mean(y_model)) - full[month]
    scale = max(float(np.std(residual, ddof=1)), 0.25)

    params_state = {
        "alpha0": float(np.mean(y_model)),
        "beta0": 0.0,
        "gamma0_season": gamma0,
        "s_level": 0.02,
        "s_trend": 0.001,
        "s_season": 0.02,
        "q_level": 0.02**2,
        "q_trend": 0.001**2,
        "q_season": 0.02**2,
    }
    if family == "gaussian":
        params_obs = {"sigma": scale, "sigma2": scale * scale}
    elif family == "gev":
        params_obs = {"sigma": scale, "xi": -0.10}
    else:
        raise NotImplementedError(f"No automatic initialisation for obs='{family}'.")
    return params_state, params_obs


def fit_gaussian_structural(
    y: np.ndarray,
    model: StateSpaceModel,
    priors: Any,
    init_params_state: dict[str, Any],
    init_params_obs: dict[str, Any],
    *,
    exog: Optional[np.ndarray] = None,
    config: GibbsConfig | None = None,
    parameterization: str = "noncentered",
    state_method: str = "ffbs",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> PosteriorBundle:
    cfg = config or GibbsConfig()
    if parameterization == "centered":
        fitter = CenteredGaussianGibbs(model=model, priors=priors, config=cfg)
    elif parameterization == "noncentered":
        fitter = NonCenteredGaussianGibbs(model=model, priors=priors, config=cfg)
    else:
        raise ValueError("parameterization must be 'centered' or 'noncentered'.")
    return fitter.fit(
        y=y,
        init_params_state=init_params_state,
        init_params_obs=init_params_obs,
        exog=exog,
        state_method=state_method,
        state_kwargs=state_kwargs,
    )


def fit_gev_structural(
    y: np.ndarray,
    model: StateSpaceModel,
    priors: Any,
    init_params_state: dict[str, Any],
    init_params_obs: dict[str, Any],
    *,
    exog: Optional[np.ndarray] = None,
    config: GibbsConfig | None = None,
    parameterization: str = "noncentered",
    state_method: str = "laplace",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> PosteriorBundle:
    cfg = config or GibbsConfig()
    kwargs = {} if state_kwargs is None else dict(state_kwargs)
    if parameterization == "centered":
        fitter = CenteredGEVGibbs(model=model, priors=priors, config=cfg)
    elif parameterization == "noncentered":
        fitter = NonCenteredGEVGibbs(
            model=model,
            priors=priors,
            config=cfg,
            step_log_sigma=float(kwargs.pop("step_log_sigma", 0.05)),
            step_xi=float(kwargs.pop("step_xi", 0.05)),
        )
    else:
        raise ValueError("parameterization must be 'centered' or 'noncentered'.")
    return fitter.fit(
        y=y,
        init_params_state=init_params_state,
        init_params_obs=init_params_obs,
        exog=exog,
        state_method=state_method,
        state_kwargs=kwargs,
    )


def fit_bayes(
    y: np.ndarray,
    model: StateSpaceModel | str | None = None,
    priors: Any = "manuscript",
    init_params_state: Optional[dict[str, Any]] = None,
    init_params_obs: Optional[dict[str, Any]] = None,
    *,
    family: Optional[str] = None,
    period: int = 12,
    dates: Optional[np.ndarray] = None,
    name: Optional[str] = None,
    transform_sign: float = 1.0,
    tail: Optional[str] = None,
    exog: Optional[np.ndarray] = None,
    config: GibbsConfig | None = None,
    n_iter: Optional[int] = None,
    burn: Optional[int] = None,
    thin: Optional[int] = None,
    seed: Optional[int] = None,
    progress: Optional[bool] = None,
    method: str = "gibbs",
    state_method: str = "auto",
    parameterization: str = "noncentered",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> PosteriorBundle:
    """Fit a Bayesian structural Gaussian or DGEV model.

    This is the v0.3.3 high-level API. A complete manuscript-style fit can be
    requested without manually constructing priors or initial values::

        fit = fit_bayes(y, family="gev", dates=dates, name="TXx")

    For block minima, pass ``tail='min'``; the function fits ``-y`` internally
    and all high-level plots and risk calculations are transformed back.
    """
    if method != "gibbs":
        raise NotImplementedError("v0.3.3 exposes Gibbs / MH-within-Gibbs fitting.")

    if tail is not None:
        if tail.lower() in {"min", "minimum", "lower"}:
            transform_sign = -1.0
        elif tail.lower() in {"max", "maximum", "upper"}:
            transform_sign = 1.0
        else:
            raise ValueError("tail must be 'min' or 'max'.")
    if transform_sign not in {-1, 1, -1.0, 1.0}:
        raise ValueError("transform_sign must be +1 or -1.")

    if isinstance(model, str):
        family = model
        model = None
    if model is None:
        if family is None:
            raise ValueError("Provide model=... or family='gaussian'/'gev'.")
        model = _default_model(family, period)
    obs_name = _obs_name(model)
    if obs_name not in {"gaussian", "gev"}:
        raise NotImplementedError(f"No high-level fitter registered for obs='{obs_name}'.")

    y_original = np.asarray(y, dtype=float).reshape(-1)
    if not np.all(np.isfinite(y_original)):
        raise ValueError("y must contain only finite observations in v0.3.3.")
    y_model = float(transform_sign) * y_original

    if config is None:
        config = GibbsConfig()
    overrides = {
        "n_iter": n_iter,
        "burn": burn,
        "thin": thin,
        "seed": seed,
        "progress": progress,
    }
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None})
    if not (0 <= config.burn < config.n_iter):
        raise ValueError("Require 0 <= burn < n_iter.")
    if config.thin < 1:
        raise ValueError("thin must be >= 1.")

    prior_profile = "manuscript" if priors is None else priors
    built_in_priors = isinstance(prior_profile, str)
    if built_in_priors:
        profile = str(prior_profile).lower().replace("-", "_")
        builders = {
            ("gaussian", "manuscript"): manuscript_gaussian_priors,
            ("gev", "manuscript"): manuscript_gev_priors,
            ("gaussian", "normal"): normal_gaussian_priors,
            ("gev", "normal"): normal_gev_priors,
            ("gaussian", "regularized"): regularized_gaussian_priors,
            ("gev", "regularized"): regularized_gev_priors,
            ("gaussian", "ssvs"): ssvs_gaussian_priors,
            ("gev", "ssvs"): ssvs_gev_priors,
        }
        key = (obs_name, profile)
        if key not in builders:
            raise ValueError(
                "Built-in prior profiles are 'manuscript', 'normal', 'regularized', and 'ssvs'."
            )
        priors = builders[key](period)
        prior_profile = profile
    else:
        prior_profile = "custom"

    auto_state, auto_obs = _default_initial_values(y_model, model, period=period)
    if init_params_state is not None:
        auto_state.update(init_params_state)
    if init_params_obs is not None:
        auto_obs.update(init_params_obs)

    resolved_state_method = (
        "ffbs" if obs_name == "gaussian" else "laplace"
    ) if state_method == "auto" else state_method

    if obs_name == "gaussian":
        if not isinstance(priors, NonCenteredGaussianPriors) and parameterization == "noncentered":
            raise TypeError("A non-centred Gaussian fit requires NonCenteredGaussianPriors.")
        fit = fit_gaussian_structural(
            y_model,
            model,
            priors,
            auto_state,
            auto_obs,
            exog=exog,
            config=config,
            parameterization=parameterization,
            state_method=resolved_state_method,
            state_kwargs=state_kwargs,
        )
    else:
        if not isinstance(priors, NonCenteredGEVPriors) and parameterization == "noncentered":
            raise TypeError("A non-centred GEV fit requires NonCenteredGEVPriors.")
        fit = fit_gev_structural(
            y_model,
            model,
            priors,
            auto_state,
            auto_obs,
            exog=exog,
            config=config,
            parameterization=parameterization,
            state_method=resolved_state_method,
            state_kwargs=state_kwargs,
        )

    fit.y = y_original
    fit.dates = None if dates is None else np.asarray(dates)
    if fit.dates is not None and len(fit.dates) != len(y_original):
        raise ValueError("dates must have the same length as y.")
    fit.model = model
    fit.state_names = tuple(model.state_names)
    fit.series_name = name
    fit.transform_sign = float(transform_sign)
    fit.meta.update(
        {
            "period": int(period),
            "family": obs_name,
            "series_name": name,
            "transform_sign": float(transform_sign),
            "prior_profile": str(prior_profile),
            "bucex_version": "0.3.3",
        }
    )
    return fit
