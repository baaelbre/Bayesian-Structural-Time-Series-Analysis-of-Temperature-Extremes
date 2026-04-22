from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.results import PosteriorBundle
from ..models.base import StateSpaceModel
from ..inference.fit.base import GibbsConfig
from ..inference.fit.centered_gaussian import CenteredGaussianGibbs
from ..inference.fit.centered_gev import CenteredGEVGibbs
from ..inference.fit.noncentered_gaussian import NonCenteredGaussianGibbs
from ..inference.fit.noncentered_gev import NonCenteredGEVGibbs


def _obs_name(model: StateSpaceModel) -> str | None:
    spec = getattr(getattr(model, "obs", None), "spec", None)
    return getattr(spec, "name", None) if spec is not None else None


def fit_gaussian_structural(
    y: np.ndarray,
    model: StateSpaceModel,
    priors: Any,
    init_params_state: dict[str, Any],
    init_params_obs: dict[str, Any],
    *,
    exog: Optional[np.ndarray] = None,
    config: GibbsConfig | None = None,
    parameterization: str = "centered",
    state_method: str = "ffbs",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> PosteriorBundle:
    """Fit a Gaussian structural model via the Gibbs family in v0.1."""
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
    parameterization: str = "centered",
    state_method: str = "laplace",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> PosteriorBundle:
    """Fit a GEV structural model via MH-within-Gibbs in v0.1."""
    cfg = config or GibbsConfig()
    if parameterization == "centered":
        fitter = CenteredGEVGibbs(model=model, priors=priors, config=cfg)
    elif parameterization == "noncentered":
        fitter = NonCenteredGEVGibbs(model=model, priors=priors, config=cfg)
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


def fit_bayes(
    y: np.ndarray,
    model: StateSpaceModel,
    priors: Any,
    init_params_state: dict[str, Any],
    init_params_obs: dict[str, Any],
    *,
    exog: Optional[np.ndarray] = None,
    config: GibbsConfig | None = None,
    method: str = "gibbs",
    state_method: str = "auto",
    parameterization: str = "centered",
    state_kwargs: Optional[dict[str, Any]] = None,
) -> PosteriorBundle:
    """High-level research API for v0.1.

    The model describes the generative structure; the backend is chosen through
    `method`, `state_method`, and `parameterization`.
    """
    if method != "gibbs":
        raise NotImplementedError("v0.1 currently exposes Gibbs / MH-within-Gibbs fitting only.")

    obs_name = _obs_name(model)
    if obs_name == "gaussian":
        resolved_state_method = "ffbs" if state_method == "auto" else state_method
        return fit_gaussian_structural(
            y=y,
            model=model,
            priors=priors,
            init_params_state=init_params_state,
            init_params_obs=init_params_obs,
            exog=exog,
            config=config,
            parameterization=parameterization,
            state_method=resolved_state_method,
            state_kwargs=state_kwargs,
        )

    if obs_name == "gev":
        resolved_state_method = "laplace" if state_method == "auto" else state_method
        return fit_gev_structural(
            y=y,
            model=model,
            priors=priors,
            init_params_state=init_params_state,
            init_params_obs=init_params_obs,
            exog=exog,
            config=config,
            parameterization=parameterization,
            state_method=resolved_state_method,
            state_kwargs=state_kwargs,
        )

    raise NotImplementedError(f"No v0.1 high-level fitter registered for obs='{obs_name}'.")
