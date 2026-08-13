from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.results import FilterResult, SmootherResult, StateSample
from ..models.base import StateSpaceModel
from .state.ffbs import ffbs_sample
from .state.kalman import filter_and_smooth, kalman_filter, kalman_smoother
from .state.particle import ParticleConfig

Array = np.ndarray
ParamDict = Dict[str, Any]


def _obs_name(model: StateSpaceModel) -> Optional[str]:
    """
    Safely extract observation model name from model.obs.spec.name, if available.
    """
    spec = getattr(getattr(model, "obs", None), "spec", None)
    if spec is None:
        return None
    return getattr(spec, "name", None)


def filter_states(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    method: str = "auto",
    *,
    particle_method: str = "bootstrap",
    particle_n_particles: int = 1000,
    particle_config: Optional[ParticleConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> FilterResult:
    """
    Dispatch filtering to the appropriate backend.

    Parameters
    ----------
    method
        "auto", "kalman", "laplace", or "particle"

    particle_method
        If method == "particle", choose:
          - "bootstrap"
          - "auxiliary"

    particle_n_particles
        Number of particles when using the particle backend.

    particle_config
        Optional ParticleConfig. If provided, it takes precedence over
        particle_method / particle_n_particles where relevant.
    """
    obs_name = _obs_name(model)

    if method == "auto":
        if obs_name == "gaussian":
            return kalman_filter(
                y=y,
                model=model,
                params_state=params_state,
                params_obs=params_obs,
                exog=exog,
            )
        raise NotImplementedError(
            f"No automatic filter backend implemented yet for obs='{obs_name}'."
        )

    if method == "kalman":
        return kalman_filter(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
        )

    if method == "laplace":
        from .state.laplace import laplace_filter
        return laplace_filter(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
        )

    if method == "particle":
        from .state.particle import particle_filter

        cfg = particle_config
        if cfg is None:
            cfg = ParticleConfig(
                n_particles=particle_n_particles,
                method=particle_method,
            )

        return particle_filter(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
            n_particles=cfg.n_particles,
            method=cfg.method,
            rng=rng,
            config=cfg,
        )

    raise ValueError(f"Unknown filter method '{method}'.")


def smooth_states(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    method: str = "auto",
    *,
    particle_method: str = "bootstrap",
    particle_n_particles: int = 1000,
    particle_n_smoother_draws: int = 100,
    particle_config: Optional[ParticleConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> SmootherResult:
    """
    Dispatch smoothing to the appropriate backend.

    Parameters
    ----------
    method
        "auto", "kalman", "laplace", or "particle"

    particle_method
        If method == "particle", choose:
          - "bootstrap"
          - "auxiliary"
    """
    obs_name = _obs_name(model)

    if method == "auto":
        if obs_name == "gaussian":
            _, sr = filter_and_smooth(
                y=y,
                model=model,
                params_state=params_state,
                params_obs=params_obs,
                exog=exog,
            )
            return sr
        raise NotImplementedError(
            f"No automatic smoother backend implemented yet for obs='{obs_name}'."
        )

    if method == "kalman":
        fr = kalman_filter(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
        )
        return kalman_smoother(fr)

    if method == "laplace":
        from .state.laplace import laplace_smoother
        return laplace_smoother(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
        )

    if method == "particle":
        from .state.particle import particle_smoother

        cfg = particle_config
        if cfg is None:
            cfg = ParticleConfig(
                n_particles=particle_n_particles,
                method=particle_method,
                n_smoother_draws=particle_n_smoother_draws,
            )

        return particle_smoother(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
            n_particles=cfg.n_particles,
            method=cfg.method,
            rng=rng,
            n_smoother_draws=cfg.n_smoother_draws,
            config=cfg,
        )

    raise ValueError(f"Unknown smoother method '{method}'.")


def sample_states(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
    method: str = "auto",
    *,
    particle_method: str = "bootstrap",
    particle_n_particles: int = 1000,
    particle_config: Optional[ParticleConfig] = None,
) -> StateSample:
    """
    Dispatch conditional trajectory sampling to the appropriate backend.

    Parameters
    ----------
    method
        "auto", "ffbs", "laplace", or "particle"

    particle_method
        If method == "particle", choose:
          - "bootstrap"
          - "auxiliary"
    """
    obs_name = _obs_name(model)

    if method == "auto":
        if obs_name == "gaussian":
            return ffbs_sample(
                y=y,
                model=model,
                params_state=params_state,
                params_obs=params_obs,
                exog=exog,
                rng=rng,
            )
        raise NotImplementedError(
            f"No automatic state-sampling backend implemented yet for obs='{obs_name}'."
        )

    if method == "ffbs":
        return ffbs_sample(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
            rng=rng,
        )

    if method == "laplace":
        from .state.laplace import laplace_ffbs
        return laplace_ffbs(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
            rng=rng,
        )

    if method == "particle":
        from .state.particle import particle_state_sample

        cfg = particle_config
        if cfg is None:
            cfg = ParticleConfig(
                n_particles=particle_n_particles,
                method=particle_method,
            )

        return particle_state_sample(
            y=y,
            model=model,
            params_state=params_state,
            params_obs=params_obs,
            exog=exog,
            rng=rng,
            n_particles=cfg.n_particles,
            method=cfg.method,
            config=cfg,
        )

    raise ValueError(f"Unknown state-sampling method '{method}'.")


def filter_smooth_sample(
    y: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
    state_method: str = "auto",
    *,
    particle_method: str = "bootstrap",
    particle_n_particles: int = 1000,
    particle_n_smoother_draws: int = 100,
    particle_config: Optional[ParticleConfig] = None,
) -> Tuple[FilterResult, SmootherResult, StateSample]:
    """
    Convenience wrapper for debugging / demos:
      - filter
      - smooth
      - sample states
    """
    filt_method = "kalman" if state_method == "ffbs" else state_method
    smooth_method = "kalman" if state_method == "ffbs" else state_method

    fr = filter_states(
        y=y,
        model=model,
        params_state=params_state,
        params_obs=params_obs,
        exog=exog,
        method=filt_method,
        particle_method=particle_method,
        particle_n_particles=particle_n_particles,
        particle_config=particle_config,
        rng=rng,
    )

    sr = smooth_states(
        y=y,
        model=model,
        params_state=params_state,
        params_obs=params_obs,
        exog=exog,
        method=smooth_method,
        particle_method=particle_method,
        particle_n_particles=particle_n_particles,
        particle_n_smoother_draws=particle_n_smoother_draws,
        particle_config=particle_config,
        rng=rng,
    )

    xs = sample_states(
        y=y,
        model=model,
        params_state=params_state,
        params_obs=params_obs,
        exog=exog,
        rng=rng,
        method=state_method,
        particle_method=particle_method,
        particle_n_particles=particle_n_particles,
        particle_config=particle_config,
    )

    return fr, sr, xs