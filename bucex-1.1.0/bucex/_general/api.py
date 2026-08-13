"""Stable public fitting API."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from .compiler import compile_model
from .model import (
    Component,
    DummySeasonal,
    LocalLevel,
    LocalLinearTrend,
    Model,
    structural_model,
)
from .observations import GEV, Gaussian
from .particle import Particles
from .plan import InferencePlan, inference_plan
from .priors import Priors, resolve_priors
from .results import BulkTailFit, FitResult
from .sampler import GibbsConfig, Laplace, MCMC, sample_posterior


Array = np.ndarray


def make_gaussian_model(components: Sequence[Component], *, name: str | None = None) -> Model:
    return Model(observation=Gaussian(), components=components, name=name)


def make_gev_model(
    components: Sequence[Component],
    *,
    xi_bounds: tuple[float, float] = (-0.5, 0.5),
    name: str | None = None,
) -> Model:
    return Model(observation=GEV(xi_bounds=xi_bounds), components=components, name=name)


def plan(
    y: Array,
    model: Model,
    *,
    exog=None,
    engine: str = "auto",
    parameterization: str = "auto",
    asis: bool = False,
) -> InferencePlan:
    compiled = compile_model(model, y, exog=exog)
    return inference_plan(
        compiled,
        engine=engine,
        parameterization=parameterization,
        asis=asis,
    )


def _with_initial_state(model: Model, values: Mapping[str, Any] | None) -> Model:
    if values is None:
        return model
    supplied = dict(values)
    aliases = {
        "alpha0": "level",
        "beta0": "slope",
        "gamma0_season": "seasonal",
    }
    normalized = {aliases.get(name, name): value for name, value in supplied.items()}
    allowed = {"level", "slope", "seasonal"}
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"Unknown initial state values: {unknown}")
    components = []
    for component in model.components:
        if isinstance(component, LocalLevel):
            components.append(
                replace(component, initial_mean=float(normalized["level"]))
                if "level" in normalized
                else component
            )
        elif isinstance(component, LocalLinearTrend):
            updates = {}
            if "level" in normalized:
                updates["initial_level"] = float(normalized["level"])
            if "slope" in normalized:
                updates["initial_slope"] = float(normalized["slope"])
            components.append(replace(component, **updates) if updates else component)
        elif isinstance(component, DummySeasonal) and "seasonal" in normalized:
            seasonal = tuple(np.asarray(normalized["seasonal"], dtype=float).reshape(-1))
            components.append(replace(component, initial_mean=seasonal))
        else:
            components.append(component)
    if "slope" in normalized and not any(isinstance(c, LocalLinearTrend) for c in components):
        raise ValueError("An initial slope was supplied but the model has no slope state.")
    if "seasonal" in normalized and not any(isinstance(c, DummySeasonal) for c in components):
        raise ValueError("An initial seasonal state was supplied but the model has no seasonal component.")
    return replace(model, components=components)


def fit(
    y: Array,
    model: Model | None = None,
    *,
    family: str | None = None,
    period: int | None = None,
    trend: str = "local_linear",
    exog=None,
    priors: Priors | str | None = None,
    engine: str = "auto",
    parameterization: str = "auto",
    asis: bool = False,
    mcmc: MCMC | None = None,
    particles: Particles | None = None,
    laplace: Laplace | None = None,
    dates: Array | None = None,
    name: str | None = None,
    tail: str = "max",
    init: Mapping[str, float] | None = None,
    initial_state: Mapping[str, Any] | None = None,
) -> FitResult:
    """Fit a validated Gaussian or dynamic-location GEV structural model."""

    if dates is None and hasattr(y, "index"):
        dates = np.asarray(y.index)
    if name is None and getattr(y, "name", None) is not None:
        name = str(y.name)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if model is None:
        model = structural_model(
            family="gaussian" if family is None else family,
            trend=trend,
            period=period,
        )
    else:
        if family is not None and str(family).lower() != model.family:
            raise ValueError("family conflicts with model.observation.")
        if period is not None and model.period != int(period):
            raise ValueError("period conflicts with the DummySeasonal component in model.")
    model = _with_initial_state(model, initial_state)
    tail_key = str(tail).lower()
    if tail_key not in {"max", "min"}:
        raise ValueError("tail must be 'max' or 'min'.")
    if model.family == "gaussian" and tail_key != "max":
        raise ValueError("tail='min' is a GEV sign-transform option, not a Gaussian option.")
    sign = -1.0 if tail_key == "min" else 1.0
    model_y = sign * y_arr
    if dates is not None and np.asarray(dates).reshape(-1).size != y_arr.size:
        raise ValueError("dates must have length T.")

    compiled = compile_model(model, model_y, exog=exog)
    resolved_priors = resolve_priors(compiled, priors)
    resolved_plan = inference_plan(
        compiled,
        engine=engine,
        parameterization=parameterization,
        asis=asis,
    )
    resolved_mcmc = MCMC() if mcmc is None else mcmc
    resolved_particles = Particles() if particles is None else particles
    resolved_laplace = Laplace() if laplace is None else laplace
    return sample_posterior(
        model_y,
        compiled,
        resolved_priors,
        resolved_plan,
        mcmc=resolved_mcmc,
        particles=resolved_particles,
        laplace=resolved_laplace,
        dates=dates,
        series_name=name,
        transform_sign=sign,
        initial_parameters=None if init is None else dict(init),
    )


def fit_bayes(
    y: Array,
    *,
    family: str = "gaussian",
    model: Model | None = None,
    period: int | None = None,
    priors: Priors | str | None = None,
    parameterization: str = "auto",
    state_method: str = "auto",
    state_kwargs: dict[str, Any] | None = None,
    n_iter: int = 2000,
    burn: int = 1000,
    thin: int = 1,
    chains: int = 1,
    seed: int | None = None,
    progress: bool = False,
    dates: Array | None = None,
    name: str | None = None,
    tail: str = "max",
    exog=None,
    asis: bool = False,
    config: GibbsConfig | MCMC | None = None,
    init_params_state: Mapping[str, Any] | None = None,
    init_params_obs: Mapping[str, Any] | None = None,
    init: Mapping[str, float] | None = None,
    method: str = "gibbs",
    **kwargs,
) -> FitResult:
    """Compatibility wrapper for the prototype's compact call."""

    if str(method).lower() not in {"gibbs", "mcmc"}:
        raise ValueError("The first release supports method='gibbs' only.")
    if config is None:
        if burn < 0 or n_iter <= burn:
            raise ValueError("Require 0 <= burn < n_iter.")
        draws = len(range(int(burn), int(n_iter), int(thin)))
        resolved_mcmc = MCMC(
            draws=draws,
            warmup=int(burn),
            thin=int(thin),
            chains=int(chains),
            seed=seed,
            progress=progress,
        )
    elif isinstance(config, GibbsConfig):
        resolved_mcmc = config.to_mcmc(chains=chains)
    elif isinstance(config, MCMC):
        resolved_mcmc = config
    else:
        raise TypeError("config must be GibbsConfig, MCMC, or None.")

    initial_state: dict[str, Any] = {}
    initial_parameters = {} if init is None else dict(init)
    legacy_state = {} if init_params_state is None else dict(init_params_state)
    for legacy, canonical in (
        ("alpha0", "level"),
        ("beta0", "slope"),
        ("gamma0_season", "seasonal"),
    ):
        if legacy in legacy_state:
            initial_state[canonical] = legacy_state.pop(legacy)
    for legacy, process in (
        ("level", "level"),
        ("trend", "slope"),
        ("season", "seasonal"),
    ):
        signed = legacy_state.pop(f"s_{legacy}", None)
        variance = legacy_state.pop(f"q_{legacy}", None)
        if signed is not None and variance is not None and not np.isclose(
            float(signed) ** 2, float(variance)
        ):
            raise ValueError(f"Conflicting initial s_{legacy} and q_{legacy}.")
        if signed is not None or variance is not None:
            initial_parameters[f"sd.{process}"] = (
                abs(float(signed)) if signed is not None else np.sqrt(float(variance))
            )
    if legacy_state:
        raise ValueError(f"Unsupported init_params_state keys: {sorted(legacy_state)}")
    legacy_obs = {} if init_params_obs is None else dict(init_params_obs)
    if "sigma" in legacy_obs:
        initial_parameters["sigma"] = float(legacy_obs.pop("sigma"))
    elif "sigma2" in legacy_obs:
        initial_parameters["sigma"] = np.sqrt(float(legacy_obs.pop("sigma2")))
    if "xi" in legacy_obs:
        initial_parameters["xi"] = float(legacy_obs.pop("xi"))
    if legacy_obs:
        raise ValueError(f"Unsupported init_params_obs keys: {sorted(legacy_obs)}")

    state_kwargs = {} if state_kwargs is None else dict(state_kwargs)
    particle_n = int(state_kwargs.pop("particle_n_particles", state_kwargs.pop("n_particles", 256)))
    laplace_options = Laplace(
        max_iterations=int(state_kwargs.pop("max_iter", 30)),
        tolerance=float(state_kwargs.pop("tol", 1e-5)),
        curvature_floor=float(state_kwargs.pop("curvature_floor", 1e-6)),
        maximum_variance=float(state_kwargs.pop("maximum_variance", 1e8)),
        draw_attempts=int(
            state_kwargs.pop("draw_attempts", state_kwargs.pop("max_state_tries", 30))
        ),
    )
    if state_kwargs:
        unknown = sorted(state_kwargs)
        raise ValueError(f"Unsupported state_kwargs in v1: {unknown}")
    if kwargs:
        unknown = sorted(kwargs)
        raise ValueError(f"Unsupported fit_bayes arguments in v1: {unknown}")
    return fit(
        y,
        model=model,
        family=family if model is None else None,
        period=period,
        exog=exog,
        priors=priors,
        engine=state_method,
        parameterization=parameterization,
        asis=asis,
        mcmc=resolved_mcmc,
        particles=Particles(n=particle_n),
        laplace=laplace_options,
        dates=dates,
        name=name,
        tail=tail,
        init=initial_parameters or None,
        initial_state=initial_state or None,
    )


def fit_bulk_tail(
    bulk: Array,
    tail: Array,
    *,
    period: int | None = None,
    components: Sequence[Component] | None = None,
    bulk_model: Model | None = None,
    tail_model: Model | None = None,
    bulk_priors: Priors | str | None = None,
    tail_priors: Priors | str | None = None,
    bulk_mcmc: MCMC | None = None,
    tail_mcmc: MCMC | None = None,
    tail_engine: str = "auto",
    parameterization: str = "auto",
    asis: bool = False,
    particles: Particles | None = None,
    dates: Array | None = None,
    tail_direction: str = "max",
) -> BulkTailFit:
    """Fit aligned Gaussian bulk and GEV tail series independently."""

    if dates is None and hasattr(bulk, "index") and hasattr(tail, "index"):
        if not bulk.index.equals(tail.index):
            raise ValueError("Pandas bulk and tail series must have identical indexes.")
        dates = np.asarray(bulk.index)
    bulk_y = np.asarray(bulk, dtype=float).reshape(-1)
    tail_y = np.asarray(tail, dtype=float).reshape(-1)
    if bulk_y.size != tail_y.size:
        raise ValueError("bulk and tail must have equal length and aligned calendars.")
    if components is not None and (bulk_model is not None or tail_model is not None):
        raise ValueError("Give components or explicit bulk_model/tail_model, not both.")
    if components is not None:
        bulk_model = make_gaussian_model(components, name="bulk")
        tail_model = make_gev_model(components, name="tail")
    if bulk_model is None:
        bulk_model = structural_model("gaussian", period=period)
    if tail_model is None:
        tail_model = structural_model("gev", period=period)
    if bulk_model.family != "gaussian" or tail_model.family != "gev":
        raise ValueError("bulk_model must be Gaussian and tail_model must be GEV.")
    bulk_fit = fit(
        bulk_y,
        model=bulk_model,
        priors=bulk_priors,
        engine="ffbs",
        parameterization=parameterization,
        asis=asis,
        mcmc=MCMC() if bulk_mcmc is None else bulk_mcmc,
        dates=dates,
        name="bulk",
    )
    tail_fit = fit(
        tail_y,
        model=tail_model,
        priors=tail_priors,
        engine=tail_engine,
        parameterization=parameterization,
        asis=asis,
        mcmc=MCMC() if tail_mcmc is None else tail_mcmc,
        particles=particles,
        dates=dates,
        name="tail",
        tail=tail_direction,
    )
    return BulkTailFit(bulk=bulk_fit, tail=tail_fit)


def fit_gaussian_structural(y: Array, **kwargs) -> FitResult:
    """Convenience wrapper fixing ``family='gaussian'``."""

    if kwargs.get("model") is None:
        kwargs["family"] = "gaussian"
    return fit(y, **kwargs)


def fit_gev_structural(y: Array, **kwargs) -> FitResult:
    """Convenience wrapper fixing ``family='gev'``."""

    if kwargs.get("model") is None:
        kwargs["family"] = "gev"
    return fit(y, **kwargs)
