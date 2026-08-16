"""The single public fitting framework."""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, replace
from typing import Any, Iterable, Mapping, Optional, Sequence
import warnings

import numpy as np

from ..__about__ import __version__
from ..components import DummySeasonal, LocalLinearTrend
from ..core.fit import BulkTailFit, FitResult, combine_fits
from ..inference.fit._fs_output import FSOutput
from ..inference.config import GibbsConfig, Laplace, MCMC, Particles
from ..inference.fit.disturbance import sample_posterior
from ..inference.fit.factor import sample_factor_posterior
from ..inference.fit.factor_fs import sample_factor_fs_posterior
from ..inference.fit.hierarchical_ssvs import sample_hierarchical_ssvs_posterior
from ..inference.fit.fs_gaussian import FSGaussianKernel
from ..inference.fit.fs_gev import FSGEVKernel
from ..inference.plan import InferencePlan, inference_plan
from ..models.compiler import CompiledModel, compile_model
from ..models.factor import FactorModel
from ..models.multiseries import MultiSeriesModel
from ..models.factor_compiler import as_factor_array, factor_dates
from ..models.structural import Model, structural_model
from ..observation import GEV, Gaussian
from ..priors import resolve_prior_spec
from ..priors.factor import resolve_factor_priors
from ..priors.hierarchical import resolve_hierarchical_ssvs_priors
from ..priors.structural import FSGaussianPriors, FSGEVPriors

Array = np.ndarray


def make_gaussian_model(
    components: Sequence[Any],
    *,
    name: str | None = None,
) -> Model:
    return Model(Gaussian(), components, name=name)


def make_gev_model(
    components: Sequence[Any],
    *,
    name: str | None = None,
    xi_bounds: tuple[float, float] = (-0.5, 0.5),
) -> Model:
    return Model(GEV(xi_bounds=xi_bounds), components, name=name)


def _default_model(family: str, period: int | None, trend: str = "local_linear") -> Model:
    return structural_model(
        family=str(family).lower(),
        trend=trend,
        period=period,
    )


def _seasonal_initial(y: Array, period: int | None) -> Array:
    if period is None:
        return np.zeros(0, dtype=float)
    overall = float(np.nanmean(y))
    phase = np.arange(y.size) % int(period)
    full = np.asarray(
        [np.nanmean(y[phase == index]) - overall for index in range(int(period))],
        dtype=float,
    )
    full = np.nan_to_num(full, nan=0.0)
    full -= full.mean()
    return full[:-1]


def _default_initial_values(
    y: Array,
    model: Model,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gamma0 = _seasonal_initial(y, model.period)
    if model.period is None:
        residual = y - float(np.mean(y))
    else:
        phase = np.arange(y.size) % int(model.period)
        full = np.r_[gamma0, -gamma0.sum()]
        residual = y - float(np.mean(y)) - full[phase]
    scale = max(float(np.std(residual, ddof=1)), 0.25)
    state = {
        "alpha0": float(np.mean(y)),
        "beta0": 0.0,
        "gamma0_season": gamma0,
        "s_level": 0.02,
        "s_trend": 0.001,
        "s_season": 0.02,
        "q_level": 0.02**2,
        "q_trend": 0.001**2,
        "q_season": 0.02**2,
    }
    observation = {"sigma": scale, "sigma2": scale**2}
    if model.family == "gev":
        observation["xi"] = -0.10
    return state, observation


def _set_model_initial_values(
    model: Model,
    params_state: Mapping[str, Any],
) -> Model:
    """Record data initialization in the model without changing its structure."""

    result = copy.deepcopy(model)
    for component in result.components:
        if isinstance(component, LocalLinearTrend):
            component.initial_level = float(params_state["alpha0"])
            component.initial_slope = float(params_state.get("beta0", 0.0))
        elif isinstance(component, DummySeasonal) and component.mode != "off":
            component.initial_mean = tuple(
                np.asarray(params_state.get("gamma0_season", ()), dtype=float)
            )
    return result


def _resolve_mcmc(
    mcmc: MCMC | None,
    config: GibbsConfig | None,
    *,
    n_iter: int | None,
    burn: int | None,
    thin: int | None,
    chains: int | None,
    seed: int | None,
    progress: bool | None,
) -> MCMC:
    compact_used = any(
        value is not None for value in (n_iter, burn, thin, chains, seed, progress)
    )
    if mcmc is not None and (config is not None or compact_used):
        raise ValueError("Give mcmc= or legacy iteration arguments, not both.")
    if mcmc is not None:
        return mcmc
    if config is not None:
        if any(value is not None for value in (n_iter, burn, thin, seed, progress)):
            raise ValueError("Give config= or individual compact arguments, not both.")
        return config.to_mcmc(chains=1 if chains is None else int(chains))
    if compact_used:
        n_iter_value = 2000 if n_iter is None else int(n_iter)
        burn_value = 1000 if burn is None else int(burn)
        thin_value = 1 if thin is None else int(thin)
        if not 0 <= burn_value < n_iter_value:
            raise ValueError("Require 0 <= burn < n_iter.")
        return MCMC(
            draws=len(range(burn_value, n_iter_value, thin_value)),
            warmup=burn_value,
            thin=thin_value,
            chains=1 if chains is None else int(chains),
            seed=seed,
            progress=False if progress is None else bool(progress),
        )
    return MCMC()


def _resolve_tail(tail: str | None, transform_sign: float) -> float:
    if tail is None:
        sign = float(transform_sign)
    else:
        key = str(tail).lower()
        if key in {"min", "minimum", "lower"}:
            sign = -1.0
        elif key in {"max", "maximum", "upper"}:
            sign = 1.0
        else:
            raise ValueError("tail must be 'min' or 'max'.")
    if sign not in {-1.0, 1.0}:
        raise ValueError("transform_sign must be +1 or -1.")
    return sign


def _chain_seeds(seed: int | None, chains: int) -> list[int]:
    if chains == 1 and seed is not None:
        return [int(seed)]
    return [
        int(sequence.generate_state(1)[0])
        for sequence in np.random.SeedSequence(seed).spawn(int(chains))
    ]


def _canonical_fs_parameters(draws: Mapping[str, Array]) -> dict[str, Array]:
    output = {name: np.asarray(values) for name, values in draws.items()}
    mappings = {
        "level": ("s_level", "q_level"),
        "slope": ("s_trend", "q_trend"),
        "seasonal": ("s_season", "q_season"),
    }
    for name, (signed, variance) in mappings.items():
        if signed in output:
            output[f"sd.{name}"] = np.abs(np.asarray(output[signed], dtype=float))
            output[f"signed_sd.{name}"] = np.asarray(output[signed], dtype=float)
        elif variance in output:
            output[f"sd.{name}"] = np.sqrt(
                np.maximum(np.asarray(output[variance], dtype=float), 0.0)
            )
    return output


def _stack_fs_chains(
    outputs: Sequence[FSOutput],
    *,
    model: Model,
    compiled: CompiledModel,
    priors: Any,
    y: Array,
    exog: Array | None,
    dates: Array | None,
    series_name: str | None,
    transform_sign: float,
    plan: InferencePlan,
    mcmc: MCMC,
    particles: Particles,
    laplace: Laplace,
    initial_state: Mapping[str, Any],
    initial_observation: Mapping[str, Any],
    seeds: Sequence[int],
) -> FitResult:
    if not outputs:
        raise RuntimeError("No FS chains were produced.")
    n_draws = outputs[0].n_draws
    if any(output.n_draws != n_draws for output in outputs):
        raise RuntimeError("FS chains retained different numbers of draws.")
    parameters_by_chain = [_canonical_fs_parameters(output.draws_static) for output in outputs]
    keys = set(parameters_by_chain[0])
    if any(set(parameters) != keys for parameters in parameters_by_chain[1:]):
        raise RuntimeError("FS chains produced incompatible parameter blocks.")
    parameter_draws = {
        name: np.stack([parameters[name] for parameters in parameters_by_chain], axis=0)
        for name in sorted(keys)
    }
    state_draws = np.stack([np.asarray(output.draws_states) for output in outputs], axis=0)
    log_posterior = np.stack([np.asarray(output.logpost) for output in outputs], axis=0)

    acceptance_names = set().union(*(output.acceptance for output in outputs))
    acceptance = {
        name: np.asarray([output.acceptance.get(name, np.nan) for output in outputs])
        for name in sorted(acceptance_names)
    }
    if "sigma_mh" in acceptance:
        acceptance.setdefault("sigma", acceptance["sigma_mh"])
    if "xi_mh" in acceptance:
        acceptance.setdefault("xi", acceptance["xi_mh"])

    retained = np.arange(mcmc.warmup, mcmc.iterations, mcmc.thin)
    metric_names = {
        "laplace_iterations",
        "laplace_converged",
        "laplace_relative_change",
        "laplace_support_rejections",
        "particle_min_ess",
        "particle_mean_unique_ancestors",
        "particle_path_changed",
        "particle_changed_fraction",
        "fs_elliptical_slice_steps",
        "ssvs_model_move_accepted",
        "ssvs_model_proposed_change",
        "ssvs_model_log_acceptance_ratio",
        "sign_invariance_error",
    }
    draw_metrics: dict[str, Array] = {}
    for name in sorted(metric_names):
        values = []
        for output in outputs:
            raw = np.asarray(output.meta.get("engine_diagnostics", {}).get(name, []))
            if raw.size == mcmc.iterations:
                raw = raw[retained]
            if raw.size != n_draws:
                raw = np.full(n_draws, np.nan)
            values.append(raw)
        draw_metrics[name] = np.stack(values, axis=0)

    auxiliary: dict[str, Array] = {}
    if all("draws_states_ncp" in output.meta for output in outputs):
        auxiliary["states.fruehwirth_schnatter"] = np.stack(
            [np.asarray(output.meta["draws_states_ncp"]) for output in outputs],
            axis=0,
        )
    restored = [float(output.meta.get("restored_fraction", 0.0)) for output in outputs]
    restored_iterations = [
        int(output.meta.get("restored_iterations", 0)) for output in outputs
    ]
    attempt_failures = Counter()
    restore_failures = Counter()
    for output in outputs:
        attempt_failures.update(output.meta.get("attempt_failure_counts", {}))
        restore_failures.update(output.meta.get("restore_failure_counts", {}))
    diagnostics = {
        "acceptance": acceptance,
        "draw_metrics": draw_metrics,
        "mcmc": asdict(mcmc),
        "particles": asdict(particles),
        "laplace": asdict(laplace),
        "restored_fraction_by_chain": restored,
        "chain_seeds": list(map(int, seeds)),
        "sign_switch_counts_by_chain": [
            dict(output.meta.get("sign_switch_counts", {})) for output in outputs
        ],
    }
    initial_parameters = []
    for _ in outputs:
        values = {
            "sd.level": abs(float(initial_state.get("s_level", 0.0))),
            "sigma": float(initial_observation["sigma"]),
        }
        if "s_trend" in initial_state:
            values["sd.slope"] = abs(float(initial_state["s_trend"]))
        if "s_season" in initial_state and compiled.model.period is not None:
            values["sd.seasonal"] = abs(float(initial_state["s_season"]))
        if "xi" in initial_observation:
            values["xi"] = float(initial_observation["xi"])
        initial_parameters.append(values)
    return FitResult(
        model=model,
        compiled=compiled,
        priors=priors,
        y=np.asarray(y, dtype=float),
        exog=None if exog is None else np.asarray(exog),
        dates=None if dates is None else np.asarray(dates),
        series_name=series_name,
        transform_sign=float(transform_sign),
        state_draws=state_draws,
        parameter_draws=parameter_draws,
        log_posterior=log_posterior,
        plan=plan,
        sampler_diagnostics=diagnostics,
        initial_values={
            "params_state": copy.deepcopy(dict(initial_state)),
            "params_obs": copy.deepcopy(dict(initial_observation)),
            "parameters_by_chain": initial_parameters,
            "indicators_by_chain": [{} for _ in outputs],
        },
        auxiliary_draws=auxiliary,
        metadata={
            "bucex_version": __version__,
            "inference_contract": "fruehwirth_schnatter_augmented_noncentering",
            "signed_innovation_scales": True,
            "kernel": outputs[0].meta.get("sampler"),
            "bayesian_lasso": bool(outputs[0].meta.get("bayesian_lasso", False)),
            "componentwise_lasso": bool(outputs[0].meta.get("componentwise_lasso", False)),
            "regularized_horseshoe": bool(outputs[0].meta.get("regularized_horseshoe", False)),
            "triple_gamma": bool(outputs[0].meta.get("triple_gamma", False)),
            "regularized_triple_gamma": bool(
                outputs[0].meta.get("regularized_triple_gamma", False)
            ),
            "shrinkage_update": outputs[0].meta.get("shrinkage_update"),
            "pc_innovation_prior": bool(outputs[0].meta.get("pc_innovation_prior", False)),
            "structural_ssvs": bool(outputs[0].meta.get("structural_ssvs", False)),
            "model_selection_exact": outputs[0].meta.get("model_selection_exact"),
            "model_selection_basis": outputs[0].meta.get("model_selection_basis"),
            "pgas_exact_invariant": bool(outputs[0].meta.get("pgas_exact_invariant", False)),
            "restored_iterations": int(sum(restored_iterations)),
            "restored_iterations_by_chain": restored_iterations,
            "restored_fraction": float(np.mean(restored)),
            "attempt_failure_counts": dict(attempt_failures),
            "restore_failure_counts": dict(restore_failures),
            "max_state_tries": outputs[0].meta.get("max_state_tries"),
            "sign_switching": all(
                bool(output.meta.get("sign_switching", False))
                for output in outputs
            ),
            "sign_switch_invariance_checked": all(
                bool(output.meta.get("sign_switch_invariance_checked", False))
                for output in outputs
            ),
        },
    )


def _fit_fruehwirth_schnatter(
    y: Array,
    *,
    model: Model,
    compiled: CompiledModel,
    priors: FSGaussianPriors | FSGEVPriors,
    plan: InferencePlan,
    mcmc: MCMC,
    particles: Particles,
    laplace: Laplace,
    state_kwargs: Mapping[str, Any],
    params_state: Mapping[str, Any],
    params_obs: Mapping[str, Any],
    exog: Array | None,
    dates: Array | None,
    series_name: str | None,
    transform_sign: float,
) -> FitResult:
    if exog is not None:
        raise ValueError("The FS parameterization does not currently support regression.")
    options = dict(state_kwargs)
    options.update(
        asis=bool(plan.asis),
        particles=int(particles.n),
        particle_proposal=str(particles.proposal),
        particle_resampling=str(particles.resampling),
        laplace_max_iterations=int(laplace.max_iterations),
        laplace_tolerance=float(laplace.tolerance),
        curvature_floor=float(laplace.curvature_floor),
        maximum_variance=float(laplace.maximum_variance),
        draw_attempts=int(laplace.draw_attempts),
    )
    seeds = _chain_seeds(mcmc.seed, mcmc.chains)
    outputs: list[FSOutput] = []
    for chain, seed in enumerate(seeds):
        chain_mcmc = replace(mcmc, chains=1, seed=seed)
        chain_options = {
            **options,
            "_progress_chain": chain + 1,
            "_progress_chains": mcmc.chains,
            "_progress_label": "univariate",
        }
        if model.family == "gaussian":
            sampler = FSGaussianKernel(model, priors, config=chain_mcmc)
        else:
            sampler = FSGEVKernel(model, priors, config=chain_mcmc)
        outputs.append(
            sampler.fit(
                y,
                dict(params_state),
                dict(params_obs),
                exog=exog,
                state_method=plan.engine,
                state_kwargs=chain_options,
            )
        )
    return _stack_fs_chains(
        outputs,
        model=model,
        compiled=compiled,
        priors=priors,
        y=y,
        exog=exog,
        dates=dates,
        series_name=series_name,
        transform_sign=transform_sign,
        plan=plan,
        mcmc=mcmc,
        particles=particles,
        laplace=laplace,
        initial_state=params_state,
        initial_observation=params_obs,
        seeds=seeds,
    )


def _fit_factor_model(
    y: Any,
    model: FactorModel,
    *,
    exog: Any,
    priors: Any,
    engine: str,
    parameterization: str,
    asis: bool,
    mcmc: MCMC | None,
    particles: Particles | int | None,
    laplace: Laplace | None,
    init: Mapping[str, Any] | None,
    dates: Array | None,
    name: str | None,
    config: GibbsConfig | None,
    n_iter: int | None,
    burn: int | None,
    thin: int | None,
    chains: int | None,
    seed: int | None,
    progress: bool | None,
) -> FitResult:
    if name is not None:
        model = replace(model, name=str(name))
    if dates is None:
        dates = factor_dates(y, model)
    y_original = as_factor_array(y, model)
    signs = model.transform_signs
    y_model = y_original * signs[None, :]
    if dates is not None and np.asarray(dates).reshape(-1).size != y_model.shape[0]:
        raise ValueError("dates must have the same length as factor-model observations.")
    compiled = compile_model(model, y_model, exog=exog)
    resolved_mcmc = _resolve_mcmc(
        mcmc,
        config,
        n_iter=n_iter,
        burn=burn,
        thin=thin,
        chains=chains,
        seed=seed,
        progress=progress,
    )
    resolved_particles = (
        Particles()
        if particles is None
        else (Particles(n=int(particles)) if isinstance(particles, int) else particles)
    )
    if not isinstance(resolved_particles, Particles):
        raise TypeError("particles must be Particles(...) or an integer.")
    resolved_laplace = Laplace() if laplace is None else laplace
    if not isinstance(resolved_laplace, Laplace):
        raise TypeError("laplace must be Laplace(...).")
    resolved_plan = inference_plan(
        compiled,
        engine=engine,
        parameterization=parameterization,
        asis=asis,
    )
    resolved_priors = resolve_factor_priors(compiled, priors)
    if resolved_plan.parameterization == "fruehwirth_schnatter":
        return sample_factor_fs_posterior(
            y_model,
            compiled,
            resolved_priors,
            resolved_plan,
            mcmc=resolved_mcmc,
            particles=resolved_particles,
            laplace=resolved_laplace,
            dates=dates,
            initial_parameters=None if init is None else dict(init),
        )
    return sample_factor_posterior(
        y_model,
        compiled,
        resolved_priors,
        resolved_plan,
        mcmc=resolved_mcmc,
        particles=resolved_particles,
        laplace=resolved_laplace,
        dates=dates,
        initial_parameters=None if init is None else dict(init),
    )


def _fit_multiseries_model(
    y: Any,
    model: MultiSeriesModel,
    *,
    exog: Any,
    priors: Any,
    engine: str,
    parameterization: str,
    asis: bool,
    mcmc: MCMC | None,
    particles: Particles | int | None,
    laplace: Laplace | None,
    init: Mapping[str, Any] | None,
    dates: Array | None,
    name: str | None,
    config: GibbsConfig | None,
    n_iter: int | None,
    burn: int | None,
    thin: int | None,
    chains: int | None,
    seed: int | None,
    progress: bool | None,
) -> FitResult:
    """Resolve and run the joint multiseries hierarchy."""

    if name is not None:
        model = replace(model, name=str(name))
    if dates is None:
        dates = factor_dates(y, model)
    y_original = as_factor_array(y, model)
    y_model = y_original * model.transform_signs[None, :]
    if dates is not None and np.asarray(dates).reshape(-1).size != y_model.shape[0]:
        raise ValueError("dates must have the same length as multiseries observations.")
    if exog is not None:
        raise ValueError("Hierarchical structural SSVS does not support regression yet.")
    compiled = compile_model(model, y_model, exog=exog)
    resolved_mcmc = _resolve_mcmc(
        mcmc,
        config,
        n_iter=n_iter,
        burn=burn,
        thin=thin,
        chains=chains,
        seed=seed,
        progress=progress,
    )
    resolved_particles = (
        Particles()
        if particles is None
        else (Particles(n=int(particles)) if isinstance(particles, int) else particles)
    )
    if not isinstance(resolved_particles, Particles):
        raise TypeError("particles must be Particles(...) or an integer.")
    resolved_laplace = Laplace() if laplace is None else laplace
    if not isinstance(resolved_laplace, Laplace):
        raise TypeError("laplace must be Laplace(...).")
    resolved_plan = inference_plan(
        compiled,
        engine=engine,
        parameterization=parameterization,
        asis=asis,
    )
    if resolved_plan.parameterization != "fruehwirth_schnatter":
        raise ValueError(
            "MultiSeriesModel currently implements hierarchical SSVS through "
            "parameterization='fs'."
        )
    if resolved_plan.asis:
        raise ValueError("ASIS is not combined with exact hierarchical SSVS.")
    resolved_plan = replace(resolved_plan, backend="hierarchical_ssvs")
    resolved_priors = resolve_hierarchical_ssvs_priors(compiled, priors)
    return sample_hierarchical_ssvs_posterior(
        y_model,
        compiled,
        resolved_priors,
        resolved_plan,
        mcmc=resolved_mcmc,
        particles=resolved_particles,
        laplace=resolved_laplace,
        dates=dates,
        initial_parameters=init,
    )


def fit(
    y: Array,
    model: Model | MultiSeriesModel | FactorModel | str | None = None,
    *,
    family: str | None = None,
    trend: str | None = None,
    period: int | None = None,
    exog: Array | None = None,
    priors: Any = None,
    engine: str = "auto",
    parameterization: str = "auto",
    asis: bool = False,
    mcmc: MCMC | None = None,
    particles: Particles | int | None = None,
    laplace: Laplace | None = None,
    init: Mapping[str, Any] | None = None,
    init_params_state: Mapping[str, Any] | None = None,
    init_params_obs: Mapping[str, Any] | None = None,
    dates: Array | None = None,
    name: str | None = None,
    tail: str | None = None,
    transform_sign: float = 1.0,
    state_kwargs: Mapping[str, Any] | None = None,
    # Compact 0.3 arguments, translated once here.
    config: GibbsConfig | None = None,
    n_iter: int | None = None,
    burn: int | None = None,
    thin: int | None = None,
    chains: int | None = None,
    seed: int | None = None,
    progress: bool | None = None,
    state_method: str | None = None,
    method: str = "mcmc",
) -> FitResult:
    """Fit a Bayesian structural model through one integrated framework.

    The independent choices are ``engine`` (FFBS/Laplace/PGAS),
    ``parameterization`` (centred/FS/disturbance), the innovation prior, and
    optional ASIS interweaving.  Invalid mathematical combinations fail before
    sampling and are recorded in the returned :class:`FitResult` plan.
    """
    if str(method).lower() not in {"gibbs", "mcmc"}:
        raise NotImplementedError("fit currently exposes MCMC inference.")
    if dates is None and hasattr(y, "index"):
        dates = np.asarray(y.index)
    if name is None and getattr(y, "name", None) is not None:
        name = str(y.name)

    if isinstance(model, str):
        if family is not None:
            raise ValueError("Give model= or family=, not both.")
        family, model = model, None
    if isinstance(model, (MultiSeriesModel, FactorModel)):
        conflicts = {
            "family": family,
            "trend": trend,
            "period": period,
            "tail": tail,
            "state_method": state_method,
            "state_kwargs": state_kwargs,
            "init_params_state": init_params_state,
            "init_params_obs": init_params_obs,
        }
        used = sorted(name for name, value in conflicts.items() if value is not None)
        if used:
            raise ValueError(
                "These univariate-only arguments cannot be used with a "
                f"multiseries model: {used}."
            )
        if float(transform_sign) != 1.0:
            raise ValueError(
                "Set lower-tail orientation on each Channel(tail='lower'), not transform_sign=."
            )
        if str(method).lower() not in {"gibbs", "mcmc"}:
            raise NotImplementedError("fit currently exposes MCMC inference.")
        fitter = (
            _fit_multiseries_model
            if isinstance(model, MultiSeriesModel)
            else _fit_factor_model
        )
        return fitter(
            y,
            model,
            exog=exog,
            priors=priors,
            engine=engine,
            parameterization=parameterization,
            asis=asis,
            mcmc=mcmc,
            particles=particles,
            laplace=laplace,
            init=init,
            dates=dates,
            name=name,
            config=config,
            n_iter=n_iter,
            burn=burn,
            thin=thin,
            chains=chains,
            seed=seed,
            progress=progress,
        )
    if model is None:
        family = "gaussian" if family is None else family
        model = _default_model(family, period, trend or "local_linear")
    elif not isinstance(model, Model):
        raise TypeError(
            "model must be a bucex.Model, MultiSeriesModel, or FactorModel."
        )
    elif family is not None and str(family).lower() != model.family:
        raise ValueError("family conflicts with model.observation.")
    elif trend is not None:
        raise ValueError("trend is a model-construction shorthand; omit it with model=.")
    if period is not None and model.period != int(period):
        raise ValueError("period conflicts with the supplied model.")
    if state_method is not None:
        if engine != "auto":
            raise ValueError("Give engine= or the deprecated state_method=, not both.")
        engine = state_method

    sign = _resolve_tail(tail, transform_sign)
    y_original = np.asarray(y, dtype=float).reshape(-1)
    if y_original.size < 2 or not np.all(np.isfinite(y_original)):
        raise ValueError("y must contain at least two finite observations.")
    y_model = sign * y_original
    if dates is not None and np.asarray(dates).reshape(-1).size != y_model.size:
        raise ValueError("dates must have the same length as y.")

    state_initial, observation_initial = _default_initial_values(y_model, model)
    if init_params_state is not None:
        state_initial.update(init_params_state)
    if init_params_obs is not None:
        observation_initial.update(init_params_obs)
    for process, legacy in (("level", "level"), ("slope", "trend"), ("seasonal", "season")):
        signed_key = f"s_{legacy}"
        variance_key = f"q_{legacy}"
        explicit = {} if init_params_state is None else init_params_state
        if variance_key in explicit and signed_key not in explicit:
            state_initial[signed_key] = float(np.sqrt(float(explicit[variance_key])))
        elif variance_key in explicit and signed_key in explicit and not np.isclose(
            float(explicit[signed_key]) ** 2,
            float(explicit[variance_key]),
        ):
            raise ValueError(f"Conflicting initial {signed_key} and {variance_key}.")
    if init_params_obs is not None and "sigma2" in init_params_obs and "sigma" not in init_params_obs:
        observation_initial["sigma"] = float(np.sqrt(float(init_params_obs["sigma2"])))
    canonical_init = dict(init or {})
    legacy_to_canonical = {
        "s_level": "sd.level",
        "s_trend": "sd.slope",
        "s_season": "sd.seasonal",
    }
    for legacy, canonical in legacy_to_canonical.items():
        if canonical in canonical_init:
            value = float(canonical_init[canonical])
            if (
                init_params_state is not None
                and legacy in init_params_state
                and not np.isclose(abs(float(init_params_state[legacy])), value)
            ):
                raise ValueError(f"Conflicting initial values for {legacy} and {canonical}.")
            state_initial[legacy] = value
    for observation_name in ("sigma", "xi"):
        if observation_name in canonical_init:
            if (
                init_params_obs is not None
                and observation_name in init_params_obs
                and not np.isclose(
                    float(init_params_obs[observation_name]),
                    float(canonical_init[observation_name]),
                )
            ):
                raise ValueError(f"Conflicting initial values for {observation_name}.")
            observation_initial[observation_name] = float(
                canonical_init[observation_name]
            )
    if "sigma" in observation_initial:
        observation_initial["sigma2"] = float(observation_initial["sigma"]) ** 2
    allowed_initial = {
        *(f"sd.{process}" for process in model.noise_names),
        "sigma",
        *(["xi"] if model.family == "gev" else []),
    }
    unknown_initial = sorted(set(canonical_init) - allowed_initial)
    if unknown_initial:
        raise ValueError(f"Unknown canonical initial parameters: {unknown_initial}")
    model = _set_model_initial_values(model, state_initial)
    compiled = compile_model(model, y_model, exog=exog)
    resolved_mcmc = _resolve_mcmc(
        mcmc,
        config,
        n_iter=n_iter,
        burn=burn,
        thin=thin,
        chains=chains,
        seed=seed,
        progress=progress,
    )
    resolved_particles = (
        Particles()
        if particles is None
        else (Particles(n=int(particles)) if isinstance(particles, int) else particles)
    )
    if not isinstance(resolved_particles, Particles):
        raise TypeError("particles must be Particles(...) or an integer.")
    resolved_laplace = Laplace() if laplace is None else laplace
    if not isinstance(resolved_laplace, Laplace):
        raise TypeError("laplace must be Laplace(...).")
    plan = inference_plan(
        compiled,
        engine=engine,
        parameterization=parameterization,
        asis=asis,
    )
    resolved_priors = resolve_prior_spec(
        compiled,
        parameterization=plan.parameterization,
        priors=priors,
    )

    if plan.parameterization == "fruehwirth_schnatter":
        return _fit_fruehwirth_schnatter(
            y_model,
            model=model,
            compiled=compiled,
            priors=resolved_priors,
            plan=plan,
            mcmc=resolved_mcmc,
            particles=resolved_particles,
            laplace=resolved_laplace,
            state_kwargs={} if state_kwargs is None else state_kwargs,
            params_state=state_initial,
            params_obs=observation_initial,
            exog=exog,
            dates=dates,
            series_name=name,
            transform_sign=sign,
        )

    general_initial = dict(canonical_init)
    if init_params_state is not None:
        for process, legacy in (
            ("level", "level"),
            ("slope", "trend"),
            ("seasonal", "season"),
        ):
            if process in compiled.noise_names:
                signed_key = f"s_{legacy}"
                variance_key = f"q_{legacy}"
                if signed_key in init_params_state or variance_key in init_params_state:
                    general_initial[f"sd.{process}"] = abs(
                        float(state_initial[signed_key])
                    )
    if init_params_obs is not None:
        general_initial["sigma"] = float(observation_initial["sigma"])
        if model.family == "gev" and "xi" in init_params_obs:
            general_initial["xi"] = float(observation_initial["xi"])

    return sample_posterior(
        y_model,
        compiled,
        resolved_priors,
        plan,
        mcmc=resolved_mcmc,
        particles=resolved_particles,
        laplace=resolved_laplace,
        dates=None if dates is None else np.asarray(dates),
        series_name=name,
        transform_sign=sign,
        initial_parameters=general_initial or None,
    )


def plan(
    model: Model | MultiSeriesModel | FactorModel,
    y: Array,
    *,
    exog: Array | None = None,
    engine: str = "auto",
    parameterization: str = "auto",
    asis: bool = False,
) -> InferencePlan:
    values = (
        as_factor_array(y, model) * model.transform_signs[None, :]
        if isinstance(model, (MultiSeriesModel, FactorModel))
        else np.asarray(y, dtype=float)
    )
    return inference_plan(
        compile_model(model, values, exog=exog),
        engine=engine,
        parameterization=parameterization,
        asis=asis,
    )


def fit_bulk_tail(
    bulk_y: Array,
    tail_y: Array,
    *,
    period: int | None = None,
    components: Sequence[Any] | None = None,
    bulk_model: Model | None = None,
    tail_model: Model | None = None,
    bulk_priors: Any = None,
    tail_priors: Any = None,
    bulk_mcmc: MCMC | None = None,
    tail_mcmc: MCMC | None = None,
    tail_engine: str = "auto",
    parameterization: str = "auto",
    asis: bool = False,
    particles: Particles | int | None = None,
    laplace: Laplace | None = None,
    dates: Array | None = None,
    tail_direction: str = "max",
    bulk_kwargs: Mapping[str, Any] | None = None,
    tail_kwargs: Mapping[str, Any] | None = None,
) -> BulkTailFit:
    """Fit aligned Gaussian bulk and GEV tail series independently."""

    bulk_values = np.asarray(bulk_y, dtype=float).reshape(-1)
    tail_values = np.asarray(tail_y, dtype=float).reshape(-1)
    if bulk_values.size != tail_values.size:
        raise ValueError("bulk_y and tail_y must have equal length.")
    if dates is None and hasattr(bulk_y, "index") and hasattr(tail_y, "index"):
        if not bulk_y.index.equals(tail_y.index):
            raise ValueError("Pandas bulk and tail series must have identical indexes.")
        dates = np.asarray(bulk_y.index)
    if components is not None:
        if bulk_model is not None or tail_model is not None:
            raise ValueError("Give components= or explicit bulk_model/tail_model, not both.")
        bulk_model = make_gaussian_model(components, name="bulk")
        tail_model = make_gev_model(copy.deepcopy(list(components)), name="tail")
    if bulk_model is None:
        bulk_model = _default_model("gaussian", period)
    if tail_model is None:
        tail_model = _default_model("gev", period)
    if bulk_model.family != "gaussian" or tail_model.family != "gev":
        raise ValueError("bulk_model must be Gaussian and tail_model must be GEV.")

    bulk_options = {
        "priors": bulk_priors,
        "parameterization": parameterization,
        "asis": asis,
        "mcmc": bulk_mcmc,
        "dates": dates,
        "name": "bulk",
        **dict(bulk_kwargs or {}),
    }
    tail_options = {
        "priors": tail_priors,
        "engine": tail_engine,
        "parameterization": parameterization,
        "asis": asis,
        "mcmc": tail_mcmc,
        "particles": particles,
        "laplace": laplace,
        "dates": dates,
        "name": "tail",
        "tail": tail_direction,
        **dict(tail_kwargs or {}),
    }
    bulk = fit(
        bulk_values,
        model=bulk_model,
        **bulk_options,
    )
    tail = fit(
        tail_values,
        model=tail_model,
        **tail_options,
    )
    return BulkTailFit(bulk=bulk, tail=tail)


def fit_bayes(y: Array, *args, **kwargs) -> FitResult:
    """Deprecated compact wrapper; all work is delegated to :func:`fit`."""

    warnings.warn(
        "fit_bayes is deprecated; use bucex.fit with MCMC/Particles/Laplace configs.",
        DeprecationWarning,
        stacklevel=2,
    )
    # The compact 0.3/1.1 entry point represented monthly structural models
    # and therefore defaulted to a 12-period dummy seasonal.  Keep that one
    # translation here; the canonical ``fit`` default remains non-seasonal.
    if not args and kwargs.get("model") is None and "period" not in kwargs:
        kwargs["period"] = 12
    kwargs.setdefault("parameterization", "fruehwirth_schnatter")
    kwargs.setdefault("priors", "manuscript_lasso")
    return fit(y, *args, **kwargs)


def fit_gaussian_structural(y: Array, *args, **kwargs) -> FitResult:
    kwargs.setdefault("family", "gaussian")
    return fit(y, *args, **kwargs)


def fit_gev_structural(y: Array, *args, **kwargs) -> FitResult:
    kwargs.setdefault("family", "gev")
    return fit(y, *args, **kwargs)


def combine_fs_fits(fits: Iterable[FitResult]) -> FitResult:
    warnings.warn(
        "combine_fs_fits is deprecated; use combine_fits.",
        DeprecationWarning,
        stacklevel=2,
    )
    return combine_fits(fits)


__all__ = [
    "fit",
    "fit_bayes",
    "fit_bulk_tail",
    "fit_gaussian_structural",
    "fit_gev_structural",
    "combine_fits",
    "combine_fs_fits",
    "make_gaussian_model",
    "make_gev_model",
    "plan",
]
