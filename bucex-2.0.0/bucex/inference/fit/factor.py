"""Multi-chain inference for mixed-family shared-factor models."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from ...core.fit import FitResult
from ...core.numerics import bounded_log_jacobian, bounded_to_real, real_to_bounded
from ...models.factor_compiler import CompiledFactorModel
from ...priors.factor import FactorPriors
from ...priors.process import FixedSD
from ..config import Laplace, MCMC, Particles
from ..plan import InferencePlan
from ..state.kalman import ffbs
from ..state.laplace import (
    iterated_laplace,
    joint_state_log_density,
    observation_log_likelihood,
)
from ..state.particle import pgas
from .disturbance import _adapt, _prior_logpdf


Array = np.ndarray


def _initial_parameters(
    compiled: CompiledFactorModel,
    priors: FactorPriors,
    rng: np.random.Generator,
    initial: Mapping[str, float] | None,
) -> dict[str, float]:
    params: dict[str, float] = {}
    for name in compiled.noise_names:
        prior = priors.process[name]
        value = float(prior.initial())
        if not isinstance(prior, FixedSD):
            value = max(value, compiled.y_scale * 1e-10)
        params[f"sd.{name}"] = value
    for channel in compiled.model.channels:
        prior = priors.observation_sd[channel.name]
        sigma = float(prior.initial())
        if not isinstance(prior, FixedSD):
            sigma = max(sigma, compiled.channel_y_scale[channel.name] * 1e-8)
        params[f"sigma.{channel.name}"] = sigma
        if channel.family == "gev":
            params[f"xi.{channel.name}"] = float(priors.shape[channel.name].initial())
    for key, spec in compiled.loading_specs.items():
        params[key] = float(spec.value)

    if initial is not None:
        unknown = sorted(set(initial) - set(params))
        if unknown:
            raise ValueError(f"Unknown initial factor parameters: {unknown}.")
        for name, value in initial.items():
            if name in compiled.loading_specs and compiled.loading_specs[name].fixed:
                if not np.isclose(float(value), compiled.loading_specs[name].value):
                    raise ValueError(f"Fixed loading '{name}' cannot be overridden.")
                continue
            params[name] = float(value)

    for name in compiled.noise_names:
        value = params[f"sd.{name}"]
        prior = priors.process[name]
        if (
            value < 0.0
            or (not isinstance(prior, FixedSD) and value <= 0.0)
            or not np.isfinite(_prior_logpdf(prior, value))
        ):
            raise ValueError(f"Initial sd.{name} is outside its prior support.")
    for channel in compiled.model.channels:
        sigma_key = f"sigma.{channel.name}"
        if params[sigma_key] <= 0.0 or not np.isfinite(
            _prior_logpdf(priors.observation_sd[channel.name], params[sigma_key])
        ):
            raise ValueError(f"Initial {sigma_key} is outside its prior support.")
        if channel.family == "gev":
            xi_key = f"xi.{channel.name}"
            if not np.isfinite(priors.shape[channel.name].logpdf(params[xi_key])):
                raise ValueError(f"Initial {xi_key} is outside its prior support.")
    return params


def _centered_scale_sweep(
    path: Array,
    compiled: CompiledFactorModel,
    params: dict[str, float],
    priors: FactorPriors,
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    adapt: bool,
    iteration: int,
    step_prefix: str = "",
) -> tuple[dict[str, bool], dict[str, float]]:
    disturbance = compiled.to_disturbance(path, params)
    current_sd = compiled.process_vector(params)
    innovations = disturbance.z * current_sd[None, :]
    accepted: dict[str, bool] = {}
    for index, name in enumerate(compiled.noise_names):
        prior = priors.process[name]
        key = f"sd.{name}"
        step_key = f"{step_prefix}{key}"
        if isinstance(prior, FixedSD):
            accepted[key] = False
            continue
        current = float(params[key])
        proposal = float(np.exp(np.log(current) + steps[step_key] * rng.normal()))
        sum_squares = float(np.sum(innovations[:, index] ** 2))
        n_time = innovations.shape[0]

        def target(value: float) -> float:
            return float(
                -n_time * np.log(value)
                - 0.5 * sum_squares / value**2
                + _prior_logpdf(prior, value)
                + np.log(value)
            )

        take = bool(np.log(rng.random()) < target(proposal) - target(current))
        if take:
            params[key] = proposal
        accepted[key] = take
        if adapt:
            steps[step_key] = _adapt(steps[step_key], take, iteration)
    return accepted, steps


def _noncentered_scale_sweep(
    y: Array,
    path: Array,
    compiled: CompiledFactorModel,
    params: dict[str, float],
    priors: FactorPriors,
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    adapt: bool,
    iteration: int,
    step_prefix: str = "",
) -> tuple[Array, dict[str, bool], dict[str, float]]:
    disturbance = compiled.to_disturbance(path, params)
    current_path = path
    current_likelihood = observation_log_likelihood(
        y, compiled.eta(current_path, params=params), compiled, params
    )
    accepted: dict[str, bool] = {}
    for name in compiled.noise_names:
        prior = priors.process[name]
        key = f"sd.{name}"
        step_key = f"{step_prefix}{key}"
        if isinstance(prior, FixedSD):
            accepted[key] = False
            continue
        current = float(params[key])
        proposal = float(np.exp(np.log(current) + steps[step_key] * rng.normal()))
        proposal_params = dict(params)
        proposal_params[key] = proposal
        proposal_path = compiled.from_disturbance(disturbance, proposal_params)
        proposal_likelihood = observation_log_likelihood(
            y,
            compiled.eta(proposal_path, params=proposal_params),
            compiled,
            proposal_params,
        )
        current_target = current_likelihood + _prior_logpdf(prior, current) + np.log(current)
        proposal_target = (
            proposal_likelihood + _prior_logpdf(prior, proposal) + np.log(proposal)
        )
        take = bool(np.log(rng.random()) < proposal_target - current_target)
        if take:
            params[key] = proposal
            current_path = proposal_path
            current_likelihood = proposal_likelihood
        accepted[key] = take
        if adapt:
            steps[step_key] = _adapt(steps[step_key], take, iteration)
    return current_path, accepted, steps


def _observation_sweep(
    y: Array,
    path: Array,
    compiled: CompiledFactorModel,
    params: dict[str, float],
    priors: FactorPriors,
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    adapt: bool,
    iteration: int,
) -> tuple[dict[str, bool], dict[str, float]]:
    eta = compiled.eta(path, params=params)
    accepted: dict[str, bool] = {}
    current_likelihood = observation_log_likelihood(y, eta, compiled, params)
    for channel in compiled.model.channels:
        sigma_key = f"sigma.{channel.name}"
        sigma_prior = priors.observation_sd[channel.name]
        if isinstance(sigma_prior, FixedSD):
            accepted[sigma_key] = False
        else:
            current = float(params[sigma_key])
            proposal = float(np.exp(np.log(current) + steps[sigma_key] * rng.normal()))
            proposal_params = dict(params)
            proposal_params[sigma_key] = proposal
            proposal_likelihood = observation_log_likelihood(
                y, eta, compiled, proposal_params
            )
            current_target = (
                current_likelihood + _prior_logpdf(sigma_prior, current) + np.log(current)
            )
            proposal_target = (
                proposal_likelihood
                + _prior_logpdf(sigma_prior, proposal)
                + np.log(proposal)
            )
            take = bool(np.log(rng.random()) < proposal_target - current_target)
            if take:
                params[sigma_key] = proposal
                current_likelihood = proposal_likelihood
            accepted[sigma_key] = take
            if adapt:
                steps[sigma_key] = _adapt(steps[sigma_key], take, iteration)

        if channel.family != "gev":
            continue
        xi_key = f"xi.{channel.name}"
        shape_prior = priors.shape[channel.name]
        lower, upper = float(shape_prior.lower), float(shape_prior.upper)
        current_xi = float(params[xi_key])
        current_real = bounded_to_real(current_xi, lower, upper)
        proposal_real = float(current_real + steps[xi_key] * rng.normal())
        proposal_xi = real_to_bounded(proposal_real, lower, upper)
        proposal_params = dict(params)
        proposal_params[xi_key] = proposal_xi
        proposal_likelihood = observation_log_likelihood(y, eta, compiled, proposal_params)
        current_target = (
            current_likelihood
            + shape_prior.logpdf(current_xi)
            + bounded_log_jacobian(current_real, lower, upper)
        )
        proposal_target = (
            proposal_likelihood
            + shape_prior.logpdf(proposal_xi)
            + bounded_log_jacobian(proposal_real, lower, upper)
        )
        take = bool(np.log(rng.random()) < proposal_target - current_target)
        if take:
            params[xi_key] = proposal_xi
            current_likelihood = proposal_likelihood
        accepted[xi_key] = take
        if adapt:
            steps[xi_key] = _adapt(steps[xi_key], take, iteration)
    return accepted, steps


def _loading_sweep(
    y: Array,
    path: Array,
    compiled: CompiledFactorModel,
    params: dict[str, float],
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    adapt: bool,
    iteration: int,
) -> tuple[dict[str, bool], dict[str, float]]:
    accepted: dict[str, bool] = {}
    current_likelihood = observation_log_likelihood(
        y, compiled.eta(path, params=params), compiled, params
    )
    for key in compiled.estimated_loading_names:
        spec = compiled.loading_specs[key]
        current = float(params[key])
        proposal = float(current + steps[key] * rng.normal())
        proposal_params = dict(params)
        proposal_params[key] = proposal
        proposal_likelihood = observation_log_likelihood(
            y,
            compiled.eta(path, params=proposal_params),
            compiled,
            proposal_params,
        )

        def prior(value: float) -> float:
            return float(-0.5 * ((value - spec.prior_mean) / spec.prior_sd) ** 2)

        take = bool(
            np.log(rng.random())
            < proposal_likelihood + prior(proposal) - current_likelihood - prior(current)
        )
        if take:
            params[key] = proposal
            current_likelihood = proposal_likelihood
        accepted[key] = take
        if adapt:
            steps[key] = _adapt(steps[key], take, iteration)
    return accepted, steps


def _initial_path(
    y: Array,
    compiled: CompiledFactorModel,
    params: dict[str, float],
    laplace: Laplace,
    rng: np.random.Generator,
) -> Array:
    if compiled.all_gaussian:
        return ffbs(y, compiled, params, rng)[0]
    return iterated_laplace(
        y,
        compiled,
        params,
        rng,
        max_iterations=laplace.max_iterations,
        tolerance=laplace.tolerance,
        curvature_floor=laplace.curvature_floor,
        maximum_variance=laplace.maximum_variance,
        draw_attempts=laplace.draw_attempts,
    ).path


def _log_posterior(
    y: Array,
    path: Array,
    compiled: CompiledFactorModel,
    params: dict[str, float],
    priors: FactorPriors,
) -> float:
    value = joint_state_log_density(y, path, compiled, params)
    for name in compiled.noise_names:
        value += _prior_logpdf(priors.process[name], params[f"sd.{name}"])
    for channel in compiled.model.channels:
        value += _prior_logpdf(
            priors.observation_sd[channel.name], params[f"sigma.{channel.name}"]
        )
        if channel.family == "gev":
            value += priors.shape[channel.name].logpdf(params[f"xi.{channel.name}"])
    for key in compiled.estimated_loading_names:
        spec = compiled.loading_specs[key]
        value += -0.5 * ((params[key] - spec.prior_mean) / spec.prior_sd) ** 2
    return float(value)


def sample_factor_posterior(
    y: Array,
    compiled: CompiledFactorModel,
    priors: FactorPriors,
    plan: InferencePlan,
    *,
    mcmc: MCMC,
    particles: Particles,
    laplace: Laplace,
    dates: Array | None = None,
    initial_parameters: Mapping[str, float] | None = None,
) -> FitResult:
    """Sample a shared-factor posterior through the common inference plan."""

    y = np.asarray(y, dtype=float)
    expected = (compiled.n_time, len(compiled.channel_names))
    if y.shape != expected:
        raise ValueError(f"y must have shape {expected}.")
    chains, draws = int(mcmc.chains), int(mcmc.draws)
    state_draws = np.zeros((chains, draws, compiled.n_time + 1, compiled.state_dim))
    log_posterior = np.zeros((chains, draws))
    parameter_names = [
        *(f"sd.{name}" for name in compiled.noise_names),
        *compiled.observation_parameter_names,
        *compiled.loading_names,
    ]
    parameter_draws = {
        name: np.zeros((chains, draws), dtype=float) for name in parameter_names
    }
    metric_names = (
        "laplace_iterations",
        "laplace_converged",
        "laplace_relative_change",
        "laplace_support_rejections",
        "particle_min_ess",
        "particle_mean_unique_ancestors",
        "particle_path_changed",
        "particle_changed_fraction",
    )
    draw_metrics = {name: np.full((chains, draws), np.nan) for name in metric_names}
    acceptance_names = [
        *(f"sd.{name}" for name in compiled.noise_names),
        *compiled.observation_parameter_names,
        *compiled.estimated_loading_names,
    ]
    if plan.asis:
        acceptance_names.extend(f"asis.sd.{name}" for name in compiled.noise_names)
    acceptance_by_chain = {name: [] for name in acceptance_names}
    final_steps = {name: [] for name in acceptance_names}
    recorded_initial: list[dict[str, float]] = []

    sequences = np.random.SeedSequence(mcmc.seed).spawn(chains)
    for chain, sequence in enumerate(sequences):
        rng = np.random.default_rng(sequence)
        params = _initial_parameters(compiled, priors, rng, initial_parameters)
        recorded_initial.append(dict(params))
        path = _initial_path(y, compiled, params, laplace, rng)
        steps = {f"sd.{name}": 0.20 for name in compiled.noise_names}
        if plan.asis:
            steps.update({f"asis.sd.{name}": 0.20 for name in compiled.noise_names})
        for channel in compiled.model.channels:
            steps[f"sigma.{channel.name}"] = 0.15
            if channel.family == "gev":
                steps[f"xi.{channel.name}"] = 0.18
        for key in compiled.estimated_loading_names:
            steps[key] = 0.10
        attempts = {name: 0 for name in acceptance_names}
        accepts = {name: 0 for name in acceptance_names}
        last_metrics = {name: np.nan for name in metric_names}
        saved = 0
        progress_every = max(1, mcmc.iterations // 20)

        for iteration in range(mcmc.iterations):
            if plan.engine == "ffbs":
                path = ffbs(y, compiled, params, rng)[0]
            elif plan.engine == "laplace":
                state = iterated_laplace(
                    y,
                    compiled,
                    params,
                    rng,
                    initial_path=path,
                    max_iterations=laplace.max_iterations,
                    tolerance=laplace.tolerance,
                    curvature_floor=laplace.curvature_floor,
                    maximum_variance=laplace.maximum_variance,
                    draw_attempts=laplace.draw_attempts,
                )
                path = state.path
                last_metrics.update(
                    laplace_iterations=state.iterations,
                    laplace_converged=float(state.converged),
                    laplace_relative_change=state.relative_change,
                    laplace_support_rejections=state.support_rejections,
                )
            elif plan.engine == "pgas":
                state = pgas(y, compiled, params, path, particles=particles, rng=rng)
                path = state.path
                last_metrics.update(
                    particle_min_ess=float(np.min(state.ess[1:])),
                    particle_mean_unique_ancestors=float(np.mean(state.unique_ancestors[1:])),
                    particle_path_changed=float(state.path_changed),
                    particle_changed_fraction=state.changed_fraction,
                )
            else:
                raise RuntimeError(f"Unhandled factor engine '{plan.engine}'.")

            adapting = bool(mcmc.adapt and iteration < mcmc.warmup)
            if plan.parameterization == "centered":
                outcomes, steps = _centered_scale_sweep(
                    path,
                    compiled,
                    params,
                    priors,
                    steps,
                    rng,
                    adapt=adapting,
                    iteration=iteration,
                )
                for key, outcome in outcomes.items():
                    attempts[key] += 1
                    accepts[key] += int(outcome)
                if plan.asis:
                    path, outcomes, steps = _noncentered_scale_sweep(
                        y,
                        path,
                        compiled,
                        params,
                        priors,
                        steps,
                        rng,
                        adapt=adapting,
                        iteration=iteration,
                        step_prefix="asis.",
                    )
                    for key, outcome in outcomes.items():
                        label = f"asis.{key}"
                        attempts[label] += 1
                        accepts[label] += int(outcome)
            elif plan.parameterization == "disturbance":
                path, outcomes, steps = _noncentered_scale_sweep(
                    y,
                    path,
                    compiled,
                    params,
                    priors,
                    steps,
                    rng,
                    adapt=adapting,
                    iteration=iteration,
                )
                for key, outcome in outcomes.items():
                    attempts[key] += 1
                    accepts[key] += int(outcome)
                if plan.asis:
                    outcomes, steps = _centered_scale_sweep(
                        path,
                        compiled,
                        params,
                        priors,
                        steps,
                        rng,
                        adapt=adapting,
                        iteration=iteration,
                        step_prefix="asis.",
                    )
                    for key, outcome in outcomes.items():
                        label = f"asis.{key}"
                        attempts[label] += 1
                        accepts[label] += int(outcome)
            else:
                raise RuntimeError(
                    "Factor models support centered or disturbance parameterization."
                )

            outcomes, steps = _observation_sweep(
                y,
                path,
                compiled,
                params,
                priors,
                steps,
                rng,
                adapt=adapting,
                iteration=iteration,
            )
            for key, outcome in outcomes.items():
                attempts[key] += 1
                accepts[key] += int(outcome)
            outcomes, steps = _loading_sweep(
                y,
                path,
                compiled,
                params,
                steps,
                rng,
                adapt=adapting,
                iteration=iteration,
            )
            for key, outcome in outcomes.items():
                attempts[key] += 1
                accepts[key] += int(outcome)

            keep = iteration >= mcmc.warmup and (
                (iteration - mcmc.warmup) % mcmc.thin == 0
            )
            if keep:
                state_draws[chain, saved] = path
                for name in parameter_names:
                    parameter_draws[name][chain, saved] = params[name]
                log_posterior[chain, saved] = _log_posterior(
                    y, path, compiled, params, priors
                )
                for name, value in last_metrics.items():
                    draw_metrics[name][chain, saved] = value
                saved += 1

            if mcmc.progress and (
                (iteration + 1) % progress_every == 0
                or iteration + 1 == mcmc.iterations
            ):
                print(
                    f"[factor chain {chain + 1}/{chains} iteration "
                    f"{iteration + 1}/{mcmc.iterations}]"
                )

        for name in acceptance_names:
            acceptance_by_chain[name].append(
                float(accepts[name] / attempts[name]) if attempts[name] else np.nan
            )
            final_steps[name].append(float(steps[name]))

    diagnostics: dict[str, Any] = {
        "acceptance": {
            name: np.asarray(values) for name, values in acceptance_by_chain.items()
        },
        "final_proposal_steps": {
            name: np.asarray(values) for name, values in final_steps.items()
        },
        "draw_metrics": draw_metrics,
        "mcmc": asdict(mcmc),
        "particles": asdict(particles),
        "laplace": asdict(laplace),
        "plan_warnings": list(plan.warnings),
    }
    return FitResult(
        model=compiled.model,
        compiled=compiled,
        priors=priors,
        y=y,
        exog=compiled.exog,
        dates=None if dates is None else np.asarray(dates),
        series_name=compiled.model.name,
        state_draws=state_draws,
        parameter_draws=parameter_draws,
        log_posterior=log_posterior,
        plan=plan,
        sampler_diagnostics=diagnostics,
        transform_sign=compiled.model.transform_signs,
        schema_version="2.0",
        initial_values={
            "parameters_by_chain": recorded_initial,
            "state_mean": compiled.initial_mean.tolist(),
            "state_sd": np.sqrt(np.diag(compiled.initial_cov)).tolist(),
        },
        metadata={
            "joint_model": True,
            "joint_likelihood": True,
            "conditional_channel_independence": True,
            "channel_names": list(compiled.channel_names),
            "factor_names": list(compiled.factor_names),
            "loading_identification": "fixed non-zero anchor per estimated factor",
        },
    )


__all__ = ["sample_factor_posterior"]
