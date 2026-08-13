"""Unified multi-chain MCMC sampler for all accepted first-release models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .compiler import CompiledModel
from .kalman import ffbs
from .laplace import (
    iterated_laplace,
    joint_state_log_density,
    observation_log_likelihood,
)
from .particle import Particles, pgas
from .plan import InferencePlan
from .priors import FixedSD, Priors, SpikeSlabSD
from .results import FitResult


Array = np.ndarray


@dataclass(frozen=True)
class MCMC:
    draws: int = 1000
    warmup: int = 1000
    thin: int = 1
    chains: int = 4
    seed: int | None = None
    progress: bool = False
    adapt: bool = True

    def __post_init__(self) -> None:
        if int(self.draws) < 1 or int(self.warmup) < 0 or int(self.thin) < 1:
            raise ValueError("draws and thin must be positive; warmup must be non-negative.")
        if int(self.chains) < 1:
            raise ValueError("chains must be positive.")

    @property
    def iterations(self) -> int:
        # The first retained draw is the sweep at index ``warmup``; subsequent
        # draws are separated by exactly ``thin`` sweeps.
        return int(self.warmup) + 1 + (int(self.draws) - 1) * int(self.thin)


@dataclass(frozen=True)
class GibbsConfig:
    """Compatibility configuration used by the 0.3 research scripts.

    New code should use :class:`MCMC`; ``to_mcmc`` makes the migration
    explicit and preserves the old ``n_iter``/``burn`` interpretation.
    """

    n_iter: int = 2000
    burn: int = 1000
    thin: int = 1
    seed: int | None = None
    progress: bool = False
    progress_every: int | None = None

    def __post_init__(self) -> None:
        if int(self.n_iter) <= int(self.burn) or int(self.burn) < 0:
            raise ValueError("Require n_iter > burn >= 0.")
        if int(self.thin) < 1:
            raise ValueError("thin must be positive.")

    def to_mcmc(self, *, chains: int = 1) -> MCMC:
        draws = len(range(int(self.burn), int(self.n_iter), int(self.thin)))
        return MCMC(
            draws=draws,
            warmup=int(self.burn),
            thin=int(self.thin),
            chains=int(chains),
            seed=self.seed,
            progress=bool(self.progress),
        )


@dataclass(frozen=True)
class Laplace:
    max_iterations: int = 30
    tolerance: float = 1e-5
    curvature_floor: float = 1e-6
    maximum_variance: float = 1e8
    draw_attempts: int = 30

    def __post_init__(self) -> None:
        if int(self.max_iterations) < 1 or float(self.tolerance) <= 0.0:
            raise ValueError("Laplace iterations and tolerance must be positive.")


def _prior_logpdf(prior, value: float, indicator: int | None = None) -> float:
    return float(prior.logpdf(value, indicator=indicator))


def _initial_parameters(
    compiled: CompiledModel,
    priors: Priors,
    rng: np.random.Generator,
    initial: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    params: dict[str, float] = {}
    indicators: dict[str, int] = {}
    for name in compiled.noise_names:
        prior = priors.process[name]
        if isinstance(prior, SpikeSlabSD):
            indicator = int(rng.binomial(1, prior.slab_probability))
            indicators[name] = indicator
            value = float(prior.sample(rng, indicator=indicator))
        else:
            value = float(prior.initial())
        if not isinstance(prior, FixedSD):
            value = max(value, compiled.y_scale * 1e-10)
        params[f"sd.{name}"] = value
    sigma = float(priors.observation_sd.initial())
    if not isinstance(priors.observation_sd, FixedSD):
        sigma = max(sigma, compiled.y_scale * 1e-6)
    params["sigma"] = sigma
    if compiled.family == "gev":
        assert priors.shape is not None
        params["xi"] = float(priors.shape.initial())
    if initial is not None:
        supplied = dict(initial)
        allowed = set(params) | {
            f"slab.{name}"
            for name in compiled.noise_names
            if isinstance(priors.process[name], SpikeSlabSD)
        }
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise ValueError(f"Unknown initial parameters: {unknown}")
        for name in params:
            if name in supplied:
                params[name] = float(supplied[name])
        for name in compiled.noise_names:
            slab_key = f"slab.{name}"
            if slab_key in supplied:
                indicator = int(supplied[slab_key])
                if indicator not in {0, 1}:
                    raise ValueError(f"{slab_key} must be 0 or 1.")
                indicators[name] = indicator

    for name in compiled.noise_names:
        value = float(params[f"sd.{name}"])
        prior = priors.process[name]
        indicator = indicators.get(name)
        if (
            value < 0.0
            or (not isinstance(prior, FixedSD) and value <= 0.0)
            or not np.isfinite(_prior_logpdf(prior, value, indicator))
        ):
            raise ValueError(f"Initial sd.{name} is outside its prior support.")
    if params["sigma"] <= 0.0 or not np.isfinite(_prior_logpdf(priors.observation_sd, params["sigma"])):
        raise ValueError("Initial sigma is outside its prior support.")
    if compiled.family == "gev":
        assert priors.shape is not None
        if not np.isfinite(priors.shape.logpdf(params["xi"])):
            raise ValueError("Initial xi is outside its prior support.")
    return params, indicators


def _adapt(step: float, accepted: bool, iteration: int, target: float = 0.30) -> float:
    gain = min(0.05, 1.0 / np.sqrt(float(iteration) + 1.0))
    updated = float(step) * np.exp(gain * (float(accepted) - target))
    return float(np.clip(updated, 0.01, 2.5))


def _centered_scale_sweep(
    path: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    priors: Priors,
    indicators: dict[str, int],
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    adapt: bool,
    iteration: int,
    step_prefix: str = "",
) -> tuple[dict[str, bool], dict[str, float]]:
    noncentered = compiled.to_noncentered(path, params)
    current_sd = compiled.process_vector(params)
    innovations = noncentered.z * current_sd[None, :]
    accepted: dict[str, bool] = {}
    for j, name in enumerate(compiled.noise_names):
        prior = priors.process[name]
        key = f"sd.{name}"
        step_key = f"{step_prefix}{key}"
        if isinstance(prior, FixedSD):
            accepted[key] = False
            continue
        current = float(params[key])
        proposal_log = float(np.log(current) + steps[step_key] * rng.normal())
        proposal = float(np.exp(proposal_log))
        indicator = indicators.get(name)
        sum_squares = float(np.sum(innovations[:, j] ** 2))
        n_time = innovations.shape[0]

        def target(value: float) -> float:
            if value <= 0.0:
                return -np.inf
            transition = -n_time * np.log(value) - 0.5 * sum_squares / value**2
            return float(transition + _prior_logpdf(prior, value, indicator) + np.log(value))

        log_ratio = target(proposal) - target(current)
        take = bool(np.log(rng.random()) < log_ratio)
        if take:
            params[key] = proposal
        accepted[key] = take
        if adapt:
            steps[step_key] = _adapt(steps[step_key], take, iteration)
    return accepted, steps


def _noncentered_scale_sweep(
    y: Array,
    path: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    priors: Priors,
    indicators: dict[str, int],
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    adapt: bool,
    iteration: int,
) -> tuple[Array, dict[str, bool], dict[str, float]]:
    noncentered = compiled.to_noncentered(path, params)
    current_path = path
    current_likelihood = observation_log_likelihood(
        y, compiled.eta(current_path), compiled, params
    )
    accepted: dict[str, bool] = {}
    for name in compiled.noise_names:
        prior = priors.process[name]
        key = f"sd.{name}"
        if isinstance(prior, FixedSD):
            accepted[key] = False
            continue
        current = float(params[key])
        proposal_log = float(np.log(current) + steps[key] * rng.normal())
        proposal = float(np.exp(proposal_log))
        proposal_params = dict(params)
        proposal_params[key] = proposal
        proposal_path = compiled.from_noncentered(noncentered, proposal_params)
        proposal_likelihood = observation_log_likelihood(
            y, compiled.eta(proposal_path), compiled, proposal_params
        )
        indicator = indicators.get(name)
        current_target = current_likelihood + _prior_logpdf(prior, current, indicator) + np.log(current)
        proposal_target = proposal_likelihood + _prior_logpdf(prior, proposal, indicator) + np.log(proposal)
        take = bool(np.log(rng.random()) < proposal_target - current_target)
        if take:
            params[key] = proposal
            current_path = proposal_path
            current_likelihood = proposal_likelihood
        accepted[key] = take
        if adapt:
            steps[key] = _adapt(steps[key], take, iteration)
    return current_path, accepted, steps


def _observation_parameter_sweep(
    y: Array,
    path: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    priors: Priors,
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    adapt: bool,
    iteration: int,
) -> tuple[dict[str, bool], dict[str, float]]:
    eta = compiled.eta(path)
    accepted: dict[str, bool] = {}

    sigma_prior = priors.observation_sd
    if isinstance(sigma_prior, FixedSD):
        accepted["sigma"] = False
    else:
        current = float(params["sigma"])
        proposal_log = float(np.log(current) + steps["sigma"] * rng.normal())
        proposal = float(np.exp(proposal_log))

        def target(value: float) -> float:
            proposal_params = dict(params)
            proposal_params["sigma"] = value
            return (
                observation_log_likelihood(y, eta, compiled, proposal_params)
                + _prior_logpdf(sigma_prior, value)
                + np.log(value)
            )

        take = bool(np.log(rng.random()) < target(proposal) - target(current))
        if take:
            params["sigma"] = proposal
        accepted["sigma"] = take
        if adapt:
            steps["sigma"] = _adapt(steps["sigma"], take, iteration)

    if compiled.family == "gev":
        assert priors.shape is not None
        lower = float(priors.shape.lower)
        upper = float(priors.shape.upper)
        from .numerics import bounded_log_jacobian, bounded_to_real, real_to_bounded

        current_xi = float(params["xi"])
        current_real = bounded_to_real(current_xi, lower, upper)
        proposal_real = float(current_real + steps["xi"] * rng.normal())
        proposal_xi = real_to_bounded(proposal_real, lower, upper)

        def target(real_value: float, xi_value: float) -> float:
            proposal_params = dict(params)
            proposal_params["xi"] = xi_value
            return (
                observation_log_likelihood(y, eta, compiled, proposal_params)
                + priors.shape.logpdf(xi_value)
                + bounded_log_jacobian(real_value, lower, upper)
            )

        take = bool(
            np.log(rng.random())
            < target(proposal_real, proposal_xi) - target(current_real, current_xi)
        )
        if take:
            params["xi"] = proposal_xi
        accepted["xi"] = take
        if adapt:
            steps["xi"] = _adapt(steps["xi"], take, iteration)
    return accepted, steps


def _update_indicators(
    compiled: CompiledModel,
    params: dict[str, float],
    priors: Priors,
    indicators: dict[str, int],
    rng: np.random.Generator,
) -> None:
    for name in compiled.noise_names:
        prior = priors.process[name]
        if isinstance(prior, SpikeSlabSD):
            probability = prior.indicator_probability(float(params[f"sd.{name}"]))
            indicators[name] = int(rng.binomial(1, probability))


def _log_posterior(
    y: Array,
    path: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    priors: Priors,
    indicators: dict[str, int],
) -> float:
    value = joint_state_log_density(y, path, compiled, params)
    for name in compiled.noise_names:
        prior = priors.process[name]
        indicator = indicators.get(name)
        value += _prior_logpdf(prior, float(params[f"sd.{name}"]), indicator)
        if isinstance(prior, SpikeSlabSD):
            value += np.log(prior.slab_probability if indicator == 1 else 1.0 - prior.slab_probability)
    value += _prior_logpdf(priors.observation_sd, float(params["sigma"]))
    if compiled.family == "gev":
        assert priors.shape is not None
        value += priors.shape.logpdf(float(params["xi"]))
    return float(value)


def _initial_path(
    y: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    plan: InferencePlan,
    laplace: Laplace,
    rng: np.random.Generator,
) -> Array:
    if compiled.family == "gaussian":
        return ffbs(y, compiled, params, rng)[0]
    initial = iterated_laplace(
        y,
        compiled,
        params,
        rng,
        max_iterations=laplace.max_iterations,
        tolerance=laplace.tolerance,
        curvature_floor=laplace.curvature_floor,
        maximum_variance=laplace.maximum_variance,
        draw_attempts=laplace.draw_attempts,
    )
    return initial.path


def sample_posterior(
    y: Array,
    compiled: CompiledModel,
    priors: Priors,
    plan: InferencePlan,
    *,
    mcmc: MCMC,
    particles: Particles,
    laplace: Laplace,
    dates: Array | None = None,
    series_name: str | None = None,
    transform_sign: float = 1.0,
    initial_parameters: dict[str, float] | None = None,
) -> FitResult:
    y = np.asarray(y, dtype=float).reshape(-1)
    chains = int(mcmc.chains)
    draws = int(mcmc.draws)
    state_draws = np.zeros((chains, draws, y.size + 1, compiled.state_dim))
    log_posterior = np.zeros((chains, draws))
    parameter_names = [f"sd.{name}" for name in compiled.noise_names] + ["sigma"]
    if compiled.family == "gev":
        parameter_names.append("xi")
    slab_names = [
        f"slab.{name}"
        for name in compiled.noise_names
        if isinstance(priors.process[name], SpikeSlabSD)
    ]
    parameter_draws = {
        name: np.zeros((chains, draws), dtype=np.int8 if name.startswith("slab.") else float)
        for name in parameter_names + slab_names
    }
    metric_names = [
        "laplace_iterations",
        "laplace_converged",
        "laplace_relative_change",
        "laplace_support_rejections",
        "particle_min_ess",
        "particle_mean_unique_ancestors",
        "particle_path_changed",
        "particle_changed_fraction",
    ]
    draw_metrics = {name: np.full((chains, draws), np.nan) for name in metric_names}
    acceptance_by_chain: dict[str, list[float]] = {
        name: [] for name in parameter_names
    }
    if plan.asis:
        for name in compiled.noise_names:
            acceptance_by_chain[f"asis.sd.{name}"] = []
    final_step_names = list(parameter_names) + (
        [f"asis.sd.{name}" for name in compiled.noise_names] if plan.asis else []
    )
    final_steps: dict[str, list[float]] = {name: [] for name in final_step_names}

    seed_sequence = np.random.SeedSequence(mcmc.seed)
    chain_sequences = seed_sequence.spawn(chains)
    total_iterations = mcmc.iterations
    recorded_initial_parameters: list[dict[str, float]] = []
    recorded_initial_indicators: list[dict[str, int]] = []

    for chain, chain_sequence in enumerate(chain_sequences):
        rng = np.random.default_rng(chain_sequence)
        params, indicators = _initial_parameters(
            compiled, priors, rng, initial=initial_parameters
        )
        recorded_initial_parameters.append(dict(params))
        recorded_initial_indicators.append(dict(indicators))
        path = _initial_path(y, compiled, params, plan, laplace, rng)
        steps = {f"sd.{name}": 0.20 for name in compiled.noise_names}
        if plan.asis:
            steps.update({f"asis.sd.{name}": 0.20 for name in compiled.noise_names})
        steps["sigma"] = 0.15
        if compiled.family == "gev":
            steps["xi"] = 0.18
        attempts = {name: 0 for name in acceptance_by_chain}
        accepts = {name: 0 for name in acceptance_by_chain}
        saved = 0
        last_metrics = {name: np.nan for name in metric_names}
        progress_every = max(1, total_iterations // 20)

        for iteration in range(total_iterations):
            # State update.
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
                raise RuntimeError(f"Unhandled engine '{plan.engine}'.")

            adapting = bool(mcmc.adapt and iteration < mcmc.warmup)
            # Scale update in the requested parameterization, with optional ASIS.
            if plan.asis or plan.parameterization == "centered":
                outcomes, steps = _centered_scale_sweep(
                    path,
                    compiled,
                    params,
                    priors,
                    indicators,
                    steps,
                    rng,
                    adapt=adapting,
                    iteration=iteration,
                    step_prefix="asis." if plan.asis else "",
                )
                for key, outcome in outcomes.items():
                    label = key if plan.parameterization == "centered" and not plan.asis else f"asis.{key}"
                    attempts[label] += 1
                    accepts[label] += int(outcome)

            if plan.asis or plan.parameterization == "noncentered":
                path, outcomes, steps = _noncentered_scale_sweep(
                    y,
                    path,
                    compiled,
                    params,
                    priors,
                    indicators,
                    steps,
                    rng,
                    adapt=adapting,
                    iteration=iteration,
                )
                for key, outcome in outcomes.items():
                    attempts[key] += 1
                    accepts[key] += int(outcome)

            outcomes, steps = _observation_parameter_sweep(
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
            _update_indicators(compiled, params, priors, indicators, rng)

            keep = iteration >= mcmc.warmup and (
                (iteration - mcmc.warmup) % mcmc.thin == 0
            )
            if keep:
                state_draws[chain, saved] = path
                for name in parameter_names:
                    parameter_draws[name][chain, saved] = params[name]
                for name in compiled.noise_names:
                    slab_key = f"slab.{name}"
                    if slab_key in parameter_draws:
                        parameter_draws[slab_key][chain, saved] = indicators[name]
                log_posterior[chain, saved] = _log_posterior(
                    y, path, compiled, params, priors, indicators
                )
                for name, value in last_metrics.items():
                    draw_metrics[name][chain, saved] = value
                saved += 1

            if mcmc.progress and (
                (iteration + 1) % progress_every == 0 or iteration + 1 == total_iterations
            ):
                message = (
                    f"[chain {chain + 1}/{chains} iteration {iteration + 1}/{total_iterations}] "
                    f"sigma={params['sigma']:.4g}"
                )
                if compiled.family == "gev":
                    message += f" xi={params['xi']:.3f}"
                print(message)

        for name in acceptance_by_chain:
            acceptance_by_chain[name].append(
                float(accepts[name] / attempts[name]) if attempts[name] else np.nan
            )
        for name in final_steps:
            final_steps[name].append(float(steps[name]))

    diagnostics: dict[str, Any] = {
        "acceptance": {name: np.asarray(values) for name, values in acceptance_by_chain.items()},
        "final_proposal_steps": {name: np.asarray(values) for name, values in final_steps.items()},
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
        series_name=series_name,
        state_draws=state_draws,
        parameter_draws=parameter_draws,
        log_posterior=log_posterior,
        plan=plan,
        sampler_diagnostics=diagnostics,
        transform_sign=float(transform_sign),
        initial_values={
            "parameters_by_chain": recorded_initial_parameters,
            "indicators_by_chain": recorded_initial_indicators,
            "state_mean": compiled.initial_mean.tolist(),
            "state_sd": np.sqrt(np.diag(compiled.initial_cov)).tolist(),
        },
    )
