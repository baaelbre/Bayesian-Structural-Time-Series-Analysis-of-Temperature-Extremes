"""Bootstrap/guided particle filtering and conditional SMC with ancestor sampling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...models.compiler import CompiledModel
from ..config import Particles
from ...core.numerics import (
    covariance_root,
    effective_sample_size,
    factored_singular_normal_logpdf,
    gaussian_support,
    normalize_logweights,
    singular_normal_logpdf,
    systematic_resample,
)


Array = np.ndarray


@dataclass
class ParticleFilterResult:
    log_likelihood: float
    particles: Array
    weights: Array
    ancestors: Array
    ess: Array
    resampled: Array
    unique_ancestors: Array


@dataclass
class PGASResult:
    path: Array
    ess: Array
    unique_ancestors: Array
    reference_ancestors: Array
    path_changed: bool
    changed_fraction: float
    exact_invariant: bool = True


def _resample(weights: Array, rng: np.random.Generator, size: int, method: str) -> Array:
    if method == "systematic":
        return systematic_resample(weights, rng, size=size)
    return rng.choice(weights.size, size=size, replace=True, p=weights)


def _observation_logweights(
    y_t: float,
    particles: Array,
    design_t: Array,
    compiled: CompiledModel,
    params: dict[str, float],
) -> Array:
    if not np.isfinite(y_t):
        return np.zeros(particles.shape[0])
    eta = particles @ design_t
    values = compiled.model.observation.logpdf(
        y_t,
        eta,
        sigma=float(params["sigma"]),
        xi=params.get("xi"),
    )
    values = np.asarray(values, dtype=float).reshape(-1)
    values[~np.isfinite(values)] = -np.inf
    return values


def _transition_loading(compiled: CompiledModel, params: dict[str, float]) -> Array:
    """Map independent standard-normal disturbances to state innovations."""

    return compiled.loading * compiled.process_vector(params)[None, :]


def _guided_statistics(
    y_t: float,
    means: Array,
    design_t: Array,
    transition_loading: Array,
    compiled: CompiledModel,
    params: dict[str, float],
) -> tuple[Array, Array, Array, Array, float]:
    """Locally Gaussian proposal in structural-disturbance coordinates.

    The likelihood is expanded at ``eta=y_t``.  This point is always inside the
    GEV support and has negative curvature for the release shape bounds.  The
    proposal is only a variance-reduction device: exact prior/proposal density
    ratios are retained in the particle weights.
    """

    means = np.asarray(means, dtype=float)
    n_particles = means.shape[0]
    noise_dim = transition_loading.shape[1]
    identity = np.eye(noise_dim)
    if not np.isfinite(y_t) or noise_dim == 0:
        return (
            np.zeros((n_particles, noise_dim)),
            identity,
            identity,
            identity,
            0.0,
        )
    loading = transition_loading.T @ design_t
    if np.linalg.norm(loading) <= np.finfo(float).tiny:
        return (
            np.zeros((n_particles, noise_dim)),
            identity,
            identity,
            identity,
            0.0,
        )
    observation = compiled.model.observation
    gradient = float(
        np.asarray(
            observation.grad_eta(
                y_t,
                y_t,
                sigma=float(params["sigma"]),
                xi=params.get("xi"),
            )
        )
    )
    hessian = float(
        np.asarray(
            observation.hess_eta(
                y_t,
                y_t,
                sigma=float(params["sigma"]),
                xi=params.get("xi"),
            )
        )
    )
    if not np.isfinite(gradient) or not np.isfinite(hessian):
        return (
            np.zeros((n_particles, noise_dim)),
            identity,
            identity,
            identity,
            0.0,
        )
    information = max(-hessian, 1e-10)
    precision = identity + information * np.outer(loading, loading)
    covariance = np.linalg.inv(precision)
    root = covariance_root(covariance)
    logdet = float(np.linalg.slogdet(covariance)[1])
    predicted_eta = means @ design_t
    coefficient = gradient + information * (float(y_t) - predicted_eta)
    direction = covariance @ loading
    proposal_mean = coefficient[:, None] * direction[None, :]
    return proposal_mean, covariance, root, precision, logdet


def _proposal_draw(
    y_t: float,
    means: Array,
    design_t: Array,
    transition_loading: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    particles: Particles,
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    n_particles = means.shape[0]
    noise_dim = transition_loading.shape[1]
    if particles.proposal == "bootstrap":
        disturbance = rng.normal(size=(n_particles, noise_dim))
        return means + disturbance @ transition_loading.T, np.zeros(n_particles)
    proposal_mean, _, root, precision, logdet = _guided_statistics(
        y_t,
        means,
        design_t,
        transition_loading,
        compiled,
        params,
    )
    centered = rng.normal(size=(n_particles, noise_dim)) @ root.T
    disturbance = proposal_mean + centered
    prior_quadratic = np.sum(disturbance**2, axis=1)
    proposal_quadratic = np.einsum(
        "ni,ij,nj->n", centered, precision, centered
    )
    log_prior_over_proposal = 0.5 * (
        logdet + proposal_quadratic - prior_quadratic
    )
    return means + disturbance @ transition_loading.T, log_prior_over_proposal


def _reference_proposal_correction(
    y_t: float,
    reference_t: Array,
    parent: Array,
    design_t: Array,
    transition_loading: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    particles: Particles,
) -> float:
    if particles.proposal == "bootstrap" or transition_loading.shape[1] == 0:
        return 0.0
    mean = compiled.transition @ parent
    residual = np.asarray(reference_t) - mean
    disturbance, *_ = np.linalg.lstsq(transition_loading, residual, rcond=None)
    if not np.allclose(
        transition_loading @ disturbance,
        residual,
        atol=1e-8 * (1.0 + np.linalg.norm(residual)),
        rtol=1e-8,
    ):
        return -np.inf
    proposal_mean, _, _, precision, logdet = _guided_statistics(
        y_t,
        mean[None, :],
        design_t,
        transition_loading,
        compiled,
        params,
    )
    centered = disturbance - proposal_mean[0]
    prior_quadratic = float(disturbance @ disturbance)
    proposal_quadratic = float(centered @ precision @ centered)
    return 0.5 * (logdet + proposal_quadratic - prior_quadratic)


def particle_filter(
    y: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    *,
    particles: Particles = Particles(),
    rng: np.random.Generator | None = None,
) -> ParticleFilterResult:
    """Particle filter with an unbiased likelihood estimate.

    The guided proposal is locally Gaussian in structural-disturbance
    coordinates and retains the exact prior/proposal density correction.
    ``proposal='bootstrap'`` recovers the ordinary bootstrap filter.
    """

    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size != compiled.n_time:
        raise ValueError("y and compiled model have different lengths.")
    n_time = y.size
    n_particles = int(particles.n)
    state_dim = compiled.state_dim
    design = compiled.design()
    transition_loading = _transition_loading(compiled, params)
    initial_root = covariance_root(compiled.initial_cov)

    cloud = np.zeros((n_time + 1, n_particles, state_dim))
    weights = np.zeros((n_time + 1, n_particles))
    normalized_log_weights = np.full(
        (n_time + 1, n_particles), -np.inf, dtype=float
    )
    ancestors = np.zeros((n_time + 1, n_particles), dtype=int)
    ess = np.zeros(n_time + 1)
    resampled = np.zeros(n_time + 1, dtype=bool)
    unique = np.zeros(n_time + 1, dtype=int)
    cloud[0] = compiled.initial_mean[None, :] + rng.normal(
        size=(n_particles, state_dim)
    ) @ initial_root.T
    weights[0] = 1.0 / n_particles
    normalized_log_weights[0] = -np.log(n_particles)
    ancestors[0] = np.arange(n_particles)
    ess[0] = n_particles
    unique[0] = n_particles
    log_likelihood = 0.0

    for t in range(1, n_time + 1):
        previous_weights = weights[t - 1]
        do_resample = effective_sample_size(previous_weights) < particles.ess_threshold * n_particles
        if do_resample:
            parent = _resample(previous_weights, rng, n_particles, particles.resampling)
            base_log_weights = np.full(n_particles, -np.log(n_particles))
            resampled[t] = True
        else:
            parent = np.arange(n_particles)
            base_log_weights = normalized_log_weights[t - 1].copy()
        ancestors[t] = parent
        unique[t] = np.unique(parent).size
        means = cloud[t - 1, parent] @ compiled.transition.T
        cloud[t], correction = _proposal_draw(
            y[t - 1],
            means,
            design[t - 1],
            transition_loading,
            compiled,
            params,
            particles,
            rng,
        )
        log_weights = base_log_weights + _observation_logweights(
            y[t - 1], cloud[t], design[t - 1], compiled, params
        ) + correction
        normalized, normalizer = normalize_logweights(log_weights)
        weights[t] = normalized
        normalized_log_weights[t] = log_weights - normalizer
        log_likelihood += normalizer
        ess[t] = effective_sample_size(normalized)

    return ParticleFilterResult(
        log_likelihood=float(log_likelihood),
        particles=cloud,
        weights=weights,
        ancestors=ancestors,
        ess=ess,
        resampled=resampled,
        unique_ancestors=unique,
    )


def _validate_reference(
    reference: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    transition_factor=None,
) -> None:
    if reference.shape != (compiled.n_time + 1, compiled.state_dim):
        raise ValueError("reference path has the wrong shape.")
    covariance = compiled.transition_cov(params)
    transition_factor = gaussian_support(covariance) if transition_factor is None else transition_factor
    if not np.isfinite(
        singular_normal_logpdf(reference[0], compiled.initial_mean, compiled.initial_cov)
    ):
        raise ValueError("reference x0 is outside the initial-state support.")
    for t in range(1, reference.shape[0]):
        value = factored_singular_normal_logpdf(
            reference[t], compiled.transition @ reference[t - 1], transition_factor
        )
        if not np.isfinite(value):
            raise ValueError(f"reference path is outside transition support at time {t}.")


def pgas(
    y: Array,
    compiled: CompiledModel,
    params: dict[str, float],
    reference: Array,
    *,
    particles: Particles = Particles(),
    rng: np.random.Generator | None = None,
) -> PGASResult:
    """One particle Gibbs with ancestor sampling state update.

    Multinomial conditional resampling is used inside the kernel, independently
    of the ordinary filter's configured resampling method. This is the simple
    conditional-SMC construction with a transparent invariant-target claim.
    """

    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float)
    if y.size != compiled.n_time:
        raise ValueError("y and compiled model have different lengths.")
    n_time = y.size
    n_particles = int(particles.n)
    conditioned = n_particles - 1
    state_dim = compiled.state_dim
    design = compiled.design()
    covariance = compiled.transition_cov(params)
    transition_loading = _transition_loading(compiled, params)
    initial_root = covariance_root(compiled.initial_cov)
    transition_factor = gaussian_support(covariance)
    _validate_reference(reference, compiled, params, transition_factor)

    cloud = np.zeros((n_time + 1, n_particles, state_dim))
    weights = np.zeros((n_time + 1, n_particles))
    normalized_log_weights = np.full(
        (n_time + 1, n_particles), -np.inf, dtype=float
    )
    ancestors = np.zeros((n_time + 1, n_particles), dtype=int)
    ess = np.zeros(n_time + 1)
    unique = np.zeros(n_time + 1, dtype=int)
    reference_ancestors = np.zeros(n_time + 1, dtype=int)

    cloud[0, :conditioned] = compiled.initial_mean[None, :] + rng.normal(
        size=(conditioned, state_dim)
    ) @ initial_root.T
    cloud[0, conditioned] = reference[0]
    weights[0] = 1.0 / n_particles
    normalized_log_weights[0] = -np.log(n_particles)
    ancestors[0] = np.arange(n_particles)
    ess[0] = n_particles
    unique[0] = n_particles
    reference_ancestors[0] = conditioned

    for t in range(1, n_time + 1):
        previous_weights = weights[t - 1]
        parent = rng.choice(n_particles, size=conditioned, replace=True, p=previous_weights)
        ancestors[t, :conditioned] = parent
        means = cloud[t - 1, parent] @ compiled.transition.T
        cloud[t, :conditioned], correction = _proposal_draw(
            y[t - 1],
            means,
            design[t - 1],
            transition_loading,
            compiled,
            params,
            particles,
            rng,
        )

        log_previous = normalized_log_weights[t - 1]
        previous_means = cloud[t - 1] @ compiled.transition.T
        log_as = log_previous + factored_singular_normal_logpdf(
            np.broadcast_to(reference[t], previous_means.shape),
            previous_means,
            transition_factor,
        )
        ancestor_weights, _ = normalize_logweights(log_as)
        reference_parent = int(rng.choice(n_particles, p=ancestor_weights))
        ancestors[t, conditioned] = reference_parent
        reference_ancestors[t] = reference_parent
        cloud[t, conditioned] = reference[t]
        reference_correction = _reference_proposal_correction(
            y[t - 1],
            reference[t],
            cloud[t - 1, reference_parent],
            design[t - 1],
            transition_loading,
            compiled,
            params,
            particles,
        )
        unique[t] = np.unique(ancestors[t]).size

        log_weights = _observation_logweights(
            y[t - 1], cloud[t], design[t - 1], compiled, params
        )
        log_weights[:conditioned] += correction
        log_weights[conditioned] += reference_correction
        weights[t], log_normalizer = normalize_logweights(log_weights)
        normalized_log_weights[t] = log_weights - log_normalizer
        ess[t] = effective_sample_size(weights[t])

    index_path = np.zeros(n_time + 1, dtype=int)
    index_path[n_time] = int(rng.choice(n_particles, p=weights[n_time]))
    path = np.zeros_like(reference)
    path[n_time] = cloud[n_time, index_path[n_time]]
    for t in range(n_time, 0, -1):
        index_path[t - 1] = ancestors[t, index_path[t]]
        path[t - 1] = cloud[t - 1, index_path[t - 1]]

    difference = np.linalg.norm(path - reference, axis=1)
    changed = difference > 1e-10 * (1.0 + np.linalg.norm(reference, axis=1))
    return PGASResult(
        path=path,
        ess=ess,
        unique_ancestors=unique,
        reference_ancestors=reference_ancestors,
        path_changed=bool(np.any(changed)),
        changed_fraction=float(np.mean(changed)),
    )
