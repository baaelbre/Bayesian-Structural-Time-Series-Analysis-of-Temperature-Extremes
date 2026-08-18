"""Joint hierarchical inference for structurally related time series.

The likelihood factorizes by channel, but the MCMC is one joint Gibbs sampler:
channel allocations update conditional on shared categorical probabilities and
shared dynamic-slab scales, followed by conjugate Dirichlet and exact slice
updates of those hyperparameters.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from ...__about__ import __version__
from ...core.fit import FitResult
from ...models.multiseries_compiler import CompiledMultiSeriesModel
from ...models.structural import Model
from ...priors.hierarchical import HierarchicalPriors
from ...priors.structural import FSGaussianPriors, FSGEVPriors
from ..config import HierarchicalSampler, Laplace, MCMC, Particles
from ..plan import InferencePlan
from ._progress import (
    compact_group,
    mcmc_progress_line,
    progress_interval,
    should_report_progress,
)
from .fs_gaussian import _sigma2_update_from_mu
from .fs_gev import FSGEVKernel, _exact_gev_loglik, _gev_support_ok, _laplace_pseudo_mu
from .fs_utils import (
    apply_theta_draw,
    baseline_mu_path,
    build_ncp_system,
    canonicalize_ncp_params,
    design_matrix_ncp,
    elliptical_slice_gaussian_prior,
    ffbs_gaussian_1d,
    infer_ncp_layout,
    iterated_laplace_ncp,
    map_centered_to_ncp,
    map_ncp_to_centered,
    measurement_vector,
    mu_from_ncp,
    ncp_pgas,
    random_sign_switches,
    theta_vector_from_params,
)
from .model_space import (
    ComponentState,
    StructuralModelState,
    TrendModelClass,
    initial_structural_state,
    sample_structural_regression,
    sample_structural_regression_exact,
    trend_model_class,
)


Array = np.ndarray
_COMPONENTS = ("level", "trend", "season")


@dataclass
class _ChannelState:
    name: str
    family: str
    y: Array
    model: Model
    compiled: Any
    layout: Any
    base_prior: FSGaussianPriors | FSGEVPriors
    params_state: dict[str, Any]
    params_obs: dict[str, float]
    z_path: Array
    model_state: StructuralModelState
    model_index: int
    gev_kernel: FSGEVKernel | None = None


def _seasonal_initial(y: Array, period: int | None) -> Array:
    if period is None:
        return np.zeros(0, dtype=float)
    phase = np.arange(y.size) % int(period)
    overall = float(np.mean(y))
    full = np.asarray(
        [float(np.mean(y[phase == index])) - overall for index in range(int(period))]
    )
    full -= float(np.mean(full))
    return full[:-1]


def _linear_initial(y: Array, gamma: Array, period: int | None) -> tuple[float, float]:
    """Data-informed chain start for the level and slope; both remain sampled."""

    values = np.asarray(y, dtype=float)
    if period is None:
        adjusted = values
    else:
        full = np.r_[np.asarray(gamma, dtype=float), -np.sum(gamma)]
        adjusted = values - full[np.arange(values.size) % int(period)]
    time = np.arange(values.size, dtype=float)
    centered = time - float(np.mean(time))
    denominator = float(centered @ centered)
    slope = 0.0 if denominator == 0.0 else float(centered @ adjusted) / denominator
    level = float(np.mean(adjusted) - slope * np.mean(time))
    return level, slope


def _initial_channel_state(
    name: str,
    family: str,
    y: Array,
    compiled: Any,
    prior: FSGaussianPriors | FSGEVPriors,
    rng: np.random.Generator,
    initial: Mapping[str, Any] | None,
) -> _ChannelState:
    model = compiled.model
    layout = infer_ncp_layout(model)
    gamma = _seasonal_initial(y, model.period)
    alpha0, beta0 = _linear_initial(y, gamma, model.period)
    fitted = alpha0 + beta0 * np.arange(y.size, dtype=float)
    if model.period is not None:
        full_season = np.r_[gamma, -np.sum(gamma)]
        fitted = fitted + full_season[np.arange(y.size) % int(model.period)]
    residual = y - fitted
    scale = max(float(np.std(residual, ddof=1)), 0.05)
    state = {
        "alpha0": alpha0,
        "beta0": beta0,
        "gamma0_season": gamma,
        "s_level": 0.01,
        "s_trend": 0.0001,
        "s_season": 0.01,
        "q_level": 0.01**2,
        "q_trend": 0.0001**2,
        "q_season": 0.01**2,
    }
    observation = {"sigma": scale, "sigma2": scale**2}
    if family == "gev":
        # Gumbel initialization has no finite support boundary.
        observation["xi"] = 0.0
    supplied = {} if initial is None else dict(initial)
    prefixes = (f"channel.{name}.", f"{name}.")
    local: dict[str, Any] = {}
    for key, value in supplied.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                local[key.removeprefix(prefix)] = value
                break
    centered_path = local.pop("__centered_path", None)
    aliases = {
        "intercept": "alpha0",
        "initial.level": "alpha0",
        "initial.slope": "beta0",
        "initial.seasonal": "gamma0_season",
        "initial_slope": "beta0",
        "sd.level": "s_level",
        "sd.slope": "s_trend",
        "sd.seasonal": "s_season",
    }
    for key, value in local.items():
        target = aliases.get(key, key)
        if target in state:
            state[target] = value
        elif target in observation:
            observation[target] = float(value)
        else:
            raise ValueError(f"Unknown initial parameter '{key}' for channel '{name}'.")
    state = canonicalize_ncp_params(state, layout)
    observation["sigma2"] = float(observation["sigma"]) ** 2
    model_state = initial_structural_state(state, layout)
    if centered_path is None:
        z_path = np.zeros((y.size + 1, layout.ncp_state_dim), dtype=float)
    else:
        centered = np.asarray(centered_path, dtype=float)
        expected = (y.size + 1, layout.centered_state_dim)
        if centered.shape != expected or not np.all(np.isfinite(centered)):
            raise ValueError(
                f"Warm-start path for channel '{name}' must have shape {expected} "
                "and contain only finite values."
            )
        z_path = map_centered_to_ncp(centered, state, layout)
    if family == "gev" and not _gev_support_ok(
        y, mu_from_ncp(z_path, state, layout), model, observation
    ):
        raise ValueError(
            f"Initial GEV values violate support for channel '{name}'. "
            "Use channel-prefixed init values closer to the data."
        )
    kernel = None
    if family == "gev":
        kernel = FSGEVKernel(model, prior, config=MCMC(draws=1, chains=1))
        kernel.rng = rng
    return _ChannelState(
        name=name,
        family=family,
        y=np.asarray(y, dtype=float),
        model=model,
        compiled=compiled,
        layout=layout,
        base_prior=prior,
        params_state=state,
        params_obs=observation,
        z_path=z_path,
        model_state=model_state,
        model_index=-1,
        gev_kernel=kernel,
    )


def _slice_log_scale(
    current: float,
    standardized_coefficients: Array,
    *,
    df: float,
    prior_scale: float,
    rng: np.random.Generator,
    width: float = 1.0,
) -> float:
    """Exact univariate slice update for a half-t scale on log scale."""

    values = np.asarray(standardized_coefficients, dtype=float).reshape(-1)
    if values.size == 0:
        return max(abs(float(prior_scale) * float(rng.standard_t(df))), 1e-12)

    sum_squares = float(values @ values)
    log_prior_denominator = float(
        np.log(float(df)) + 2.0 * np.log(float(prior_scale))
    )

    def log_density(log_tau: float) -> float:
        # Work on the log scale throughout.  Directly evaluating ``tau**2``
        # can overflow when a weak hierarchy explores a very large scale even
        # though the corresponding log density is perfectly representable.
        if not np.isfinite(log_tau) or abs(log_tau) > 350.0:
            return -np.inf
        if sum_squares > 0.0 and -2.0 * log_tau > 700.0:
            return -np.inf
        quadratic = sum_squares * float(np.exp(-2.0 * log_tau))
        log_likelihood = -values.size * log_tau - 0.5 * quadratic
        log_prior = -0.5 * (float(df) + 1.0) * np.logaddexp(
            0.0,
            2.0 * log_tau - log_prior_denominator,
        )
        return float(log_likelihood + log_prior + log_tau)

    x0 = float(np.log(max(float(current), 1e-12)))
    log_y = log_density(x0) + float(
        np.log(max(float(rng.random()), np.finfo(float).tiny))
    )
    left = x0 - width * float(rng.random())
    right = left + width
    for _ in range(50):
        if log_density(left) <= log_y:
            break
        left -= width
    for _ in range(50):
        if log_density(right) <= log_y:
            break
        right += width
    for _ in range(10_000):
        proposal = float(rng.uniform(left, right))
        if log_density(proposal) >= log_y:
            return max(float(np.exp(proposal)), 1e-12)
        if proposal < x0:
            left = proposal
        else:
            right = proposal
    raise RuntimeError("Hierarchical slab-scale slice sampler exceeded 10,000 steps.")


def _update_hierarchy(
    states: list[_ChannelState],
    priors: HierarchicalPriors,
    current_slab: Mapping[str, float],
    rng: np.random.Generator,
) -> tuple[dict[str, Array], dict[str, float]]:
    hierarchy = priors.hierarchy
    counts = {
        "level": np.zeros(2, dtype=float),
        "trend": np.zeros(3, dtype=float),
        "season": np.zeros(3, dtype=float),
    }
    model_counts = np.zeros(4, dtype=float)
    coefficients: dict[str, list[float]] = {name: [] for name in _COMPONENTS}
    for item in states:
        counts["level"][0 if item.model_state.level == ComponentState.FIXED else 1] += 1.0
        if item.layout.has_beta:
            counts["trend"][int(item.model_state.trend)] += 1.0
        if item.layout.season_dim > 0:
            counts["season"][int(item.model_state.season)] += 1.0
        if hierarchy.uses_joint_trend_space:
            model_counts[int(trend_model_class(item.model_state))] += 1.0
        if item.model_state.level == ComponentState.DYNAMIC:
            coefficients["level"].append(float(item.params_state["s_level"]))
        if item.model_state.trend == ComponentState.DYNAMIC:
            coefficients["trend"].append(float(item.params_state["s_trend"]))
        if item.model_state.season == ComponentState.DYNAMIC:
            coefficients["season"].append(float(item.params_state["s_season"]))
    probabilities: dict[str, Array] = {}
    if hierarchy.uses_joint_trend_space:
        model_probabilities = hierarchy.initial_model_probabilities()
        if hierarchy.pools_selection:
            model_probabilities = rng.dirichlet(
                np.asarray(hierarchy.trend_model_concentration, dtype=float)
                + model_counts
            )
        probabilities["trend_model"] = model_probabilities
        probabilities["level"] = np.asarray(
            (
                model_probabilities[0] + model_probabilities[2],
                model_probabilities[1] + model_probabilities[3],
            )
        )
        probabilities["trend"] = np.asarray(
            (
                0.0,
                model_probabilities[0] + model_probabilities[1],
                model_probabilities[2] + model_probabilities[3],
            )
        )
        probabilities["season"] = hierarchy.initial_probabilities("season")
        if hierarchy.pools_selection:
            indices = hierarchy.allowed_indices("season")
            probabilities["season"][indices] = rng.dirichlet(
                hierarchy.concentration("season") + counts["season"][indices]
            )
    else:
        for name in _COMPONENTS:
            probabilities[name] = hierarchy.initial_probabilities(name)
            if hierarchy.pools_selection:
                indices = hierarchy.allowed_indices(name)
                probabilities[name][indices] = rng.dirichlet(
                    hierarchy.concentration(name) + counts[name][indices]
                )
    slab = {}
    for name in _COMPONENTS:
        current = float(current_slab[name])
        if hierarchy.pools_slab:
            standardized = np.asarray(coefficients[name], dtype=float) / float(
                hierarchy.coefficient_scale[name]
            )
            slab[name] = _slice_log_scale(
                current,
                standardized,
                df=float(hierarchy.slab_df),
                prior_scale=float(hierarchy.slab_prior_scale[name]),
                rng=rng,
            )
        else:
            slab[name] = float(hierarchy.initial_slab_scale[name])
    return probabilities, slab


def _current_prior(
    state: _ChannelState,
    resolved: HierarchicalPriors,
    probabilities: Mapping[str, Array],
    slab: Mapping[str, float],
):
    ssvs = resolved.hierarchy.component_ssvs(probabilities, slab)
    return replace(state.base_prior, ssvs=ssvs)


def _sign_move(state: _ChannelState, rng: np.random.Generator):
    before = mu_from_ncp(state.z_path, state.params_state, state.layout)
    z_path, params_state, switches = random_sign_switches(
        state.z_path,
        state.params_state,
        state.layout,
        rng,
        return_switches=True,
    )
    after = mu_from_ncp(z_path, params_state, state.layout)
    error = float(np.max(np.abs(before - after)))
    tolerance = 1e-10 * (1.0 + float(np.max(np.abs(before))))
    if error > tolerance:
        raise RuntimeError(
            f"FS sign switch changed the predictor for channel '{state.name}' "
            f"by {error:.3g}."
        )
    state.z_path = z_path
    state.params_state = params_state
    return switches, error


def _gaussian_step(
    state: _ChannelState,
    prior: FSGaussianPriors,
    rng: np.random.Generator,
) -> tuple[float, dict[str, float]]:
    layout = state.layout
    G, Q = build_ncp_system(layout)
    offset = baseline_mu_path(state.y.size, state.params_state, layout)
    H = measurement_vector(state.params_state, layout)
    state.z_path = ffbs_gaussian_1d(
        y=state.y - offset,
        G=G,
        Q=Q,
        H=H,
        R=float(state.params_obs["sigma2"]),
        m0=np.zeros(layout.ncp_state_dim),
        C0=np.zeros((layout.ncp_state_dim, layout.ncp_state_dim)),
        rng=rng,
    )
    X, theta_names, tbar = design_matrix_ncp(state.z_path, layout, center_time=True)
    selection = sample_structural_regression(
        y=state.y,
        X=X,
        theta_names=theta_names,
        tbar=tbar,
        noise_variance=float(state.params_obs["sigma2"]),
        priors=prior,
        layout=layout,
        rng=rng,
        apply_theta_draw=apply_theta_draw,
    )
    state.params_state.update(selection.params_state)
    state.model_state = selection.state
    state.model_index = int(selection.selected_index)
    switches, sign_error = _sign_move(state, rng)
    mu = mu_from_ncp(state.z_path, state.params_state, layout)
    sigma2 = _sigma2_update_from_mu(state.y, mu, prior.sigma2, rng)
    state.params_obs["sigma2"] = float(sigma2)
    state.params_obs["sigma"] = float(np.sqrt(sigma2))
    residual = state.y - mu
    log_likelihood = float(
        -0.5 * state.y.size * np.log(2.0 * np.pi * sigma2)
        - 0.5 * float(residual @ residual) / sigma2
    )
    metrics = {
        "sign_error": sign_error,
        **{f"sign_{name}": float(value) for name, value in switches.items()},
    }
    return log_likelihood, metrics


def _gev_step(
    state: _ChannelState,
    prior: FSGEVPriors,
    particles: Particles,
    laplace: Laplace,
    rng: np.random.Generator,
) -> tuple[float, dict[str, float], bool, bool, bool]:
    assert state.gev_kernel is not None
    state.gev_kernel.priors = prior
    state.gev_kernel.rng = rng
    last = (
        state.z_path.copy(),
        dict(state.params_state),
        dict(state.params_obs),
        state.model_state,
        state.model_index,
    )
    try:
        pgas_result = ncp_pgas(
            state.y,
            state.model,
            state.params_state,
            state.params_obs,
            state.layout,
            state.z_path,
            n_particles=int(particles.n),
            proposal=str(particles.proposal),
            rng=rng,
        )
        state.z_path = pgas_result.z_path
        X, theta_names, tbar = design_matrix_ncp(
            state.z_path, state.layout, center_time=True
        )
        current_mu = mu_from_ncp(state.z_path, state.params_state, state.layout)
        pseudo_y, pseudo_variance = _laplace_pseudo_mu(
            state.y,
            current_mu,
            state.model,
            state.params_obs,
            curvature_floor=float(laplace.curvature_floor),
        )

        def exact_loglik(eta):
            return _exact_gev_loglik(
                state.y, np.asarray(eta, dtype=float), state.model, state.params_obs
            )

        selection = sample_structural_regression_exact(
            y=state.y,
            X=X,
            theta_names=theta_names,
            tbar=tbar,
            pseudo_y=pseudo_y,
            pseudo_variance=pseudo_variance,
            current_state=state.model_state,
            current_params_state=state.params_state,
            priors=prior,
            layout=state.layout,
            rng=rng,
            apply_theta_draw=apply_theta_draw,
            theta_vector_from_params=theta_vector_from_params,
            elliptical_slice=elliptical_slice_gaussian_prior,
            log_likelihood=exact_loglik,
        )
        state.params_state.update(selection.params_state)
        state.model_state = selection.state
        state.model_index = int(selection.selected_index)
        switches, sign_error = _sign_move(state, rng)
        mu = mu_from_ncp(state.z_path, state.params_state, state.layout)
        if not _gev_support_ok(state.y, mu, state.model, state.params_obs):
            raise ValueError("GEV support failed after hierarchical structural update.")
        state.params_obs, accepted_sigma = state.gev_kernel._mh_update_log_sigma(
            state.y, mu, state.params_obs
        )
        state.params_obs, accepted_xi = state.gev_kernel._mh_update_xi(
            state.y, mu, state.params_obs
        )
        log_likelihood = _exact_gev_loglik(
            state.y, mu, state.model, state.params_obs
        )
        if not np.isfinite(log_likelihood):
            raise ValueError("GEV observation update left the finite support.")
        metrics = {
            "particle_min_ess": float(np.min(pgas_result.ess[1:])),
            "particle_mean_unique_ancestors": float(
                np.mean(pgas_result.unique_ancestors[1:])
            ),
            "particle_path_changed": float(pgas_result.path_changed),
            "particle_path_update_fraction": float(
                pgas_result.path_update_fraction
            ),
            # Kept in the serialized diagnostics for 2.4 compatibility.
            "particle_changed_fraction": float(
                pgas_result.path_update_fraction
            ),
            "ssvs_model_move_accepted": float(selection.move_accepted),
            "ssvs_model_proposed_change": float(
                selection.proposed_index != last[4]
            ),
            "sign_error": sign_error,
            **{f"sign_{name}": float(value) for name, value in switches.items()},
        }
        return float(log_likelihood), metrics, bool(accepted_sigma), bool(accepted_xi), False
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        (
            state.z_path,
            state.params_state,
            state.params_obs,
            state.model_state,
            state.model_index,
        ) = last
        mu = mu_from_ncp(state.z_path, state.params_state, state.layout)
        return (
            float(_exact_gev_loglik(state.y, mu, state.model, state.params_obs)),
            {},
            False,
            False,
            True,
        )


def _gev_laplace_step(
    state: _ChannelState,
    prior: FSGEVPriors,
    laplace: Laplace,
    rng: np.random.Generator,
) -> tuple[float, dict[str, float], bool, bool, bool]:
    """Exploratory approximate GEV update for hierarchical screening.

    The state path and structural regression use the iterated Gaussian
    approximation. Observation parameters are still updated against the exact
    GEV likelihood. This kernel is intentionally labelled approximate in the
    inference plan and fit metadata.
    """

    assert state.gev_kernel is not None
    state.gev_kernel.priors = prior
    state.gev_kernel.rng = rng
    last = (
        state.z_path.copy(),
        dict(state.params_state),
        dict(state.params_obs),
        state.model_state,
        state.model_index,
    )
    try:
        result = iterated_laplace_ncp(
            state.y,
            state.model,
            state.params_state,
            state.params_obs,
            state.layout,
            initial_path=state.z_path,
            rng=rng,
            max_iterations=int(laplace.max_iterations),
            tolerance=float(laplace.tolerance),
            curvature_floor=float(laplace.curvature_floor),
            maximum_variance=float(laplace.maximum_variance),
            draw_attempts=int(laplace.draw_attempts),
        )
        state.z_path = result.z_path
        X, theta_names, tbar = design_matrix_ncp(
            state.z_path, state.layout, center_time=True
        )
        selection = sample_structural_regression(
            y=result.pseudo_y,
            X=X,
            theta_names=theta_names,
            tbar=tbar,
            noise_variance=result.pseudo_variance,
            priors=prior,
            layout=state.layout,
            rng=rng,
            apply_theta_draw=apply_theta_draw,
        )
        state.params_state.update(selection.params_state)
        state.model_state = selection.state
        state.model_index = int(selection.selected_index)
        switches, sign_error = _sign_move(state, rng)
        mu = mu_from_ncp(state.z_path, state.params_state, state.layout)
        if not _gev_support_ok(state.y, mu, state.model, state.params_obs):
            raise ValueError("GEV support failed after Laplace structural update.")
        state.params_obs, accepted_sigma = state.gev_kernel._mh_update_log_sigma(
            state.y, mu, state.params_obs
        )
        state.params_obs, accepted_xi = state.gev_kernel._mh_update_xi(
            state.y, mu, state.params_obs
        )
        log_likelihood = _exact_gev_loglik(
            state.y, mu, state.model, state.params_obs
        )
        if not np.isfinite(log_likelihood):
            raise ValueError("GEV observation update left the finite support.")
        changed_model = float(state.model_index != last[4])
        metrics = {
            "laplace_iterations": float(result.iterations),
            "laplace_converged": float(result.converged),
            "laplace_relative_change": float(result.relative_change),
            "laplace_support_rejections": float(result.support_rejections),
            "ssvs_model_move_accepted": changed_model,
            "ssvs_model_proposed_change": changed_model,
            "sign_error": sign_error,
            **{f"sign_{name}": float(value) for name, value in switches.items()},
        }
        return (
            float(log_likelihood),
            metrics,
            bool(accepted_sigma),
            bool(accepted_xi),
            False,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        (
            state.z_path,
            state.params_state,
            state.params_obs,
            state.model_state,
            state.model_index,
        ) = last
        mu = mu_from_ncp(state.z_path, state.params_state, state.layout)
        return (
            float(_exact_gev_loglik(state.y, mu, state.model, state.params_obs)),
            {},
            False,
            False,
            True,
        )


def _laplace_initialize_channel(
    state: _ChannelState,
    laplace: Laplace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Create a support-valid GEV starting path without changing the PGAS target."""

    if state.family != "gev":
        return {"channel": state.name, "used": False, "reason": "gaussian"}
    try:
        result = iterated_laplace_ncp(
            state.y,
            state.model,
            state.params_state,
            state.params_obs,
            state.layout,
            initial_path=state.z_path,
            rng=rng,
            max_iterations=int(laplace.max_iterations),
            tolerance=float(laplace.tolerance),
            curvature_floor=float(laplace.curvature_floor),
            maximum_variance=float(laplace.maximum_variance),
            draw_attempts=int(laplace.draw_attempts),
        )
        state.z_path = result.z_path
        return {
            "channel": state.name,
            "used": True,
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "relative_change": float(result.relative_change),
            "support_rejections": int(result.support_rejections),
        }
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        return {
            "channel": state.name,
            "used": False,
            "reason": type(error).__name__,
        }


def _channel_step(
    state: _ChannelState,
    prior: FSGaussianPriors | FSGEVPriors,
    engine: str,
    particles: Particles,
    laplace: Laplace,
    rng: np.random.Generator,
) -> tuple[float, dict[str, float], bool, bool, bool]:
    """One conditionally independent channel update with a uniform contract."""

    if state.family == "gaussian":
        log_likelihood, metrics = _gaussian_step(state, prior, rng)
        return log_likelihood, metrics, False, False, False
    if engine == "laplace":
        return _gev_laplace_step(state, prior, laplace, rng)
    return _gev_step(state, prior, particles, laplace, rng)


def _parameter_storage(
    compiled: CompiledMultiSeriesModel,
    states: list[_ChannelState],
    priors: HierarchicalPriors,
    chains: int,
    draws: int,
) -> dict[str, Array]:
    output: dict[str, Array] = {}
    for item in states:
        name = item.name
        output[f"initial.channel.{name}.level"] = np.zeros((chains, draws))
        output[f"state.{name}.level"] = np.zeros((chains, draws), dtype=np.int8)
        output[f"state.{name}.trend"] = np.zeros((chains, draws), dtype=np.int8)
        output[f"state.{name}.season"] = np.zeros((chains, draws), dtype=np.int8)
        output[f"model_index.{name}"] = np.zeros((chains, draws), dtype=np.int16)
        output[f"sigma.{name}"] = np.zeros((chains, draws))
        if item.family == "gev":
            output[f"xi.{name}"] = np.zeros((chains, draws))
        if item.layout.has_beta:
            output[f"initial.channel.{name}.slope"] = np.zeros((chains, draws))
        if item.layout.season_dim > 0:
            output[f"initial.channel.{name}.seasonal"] = np.zeros(
                (chains, draws, item.layout.season_dim)
            )
        for component, process, _ in (
            ("level", "level", "s_level"),
            ("slope", "slope", "s_trend"),
            ("seasonal", "seasonal", "s_season"),
        ):
            scoped = f"channel.{name}.{process}"
            if scoped in compiled.noise_names:
                output[f"sd.{scoped}"] = np.zeros((chains, draws))
                output[f"signed_sd.{scoped}"] = np.zeros((chains, draws))
    labels = {
        "level": ("fixed", "dynamic"),
        "trend": ("zero", "fixed", "dynamic"),
        "season": ("zero", "fixed", "dynamic"),
    }
    for component, names in labels.items():
        for index, name in enumerate(names):
            output[f"hierarchy.prob.{component}.{name}"] = np.zeros(
                (chains, draws)
            )
        output[f"hierarchy.slab_scale.{component}"] = np.zeros(
            (chains, draws)
        )
    if priors.hierarchy.uses_joint_trend_space:
        for name in (
            "linear_trend",
            "rw1_drift",
            "rw2_smooth_trend",
            "local_linear_trend",
        ):
            output[f"hierarchy.model_prob.{name}"] = np.zeros((chains, draws))
    return output


def _store_parameters(
    output: dict[str, Array],
    states: list[_ChannelState],
    probabilities: Mapping[str, Array],
    slab: Mapping[str, float],
    chain: int,
    draw: int,
) -> None:
    for item in states:
        name = item.name
        output[f"initial.channel.{name}.level"][chain, draw] = float(
            item.params_state["alpha0"]
        )
        output[f"state.{name}.level"][chain, draw] = int(item.model_state.level)
        output[f"state.{name}.trend"][chain, draw] = int(item.model_state.trend)
        output[f"state.{name}.season"][chain, draw] = int(item.model_state.season)
        output[f"model_index.{name}"][chain, draw] = int(item.model_index)
        output[f"sigma.{name}"][chain, draw] = float(item.params_obs["sigma"])
        if item.family == "gev":
            output[f"xi.{name}"][chain, draw] = float(item.params_obs["xi"])
        if item.layout.has_beta:
            output[f"initial.channel.{name}.slope"][chain, draw] = float(
                item.params_state["beta0"]
            )
        if item.layout.season_dim > 0:
            output[f"initial.channel.{name}.seasonal"][chain, draw] = np.asarray(
                item.params_state["gamma0_season"], dtype=float
            )
        for process, signed in (
            ("level", "s_level"),
            ("slope", "s_trend"),
            ("seasonal", "s_season"),
        ):
            scoped = f"channel.{name}.{process}"
            key = f"signed_sd.{scoped}"
            if key in output:
                value = float(item.params_state[signed])
                output[key][chain, draw] = value
                output[f"sd.{scoped}"][chain, draw] = abs(value)
    labels = {
        "level": ("fixed", "dynamic"),
        "trend": ("zero", "fixed", "dynamic"),
        "season": ("zero", "fixed", "dynamic"),
    }
    for component, names in labels.items():
        for index, name in enumerate(names):
            output[f"hierarchy.prob.{component}.{name}"][chain, draw] = float(
                probabilities[component][index]
            )
        output[f"hierarchy.slab_scale.{component}"][chain, draw] = float(
            slab[component]
        )
    if "trend_model" in probabilities:
        for index, name in enumerate(
            (
                "linear_trend",
                "rw1_drift",
                "rw2_smooth_trend",
                "local_linear_trend",
            )
        ):
            output[f"hierarchy.model_prob.{name}"][chain, draw] = float(
                probabilities["trend_model"][index]
            )


def _normalized_probabilities(
    values: Any,
    *,
    size: int,
    label: str,
) -> Array:
    output = np.asarray(values, dtype=float).reshape(-1)
    if (
        output.size != int(size)
        or np.any(~np.isfinite(output))
        or np.any(output < 0.0)
        or float(output.sum()) <= 0.0
    ):
        raise ValueError(
            f"Warm-start {label} must contain {size} finite non-negative "
            "values with a positive sum."
        )
    return output / float(output.sum())


def _initial_hierarchy(
    hierarchy: Any,
    initial: Mapping[str, Any] | None,
) -> tuple[dict[str, Array], dict[str, float]]:
    """Resolve default or exported shared-hierarchy starting values."""

    probabilities = {
        component: hierarchy.initial_probabilities(component)
        for component in _COMPONENTS
    }
    if hierarchy.uses_joint_trend_space:
        probabilities["trend_model"] = hierarchy.initial_model_probabilities()
    slab = {
        component: float(hierarchy.initial_slab_scale[component])
        for component in _COMPONENTS
    }
    if initial is None:
        return probabilities, slab

    supplied = dict(initial)
    if hierarchy.pools_selection:
        if hierarchy.uses_joint_trend_space and (
            "hierarchy.trend_model_probabilities" in supplied
        ):
            model = _normalized_probabilities(
                supplied["hierarchy.trend_model_probabilities"],
                size=4,
                label="trend-model probabilities",
            )
            probabilities["trend_model"] = model
            probabilities["level"] = np.asarray(
                (model[0] + model[2], model[1] + model[3]), dtype=float
            )
            probabilities["trend"] = np.asarray(
                (0.0, model[0] + model[1], model[2] + model[3]), dtype=float
            )
        elif not hierarchy.uses_joint_trend_space:
            for component, size in (("level", 2), ("trend", 3)):
                key = f"hierarchy.prob.{component}"
                if key in supplied:
                    probabilities[component] = _normalized_probabilities(
                        supplied[key], size=size, label=f"{component} probabilities"
                    )
        season_key = "hierarchy.prob.season"
        if season_key in supplied:
            probabilities["season"] = _normalized_probabilities(
                supplied[season_key], size=3, label="season probabilities"
            )

    if hierarchy.pools_slab:
        for component in _COMPONENTS:
            key = f"hierarchy.slab_scale.{component}"
            if key not in supplied:
                continue
            value = float(supplied[key])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"Warm-start {key} must be finite and positive.")
            slab[component] = value
    return probabilities, slab


def sample_hierarchical_posterior(
    y: Array,
    compiled: CompiledMultiSeriesModel,
    priors: HierarchicalPriors,
    plan: InferencePlan,
    *,
    mcmc: MCMC,
    particles: Particles,
    laplace: Laplace,
    sampler: HierarchicalSampler,
    dates: Array | None = None,
    initial_parameters: Mapping[str, Any] | None = None,
) -> FitResult:
    """Run joint hierarchical inference for Gaussian and/or GEV channels.

    Gaussian FFBS and non-Gaussian PGAS target the exact posterior. The
    hierarchical Laplace engine is an explicitly exploratory approximation.
    """

    values = np.asarray(y, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(compiled.channel_names):
        raise ValueError("Hierarchical observations must have shape (T, n_channels).")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Hierarchical FS inference currently requires aligned finite observations."
        )
    if plan.parameterization != "fruehwirth_schnatter":
        raise ValueError("Hierarchical inference requires parameterization='fs'.")
    if plan.engine not in {"ffbs", "laplace", "pgas"}:
        raise ValueError("Hierarchical inference supports FFBS, Laplace, or PGAS.")
    if plan.asis:
        raise ValueError("ASIS is not combined with the joint hierarchy sampler.")

    block_by_name = {
        block.name: block for block in compiled.blocks if block.kind == "channel"
    }
    seed_sequences = np.random.SeedSequence(mcmc.seed).spawn(mcmc.chains)
    state_draws = np.zeros(
        (mcmc.chains, mcmc.draws, values.shape[0] + 1, compiled.state_dim)
    )
    log_posterior = np.zeros((mcmc.chains, mcmc.draws))
    template_rng = np.random.default_rng(seed_sequences[0])
    template_states = [
        _initial_channel_state(
            channel.name,
            channel.family,
            values[:, index],
            block_by_name[channel.name].compiled,
            priors.channels[channel.name],
            template_rng,
            initial_parameters,
        )
        for index, channel in enumerate(compiled.model.channels)
    ]
    parameter_draws = _parameter_storage(
        compiled, template_states, priors, mcmc.chains, mcmc.draws
    )
    metric_names = (
        "laplace_iterations",
        "laplace_converged",
        "laplace_relative_change",
        "laplace_support_rejections",
        "particle_min_ess",
        "particle_mean_unique_ancestors",
        "particle_path_changed",
        "particle_path_update_fraction",
        "particle_changed_fraction",
        "ssvs_model_move_accepted",
        "ssvs_model_proposed_change",
        "sign_invariance_error",
    )
    draw_metrics = {
        name: np.full((mcmc.chains, mcmc.draws), np.nan) for name in metric_names
    }
    acceptance = {
        f"sigma.{channel.name}": np.zeros(mcmc.chains)
        for channel in compiled.model.channels
        if channel.family == "gev"
    }
    acceptance.update(
        {
            f"xi.{channel.name}": np.zeros(mcmc.chains)
            for channel in compiled.model.channels
            if channel.family == "gev"
        }
    )
    sign_counts = {
        component: np.zeros(mcmc.chains, dtype=int) for component in _COMPONENTS
    }
    restored_by_chain: list[int] = []
    initial_by_chain: list[dict[str, Any]] = []
    laplace_initialization_by_chain: list[list[dict[str, Any]]] = []

    for chain, sequence in enumerate(seed_sequences):
        rng = np.random.default_rng(sequence)
        states = [
            _initial_channel_state(
                channel.name,
                channel.family,
                values[:, index],
                block_by_name[channel.name].compiled,
                priors.channels[channel.name],
                rng,
                initial_parameters,
            )
            for index, channel in enumerate(compiled.model.channels)
        ]
        probabilities, slab = _initial_hierarchy(
            priors.hierarchy, initial_parameters
        )
        initialization: list[dict[str, Any]] = []
        if plan.engine == "pgas" and sampler.initializer == "laplace":
            initializer_seeds = rng.integers(
                0, np.iinfo(np.int64).max, size=len(states), dtype=np.int64
            )
            for item, child_seed in zip(states, initializer_seeds):
                initialization.append(
                    _laplace_initialize_channel(
                        item,
                        laplace,
                        np.random.default_rng(int(child_seed)),
                    )
                )
        laplace_initialization_by_chain.append(initialization)
        initial_by_chain.append(
            {
                "probabilities": {
                    key: value.tolist() for key, value in probabilities.items()
                },
                "slab_scale": dict(slab),
                "channels": {
                    item.name: {
                        "params_state": dict(item.params_state),
                        "params_obs": dict(item.params_obs),
                    }
                    for item in states
                },
            }
        )
        saved = 0
        restored = 0
        accept_counts = Counter()
        attempts = Counter()
        progress_every = progress_interval(mcmc.iterations, mcmc.progress_every)
        started = perf_counter()
        workers = min(int(sampler.channel_workers), len(states))
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            for iteration in range(mcmc.iterations):
                total_loglik = 0.0
                iteration_metrics: list[dict[str, float]] = []
                current_priors = [
                    _current_prior(item, priors, probabilities, slab)
                    for item in states
                ]
                child_seeds = rng.integers(
                    0, np.iinfo(np.int64).max, size=len(states), dtype=np.int64
                )

                def update_channel(index: int):
                    return _channel_step(
                        states[index],
                        current_priors[index],
                        plan.engine,
                        particles,
                        laplace,
                        np.random.default_rng(int(child_seeds[index])),
                    )

                if executor is None:
                    channel_results = [
                        update_channel(index) for index in range(len(states))
                    ]
                else:
                    channel_results = list(
                        executor.map(update_channel, range(len(states)))
                    )

                for item, result in zip(states, channel_results):
                    ll, metrics, acc_sigma, acc_xi, was_restored = result
                    if item.family == "gev":
                        attempts[f"sigma.{item.name}"] += 1
                        attempts[f"xi.{item.name}"] += 1
                        accept_counts[f"sigma.{item.name}"] += int(acc_sigma)
                        accept_counts[f"xi.{item.name}"] += int(acc_xi)
                        restored += int(was_restored)
                    total_loglik += float(ll)
                    iteration_metrics.append(metrics)
                    for component in _COMPONENTS:
                        sign_counts[component][chain] += int(
                            metrics.get(f"sign_{component}", 0.0)
                        )
                probabilities, slab = _update_hierarchy(states, priors, slab, rng)

                keep = iteration >= mcmc.warmup and (
                    (iteration - mcmc.warmup) % mcmc.thin == 0
                )
                if keep:
                    for item in states:
                        block = block_by_name[item.name]
                        state_draws[chain, saved, :, block.state_slice] = map_ncp_to_centered(
                            item.z_path, item.params_state, item.layout
                        )
                    _store_parameters(
                        parameter_draws, states, probabilities, slab, chain, saved
                    )
                    log_posterior[chain, saved] = total_loglik
                    for metric in metric_names:
                        current = [
                            entry[metric]
                            for entry in iteration_metrics
                            if metric in entry
                        ]
                        if current:
                            if metric == "particle_min_ess":
                                value = float(np.min(current))
                            elif metric == "sign_invariance_error":
                                value = float(np.max(current))
                            else:
                                value = float(np.mean(current))
                            draw_metrics[metric][chain, saved] = value
                    sign_errors = [
                        entry.get("sign_error", np.nan)
                        for entry in iteration_metrics
                    ]
                    finite_errors = [
                        value for value in sign_errors if np.isfinite(value)
                    ]
                    if finite_errors:
                        draw_metrics["sign_invariance_error"][chain, saved] = max(
                            finite_errors
                        )
                    saved += 1

                completed = iteration + 1
                if mcmc.progress and should_report_progress(
                    completed,
                    total=mcmc.iterations,
                    warmup=mcmc.warmup,
                    every=progress_every,
                ):
                    current_params: dict[str, Any] = {
                        "sigma": compact_group(
                            {item.name: item.params_obs["sigma"] for item in states}
                        ),
                        "slab": compact_group(slab),
                    }
                    xis = {
                        item.name: item.params_obs["xi"]
                        for item in states
                        if item.family == "gev"
                    }
                    if xis:
                        current_params["xi"] = compact_group(xis)
                    if "trend_model" in probabilities:
                        current_params["pi_trend"] = compact_group(
                            {
                                "LT": probabilities["trend_model"][0],
                                "RW1": probabilities["trend_model"][1],
                                "RW2": probabilities["trend_model"][2],
                                "LLT": probabilities["trend_model"][3],
                            }
                        )
                    else:
                        current_params["pi_level"] = compact_group(
                            {
                                "F": probabilities["level"][0],
                                "D": probabilities["level"][1],
                            }
                        )
                    progress_metrics: dict[str, float] = {}
                    if plan.engine == "pgas":
                        gev_metrics = [
                            entry
                            for entry in iteration_metrics
                            if "particle_min_ess" in entry
                        ]
                        if gev_metrics:
                            progress_metrics = {
                                "particle_min_ess": min(
                                    entry["particle_min_ess"]
                                    for entry in gev_metrics
                                ),
                                "particle_mean_unique_ancestors": float(
                                    np.mean(
                                        [
                                            entry["particle_mean_unique_ancestors"]
                                            for entry in gev_metrics
                                        ]
                                    )
                                ),
                                "particle_path_update_fraction": float(
                                    np.mean(
                                        [
                                            entry["particle_path_update_fraction"]
                                            for entry in gev_metrics
                                        ]
                                    )
                                ),
                            }
                    elif plan.engine == "laplace":
                        laplace_metrics = [
                            entry
                            for entry in iteration_metrics
                            if "laplace_iterations" in entry
                        ]
                        if laplace_metrics:
                            progress_metrics = {
                                "laplace_iterations": float(
                                    np.max(
                                        [
                                            entry["laplace_iterations"]
                                            for entry in laplace_metrics
                                        ]
                                    )
                                ),
                                "laplace_converged": float(
                                    np.mean(
                                        [
                                            entry["laplace_converged"]
                                            for entry in laplace_metrics
                                        ]
                                    )
                                ),
                            }
                    print(
                        mcmc_progress_line(
                            label="hierarchical SSVS",
                            engine=plan.engine,
                            chain=chain + 1,
                            chains=mcmc.chains,
                            completed=completed,
                            total=mcmc.iterations,
                            warmup=mcmc.warmup,
                            saved=saved,
                            draws=mcmc.draws,
                            elapsed=perf_counter() - started,
                            parameters=current_params,
                            metrics=progress_metrics,
                            particles=(
                                particles.n if plan.engine == "pgas" else None
                            ),
                            details=(f"restored={restored}",) if restored else (),
                        ),
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        for name in acceptance:
            acceptance[name][chain] = (
                accept_counts[name] / attempts[name] if attempts[name] else np.nan
            )
        restored_by_chain.append(restored)

    return FitResult(
        model=compiled.model,
        compiled=compiled,
        priors=priors,
        y=values,
        state_draws=state_draws,
        parameter_draws=parameter_draws,
        log_posterior=log_posterior,
        plan=plan,
        sampler_diagnostics={
            "acceptance": acceptance,
            "draw_metrics": draw_metrics,
            "mcmc": asdict(mcmc),
            "particles": asdict(particles),
            "laplace": asdict(laplace),
            "hierarchical_sampler": asdict(sampler),
            "chain_seeds": [int(sequence.generate_state(1)[0]) for sequence in seed_sequences],
            "sign_switch_counts": {
                key: value.tolist() for key, value in sign_counts.items()
            },
            "restored_iterations_by_chain": restored_by_chain,
            "laplace_initialization_by_chain": laplace_initialization_by_chain,
        },
        dates=None if dates is None else np.asarray(dates),
        series_name=compiled.model.name,
        transform_sign=compiled.model.transform_signs,
        schema_version="2.4.1",
        initial_values={"chains": initial_by_chain},
        metadata={
            "bucex_version": __version__,
            "inference_contract": "joint_hierarchical_structural_model",
            "joint_model": True,
            "joint_likelihood": True,
            "hierarchical_model_selection": bool(priors.hierarchy.pools_selection),
            "shared_temporal_state": False,
            "conditional_channel_independence": True,
            "structural_ssvs": bool(priors.hierarchy.pools_selection),
            "model_selection_exact": bool(
                priors.hierarchy.pools_selection and plan.targets_exact_posterior
            ),
            "pooled_normal_slab": bool(priors.hierarchy.pools_slab),
            "hierarchy_pool": priors.hierarchy.pool,
            "trend_model_space": priors.hierarchy.model_space,
            "signed_innovation_scales": True,
            "sign_switching": True,
            "sign_switch_invariance_checked": True,
            "restored_iterations": int(sum(restored_by_chain)),
            "restored_iterations_by_chain": restored_by_chain,
            "pgas_exact_invariant": bool(plan.engine == "pgas"),
            "hierarchical_laplace_approximation": bool(plan.engine == "laplace"),
            "pgas_initializer": (
                sampler.initializer if plan.engine == "pgas" else None
            ),
            "channel_workers": int(sampler.channel_workers),
            "external_warm_start": bool(
                initial_parameters is not None
                and "__warm_start__" in initial_parameters
            ),
            "warm_start_source": (
                dict(initial_parameters["__warm_start__"])
                if initial_parameters is not None
                and "__warm_start__" in initial_parameters
                else None
            ),
        },
    )


__all__ = ["sample_hierarchical_posterior"]
