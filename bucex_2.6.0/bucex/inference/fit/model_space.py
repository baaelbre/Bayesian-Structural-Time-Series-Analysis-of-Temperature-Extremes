from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable

import numpy as np

from ...core.numerics import symmetrize as _symmetrize

Array = np.ndarray


class ComponentState(IntEnum):
    """Structural status of one model component."""

    ZERO = 0
    FIXED = 1
    DYNAMIC = 2

    @property
    def label(self) -> str:
        return {0: "zero", 1: "fixed", 2: "dynamic"}[int(self)]


class TrendModelClass(IntEnum):
    """Joint evolution class for a level and its always-present slope."""

    LINEAR_TREND = 0
    RW1_WITH_DRIFT = 1
    RW2_SMOOTH_TREND = 2
    LOCAL_LINEAR_TREND = 3

    @property
    def label(self) -> str:
        return (
            "linear_trend",
            "rw1_drift",
            "rw2_smooth_trend",
            "local_linear_trend",
        )[int(self)]


_TREND_CLASS_STATES = {
    TrendModelClass.LINEAR_TREND: (ComponentState.FIXED, ComponentState.FIXED),
    TrendModelClass.RW1_WITH_DRIFT: (ComponentState.DYNAMIC, ComponentState.FIXED),
    TrendModelClass.RW2_SMOOTH_TREND: (ComponentState.FIXED, ComponentState.DYNAMIC),
    TrendModelClass.LOCAL_LINEAR_TREND: (
        ComponentState.DYNAMIC,
        ComponentState.DYNAMIC,
    ),
}


def trend_model_class(state: "StructuralModelState") -> TrendModelClass:
    """Map level/slope innovation allocation to its scientific trend class."""

    pair = (state.level, state.trend)
    for model_class, states in _TREND_CLASS_STATES.items():
        if pair == states:
            return model_class
    raise ValueError(
        "A zero slope is outside the joint trend model space; use the "
        "componentwise SSVS model space when an absent slope is required."
    )


@dataclass(frozen=True)
class StructuralModelState:
    """One zero/fixed/dynamic structural specification.

    The level is always present and can only be fixed or dynamic. The trend and
    seasonal components can be absent, fixed, or dynamic.
    """

    level: ComponentState
    trend: ComponentState
    season: ComponentState

    def __post_init__(self) -> None:
        if self.level not in {ComponentState.FIXED, ComponentState.DYNAMIC}:
            raise ValueError("The level must be fixed or dynamic; it cannot be absent.")

    @property
    def label(self) -> str:
        return (
            f"level={self.level.label},"
            f"trend={self.trend.label},"
            f"season={self.season.label}"
        )


@dataclass
class StructuralSelectionResult:
    state: StructuralModelState
    params_state: dict[str, Any]
    probabilities: Array
    log_probabilities: Array
    candidates: tuple[StructuralModelState, ...]
    selected_index: int


@dataclass
class ExactStructuralSelectionResult:
    """One exact-invariant structural update for a non-Gaussian model.

    ``proposal_probabilities`` are the deliberately defensive mixture used to
    propose models; they are *not* posterior model probabilities.  The exact
    GEV likelihood, normalized slab densities, model prior, and forward/reverse
    proposal densities all enter the Metropolis--Hastings ratio.
    """

    state: StructuralModelState
    params_state: dict[str, Any]
    candidates: tuple[StructuralModelState, ...]
    selected_index: int
    proposed_index: int
    move_accepted: bool
    log_acceptance_ratio: float
    proposal_probabilities: Array
    elliptical_slice_steps: int


def enumerate_structural_models(
    layout: Any,
    ssvs: Any | None = None,
) -> tuple[StructuralModelState, ...]:
    """Enumerate the complete structural model space supported by ``layout``."""

    joint = getattr(ssvs, "trend_model_probabilities", None)
    if joint is not None:
        if not bool(layout.has_beta):
            raise ValueError("The joint trend model space requires a slope state.")
        season_states = (
            (ComponentState.FIXED, ComponentState.DYNAMIC)
            if int(layout.season_dim) > 0
            else (ComponentState.ZERO,)
        )
        return tuple(
            StructuralModelState(level=level, trend=trend, season=season)
            for level, trend in _TREND_CLASS_STATES.values()
            for season in season_states
        )

    level_states = (ComponentState.FIXED, ComponentState.DYNAMIC)
    trend_states = (
        (ComponentState.ZERO, ComponentState.FIXED, ComponentState.DYNAMIC)
        if bool(layout.has_beta)
        else (ComponentState.ZERO,)
    )
    season_states = (
        (ComponentState.ZERO, ComponentState.FIXED, ComponentState.DYNAMIC)
        if int(layout.season_dim) > 0
        else (ComponentState.ZERO,)
    )

    return tuple(
        StructuralModelState(level=level, trend=trend, season=season)
        for level in level_states
        for trend in trend_states
        for season in season_states
    )


def initial_structural_state(params_state: dict[str, Any], layout: Any) -> StructuralModelState:
    """Derive a sensible initial structural state from supplied parameters."""

    level = (
        ComponentState.DYNAMIC
        if abs(float(params_state.get("s_level", 0.0))) > 0.0
        else ComponentState.FIXED
    )

    if bool(layout.has_beta):
        if abs(float(params_state.get("s_trend", 0.0))) > 0.0:
            trend = ComponentState.DYNAMIC
        elif abs(float(params_state.get("beta0", 0.0))) > 0.0:
            trend = ComponentState.FIXED
        else:
            trend = ComponentState.ZERO
    else:
        trend = ComponentState.ZERO

    if int(layout.season_dim) > 0:
        gamma = np.asarray(params_state.get("gamma0_season", []), dtype=float)
        if abs(float(params_state.get("s_season", 0.0))) > 0.0:
            season = ComponentState.DYNAMIC
        elif gamma.size and np.any(np.abs(gamma) > 0.0):
            season = ComponentState.FIXED
        else:
            season = ComponentState.ZERO
    else:
        season = ComponentState.ZERO

    return StructuralModelState(level=level, trend=trend, season=season)


def active_theta_names(
    all_names: Iterable[str],
    state: StructuralModelState,
) -> list[str]:
    """Return active FS regression columns for one structural state."""

    names = list(all_names)
    active: list[str] = []

    for name in names:
        if name in {"alpha0", "alpha_c"}:
            active.append(name)
        elif name == "beta0" and state.trend != ComponentState.ZERO:
            active.append(name)
        elif name.startswith("gamma0_season_") and state.season != ComponentState.ZERO:
            active.append(name)
        elif name == "s_level" and state.level == ComponentState.DYNAMIC:
            active.append(name)
        elif name == "s_trend" and state.trend == ComponentState.DYNAMIC:
            active.append(name)
        elif name == "s_season" and state.season == ComponentState.DYNAMIC:
            active.append(name)

    return active


def enforce_structural_state(
    params_state: dict[str, Any],
    state: StructuralModelState,
    layout: Any,
) -> dict[str, Any]:
    """Set all coefficients excluded by ``state`` exactly to zero."""

    out = dict(params_state)
    out.setdefault("alpha0", 0.0)

    if state.level != ComponentState.DYNAMIC:
        out["s_level"] = 0.0
    else:
        out.setdefault("s_level", 0.0)

    if bool(layout.has_beta):
        if state.trend == ComponentState.ZERO:
            out["beta0"] = 0.0
            out["s_trend"] = 0.0
        elif state.trend == ComponentState.FIXED:
            out.setdefault("beta0", 0.0)
            out["s_trend"] = 0.0
        else:
            out.setdefault("beta0", 0.0)
            out.setdefault("s_trend", 0.0)
    else:
        out["beta0"] = 0.0
        out["s_trend"] = 0.0

    if int(layout.season_dim) > 0:
        if state.season == ComponentState.ZERO:
            out["gamma0_season"] = np.zeros(int(layout.season_dim), dtype=float)
            out["s_season"] = 0.0
        elif state.season == ComponentState.FIXED:
            out.setdefault(
                "gamma0_season", np.zeros(int(layout.season_dim), dtype=float)
            )
            out["s_season"] = 0.0
        else:
            out.setdefault(
                "gamma0_season", np.zeros(int(layout.season_dim), dtype=float)
            )
            out.setdefault("s_season", 0.0)
    else:
        out["gamma0_season"] = np.zeros(0, dtype=float)
        out["s_season"] = 0.0

    out["q_level"] = float(out.get("s_level", 0.0)) ** 2
    out["q_trend"] = float(out.get("s_trend", 0.0)) ** 2
    out["q_season"] = float(out.get("s_season", 0.0)) ** 2
    return out


def _component_probability(
    state: ComponentState,
    probabilities: tuple[float, float, float],
) -> float:
    return float(probabilities[int(state)])


def structural_model_log_prior(state: StructuralModelState, ssvs: Any) -> float:
    """Log prior probability of one structural model."""

    joint = getattr(ssvs, "trend_model_probabilities", None)
    if joint is not None:
        try:
            model_probability = float(joint[trend_model_class(state).label])
        except (KeyError, ValueError):
            return -np.inf
        p_season = _component_probability(
            state.season, tuple(ssvs.season_probabilities)
        )
        if min(model_probability, p_season) <= 0.0:
            return -np.inf
        return float(np.log(model_probability) + np.log(p_season))

    p_level = (
        float(ssvs.level_dynamic_probability)
        if state.level == ComponentState.DYNAMIC
        else 1.0 - float(ssvs.level_dynamic_probability)
    )
    p_trend = _component_probability(state.trend, tuple(ssvs.trend_probabilities))
    p_season = _component_probability(state.season, tuple(ssvs.season_probabilities))

    if min(p_level, p_trend, p_season) <= 0.0:
        return -np.inf
    return float(np.log(p_level) + np.log(p_trend) + np.log(p_season))


def _safe_cholesky(matrix: Array, jitter: float = 1e-12) -> Array:
    matrix = _symmetrize(matrix)
    eye = np.eye(matrix.shape[0])
    for power in range(10):
        try:
            return np.linalg.cholesky(matrix + (10.0**power) * jitter * eye)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Matrix is not positive definite after jittering.")


def _solve_spd(matrix: Array, rhs: Array) -> Array:
    chol = _safe_cholesky(matrix)
    return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))


def _logdet_spd(matrix: Array) -> float:
    chol = _safe_cholesky(matrix)
    return float(2.0 * np.sum(np.log(np.diag(chol))))


def _sample_gaussian_precision(
    precision: Array,
    h: Array,
    rng: np.random.Generator,
) -> Array:
    chol = _safe_cholesky(precision)
    mean = np.linalg.solve(chol.T, np.linalg.solve(chol, h))
    noise = np.linalg.solve(chol.T, rng.normal(size=precision.shape[0]))
    return mean + noise


def _log_gaussian_precision(
    value: Array,
    mean: Array,
    precision: Array,
) -> float:
    """Normalized multivariate Gaussian log density from a precision matrix."""

    value = np.asarray(value, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    precision = _symmetrize(np.asarray(precision, dtype=float))
    difference = value - mean
    return float(
        0.5 * _logdet_spd(precision)
        - 0.5 * value.size * np.log(2.0 * np.pi)
        - 0.5 * difference @ precision @ difference
    )


def _structural_prior_mean_precision(
    active_names: list[str],
    *,
    tbar: float,
    priors: Any,
    layout: Any,
) -> tuple[Array, Array]:
    """Prior for active regression coefficients under an SSVS slab."""

    index = {name: i for i, name in enumerate(active_names)}
    d = len(active_names)
    mean = np.zeros(d, dtype=float)
    precision = np.zeros((d, d), dtype=float)

    alpha_name = "alpha_c" if "alpha_c" in index else "alpha0"
    ia = index[alpha_name]
    ma = float(priors.alpha0.mean)
    va = float(priors.alpha0.sd) ** 2

    if "beta0" in index:
        ib = index["beta0"]
        mb = float(priors.beta0.mean)
        vb = float(priors.beta0.sd) ** 2
        mean[ia] = ma + float(tbar) * mb
        mean[ib] = mb
        covariance = np.array(
            [
                [va + float(tbar) ** 2 * vb, float(tbar) * vb],
                [float(tbar) * vb, vb],
            ],
            dtype=float,
        )
        joint_precision = _solve_spd(covariance, np.eye(2))
        precision[np.ix_([ia, ib], [ia, ib])] = joint_precision
    else:
        mean[ia] = ma
        precision[ia, ia] = 1.0 / va

    gamma_names = [name for name in active_names if name.startswith("gamma0_season_")]
    if gamma_names:
        gamma_prior = priors.gamma0_season
        if gamma_prior is None:
            raise ValueError("gamma0_season prior is required when seasonality is active.")
        gamma_mean = gamma_prior.mean_array()
        gamma_sd = gamma_prior.sd_array()
        if gamma_mean.size != int(layout.season_dim):
            raise ValueError("gamma0_season prior dimension mismatch.")
        for name in gamma_names:
            j = int(name.rsplit("_", 1)[1]) - 1
            i = index[name]
            mean[i] = float(gamma_mean[j])
            precision[i, i] = 1.0 / float(gamma_sd[j]) ** 2

    slab_sd = dict(priors.ssvs.innovation_slab_sd)
    for coefficient, block in (
        ("s_level", "level"),
        ("s_trend", "trend"),
        ("s_season", "season"),
    ):
        if coefficient in index:
            sd = float(slab_sd[block])
            precision[index[coefficient], index[coefficient]] = 1.0 / (sd * sd)

    if np.any(np.diag(precision) <= 0.0):
        raise ValueError("Every active SSVS coefficient needs a proper Gaussian slab prior.")
    return mean, precision


def _candidate_posterior(
    y: Array,
    X: Array,
    noise_variance: Array,
    prior_mean: Array,
    prior_precision: Array,
) -> tuple[float, Array, Array]:
    """Marginal log density and Gaussian posterior parameters.

    The coefficient vector is integrated out to score each structural model.
    ``noise_variance`` may be scalar or time varying.
    """

    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    variance = np.asarray(noise_variance, dtype=float)
    if variance.ndim == 0:
        variance = np.full(y.size, float(variance), dtype=float)
    variance = variance.reshape(-1)
    if variance.size != y.size or np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("noise_variance must be positive and have length T.")

    weights = 1.0 / variance
    xtw = X.T * weights
    posterior_precision = _symmetrize(prior_precision + xtw @ X)
    h = prior_precision @ prior_mean + xtw @ y
    posterior_mean = _solve_spd(posterior_precision, h)

    quadratic = (
        float(np.dot(y * weights, y))
        + float(prior_mean @ prior_precision @ prior_mean)
        - float(h @ posterior_mean)
    )
    log_marginal = -0.5 * (
        y.size * np.log(2.0 * np.pi)
        + float(np.sum(np.log(variance)))
        - _logdet_spd(prior_precision)
        + _logdet_spd(posterior_precision)
        + quadratic
    )
    return float(log_marginal), posterior_mean, posterior_precision


def _normalize_log_probabilities(log_weights: Array) -> tuple[Array, Array]:
    log_weights = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        probabilities = np.full(log_weights.size, 1.0 / log_weights.size)
        return probabilities, np.log(probabilities)
    maximum = float(np.max(log_weights[finite]))
    weights = np.zeros_like(log_weights)
    weights[finite] = np.exp(log_weights[finite] - maximum)
    probabilities = weights / np.sum(weights)
    with np.errstate(divide="ignore"):
        log_probabilities = np.log(probabilities)
    return probabilities, log_probabilities


def sample_structural_regression(
    *,
    y: Array,
    X: Array,
    theta_names: list[str],
    tbar: float,
    noise_variance: Array | float,
    priors: Any,
    layout: Any,
    rng: np.random.Generator,
    apply_theta_draw: Any,
) -> StructuralSelectionResult:
    """Enumerate, sample, and fit all supported structural models."""

    if getattr(priors, "ssvs", None) is None:
        raise ValueError("sample_structural_regression requires priors.ssvs.")

    candidates = enumerate_structural_models(layout, priors.ssvs)
    log_weights = np.full(len(candidates), -np.inf, dtype=float)
    cache: list[tuple[list[str], Array, Array] | None] = [None] * len(candidates)

    for i, state in enumerate(candidates):
        active = active_theta_names(theta_names, state)
        keep = [theta_names.index(name) for name in active]
        X_model = X[:, keep]
        prior_mean, prior_precision = _structural_prior_mean_precision(
            active,
            tbar=tbar,
            priors=priors,
            layout=layout,
        )
        try:
            log_marginal, posterior_mean, posterior_precision = _candidate_posterior(
                y,
                X_model,
                noise_variance,
                prior_mean,
                prior_precision,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        log_weights[i] = log_marginal + structural_model_log_prior(state, priors.ssvs)
        cache[i] = (active, posterior_mean, posterior_precision)

    if not np.any(np.isfinite(log_weights)):
        raise ValueError(
            "No structural model has positive prior probability and a valid posterior."
        )
    probabilities, log_probabilities = _normalize_log_probabilities(log_weights)
    selected_index = int(rng.choice(len(candidates), p=probabilities))
    selected = cache[selected_index]
    if selected is None:
        raise RuntimeError("The selected structural model did not have a valid posterior.")
    active, _, posterior_precision = selected

    keep = [theta_names.index(name) for name in active]
    X_model = X[:, keep]
    prior_mean, prior_precision = _structural_prior_mean_precision(
        active,
        tbar=tbar,
        priors=priors,
        layout=layout,
    )
    variance = np.asarray(noise_variance, dtype=float)
    if variance.ndim == 0:
        variance = np.full(np.asarray(y).size, float(variance), dtype=float)
    weights = 1.0 / variance.reshape(-1)
    xtw = X_model.T * weights
    h = prior_precision @ prior_mean + xtw @ np.asarray(y, dtype=float).reshape(-1)
    theta = _sample_gaussian_precision(posterior_precision, h, rng)
    params_state = apply_theta_draw({}, theta, active, layout, tbar=tbar)
    params_state = enforce_structural_state(
        params_state, candidates[selected_index], layout
    )

    return StructuralSelectionResult(
        state=candidates[selected_index],
        params_state=params_state,
        probabilities=probabilities,
        log_probabilities=log_probabilities,
        candidates=candidates,
        selected_index=selected_index,
    )


def sample_structural_regression_exact(
    *,
    y: Array,
    X: Array,
    theta_names: list[str],
    tbar: float,
    pseudo_y: Array,
    pseudo_variance: Array | float,
    current_state: StructuralModelState,
    current_params_state: dict[str, Any],
    priors: Any,
    layout: Any,
    rng: np.random.Generator,
    apply_theta_draw: Any,
    theta_vector_from_params: Any,
    elliptical_slice: Any,
    log_likelihood: Any,
    proposal_uniform_weight: float = 0.05,
    elliptical_slice_max_steps: int = 10_000,
) -> ExactStructuralSelectionResult:
    """Exact PGAS-compatible SSVS update using Laplace-informed proposals.

    Conditional on a PGAS state path, the GEV regression block is
    non-Gaussian.  Gaussian model enumeration is therefore not exact.  This
    kernel instead uses each model's Gaussian pseudo-posterior only as an
    independence proposal and corrects it with a trans-dimensional
    Metropolis--Hastings ratio based on the exact likelihood.  An exact
    elliptical-slice update then refreshes the active coefficients.

    The small uniform mixture on model proposals guarantees positive proposal
    probability for every model with positive prior mass.  Consequently the
    kernel remains irreducible even if the Laplace approximation is extremely
    concentrated or numerically poor.
    """

    if getattr(priors, "ssvs", None) is None:
        raise ValueError("sample_structural_regression_exact requires priors.ssvs.")
    uniform_weight = float(proposal_uniform_weight)
    if not 0.0 <= uniform_weight <= 1.0:
        raise ValueError("proposal_uniform_weight must lie in [0, 1].")

    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    pseudo_y = np.asarray(pseudo_y, dtype=float).reshape(-1)
    if X.shape != (y.size, len(theta_names)) or pseudo_y.size != y.size:
        raise ValueError("X, y, pseudo_y, and theta_names have incompatible shapes.")

    candidates = enumerate_structural_models(layout, priors.ssvs)
    if current_state not in candidates:
        raise ValueError("current_state is not supported by this model layout.")
    current_index = candidates.index(current_state)

    # Each cache entry contains active names/columns, the exact slab prior,
    # and a deterministic Gaussian approximation used only as q(theta | M).
    cache: list[dict[str, Any] | None] = [None] * len(candidates)
    approximate_weights = np.full(len(candidates), -np.inf, dtype=float)
    valid = np.zeros(len(candidates), dtype=bool)
    for index, state in enumerate(candidates):
        model_log_prior = structural_model_log_prior(state, priors.ssvs)
        if not np.isfinite(model_log_prior):
            continue
        active = active_theta_names(theta_names, state)
        keep = [theta_names.index(name) for name in active]
        X_model = X[:, keep]
        prior_mean, prior_precision = _structural_prior_mean_precision(
            active,
            tbar=tbar,
            priors=priors,
            layout=layout,
        )
        try:
            approximate_marginal, proposal_mean, proposal_precision = (
                _candidate_posterior(
                    pseudo_y,
                    X_model,
                    pseudo_variance,
                    prior_mean,
                    prior_precision,
                )
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            # The exact kernel is not allowed to fail merely because its
            # approximation failed.  A prior proposal has full support.
            approximate_marginal = 0.0
            proposal_mean = prior_mean.copy()
            proposal_precision = prior_precision.copy()
        cache[index] = {
            "active": active,
            "keep": keep,
            "X": X_model,
            "prior_mean": prior_mean,
            "prior_precision": prior_precision,
            "proposal_mean": proposal_mean,
            "proposal_precision": proposal_precision,
            "model_log_prior": model_log_prior,
        }
        approximate_weights[index] = approximate_marginal + model_log_prior
        valid[index] = True

    if not np.any(valid):
        raise ValueError("The SSVS prior assigns zero mass to every model.")
    if not valid[current_index]:
        raise ValueError(
            "The current structural model has zero prior probability; choose "
            "initial values compatible with the SSVS model probabilities."
        )

    approximate_probabilities, _ = _normalize_log_probabilities(
        approximate_weights[valid]
    )
    valid_indices = np.flatnonzero(valid)
    mixed = (
        (1.0 - uniform_weight) * approximate_probabilities
        + uniform_weight / valid_indices.size
    )
    proposal_probabilities = np.zeros(len(candidates), dtype=float)
    proposal_probabilities[valid_indices] = mixed / np.sum(mixed)
    proposed_index = int(rng.choice(len(candidates), p=proposal_probabilities))

    current_cache = cache[current_index]
    proposed_cache = cache[proposed_index]
    assert current_cache is not None and proposed_cache is not None
    current_theta = theta_vector_from_params(
        current_params_state,
        current_cache["active"],
        layout,
        tbar=tbar,
    )
    proposed_theta = _sample_gaussian_precision(
        proposed_cache["proposal_precision"],
        proposed_cache["proposal_precision"] @ proposed_cache["proposal_mean"],
        rng,
    )

    def exact_target(theta: Array, entry: dict[str, Any]) -> float:
        likelihood = float(log_likelihood(entry["X"] @ theta))
        if not np.isfinite(likelihood):
            return -np.inf
        return float(
            likelihood
            + _log_gaussian_precision(
                theta, entry["prior_mean"], entry["prior_precision"]
            )
            + entry["model_log_prior"]
        )

    current_target = exact_target(current_theta, current_cache)
    proposed_target = exact_target(proposed_theta, proposed_cache)
    current_log_q = float(
        np.log(proposal_probabilities[current_index])
        + _log_gaussian_precision(
            current_theta,
            current_cache["proposal_mean"],
            current_cache["proposal_precision"],
        )
    )
    proposed_log_q = float(
        np.log(proposal_probabilities[proposed_index])
        + _log_gaussian_precision(
            proposed_theta,
            proposed_cache["proposal_mean"],
            proposed_cache["proposal_precision"],
        )
    )
    log_acceptance_ratio = float(
        proposed_target - current_target + current_log_q - proposed_log_q
    )
    accepted = bool(
        np.isfinite(proposed_target)
        and np.log(rng.random()) < min(0.0, log_acceptance_ratio)
    )
    if accepted:
        selected_index = proposed_index
        selected_cache = proposed_cache
        selected_theta = proposed_theta
    else:
        selected_index = current_index
        selected_cache = current_cache
        selected_theta = current_theta

    prior_covariance = _symmetrize(
        _solve_spd(
            selected_cache["prior_precision"],
            np.eye(selected_theta.size),
        )
    )

    def selected_log_likelihood(theta: Array) -> float:
        return float(log_likelihood(selected_cache["X"] @ theta))

    selected_theta, slice_steps = elliptical_slice(
        selected_theta,
        selected_cache["prior_mean"],
        prior_covariance,
        selected_log_likelihood,
        rng,
        max_steps=int(elliptical_slice_max_steps),
    )
    params_state = apply_theta_draw(
        {},
        selected_theta,
        selected_cache["active"],
        layout,
        tbar=tbar,
    )
    selected_state = candidates[selected_index]
    params_state = enforce_structural_state(params_state, selected_state, layout)
    return ExactStructuralSelectionResult(
        state=selected_state,
        params_state=params_state,
        candidates=candidates,
        selected_index=selected_index,
        proposed_index=proposed_index,
        move_accepted=accepted,
        log_acceptance_ratio=log_acceptance_ratio,
        proposal_probabilities=proposal_probabilities,
        elliptical_slice_steps=int(slice_steps),
    )
