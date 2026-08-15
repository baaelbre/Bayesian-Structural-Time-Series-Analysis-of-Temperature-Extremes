"""Blocked loading updates for identified one-factor models.

The generic factor sampler can update loadings conditional on a complete
latent path.  That is a valid Metropolis-within-Gibbs step, but it is a poor
parameterization when a channel-specific random walk can offset a change in
the loading.  This module contains two complementary kernels used by the
specialized Fruehwirth--Schnatter factor sampler:

* an exact Gaussian FFBS block for the channel intercept, loading and
  idiosyncratic random walk; and
* a predictor-preserving loading/deviation interweaving move for GEV
  channels.

Both kernels operate on the common semantic (centred) state representation
and map back to the FS representation before returning.  They therefore do
not change the public result contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

from ...models.factor_compiler import CompiledFactorModel
from ...priors.factor import FactorPriors
from ...priors.process import FixedSD
from ..state.kalman import ffbs, kalman_filter
from .disturbance import _adapt, _prior_logpdf
from .factor_horseshoe import factor_horseshoe_coefficient_logpdf
from .factor_triple_gamma import factor_triple_gamma_coefficient_logpdf

if TYPE_CHECKING:  # pragma: no cover - imported only for static checking
    from .factor_fs import CompiledFactorFS, FactorFSBlock


Array = np.ndarray


@dataclass(frozen=True)
class _RegressionRandomWalk:
    """Three-state DLM for ``c + loading * factor_t + deviation_t``.

    The intercept and loading are static states with their declared normal
    priors.  The deviation starts at zero and follows a random walk.  FFBS on
    this augmented state is an exact joint draw and hence is equivalent to
    integrating the deviation out for the loading draw and drawing it back
    conditionally afterwards.
    """

    factor: Array
    deviation_sd: float
    intercept_mean: float
    intercept_sd: float
    loading_mean: float
    loading_sd: float

    def __post_init__(self) -> None:
        factor = np.asarray(self.factor, dtype=float).reshape(-1)
        if factor.size < 2 or not np.all(np.isfinite(factor)):
            raise ValueError("factor must contain at least two finite values.")
        if self.deviation_sd < 0.0 or not np.isfinite(self.deviation_sd):
            raise ValueError("deviation_sd must be finite and non-negative.")
        if self.intercept_sd <= 0.0 or self.loading_sd <= 0.0:
            raise ValueError("Static coefficient prior SDs must be positive.")
        object.__setattr__(self, "factor", factor)

    @property
    def n_time(self) -> int:
        return int(self.factor.size)

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def transition(self) -> Array:
        return np.eye(3, dtype=float)

    @property
    def initial_mean(self) -> Array:
        return np.asarray(
            [self.intercept_mean, self.loading_mean, 0.0], dtype=float
        )

    @property
    def initial_cov(self) -> Array:
        return np.diag(
            [self.intercept_sd**2, self.loading_sd**2, 0.0]
        )

    def transition_cov(self, params: Mapping[str, float]) -> Array:
        del params
        return np.diag([0.0, 0.0, self.deviation_sd**2])

    def design(
        self,
        n_time: int | None = None,
        exog=None,
        *,
        params: Mapping[str, float] | None = None,
    ) -> Array:
        del params
        if exog is not None:
            raise ValueError("The collapsed loading block does not use exog.")
        n = self.n_time if n_time is None else int(n_time)
        if n != self.n_time:
            raise ValueError("Use the factor path matching this loading block.")
        return np.column_stack(
            (np.ones(n, dtype=float), self.factor, np.ones(n, dtype=float))
        )

    def project_path(self, path: Array, params: Mapping[str, float]) -> Array:
        del params
        output = np.asarray(path, dtype=float).copy()
        # The first two transitions are deterministic.  Enforce that affine
        # support after the Moore--Penrose smoother to remove round-off drift.
        output[:, 0] = output[0, 0]
        output[:, 1] = output[0, 1]
        output[0, 2] = 0.0
        if self.deviation_sd == 0.0:
            output[:, 2] = 0.0
        return output


def _factor_and_channel(key: str) -> tuple[str, str]:
    pieces = str(key).split(".")
    if len(pieces) != 3 or pieces[0] != "loading":
        raise ValueError(f"Invalid loading parameter name '{key}'.")
    return pieces[1], pieces[2]


def _block(
    fs: "CompiledFactorFS",
    *,
    kind: str,
    name: str,
) -> "FactorFSBlock":
    return next(
        item for item in fs.blocks if item.kind == kind and item.name == name
    )


def _block_predictor(
    centered_path: Array,
    block: "FactorFSBlock",
    *,
    remove_level: bool = False,
) -> Array:
    local_path = np.asarray(
        centered_path[:, block.centered_slice], dtype=float
    ).copy()
    if remove_level:
        local_path[:, int(block.layout.idx_alpha)] = 0.0
    design = block.compiled.design(local_path.shape[0] - 1, params={})
    return np.einsum("tm,tm->t", design, local_path[1:])


def collapsed_gaussian_loading_keys(
    compiled: CompiledFactorModel,
) -> tuple[str, ...]:
    """Estimated loading keys eligible for the exact Gaussian block."""

    families = {channel.name: channel.family for channel in compiled.model.channels}
    return tuple(
        key
        for key in compiled.estimated_loading_names
        if families[_factor_and_channel(key)[1]] == "gaussian"
    )


def collapsed_gaussian_intercept_keys(
    compiled: CompiledFactorModel,
) -> frozenset[str]:
    """Intercepts sampled inside the exact Gaussian loading block."""

    return frozenset(
        f"intercept.{_factor_and_channel(key)[1]}"
        for key in collapsed_gaussian_loading_keys(compiled)
    )


def collapsed_gaussian_processes(
    compiled: CompiledFactorModel,
    fs: "CompiledFactorFS",
) -> frozenset[str]:
    """Idiosyncratic scales integrated by Gaussian loading blocks."""

    return frozenset(
        _block(
            fs,
            kind="channel",
            name=_factor_and_channel(key)[1],
        ).process_by_component["level"]
        for key in collapsed_gaussian_loading_keys(compiled)
    )


def _signed_process_prior_logpdf(
    process: str,
    value: float,
    priors: FactorPriors,
    horseshoe_state: Mapping[str, object],
    triple_gamma_state: Mapping[str, object],
) -> float:
    if process in priors.horseshoe_processes:
        return factor_horseshoe_coefficient_logpdf(
            value, process, priors, horseshoe_state
        )
    if process in priors.triple_gamma_processes:
        return factor_triple_gamma_coefficient_logpdf(
            value, process, priors, triple_gamma_state
        )
    return _prior_logpdf(priors.process[process], abs(float(value)))


def collapsed_gaussian_scale_sweep(
    y: Array,
    path: Array,
    compiled: CompiledFactorModel,
    fs: "CompiledFactorFS",
    params: dict[str, float],
    priors: FactorPriors,
    horseshoe_state: Mapping[str, object],
    triple_gamma_state: Mapping[str, object],
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    adapt: bool,
    iteration: int,
) -> tuple[dict[str, bool], dict[str, float]]:
    """Update Gaussian idiosyncratic SDs from their collapsed likelihood.

    The Kalman likelihood integrates the channel intercept, loading and random
    walk simultaneously.  Updating the scale from this marginal target avoids
    conditioning it on one compensating loading/deviation decomposition.
    """

    keys = collapsed_gaussian_loading_keys(compiled)
    if not keys:
        return {}, steps
    centered = fs.to_centered(path, params)
    channel_index = {
        name: index for index, name in enumerate(compiled.channel_names)
    }
    accepted: dict[str, bool] = {}

    for key in keys:
        factor_name, channel_name = _factor_and_channel(key)
        factor_block = _block(fs, kind="factor", name=factor_name)
        channel_block = _block(fs, kind="channel", name=channel_name)
        process = channel_block.process_by_component["level"]
        label = f"signed_sd.{process}"
        if isinstance(priors.process[process], FixedSD):
            continue
        factor_path = _block_predictor(centered, factor_block)
        seasonal_path = _block_predictor(
            centered, channel_block, remove_level=True
        )
        response = (
            np.asarray(y[:, channel_index[channel_name]], dtype=float)
            - seasonal_path
        )
        intercept_prior = priors.intercept[channel_name]
        loading_prior = compiled.loading_specs[key]
        sigma2 = float(params[f"sigma.{channel_name}"]) ** 2
        current_signed = float(params[label])
        sign = -1.0 if current_signed < 0.0 else 1.0
        current = max(abs(current_signed), 1e-14)
        proposal = float(np.exp(np.log(current) + steps[label] * rng.normal()))

        def target(magnitude: float) -> float:
            regression = _RegressionRandomWalk(
                factor=factor_path,
                deviation_sd=magnitude,
                intercept_mean=float(intercept_prior.mean),
                intercept_sd=float(intercept_prior.sd),
                loading_mean=float(loading_prior.prior_mean),
                loading_sd=float(loading_prior.prior_sd),
            )
            likelihood = kalman_filter(
                response,
                regression,
                {},
                observation_variance=sigma2,
            ).log_likelihood
            return float(
                likelihood
                + _signed_process_prior_logpdf(
                    process,
                    sign * magnitude,
                    priors,
                    horseshoe_state,
                    triple_gamma_state,
                )
                + np.log(magnitude)
            )

        take = bool(np.log(rng.random()) < target(proposal) - target(current))
        if take:
            params[label] = sign * proposal
            params[f"sd.{process}"] = proposal
        accepted[label] = take
        if adapt:
            steps[label] = _adapt(steps[label], take, iteration)
    return accepted, steps


def collapsed_gaussian_loading_sweep(
    y: Array,
    path: Array,
    compiled: CompiledFactorModel,
    fs: "CompiledFactorFS",
    params: dict[str, float],
    priors: FactorPriors,
    rng: np.random.Generator,
) -> tuple[Array, dict[str, bool]]:
    """Jointly draw Gaussian-channel ``(c, loading, deviation[0:T])``.

    Conditional on the shared factor and any seasonal state, this is a
    three-state linear Gaussian model.  The returned loading draw is therefore
    marginal with respect to the idiosyncratic path rather than conditional on
    its previous MCMC value.
    """

    keys = collapsed_gaussian_loading_keys(compiled)
    if not keys:
        return np.asarray(path, dtype=float), {}
    centered = fs.to_centered(path, params)
    channel_index = {
        name: index for index, name in enumerate(compiled.channel_names)
    }
    outcomes: dict[str, bool] = {}

    for key in keys:
        factor_name, channel_name = _factor_and_channel(key)
        factor_block = _block(fs, kind="factor", name=factor_name)
        channel_block = _block(fs, kind="channel", name=channel_name)
        factor_path = _block_predictor(centered, factor_block)
        seasonal_path = _block_predictor(
            centered, channel_block, remove_level=True
        )
        response = (
            np.asarray(y[:, channel_index[channel_name]], dtype=float)
            - seasonal_path
        )
        process = channel_block.process_by_component["level"]
        intercept_prior = priors.intercept[channel_name]
        loading_prior = compiled.loading_specs[key]
        regression = _RegressionRandomWalk(
            factor=factor_path,
            deviation_sd=float(params[f"sd.{process}"]),
            intercept_mean=float(intercept_prior.mean),
            intercept_sd=float(intercept_prior.sd),
            loading_mean=float(loading_prior.prior_mean),
            loading_sd=float(loading_prior.prior_sd),
        )
        block_path, _ = ffbs(
            response,
            regression,
            {},
            rng,
            observation_variance=float(params[f"sigma.{channel_name}"]) ** 2,
        )
        intercept = float(block_path[0, 0])
        loading = float(block_path[0, 1])
        deviation = np.asarray(block_path[:, 2], dtype=float)

        params[f"intercept.{channel_name}"] = intercept
        params[key] = loading
        local = centered[:, channel_block.centered_slice]
        local[:, int(channel_block.layout.idx_alpha)] = intercept + deviation
        outcomes[key] = True

    return fs.from_centered(centered, params), outcomes


def _loading_prior_logpdf(value: float, specification) -> float:
    return float(
        -np.log(float(specification.prior_sd))
        - 0.5
        * (
            (float(value) - float(specification.prior_mean))
            / float(specification.prior_sd)
        )
        ** 2
    )


def _random_walk_log_kernel(level: Array, sd: float) -> float:
    if sd <= 0.0:
        return -np.inf
    innovations = np.diff(np.asarray(level, dtype=float))
    return float(-0.5 * np.sum((innovations / float(sd)) ** 2))


def loading_deviation_interweave_sweep(
    path: Array,
    compiled: CompiledFactorModel,
    fs: "CompiledFactorFS",
    params: dict[str, float],
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    keys: tuple[str, ...],
    adapt: bool,
    iteration: int,
) -> tuple[Array, dict[str, bool], dict[str, float], tuple[str, ...]]:
    """Move a loading and channel deviation without changing the predictor.

    For a proposed ``loading' = loading + delta`` the semantic channel level
    is transformed as ``level'_t = level_t - delta * factor_t``.  The complete
    observation predictor and, for a GEV channel, its support are unchanged.
    Acceptance therefore depends only on the loading and state-evolution
    priors.  Channels with an exactly fixed zero deviation SD are returned as
    unhandled so the caller can use its ordinary path-conditional update.
    """

    if not keys:
        return np.asarray(path, dtype=float), {}, steps, ()
    centered = fs.to_centered(path, params)
    accepted: dict[str, bool] = {}
    handled: list[str] = []

    for key in keys:
        factor_name, channel_name = _factor_and_channel(key)
        factor_block = _block(fs, kind="factor", name=factor_name)
        channel_block = _block(fs, kind="channel", name=channel_name)
        process = channel_block.process_by_component["level"]
        deviation_sd = float(params[f"sd.{process}"])
        if deviation_sd <= 1e-14:
            continue

        current = float(params[key])
        proposal = float(current + steps[key] * rng.normal())
        delta = proposal - current
        factor_level = centered[
            :, factor_block.centered_slice
        ][:, int(factor_block.layout.idx_alpha)]
        channel_level = centered[
            :, channel_block.centered_slice
        ][:, int(channel_block.layout.idx_alpha)]
        proposal_level = channel_level - delta * factor_level
        specification = compiled.loading_specs[key]
        current_target = _random_walk_log_kernel(
            channel_level, deviation_sd
        ) + _loading_prior_logpdf(current, specification)
        proposal_target = _random_walk_log_kernel(
            proposal_level, deviation_sd
        ) + _loading_prior_logpdf(proposal, specification)
        take = bool(
            np.isfinite(proposal_target)
            and np.log(rng.random()) < proposal_target - current_target
        )
        if take:
            params[key] = proposal
            centered[:, channel_block.centered_slice][
                :, int(channel_block.layout.idx_alpha)
            ] = proposal_level
        accepted[key] = take
        handled.append(key)
        if adapt:
            steps[key] = _adapt(steps[key], take, iteration)

    return (
        fs.from_centered(centered, params),
        accepted,
        steps,
        tuple(handled),
    )


__all__ = [
    "collapsed_gaussian_intercept_keys",
    "collapsed_gaussian_loading_keys",
    "collapsed_gaussian_processes",
    "collapsed_gaussian_scale_sweep",
    "collapsed_gaussian_loading_sweep",
    "loading_deviation_interweave_sweep",
]
