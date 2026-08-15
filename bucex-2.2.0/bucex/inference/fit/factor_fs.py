"""Fruehwirth--Schnatter inference for the v2.1 one-factor model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from scipy.linalg import block_diag

from ...components import DummySeasonal, LocalLevel, LocalLinearTrend
from ...core.fit import FitResult
from ...models.factor_compiler import CompiledFactorModel
from ...models.structural import Model
from ...observation import Gaussian
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
from ._progress import (
    factor_progress_parameters,
    mcmc_progress_line,
    progress_interval,
    should_report_progress,
)
from .factor import _initial_parameters, _loading_sweep, _observation_sweep
from .factor_horseshoe import (
    factor_horseshoe_coefficient_logpdf,
    factor_horseshoe_logpdf,
    initialise_factor_horseshoe,
    update_factor_horseshoe,
)
from .factor_triple_gamma import (
    factor_triple_gamma_coefficient_logpdf,
    factor_triple_gamma_logpdf,
    initialise_factor_triple_gamma,
    update_factor_triple_gamma,
)
from .factor_loading import (
    collapsed_gaussian_intercept_keys,
    collapsed_gaussian_loading_keys,
    collapsed_gaussian_loading_sweep,
    collapsed_gaussian_processes,
    collapsed_gaussian_scale_sweep,
    loading_deviation_interweave_sweep,
)
from .fs_utils import (
    NCPLayout,
    _centered_innovations_by_component,
    baseline_mu_path,
    build_ncp_system,
    infer_ncp_layout,
    map_centered_to_ncp,
    map_ncp_to_centered,
    measurement_vector,
)


Array = np.ndarray


@dataclass(frozen=True)
class FactorFSBlock:
    kind: str
    name: str
    compiled: Any
    centered_slice: slice
    ncp_slice: slice
    layout: NCPLayout
    process_by_component: Mapping[str, str]
    alpha0_key: str
    beta0_key: str | None
    beta0_fixed: bool
    beta0_value: float | None
    seasonal_keys: tuple[str, ...]


def _fs_submodel(block) -> Model:
    components = []
    for component in block.compiled.model.components:
        if isinstance(component, LocalLevel):
            components.append(
                LocalLinearTrend(
                    level_mode="dynamic",
                    trend_mode="off",
                    level_name=component.name,
                    initial_level=component.initial_mean,
                    initial_level_sd=component.initial_sd,
                )
            )
        else:
            components.append(component)
    return Model(Gaussian(), tuple(components), name=f"fs.{block.kind}.{block.name}")


class CompiledFactorFS:
    """Unit-innovation state graph with parameter-dependent observation design."""

    def __init__(self, centered: CompiledFactorModel):
        if not centered.model.supports_fs_parameterization:
            raise ValueError(
                "The FS factor compiler requires one shared local-linear trend, "
                "one dynamic channel local level, and optional dynamic dummy "
                "seasonality per channel."
            )
        self.centered = centered
        self.model = centered.model
        self.n_time = centered.n_time
        self.exog = None

        transitions: list[Array] = [np.ones((1, 1))]
        loadings: list[Array] = [np.zeros((1, 0))]
        state_names = ["fs.constant"]
        noise_names: list[str] = []
        blocks: list[FactorFSBlock] = []
        position = 1

        for block in centered.blocks:
            submodel = _fs_submodel(block)
            layout = infer_ncp_layout(submodel)
            if tuple(layout.centered_state_names) != tuple(block.compiled.state_names):
                raise RuntimeError(
                    f"FS/centered state ordering differs for {block.kind}.{block.name}."
                )
            transition, covariance = build_ncp_system(layout)
            active = np.flatnonzero(np.diag(covariance) > 0.0)
            local_loading = np.eye(layout.ncp_state_dim)[:, active]
            ncp_slice = slice(position, position + layout.ncp_state_dim)
            position = ncp_slice.stop

            process_by_component: dict[str, str] = {}
            original_components = block.compiled.model.components
            trend = next(
                component
                for component in original_components
                if isinstance(component, (LocalLevel, LocalLinearTrend))
            )
            if isinstance(trend, LocalLevel):
                process_by_component["level"] = (
                    f"channel.{block.name}.{trend.name}"
                )
            else:
                if trend.level_mode == "dynamic":
                    process_by_component["level"] = (
                        f"factor.{block.name}.{trend.level_name}"
                    )
                if trend.trend_mode == "dynamic":
                    process_by_component["trend"] = (
                        f"factor.{block.name}.{trend.slope_name}"
                    )
            seasonal = next(
                (
                    component
                    for component in original_components
                    if isinstance(component, DummySeasonal)
                    and component.mode != "off"
                ),
                None,
            )
            if seasonal is not None and seasonal.mode == "dynamic":
                process_by_component["season"] = (
                    f"{block.kind}.{block.name}.{seasonal.name}"
                )

            alpha0_key = (
                f"initial.factor.{block.name}.level"
                if block.kind == "factor"
                else f"intercept.{block.name}"
            )
            beta0_key = (
                f"initial.factor.{block.name}.slope"
                if block.kind == "factor" and layout.has_beta
                else None
            )
            beta0_fixed = bool(
                block.kind == "factor"
                and isinstance(trend, LocalLinearTrend)
                and trend.trend_mode != "off"
                and trend.initial_slope_sd is not None
                and np.isclose(float(trend.initial_slope_sd), 0.0)
            )
            beta0_value = (
                float(trend.initial_slope)
                if block.kind == "factor"
                and isinstance(trend, LocalLinearTrend)
                and trend.trend_mode != "off"
                else None
            )
            seasonal_keys = tuple(
                f"initial_seasonal.{block.name}[{index + 1}]"
                for index in range(layout.season_dim)
            )
            blocks.append(
                FactorFSBlock(
                    kind=block.kind,
                    name=block.name,
                    compiled=block.compiled,
                    centered_slice=block.state_slice,
                    ncp_slice=ncp_slice,
                    layout=layout,
                    process_by_component=process_by_component,
                    alpha0_key=alpha0_key,
                    beta0_key=beta0_key,
                    beta0_fixed=beta0_fixed,
                    beta0_value=beta0_value,
                    seasonal_keys=seasonal_keys,
                )
            )
            transitions.append(transition)
            loadings.append(local_loading)
            state_names.extend(
                f"fs.{block.kind}.{block.name}.{name}"
                for name in layout.ncp_state_names
            )
            noise_names.extend(
                f"fs.{block.kind}.{block.name}.{layout.ncp_state_names[index]}"
                for index in active
            )

        self.blocks = tuple(blocks)
        self.transition = block_diag(*transitions)
        rows = sum(item.shape[0] for item in loadings)
        columns = sum(item.shape[1] for item in loadings)
        self.loading = np.zeros((rows, columns), dtype=float)
        row = column = 0
        for item in loadings:
            nrow, ncolumn = item.shape
            self.loading[row : row + nrow, column : column + ncolumn] = item
            row += nrow
            column += ncolumn
        self.initial_mean = np.zeros(rows, dtype=float)
        self.initial_mean[0] = 1.0
        self.initial_cov = np.zeros((rows, rows), dtype=float)
        self.state_names = tuple(state_names)
        self.noise_names = tuple(noise_names)

    @property
    def state_dim(self) -> int:
        return int(self.transition.shape[0])

    @property
    def noise_dim(self) -> int:
        return int(self.loading.shape[1])

    @property
    def family(self) -> str:
        return self.centered.family

    @property
    def all_gaussian(self) -> bool:
        return self.centered.all_gaussian

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self.centered.channel_names

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self.centered.factor_names

    @property
    def observation_parameter_names(self) -> tuple[str, ...]:
        return self.centered.observation_parameter_names

    @property
    def loading_names(self) -> tuple[str, ...]:
        return self.centered.loading_names

    @property
    def estimated_loading_names(self) -> tuple[str, ...]:
        return self.centered.estimated_loading_names

    @property
    def loading_specs(self):
        return self.centered.loading_specs

    def process_vector(self, params: Mapping[str, float]) -> Array:
        return np.ones(self.noise_dim, dtype=float)

    def transition_cov(self, params: Mapping[str, float]) -> Array:
        return self.loading @ self.loading.T

    def _block_params(
        self, block: FactorFSBlock, params: Mapping[str, float]
    ) -> dict[str, Any]:
        output: dict[str, Any] = {
            "alpha0": float(params[block.alpha0_key]),
            "beta0": (
                0.0 if block.beta0_key is None else float(params[block.beta0_key])
            ),
            "s_level": float(
                params[
                    f"signed_sd.{block.process_by_component['level']}"
                ]
            ),
            "s_trend": 0.0,
            "s_season": 0.0,
        }
        if "trend" in block.process_by_component:
            output["s_trend"] = float(
                params[
                    f"signed_sd.{block.process_by_component['trend']}"
                ]
            )
        if "season" in block.process_by_component:
            output["s_season"] = float(
                params[
                    f"signed_sd.{block.process_by_component['season']}"
                ]
            )
        if block.layout.season_dim:
            output["gamma0_season"] = np.asarray(
                [params[key] for key in block.seasonal_keys], dtype=float
            )
        return output

    def design(
        self,
        n_time: int | None = None,
        exog: Any = None,
        *,
        params: Mapping[str, float] | None = None,
    ) -> Array:
        if params is None:
            raise ValueError("FS factor design requires static parameters.")
        if exog is not None:
            raise ValueError("The v2.1 FS factor model does not use exogenous regressors.")
        n = self.n_time if n_time is None else int(n_time)
        output = np.zeros((n, len(self.channel_names), self.state_dim), dtype=float)
        channel_index = {name: index for index, name in enumerate(self.channel_names)}
        for block in self.blocks:
            local_params = self._block_params(block, params)
            baseline = baseline_mu_path(n, local_params, block.layout)
            measurement = measurement_vector(local_params, block.layout)
            if block.kind == "factor":
                for channel in self.channel_names:
                    coefficient = self.centered._loading_value(
                        block.name, channel, params
                    )
                    index = channel_index[channel]
                    output[:, index, 0] += coefficient * baseline
                    output[:, index, block.ncp_slice] += coefficient * measurement
            else:
                index = channel_index[block.name]
                output[:, index, 0] += baseline
                output[:, index, block.ncp_slice] += measurement
        return output

    def eta(
        self,
        path: Array,
        exog: Any = None,
        *,
        params: Mapping[str, float] | None = None,
    ) -> Array:
        path = np.asarray(path, dtype=float)
        if path.shape != (path.shape[0], self.state_dim) or path.shape[0] < 2:
            raise ValueError("path must have shape (T+1, fs_state_dim).")
        design = self.design(path.shape[0] - 1, exog=exog, params=params)
        return np.einsum("tpm,tm->tp", design, path[1:])

    def observation_variance(
        self, params: Mapping[str, float], n_time: int | None = None
    ) -> Array:
        return self.centered.observation_variance(params, n_time)

    def observation_log_likelihood(self, y, eta, params) -> float:
        return self.centered.observation_log_likelihood(y, eta, params)

    def observation_logweights(self, y_t, particles, design_t, params) -> Array:
        return self.centered.observation_logweights(
            y_t, particles, design_t, params
        )

    def observation_derivatives(self, y, eta, params):
        return self.centered.observation_derivatives(y, eta, params)

    def sample_observation(self, eta, params, rng):
        return self.centered.sample_observation(eta, params, rng)

    def project_path(self, path: Array, params: Mapping[str, float]) -> Array:
        values = np.asarray(path, dtype=float).copy()
        if values.ndim != 2 or values.shape[1] != self.state_dim:
            raise ValueError("FS path has the wrong shape.")
        values[0] = self.initial_mean
        for index in range(1, values.shape[0]):
            mean = self.transition @ values[index - 1]
            residual = values[index] - mean
            disturbance, *_ = np.linalg.lstsq(self.loading, residual, rcond=None)
            values[index] = mean + self.loading @ disturbance
        return values

    def to_centered(self, path: Array, params: Mapping[str, float]) -> Array:
        path = np.asarray(path, dtype=float)
        output = np.zeros(
            (path.shape[0], self.centered.state_dim), dtype=float
        )
        for block in self.blocks:
            output[:, block.centered_slice] = map_ncp_to_centered(
                path[:, block.ncp_slice],
                self._block_params(block, params),
                block.layout,
            )
        return output

    def from_centered(self, path: Array, params: Mapping[str, float]) -> Array:
        path = np.asarray(path, dtype=float)
        output = np.zeros((path.shape[0], self.state_dim), dtype=float)
        output[:, 0] = 1.0
        for block in self.blocks:
            output[:, block.ncp_slice] = map_centered_to_ncp(
                path[:, block.centered_slice],
                self._block_params(block, params),
                block.layout,
            )
        return output

    def process_location(self, process: str) -> tuple[FactorFSBlock, str]:
        for block in self.blocks:
            for component, name in block.process_by_component.items():
                if name == process:
                    return block, component
        raise KeyError(process)

    def sign_switch(
        self,
        path: Array,
        params: dict[str, float],
        rng: np.random.Generator,
    ) -> tuple[Array, dict[str, float]]:
        output = np.asarray(path, dtype=float).copy()
        for block in self.blocks:
            layout = block.layout
            for component, process in block.process_by_component.items():
                if rng.random() >= 0.5:
                    continue
                key = f"signed_sd.{process}"
                params[key] = -float(params[key])
                local = output[:, block.ncp_slice]
                if component == "level":
                    local[:, int(layout.idx_tilde_alpha)] *= -1.0
                elif component == "trend":
                    local[:, int(layout.idx_tilde_beta)] *= -1.0
                    local[:, int(layout.idx_A)] *= -1.0
                elif component == "season":
                    local[:, layout.season_ncp_slice] *= -1.0
        return output, params


def _normal_logpdf(value: float, mean: float, sd: float) -> float:
    return float(-np.log(sd) - 0.5 * ((float(value) - mean) / sd) ** 2)


def _symmetric_process_logpdf(prior, value: float) -> float:
    return _prior_logpdf(prior, abs(float(value)))


def _sync_signed_scales(params: dict[str, float], compiled: CompiledFactorModel) -> None:
    for name in compiled.noise_names:
        params[f"sd.{name}"] = abs(float(params[f"signed_sd.{name}"]))


def _fs_static_parameter_names(
    compiled: CompiledFactorModel, fs: CompiledFactorFS
) -> list[str]:
    names: list[str] = []
    for block in fs.blocks:
        names.append(block.alpha0_key)
        if block.beta0_key is not None:
            names.append(block.beta0_key)
        names.extend(block.seasonal_keys)
    names.extend(f"signed_sd.{name}" for name in compiled.noise_names)
    return names


def _initial_fs_parameters(
    compiled: CompiledFactorModel,
    fs: CompiledFactorFS,
    priors: FactorPriors,
    rng: np.random.Generator,
    initial: Mapping[str, Any] | None,
) -> dict[str, float]:
    base_names = {
        *(f"sd.{name}" for name in compiled.noise_names),
        *compiled.observation_parameter_names,
        *compiled.loading_names,
    }
    supplied = {} if initial is None else dict(initial)
    base_initial = {name: supplied[name] for name in supplied if name in base_names}
    params = _initial_parameters(compiled, priors, rng, base_initial)

    factor = compiled.model.factors[0]
    factor_trend = next(
        component
        for component in factor.components
        if isinstance(component, LocalLinearTrend)
    )
    for block in fs.blocks:
        if block.kind == "factor":
            params[block.alpha0_key] = float(factor_trend.initial_level or 0.0)
            if block.beta0_key is not None:
                params[block.beta0_key] = float(
                    block.beta0_value
                    if block.beta0_fixed
                    else priors.factor_initial_slope[block.name].mean
                )
        else:
            params[block.alpha0_key] = float(priors.intercept[block.name].mean)
        if block.seasonal_keys:
            seasonal_prior = priors.seasonal_initial[block.name]
            for key, value in zip(block.seasonal_keys, seasonal_prior.mean_array()):
                params[key] = float(value)

    for name in compiled.noise_names:
        signed_key = f"signed_sd.{name}"
        params[signed_key] = float(params[f"sd.{name}"])

    allowed = base_names | set(_fs_static_parameter_names(compiled, fs))
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(f"Unknown initial FS factor parameters: {unknown}.")
    for name, value in supplied.items():
        if name not in allowed or name in base_names:
            continue
        if name.startswith("initial.factor.") and name.endswith(".level"):
            if not np.isclose(float(value), 0.0):
                raise ValueError("The shared factor initial level is fixed to zero.")
            continue
        fixed_beta = next(
            (
                block
                for block in fs.blocks
                if block.beta0_key == name and block.beta0_fixed
            ),
            None,
        )
        if fixed_beta is not None:
            if not np.isclose(float(value), float(fixed_beta.beta0_value)):
                raise ValueError(
                    f"{name} is fixed at {float(fixed_beta.beta0_value):g} "
                    "by LocalLinearTrend.initial_slope_sd=0."
                )
            continue
        params[name] = float(value)
    for name in compiled.noise_names:
        signed_key = f"signed_sd.{name}"
        sd_key = f"sd.{name}"
        if signed_key in supplied and sd_key in supplied and not np.isclose(
            abs(float(supplied[signed_key])), float(supplied[sd_key])
        ):
            raise ValueError(f"Conflicting initial {signed_key} and {sd_key}.")
    _sync_signed_scales(params, compiled)
    return params


def _mutable_fs_parameters(
    compiled: CompiledFactorModel,
    fs: CompiledFactorFS,
) -> list[str]:
    names: list[str] = []
    for block in fs.blocks:
        if block.kind == "channel":
            names.append(block.alpha0_key)
        if block.beta0_key is not None and not block.beta0_fixed:
            names.append(block.beta0_key)
        names.extend(block.seasonal_keys)
    # Fixed-process filtering is performed with the actual prior in the sweep;
    # retaining every name keeps diagnostics and saved schemas stable.
    names.extend(f"signed_sd.{name}" for name in compiled.noise_names)
    return names


def _static_prior_logpdf(
    key: str,
    value: float,
    compiled: CompiledFactorModel,
    priors: FactorPriors,
    horseshoe_state: Mapping[str, Any],
    triple_gamma_state: Mapping[str, Any],
) -> float:
    if key.startswith("intercept."):
        channel = key.removeprefix("intercept.")
        prior = priors.intercept[channel]
        return _normal_logpdf(value, prior.mean, prior.sd)
    if key.startswith("initial.factor.") and key.endswith(".slope"):
        factor = key.removeprefix("initial.factor.").removesuffix(".slope")
        prior = priors.factor_initial_slope[factor]
        return _normal_logpdf(value, prior.mean, prior.sd)
    if key.startswith("initial_seasonal."):
        channel, index = key.removeprefix("initial_seasonal.").split("[")
        position = int(index.removesuffix("]")) - 1
        prior = priors.seasonal_initial[channel]
        return _normal_logpdf(
            value,
            float(prior.mean_array()[position]),
            float(prior.sd_array()[position]),
        )
    if key.startswith("signed_sd."):
        process = key.removeprefix("signed_sd.")
        if process in priors.horseshoe_processes:
            return factor_horseshoe_coefficient_logpdf(
                value, process, priors, horseshoe_state
            )
        if process in priors.triple_gamma_processes:
            return factor_triple_gamma_coefficient_logpdf(
                value, process, priors, triple_gamma_state
            )
        return _symmetric_process_logpdf(priors.process[process], value)
    raise KeyError(key)


def _fs_static_sweep(
    y: Array,
    path: Array,
    compiled: CompiledFactorModel,
    fs: CompiledFactorFS,
    params: dict[str, float],
    priors: FactorPriors,
    horseshoe_state: Mapping[str, Any],
    triple_gamma_state: Mapping[str, Any],
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    adapt: bool,
    iteration: int,
) -> tuple[dict[str, bool], dict[str, float]]:
    current_likelihood = observation_log_likelihood(
        y, fs.eta(path, params=params), fs, params
    )
    accepted: dict[str, bool] = {}
    blocked_intercepts = collapsed_gaussian_intercept_keys(compiled)
    blocked_processes = collapsed_gaussian_processes(compiled, fs)
    for key in _mutable_fs_parameters(compiled, fs):
        if key in blocked_intercepts:
            continue
        if key.startswith("signed_sd."):
            process = key.removeprefix("signed_sd.")
            if process in blocked_processes:
                continue
            if isinstance(priors.process[process], FixedSD):
                accepted[key] = False
                continue
        current = float(params[key])
        proposal = float(current + steps[key] * rng.normal())
        proposal_params = dict(params)
        proposal_params[key] = proposal
        if key.startswith("signed_sd."):
            process = key.removeprefix("signed_sd.")
            proposal_params[f"sd.{process}"] = abs(proposal)
        proposal_likelihood = observation_log_likelihood(
            y,
            fs.eta(path, params=proposal_params),
            fs,
            proposal_params,
        )
        current_target = current_likelihood + _static_prior_logpdf(
            key, current, compiled, priors, horseshoe_state,
            triple_gamma_state,
        )
        proposal_target = proposal_likelihood + _static_prior_logpdf(
            key, proposal, compiled, priors, horseshoe_state,
            triple_gamma_state,
        )
        take = bool(
            np.isfinite(proposal_target)
            and np.log(rng.random()) < proposal_target - current_target
        )
        if take:
            params.update(proposal_params)
            current_likelihood = proposal_likelihood
        accepted[key] = take
        if adapt:
            steps[key] = _adapt(steps[key], take, iteration)
    return accepted, steps


def _fs_asis_scale_sweep(
    path: Array,
    compiled: CompiledFactorModel,
    fs: CompiledFactorFS,
    params: dict[str, float],
    priors: FactorPriors,
    horseshoe_state: Mapping[str, Any],
    triple_gamma_state: Mapping[str, Any],
    steps: dict[str, float],
    rng: np.random.Generator,
    *,
    adapt: bool,
    iteration: int,
    skip_processes: frozenset[str] = frozenset(),
) -> tuple[Array, dict[str, bool], dict[str, float]]:
    centered = fs.to_centered(path, params)
    accepted: dict[str, bool] = {}
    for process in compiled.noise_names:
        if process in skip_processes:
            continue
        prior = priors.process[process]
        label = f"asis.signed_sd.{process}"
        if isinstance(prior, FixedSD):
            accepted[label] = False
            continue
        block, component = fs.process_location(process)
        innovations = _centered_innovations_by_component(
            centered[:, block.centered_slice], block.layout
        )[component]
        current_signed = float(params[f"signed_sd.{process}"])
        current = max(abs(current_signed), 1e-14)
        sign = -1.0 if current_signed < 0.0 else 1.0
        proposal = float(np.exp(np.log(current) + steps[label] * rng.normal()))
        sum_squares = float(np.sum(np.asarray(innovations) ** 2))
        n = int(np.asarray(innovations).size)

        def target(magnitude: float) -> float:
            signed = sign * magnitude
            return float(
                -n * np.log(magnitude)
                - 0.5 * sum_squares / magnitude**2
                + _static_prior_logpdf(
                    f"signed_sd.{process}",
                    signed,
                    compiled,
                    priors,
                    horseshoe_state,
                    triple_gamma_state,
                )
                + np.log(magnitude)
            )

        take = bool(np.log(rng.random()) < target(proposal) - target(current))
        if take:
            params[f"signed_sd.{process}"] = sign * proposal
            params[f"sd.{process}"] = proposal
        accepted[label] = take
        if adapt:
            steps[label] = _adapt(steps[label], take, iteration)
    transformed = fs.from_centered(centered, params)
    return transformed, accepted, steps


def _fs_log_posterior(
    y: Array,
    path: Array,
    compiled: CompiledFactorModel,
    fs: CompiledFactorFS,
    params: dict[str, float],
    priors: FactorPriors,
    horseshoe_state: Mapping[str, Any],
    triple_gamma_state: Mapping[str, Any],
) -> float:
    value = joint_state_log_density(y, path, fs, params)
    for key in _mutable_fs_parameters(compiled, fs):
        if not key.startswith("signed_sd."):
            value += _static_prior_logpdf(
                key, params[key], compiled, priors, horseshoe_state,
                triple_gamma_state,
            )
    for process in compiled.noise_names:
        if (
            process not in priors.horseshoe_processes
            and process not in priors.triple_gamma_processes
        ):
            value += _static_prior_logpdf(
                f"signed_sd.{process}",
                params[f"signed_sd.{process}"],
                compiled,
                priors,
                horseshoe_state,
                triple_gamma_state,
            )
    if priors.horseshoe is not None:
        value += factor_horseshoe_logpdf(
            {
                process: params[f"signed_sd.{process}"]
                for process in priors.horseshoe_processes
            },
            priors,
            horseshoe_state,
        )
    if priors.triple_gamma is not None:
        value += factor_triple_gamma_logpdf(
            {
                process: params[f"signed_sd.{process}"]
                for process in priors.triple_gamma_processes
            },
            priors,
            triple_gamma_state,
        )
    for channel in compiled.model.channels:
        value += _prior_logpdf(
            priors.observation_sd[channel.name], params[f"sigma.{channel.name}"]
        )
        if channel.family == "gev":
            value += priors.shape[channel.name].logpdf(params[f"xi.{channel.name}"])
    for key in compiled.estimated_loading_names:
        specification = compiled.loading_specs[key]
        value += _normal_logpdf(
            params[key], specification.prior_mean, specification.prior_sd
        )
    return float(value)


def _initial_fs_path(
    y: Array,
    fs: CompiledFactorFS,
    params: dict[str, float],
    laplace: Laplace,
    rng: np.random.Generator,
) -> Array:
    if fs.all_gaussian:
        return ffbs(y, fs, params, rng)[0]
    return iterated_laplace(
        y,
        fs,
        params,
        rng,
        max_iterations=laplace.max_iterations,
        tolerance=laplace.tolerance,
        curvature_floor=laplace.curvature_floor,
        maximum_variance=laplace.maximum_variance,
        draw_attempts=laplace.draw_attempts,
    ).path


def sample_factor_fs_posterior(
    y: Array,
    compiled: CompiledFactorModel,
    priors: FactorPriors,
    plan: InferencePlan,
    *,
    mcmc: MCMC,
    particles: Particles,
    laplace: Laplace,
    dates: Array | None = None,
    initial_parameters: Mapping[str, Any] | None = None,
) -> FitResult:
    """Sample the one-factor posterior in the full FS NCP."""

    if plan.parameterization != "fruehwirth_schnatter":
        raise ValueError("sample_factor_fs_posterior requires the FS plan.")
    y = np.asarray(y, dtype=float)
    expected = (compiled.n_time, len(compiled.channel_names))
    if y.shape != expected:
        raise ValueError(f"y must have shape {expected}.")
    fs = CompiledFactorFS(compiled)
    chains, draws = int(mcmc.chains), int(mcmc.draws)
    state_draws = np.zeros(
        (chains, draws, compiled.n_time + 1, compiled.state_dim), dtype=float
    )
    ncp_draws = np.zeros(
        (chains, draws, compiled.n_time + 1, fs.state_dim), dtype=float
    )
    log_posterior = np.zeros((chains, draws), dtype=float)

    static_names = _fs_static_parameter_names(compiled, fs)
    horseshoe_names = (
        [
            "horseshoe.global",
            "horseshoe.slab2",
            *(f"horseshoe.local.{name}" for name in priors.horseshoe_processes),
        ]
        if priors.horseshoe is not None
        else []
    )
    triple_gamma_names = (
        [
            "triple_gamma.global",
            "triple_gamma.a",
            "triple_gamma.c",
            *(
                ["triple_gamma.slab2"]
                if priors.triple_gamma.regularized
                else []
            ),
            *(
                f"triple_gamma.numerator.{name}"
                for name in priors.triple_gamma_processes
            ),
            *(
                f"triple_gamma.denominator.{name}"
                for name in priors.triple_gamma_processes
            ),
            *(
                f"triple_gamma.rho.{name}"
                for name in priors.triple_gamma_processes
            ),
        ]
        if priors.triple_gamma is not None
        else []
    )
    triple_gamma_update_names = (
        [
            *(
                ["triple_gamma.global"]
                if priors.triple_gamma.learn_global
                else []
            ),
            *(
                ["triple_gamma.a", "triple_gamma.c"]
                if priors.triple_gamma.learn_shapes
                else []
            ),
            *(
                ["triple_gamma.slab2"]
                if priors.triple_gamma.regularized
                else []
            ),
            *(
                f"triple_gamma.numerator.{name}"
                for name in priors.triple_gamma_processes
            ),
            *(
                f"triple_gamma.denominator.{name}"
                for name in priors.triple_gamma_processes
            ),
        ]
        if priors.triple_gamma is not None
        else []
    )
    parameter_names = list(
        dict.fromkeys(
            [
                *(f"sd.{name}" for name in compiled.noise_names),
                *static_names,
                *compiled.observation_parameter_names,
                *compiled.loading_names,
                *horseshoe_names,
                *triple_gamma_names,
            ]
        )
    )
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
    draw_metrics = {
        name: np.full((chains, draws), np.nan, dtype=float)
        for name in metric_names
    }
    mutable_static = _mutable_fs_parameters(compiled, fs)
    collapsed_loading_keys = collapsed_gaussian_loading_keys(compiled)
    collapsed_intercepts = collapsed_gaussian_intercept_keys(compiled)
    collapsed_processes = collapsed_gaussian_processes(compiled, fs)
    acceptance_names = [
        *(name for name in mutable_static if name not in collapsed_intercepts),
        *compiled.observation_parameter_names,
        *compiled.estimated_loading_names,
        *horseshoe_names,
        *triple_gamma_update_names,
    ]
    if plan.asis:
        acceptance_names.extend(
            f"asis.signed_sd.{name}"
            for name in compiled.noise_names
            if name not in collapsed_processes
        )
    acceptance_names = list(dict.fromkeys(acceptance_names))
    acceptance_by_chain = {name: [] for name in acceptance_names}
    final_steps = {name: [] for name in acceptance_names}
    recorded_initial: list[dict[str, Any]] = []

    sequences = np.random.SeedSequence(mcmc.seed).spawn(chains)
    for chain, sequence in enumerate(sequences):
        rng = np.random.default_rng(sequence)
        horseshoe_state = initialise_factor_horseshoe(priors)
        triple_gamma_state = initialise_factor_triple_gamma(priors)
        params = _initial_fs_parameters(
            compiled, fs, priors, rng, initial_parameters
        )
        path = _initial_fs_path(y, fs, params, laplace, rng)
        recorded_initial.append(
            {
                "parameters": dict(params),
                "horseshoe": {
                    "local": dict(horseshoe_state.get("local", {})),
                    "global": horseshoe_state.get("global"),
                    "slab2": horseshoe_state.get("slab2"),
                },
                "triple_gamma": {
                    "numerator": dict(
                        triple_gamma_state.get("numerator", {})
                    ),
                    "denominator": dict(
                        triple_gamma_state.get("denominator", {})
                    ),
                    "global": triple_gamma_state.get("global"),
                    "a": triple_gamma_state.get("a"),
                    "c": triple_gamma_state.get("c"),
                    "slab2": triple_gamma_state.get("slab2"),
                },
            }
        )

        steps: dict[str, float] = {}
        for block in fs.blocks:
            if block.kind == "channel":
                steps[block.alpha0_key] = max(
                    0.02 * priors.intercept[block.name].sd, 1e-6
                )
            if block.beta0_key is not None and not block.beta0_fixed:
                steps[block.beta0_key] = max(
                    0.05 * priors.factor_initial_slope[block.name].sd, 1e-10
                )
            if block.seasonal_keys:
                prior = priors.seasonal_initial[block.name]
                for key, sd in zip(block.seasonal_keys, prior.sd_array()):
                    steps[key] = max(0.02 * float(sd), 1e-6)
        for name in compiled.noise_names:
            if name in priors.horseshoe_processes:
                reference = priors.horseshoe.coefficient_scale_for(name)
            elif name in priors.triple_gamma_processes:
                reference = priors.triple_gamma.coefficient_scale_for(name)
            else:
                reference = max(float(priors.process[name].initial()), 1e-10)
            steps[f"signed_sd.{name}"] = (
                0.25
                if name in collapsed_processes
                else max(0.15 * reference, 1e-10)
            )
            if plan.asis and name not in collapsed_processes:
                steps[f"asis.signed_sd.{name}"] = 0.20
        for channel in compiled.model.channels:
            steps[f"sigma.{channel.name}"] = 0.15
            if channel.family == "gev":
                steps[f"xi.{channel.name}"] = 0.18
        for key in compiled.estimated_loading_names:
            steps[key] = 0.10
        if priors.horseshoe is not None:
            steps["horseshoe.global"] = 0.25
            steps["horseshoe.slab2"] = 0.20
            steps.update(
                {
                    f"horseshoe.local.{name}": 0.35
                    for name in priors.horseshoe_processes
                }
            )
        if priors.triple_gamma is not None:
            steps["triple_gamma.global"] = 1.0
            steps["triple_gamma.a"] = 0.8
            steps["triple_gamma.c"] = 0.8
            if priors.triple_gamma.regularized:
                steps["triple_gamma.slab2"] = 0.8
            for name in priors.triple_gamma_processes:
                steps[f"triple_gamma.numerator.{name}"] = 1.0
                steps[f"triple_gamma.denominator.{name}"] = 1.0

        attempts = {name: 0 for name in acceptance_names}
        accepts = {name: 0 for name in acceptance_names}
        last_metrics = {name: np.nan for name in metric_names}
        saved = 0
        progress_every = progress_interval(
            mcmc.iterations, mcmc.progress_every
        )
        chain_started = perf_counter()

        for iteration in range(mcmc.iterations):
            if plan.engine == "ffbs":
                path = ffbs(y, fs, params, rng)[0]
            elif plan.engine == "laplace":
                state = iterated_laplace(
                    y,
                    fs,
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
                state = pgas(y, fs, params, path, particles=particles, rng=rng)
                path = state.path
                last_metrics.update(
                    particle_min_ess=float(np.min(state.ess[1:])),
                    particle_mean_unique_ancestors=float(
                        np.mean(state.unique_ancestors[1:])
                    ),
                    particle_path_changed=float(state.path_changed),
                    particle_changed_fraction=state.changed_fraction,
                )
            else:
                raise RuntimeError(f"Unhandled factor FS engine '{plan.engine}'.")

            adapting = bool(mcmc.adapt and iteration < mcmc.warmup)
            outcomes, steps = _fs_static_sweep(
                y,
                path,
                compiled,
                fs,
                params,
                priors,
                horseshoe_state,
                triple_gamma_state,
                steps,
                rng,
                adapt=adapting,
                iteration=iteration,
            )
            for key, outcome in outcomes.items():
                attempts[key] += 1
                accepts[key] += int(outcome)

            if plan.asis:
                path, outcomes, steps = _fs_asis_scale_sweep(
                    path,
                    compiled,
                    fs,
                    params,
                    priors,
                    horseshoe_state,
                    triple_gamma_state,
                    steps,
                    rng,
                    adapt=adapting,
                    iteration=iteration,
                    skip_processes=collapsed_processes,
                )
                for key, outcome in outcomes.items():
                    attempts[key] += 1
                    accepts[key] += int(outcome)

            outcomes, steps = _observation_sweep(
                y,
                path,
                fs,
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

            outcomes, steps = collapsed_gaussian_scale_sweep(
                y,
                path,
                compiled,
                fs,
                params,
                priors,
                horseshoe_state,
                triple_gamma_state,
                steps,
                rng,
                adapt=adapting,
                iteration=iteration,
            )
            for key, outcome in outcomes.items():
                attempts[key] += 1
                accepts[key] += int(outcome)

            path, outcomes = collapsed_gaussian_loading_sweep(
                y,
                path,
                compiled,
                fs,
                params,
                priors,
                rng,
            )
            for key, outcome in outcomes.items():
                attempts[key] += 1
                accepts[key] += int(outcome)

            remaining_loading_keys = tuple(
                key
                for key in compiled.estimated_loading_names
                if key not in collapsed_loading_keys
            )
            path, outcomes, steps, interwoven = (
                loading_deviation_interweave_sweep(
                    path,
                    compiled,
                    fs,
                    params,
                    steps,
                    rng,
                    keys=remaining_loading_keys,
                    adapt=adapting,
                    iteration=iteration,
                )
            )
            for key, outcome in outcomes.items():
                attempts[key] += 1
                accepts[key] += int(outcome)

            fallback_loading_keys = tuple(
                key for key in remaining_loading_keys if key not in interwoven
            )
            outcomes, steps = _loading_sweep(
                y,
                path,
                fs,
                params,
                steps,
                rng,
                adapt=adapting,
                iteration=iteration,
                keys=fallback_loading_keys,
            )
            for key, outcome in outcomes.items():
                attempts[key] += 1
                accepts[key] += int(outcome)

            if priors.horseshoe is not None:
                horseshoe_state, outcomes = update_factor_horseshoe(
                    {
                        name: params[f"signed_sd.{name}"]
                        for name in priors.horseshoe_processes
                    },
                    horseshoe_state,
                    priors,
                    rng,
                    steps=steps,
                )
                for key, outcome in outcomes.items():
                    attempts[key] += 1
                    accepts[key] += int(outcome)
            if priors.triple_gamma is not None:
                triple_gamma_state, outcomes = update_factor_triple_gamma(
                    {
                        name: params[f"signed_sd.{name}"]
                        for name in priors.triple_gamma_processes
                    },
                    triple_gamma_state,
                    priors,
                    rng,
                    widths=steps,
                )
                for key, outcome in outcomes.items():
                    attempts[key] += 1
                    accepts[key] += int(outcome)

            path, params = fs.sign_switch(path, params, rng)
            _sync_signed_scales(params, compiled)

            keep = iteration >= mcmc.warmup and (
                (iteration - mcmc.warmup) % mcmc.thin == 0
            )
            if keep:
                centered_path = fs.to_centered(path, params)
                state_draws[chain, saved] = centered_path
                ncp_draws[chain, saved] = path
                for name in parameter_names:
                    if name == "horseshoe.global":
                        value = horseshoe_state["global"]
                    elif name == "horseshoe.slab2":
                        value = horseshoe_state["slab2"]
                    elif name.startswith("horseshoe.local."):
                        value = horseshoe_state["local"][
                            name.removeprefix("horseshoe.local.")
                        ]
                    elif name == "triple_gamma.global":
                        value = triple_gamma_state["global"]
                    elif name == "triple_gamma.a":
                        value = triple_gamma_state["a"]
                    elif name == "triple_gamma.c":
                        value = triple_gamma_state["c"]
                    elif name == "triple_gamma.slab2":
                        value = triple_gamma_state["slab2"]
                    elif name.startswith("triple_gamma.numerator."):
                        process = name.removeprefix(
                            "triple_gamma.numerator."
                        )
                        value = triple_gamma_state["numerator"][process]
                    elif name.startswith("triple_gamma.denominator."):
                        process = name.removeprefix(
                            "triple_gamma.denominator."
                        )
                        value = triple_gamma_state["denominator"][process]
                    elif name.startswith("triple_gamma.rho."):
                        process = name.removeprefix("triple_gamma.rho.")
                        value = priors.triple_gamma.shrinkage_factor(
                            numerator=triple_gamma_state["numerator"][process],
                            denominator=triple_gamma_state["denominator"][process],
                            global_scale=triple_gamma_state["global"],
                        )
                    else:
                        value = params[name]
                    parameter_draws[name][chain, saved] = float(value)
                log_posterior[chain, saved] = _fs_log_posterior(
                    y,
                    path,
                    compiled,
                    fs,
                    params,
                    priors,
                    horseshoe_state,
                    triple_gamma_state,
                )
                for name, value in last_metrics.items():
                    draw_metrics[name][chain, saved] = value
                saved += 1

            completed = iteration + 1
            if mcmc.progress and should_report_progress(
                completed,
                total=mcmc.iterations,
                warmup=mcmc.warmup,
                every=progress_every,
            ):
                print(
                    mcmc_progress_line(
                        label="factor FS",
                        engine=plan.engine,
                        chain=chain + 1,
                        chains=chains,
                        completed=completed,
                        total=mcmc.iterations,
                        warmup=mcmc.warmup,
                        saved=saved,
                        draws=mcmc.draws,
                        elapsed=perf_counter() - chain_started,
                        parameters=factor_progress_parameters(
                            compiled,
                            params,
                            horseshoe_state=horseshoe_state,
                            triple_gamma_state=triple_gamma_state,
                        ),
                        metrics=last_metrics,
                        particles=particles.n if plan.engine == "pgas" else None,
                    ),
                    flush=True,
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
        "fs_state_names": list(fs.state_names),
        "loading_kernels": {
            key: (
                "collapsed_gaussian_ffbs"
                if key in collapsed_loading_keys
                else "predictor_preserving_interweave_with_path_fallback"
            )
            for key in compiled.estimated_loading_names
        },
        "collapsed_gaussian_processes": sorted(collapsed_processes),
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
        schema_version="2.1",
        initial_values={
            "by_chain": recorded_initial,
            "state_mean": compiled.initial_mean.tolist(),
            "state_sd": np.sqrt(np.diag(compiled.initial_cov)).tolist(),
        },
        auxiliary_draws={"fs_state": ncp_draws},
        metadata={
            "joint_model": True,
            "joint_likelihood": True,
            "conditional_channel_independence": True,
            "channel_names": list(compiled.channel_names),
            "factor_names": list(compiled.factor_names),
            "loading_identification": (
                "fixed non-zero loading anchor: "
                + next(
                    f"{channel}={spec.value:g}"
                    for channel, spec in compiled.model.factors[0]._resolved_loadings.items()
                    if spec.fixed and abs(spec.value) > 0.0
                )
            ),
            "factor_parameterization": "fruehwirth_schnatter",
            "unit_innovation_state": True,
            "signed_innovation_scales": True,
            "fixed_initial_factor_slopes": {
                block.name: float(block.beta0_value)
                for block in fs.blocks
                if block.kind == "factor" and block.beta0_fixed
            },
            "loading_kernels": {
                key: (
                    "collapsed_gaussian_ffbs"
                    if key in collapsed_loading_keys
                    else "predictor_preserving_interweave_with_path_fallback"
                )
                for key in compiled.estimated_loading_names
            },
            "collapsed_gaussian_processes": sorted(collapsed_processes),
            "regularized_horseshoe": priors.horseshoe is not None,
            "horseshoe_processes": list(priors.horseshoe_processes),
            "triple_gamma": priors.triple_gamma is not None,
            "regularized_triple_gamma": bool(
                priors.triple_gamma is not None
                and priors.triple_gamma.regularized
            ),
            "triple_gamma_processes": list(priors.triple_gamma_processes),
            "shrinkage_update": (
                "slice"
                if priors.horseshoe is not None
                or priors.triple_gamma is not None
                else None
            ),
        },
    )


__all__ = ["CompiledFactorFS", "FactorFSBlock", "sample_factor_fs_posterior"]
