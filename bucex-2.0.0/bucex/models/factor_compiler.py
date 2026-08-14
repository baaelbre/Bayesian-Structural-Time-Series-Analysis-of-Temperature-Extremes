"""Compile multichannel factors to one linear state-space representation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.linalg import block_diag

from .factor import FactorModel, Loading
from .structural import Model


Array = np.ndarray


def as_factor_array(y: Any, model: FactorModel) -> Array:
    """Return observations in canonical ``(time, channel)`` order."""

    names = model.channel_names
    try:
        import pandas as pd

        if isinstance(y, pd.DataFrame):
            missing = [name for name in names if name not in y.columns]
            if missing:
                raise ValueError(f"Observation table is missing channels: {missing}.")
            values = y.loc[:, list(names)].to_numpy(dtype=float)
        elif isinstance(y, Mapping):
            missing = [name for name in names if name not in y]
            if missing:
                raise ValueError(f"Observation mapping is missing channels: {missing}.")
            columns = [np.asarray(y[name], dtype=float).reshape(-1) for name in names]
            lengths = {column.size for column in columns}
            if len(lengths) != 1:
                raise ValueError("All factor-model channels must have the same length.")
            values = np.column_stack(columns)
        else:
            values = np.asarray(y, dtype=float)
    except ImportError:
        if isinstance(y, Mapping):
            missing = [name for name in names if name not in y]
            if missing:
                raise ValueError(f"Observation mapping is missing channels: {missing}.")
            columns = [np.asarray(y[name], dtype=float).reshape(-1) for name in names]
            values = np.column_stack(columns)
        else:
            values = np.asarray(y, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(
            f"Factor-model observations must have shape (T, {len(names)}) in "
            f"channel order {names}; got {values.shape}."
        )
    if values.shape[0] < 2:
        raise ValueError("At least two time points are required.")
    for index, name in enumerate(names):
        if np.sum(np.isfinite(values[:, index])) < 2:
            raise ValueError(f"Channel '{name}' must contain at least two finite observations.")
    if np.any(~np.any(np.isfinite(values), axis=1)):
        raise ValueError("Every time point must contain at least one finite channel observation.")
    return np.asarray(values, dtype=float)


def factor_dates(y: Any, model: FactorModel) -> Array | None:
    """Extract an aligned pandas index, when one is present."""

    try:
        import pandas as pd

        if isinstance(y, pd.DataFrame):
            return y.index.to_numpy()
        if isinstance(y, Mapping):
            indexes = [getattr(y.get(name), "index", None) for name in model.channel_names]
            if all(index is not None for index in indexes):
                first = indexes[0]
                if not all(first.equals(index) for index in indexes[1:]):
                    raise ValueError("Pandas factor-model channels must have identical indexes.")
                return np.asarray(first)
    except ImportError:
        pass
    return None


def _factor_exog(
    exog: Any,
    model: FactorModel,
    n_time: int,
) -> dict[str, Array] | None:
    required = {channel.name: channel.n_exog for channel in model.channels if channel.n_exog}
    if not required:
        if exog is not None:
            if isinstance(exog, Mapping):
                nonempty = [name for name, value in exog.items() if np.asarray(value).size]
                if nonempty:
                    raise ValueError("exog was supplied but no channel has a Regression component.")
            elif np.asarray(exog).size:
                raise ValueError("exog was supplied but no channel has a Regression component.")
        return None
    if not isinstance(exog, Mapping):
        raise TypeError(
            "Factor-model exog must be a mapping from channel name to its design matrix."
        )
    missing = sorted(set(required) - set(exog))
    extra = sorted(set(exog) - set(required))
    if missing or extra:
        raise ValueError(f"Factor exog keys do not match regression channels; missing={missing}, extra={extra}.")
    output: dict[str, Array] = {}
    for channel in model.channels:
        if not channel.n_exog:
            continue
        value = exog[channel.name]
        try:
            import pandas as pd

            if isinstance(value, pd.DataFrame):
                names: list[str] = []
                all_named = True
                from ..components import Regression

                for component in channel.components:
                    if isinstance(component, Regression):
                        if component.feature_names is None:
                            all_named = False
                            break
                        names.extend(component.feature_names)
                array = (
                    value.loc[:, names].to_numpy(dtype=float)
                    if all_named
                    else value.to_numpy(dtype=float)
                )
            else:
                array = np.asarray(value, dtype=float)
        except ImportError:
            array = np.asarray(value, dtype=float)
        if array.ndim == 1:
            array = array[:, None]
        expected = (int(n_time), int(channel.n_exog))
        if array.shape != expected:
            raise ValueError(
                f"exog['{channel.name}'] must have shape {expected}; got {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"exog['{channel.name}'] must contain only finite values.")
        output[channel.name] = array
    return output


@dataclass(frozen=True)
class FactorStateBlock:
    kind: str
    name: str
    compiled: Any
    state_slice: slice


@dataclass
class CompiledFactorModel:
    """One global state graph for shared factors and individual components."""

    model: FactorModel
    transition: Array
    loading: Array
    initial_mean: Array
    initial_cov: Array
    state_names: tuple[str, ...]
    noise_names: tuple[str, ...]
    blocks: tuple[FactorStateBlock, ...]
    factor_slices: Mapping[str, slice]
    channel_slices: Mapping[str, slice]
    y_scale: float
    observation_scale: float
    difference_scale: float
    channel_y_scale: Mapping[str, float]
    channel_observation_scale: Mapping[str, float]
    channel_difference_scale: Mapping[str, float]
    process_reference_scale: Mapping[str, float]
    n_time: int
    exog: Mapping[str, Array] | None = None

    @property
    def state_dim(self) -> int:
        return int(self.transition.shape[0])

    @property
    def noise_dim(self) -> int:
        return int(self.loading.shape[1])

    @property
    def family(self) -> str:
        return self.model.family

    @property
    def all_gaussian(self) -> bool:
        return self.model.all_gaussian

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self.model.channel_names

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self.model.factor_names

    @property
    def observation_parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for channel in self.model.channels:
            names.append(f"sigma.{channel.name}")
            if channel.family == "gev":
                names.append(f"xi.{channel.name}")
        return tuple(names)

    @property
    def loading_names(self) -> tuple[str, ...]:
        return tuple(
            f"loading.{factor.name}.{channel}"
            for factor in self.model.factors
            for channel in factor._resolved_loadings
        )

    @property
    def estimated_loading_names(self) -> tuple[str, ...]:
        return self.model.estimated_loading_names

    @property
    def loading_specs(self) -> Mapping[str, Loading]:
        return {
            f"loading.{factor.name}.{channel}": spec
            for factor in self.model.factors
            for channel, spec in factor._resolved_loadings.items()
        }

    def process_vector(self, params: Mapping[str, float]) -> Array:
        return np.asarray([float(params[f"sd.{name}"]) for name in self.noise_names])

    def transition_cov(self, params: Mapping[str, float]) -> Array:
        sd = self.process_vector(params)
        return self.loading @ np.diag(sd**2) @ self.loading.T

    def _loading_value(
        self,
        factor_name: str,
        channel_name: str,
        params: Mapping[str, float] | None,
    ) -> float:
        factor = self.model.factor(factor_name)
        spec = factor.loading_for(channel_name)
        key = f"loading.{factor_name}.{channel_name}"
        if not spec.fixed and params is not None and key in params:
            return float(params[key])
        return float(spec.value)

    def loading_matrix(self, params: Mapping[str, float] | None = None) -> Array:
        output = np.zeros((len(self.channel_names), len(self.factor_names)), dtype=float)
        for j, channel in enumerate(self.channel_names):
            for k, factor in enumerate(self.factor_names):
                output[j, k] = self._loading_value(factor, channel, params)
        return output

    def design(
        self,
        n_time: int | None = None,
        exog: Any = None,
        *,
        params: Mapping[str, float] | None = None,
    ) -> Array:
        n = self.n_time if n_time is None else int(n_time)
        if exog is None and n == self.n_time:
            resolved_exog = self.exog
        else:
            resolved_exog = _factor_exog(exog, self.model, n)
        output = np.zeros((n, len(self.channel_names), self.state_dim), dtype=float)
        channel_index = {name: index for index, name in enumerate(self.channel_names)}
        for block in self.blocks:
            if block.kind == "factor":
                local = block.compiled.design(n)
                for channel in self.channel_names:
                    coefficient = self._loading_value(block.name, channel, params)
                    if coefficient:
                        output[:, channel_index[channel], block.state_slice] += coefficient * local
            else:
                local_exog = None if resolved_exog is None else resolved_exog.get(block.name)
                local = block.compiled.design(n, exog=local_exog)
                output[:, channel_index[block.name], block.state_slice] += local
        return output

    def eta(
        self,
        path: Array,
        exog: Any = None,
        *,
        params: Mapping[str, float] | None = None,
    ) -> Array:
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != self.state_dim or path.shape[0] < 2:
            raise ValueError("path must have shape (T+1, state_dim).")
        design = self.design(path.shape[0] - 1, exog=exog, params=params)
        return np.einsum("tpm,tm->tp", design, path[1:])

    def observation_variance(self, params: Mapping[str, float], n_time: int | None = None) -> Array:
        n = self.n_time if n_time is None else int(n_time)
        row = np.asarray(
            [float(params[f"sigma.{name}"]) ** 2 for name in self.channel_names],
            dtype=float,
        )
        return np.broadcast_to(row, (n, row.size)).copy()

    def observation_log_likelihood(
        self,
        y: Array,
        eta: Array,
        params: Mapping[str, float],
    ) -> float:
        y = np.asarray(y, dtype=float)
        eta = np.asarray(eta, dtype=float)
        if y.shape != eta.shape or y.ndim != 2 or y.shape[1] != len(self.channel_names):
            raise ValueError("y and eta must have shape (T, n_channels).")
        total = 0.0
        for index, channel in enumerate(self.model.channels):
            mask = np.isfinite(y[:, index])
            if not np.any(mask):
                continue
            values = channel.observation.logpdf(
                y[mask, index],
                eta[mask, index],
                sigma=float(params[f"sigma.{channel.name}"]),
                xi=params.get(f"xi.{channel.name}"),
            )
            if not np.all(np.isfinite(values)):
                return -np.inf
            total += float(np.sum(values))
        return float(total)

    def observation_logweights(
        self,
        y_t: Array,
        particles: Array,
        design_t: Array,
        params: Mapping[str, float],
    ) -> Array:
        values = np.asarray(y_t, dtype=float).reshape(len(self.channel_names))
        eta = np.asarray(particles, dtype=float) @ np.asarray(design_t, dtype=float).T
        output = np.zeros(eta.shape[0], dtype=float)
        for index, channel in enumerate(self.model.channels):
            if not np.isfinite(values[index]):
                continue
            contribution = channel.observation.logpdf(
                values[index],
                eta[:, index],
                sigma=float(params[f"sigma.{channel.name}"]),
                xi=params.get(f"xi.{channel.name}"),
            )
            contribution = np.asarray(contribution, dtype=float).reshape(-1)
            output += np.where(np.isfinite(contribution), contribution, -np.inf)
        output[~np.isfinite(output)] = -np.inf
        return output

    def observation_derivatives(
        self,
        y: Array,
        eta: Array,
        params: Mapping[str, float],
    ) -> tuple[Array, Array]:
        y = np.asarray(y, dtype=float)
        eta = np.asarray(eta, dtype=float)
        gradient = np.zeros_like(y, dtype=float)
        hessian = np.zeros_like(y, dtype=float)
        for index, channel in enumerate(self.model.channels):
            mask = np.isfinite(y[:, index])
            if not np.any(mask):
                continue
            kwargs = {
                "sigma": float(params[f"sigma.{channel.name}"]),
                "xi": params.get(f"xi.{channel.name}"),
            }
            gradient[mask, index] = channel.observation.grad_eta(
                y[mask, index], eta[mask, index], **kwargs
            )
            hessian[mask, index] = channel.observation.hess_eta(
                y[mask, index], eta[mask, index], **kwargs
            )
        return gradient, hessian

    def sample_observation(
        self,
        eta_t: Array,
        params: Mapping[str, float],
        rng: np.random.Generator,
    ) -> Array:
        eta_t = np.asarray(eta_t, dtype=float).reshape(len(self.channel_names))
        output = np.zeros(len(self.channel_names), dtype=float)
        for index, channel in enumerate(self.model.channels):
            output[index] = float(
                channel.observation.sample(
                    eta=eta_t[index],
                    sigma=float(params[f"sigma.{channel.name}"]),
                    xi=params.get(f"xi.{channel.name}"),
                    rng=rng,
                )
            )
        return output

    def to_disturbance(
        self,
        path: Array,
        params: Mapping[str, float],
        *,
        tolerance: float = 1e-7,
    ):
        from .compiler import DisturbancePath

        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != self.state_dim or path.shape[0] < 2:
            raise ValueError("path must have shape (T+1, state_dim).")
        sd = self.process_vector(params)
        z = np.zeros((path.shape[0] - 1, self.noise_dim), dtype=float)
        for t in range(1, path.shape[0]):
            residual = path[t] - self.transition @ path[t - 1]
            innovation, *_ = np.linalg.lstsq(self.loading, residual, rcond=None)
            reconstruction = self.loading @ innovation
            if np.linalg.norm(residual - reconstruction) > tolerance * (1.0 + np.linalg.norm(residual)):
                raise ValueError("The path is outside the structural innovation support.")
            for j, scale in enumerate(sd):
                if scale > 1e-12:
                    z[t - 1, j] = innovation[j] / scale
                elif abs(innovation[j]) > tolerance:
                    raise ValueError(
                        f"Non-zero disturbance encountered for fixed process '{self.noise_names[j]}'."
                    )
        return DisturbancePath(x0=path[0].copy(), z=z)

    def from_disturbance(self, value: Any, params: Mapping[str, float]) -> Array:
        x0 = np.asarray(value.x0, dtype=float).reshape(self.state_dim)
        z = np.asarray(value.z, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.noise_dim:
            raise ValueError("z must have shape (T, noise_dim).")
        sd = self.process_vector(params)
        path = np.zeros((z.shape[0] + 1, self.state_dim), dtype=float)
        path[0] = x0
        for t in range(1, path.shape[0]):
            path[t] = self.transition @ path[t - 1] + self.loading @ (sd * z[t - 1])
        return path

    def to_noncentered(self, path: Array, params: Mapping[str, float], **kwargs):
        return self.to_disturbance(path, params, **kwargs)

    def from_noncentered(self, value: Any, params: Mapping[str, float]) -> Array:
        return self.from_disturbance(value, params)

    def project_path(self, path: Array, params: Mapping[str, float]) -> Array:
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != self.state_dim or path.shape[0] < 2:
            raise ValueError("path must have shape (T+1, state_dim).")
        output = np.zeros_like(path)
        output[0] = path[0]
        sd = self.process_vector(params)
        for t in range(1, path.shape[0]):
            mean = self.transition @ output[t - 1]
            innovation, *_ = np.linalg.lstsq(self.loading, path[t] - mean, rcond=None)
            innovation = np.asarray(innovation)
            innovation[sd <= 1e-12] = 0.0
            output[t] = mean + self.loading @ innovation
        return output

    def with_data(self, y: Any, exog: Any = None) -> "CompiledFactorModel":
        return compile_factor_model(self.model, y, exog=exog)


def compile_factor_model(
    model: FactorModel,
    y: Any,
    exog: Any = None,
) -> CompiledFactorModel:
    """Compile all shared and individual blocks into one global state."""

    if not isinstance(model, FactorModel):
        raise TypeError("model must be a FactorModel.")
    y_array = as_factor_array(y, model)
    n_time, n_channels = y_array.shape
    xreg = _factor_exog(exog, model, n_time)

    # Imported lazily to avoid a circular import with compile_model's public
    # dispatcher.
    from .compiler import _compile_scalar_model

    transitions: list[Array] = []
    loadings: list[Array] = []
    initial_means: list[Array] = []
    initial_covariances: list[Array] = []
    state_names: list[str] = []
    noise_names: list[str] = []
    blocks: list[FactorStateBlock] = []
    factor_slices: dict[str, slice] = {}
    channel_slices: dict[str, slice] = {}
    process_reference_scale: dict[str, float] = {}
    state_position = 0

    channel_index = {name: index for index, name in enumerate(model.channel_names)}
    for factor in model.factors:
        anchors = [
            (name, spec)
            for name, spec in factor._resolved_loadings.items()
            if spec.fixed and abs(spec.value) > 0.0
        ]
        anchor_name, anchor = anchors[0] if anchors else next(
            (item for item in factor._resolved_loadings.items() if abs(item[1].value) > 0.0)
        )
        reference = y_array[:, channel_index[anchor_name]] / float(anchor.value)
        finite = np.flatnonzero(np.isfinite(reference))
        reference = reference - float(reference[finite[0]])
        submodel = Model(
            observation=model.channel(anchor_name).observation,
            components=factor.components,
            name=f"factor.{factor.name}",
        )
        sub = _compile_scalar_model(submodel, reference)
        state_slice = slice(state_position, state_position + sub.state_dim)
        factor_slices[factor.name] = state_slice
        blocks.append(FactorStateBlock("factor", factor.name, sub, state_slice))
        transitions.append(sub.transition)
        loadings.append(sub.loading)
        initial_means.append(sub.initial_mean)
        initial_covariances.append(sub.initial_cov)
        state_names.extend(f"factor.{factor.name}.{name}" for name in sub.state_names)
        scoped_noise = [f"factor.{factor.name}.{name}" for name in sub.noise_names]
        noise_names.extend(scoped_noise)
        process_reference_scale.update(
            {name: float(sub.difference_scale) for name in scoped_noise}
        )
        state_position = state_slice.stop

    for channel in model.channels:
        if not channel.components:
            channel_slices[channel.name] = slice(state_position, state_position)
            continue
        submodel = Model(
            observation=channel.observation,
            components=channel.components,
            name=f"channel.{channel.name}",
        )
        sub_exog = None if xreg is None else xreg.get(channel.name)
        sub = _compile_scalar_model(
            submodel,
            y_array[:, channel_index[channel.name]],
            exog=sub_exog,
        )
        state_slice = slice(state_position, state_position + sub.state_dim)
        channel_slices[channel.name] = state_slice
        blocks.append(FactorStateBlock("channel", channel.name, sub, state_slice))
        transitions.append(sub.transition)
        loadings.append(sub.loading)
        initial_means.append(sub.initial_mean)
        initial_covariances.append(sub.initial_cov)
        state_names.extend(f"channel.{channel.name}.{name}" for name in sub.state_names)
        scoped_noise = [f"channel.{channel.name}.{name}" for name in sub.noise_names]
        noise_names.extend(scoped_noise)
        process_reference_scale.update(
            {name: float(sub.difference_scale) for name in scoped_noise}
        )
        state_position = state_slice.stop

    if not transitions:
        raise ValueError("The factor model compiled to an empty state.")
    transition = block_diag(*transitions)
    initial_cov = block_diag(*initial_covariances)
    rows = sum(block.shape[0] for block in loadings)
    cols = sum(block.shape[1] for block in loadings)
    global_loading = np.zeros((rows, cols), dtype=float)
    row = column = 0
    for block in loadings:
        nrow, ncol = block.shape
        global_loading[row : row + nrow, column : column + ncol] = block
        row += nrow
        column += ncol

    channel_y_scale: dict[str, float] = {}
    channel_observation_scale: dict[str, float] = {}
    channel_difference_scale: dict[str, float] = {}
    from .compiler import _data_reference_scales, robust_scale

    for index, channel in enumerate(model.channels):
        values = y_array[:, index]
        channel_y_scale[channel.name] = robust_scale(values)
        observation_scale, difference_scale = _data_reference_scales(values, channel.period)
        channel_observation_scale[channel.name] = observation_scale
        channel_difference_scale[channel.name] = difference_scale

    return CompiledFactorModel(
        model=model,
        transition=transition,
        loading=global_loading,
        initial_mean=np.concatenate(initial_means),
        initial_cov=initial_cov,
        state_names=tuple(state_names),
        noise_names=tuple(noise_names),
        blocks=tuple(blocks),
        factor_slices=factor_slices,
        channel_slices=channel_slices,
        y_scale=float(max(channel_y_scale.values())),
        observation_scale=float(max(channel_observation_scale.values())),
        difference_scale=float(max(channel_difference_scale.values())),
        channel_y_scale=channel_y_scale,
        channel_observation_scale=channel_observation_scale,
        channel_difference_scale=channel_difference_scale,
        process_reference_scale=process_reference_scale,
        n_time=n_time,
        exog=xreg,
    )


__all__ = [
    "CompiledFactorModel",
    "FactorStateBlock",
    "as_factor_array",
    "factor_dates",
    "compile_factor_model",
]
