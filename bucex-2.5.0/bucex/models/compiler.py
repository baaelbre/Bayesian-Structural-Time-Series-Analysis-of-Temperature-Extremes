"""Compile the model grammar to one linear state representation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import block_diag

from ..components import DummySeasonal, LocalLevel, LocalLinearTrend, Regression
from ..core.numerics import solve_affine_disturbance
from .structural import Model


Array = np.ndarray


def robust_scale(y: Array) -> float:
    values = np.asarray(y, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("y contains no finite observations.")
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return max(mad, sd * 0.25, abs(median) * 1e-3, 1e-6)


def _data_reference_scales(y: Array, period: int | None) -> tuple[float, float]:
    """Return robust observation and one-step change reference scales.

    The observation scale is computed after removing a linear trend and, when
    present, phase-specific seasonal medians.  The change scale uses lag-one
    differences for non-seasonal models and annual (``period``-lag)
    differences divided by ``sqrt(period)`` for seasonal models.  These are
    calibration references only: the observations themselves are never
    transformed by the compiler.
    """

    values = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    residual = values.copy()
    if period is not None:
        phase = np.arange(values.size) % int(period)
        for index in range(int(period)):
            selected = finite & (phase == index)
            if np.any(selected):
                residual[selected] -= float(np.median(values[selected]))

    locations = np.flatnonzero(finite).astype(float)
    observed = residual[finite]
    if observed.size >= 2:
        centred = locations - float(np.mean(locations))
        design = np.column_stack((np.ones(observed.size), centred))
        coefficient, *_ = np.linalg.lstsq(design, observed, rcond=None)
        observed = observed - design @ coefficient
    raw_scale = robust_scale(values)
    # Phase medians estimated from a short seasonal series can absorb most of
    # the within-phase variation.  Keep a conservative fraction of the raw
    # scale as a finite-sample floor without using record length in the prior.
    floor_fraction = 0.15 if period is not None else 0.05
    observation_scale = max(robust_scale(observed), floor_fraction * raw_scale)

    lag = 1 if period is None else int(period)
    paired = finite[lag:] & finite[:-lag]
    if np.any(paired):
        changes = (values[lag:][paired] - values[:-lag][paired]) / np.sqrt(lag)
        change_scale = robust_scale(changes)
    else:
        change_scale = observation_scale
    return float(observation_scale), float(change_scale)


def _rect_block_diag(blocks: list[Array]) -> Array:
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    out = np.zeros((rows, cols), dtype=float)
    i = j = 0
    for block in blocks:
        r, c = block.shape
        out[i : i + r, j : j + c] = block
        i += r
        j += c
    return out


def _as_exog(exog: Any, n_time: int, model: Model) -> Array | None:
    if model.n_exog == 0:
        if exog is not None:
            arr = np.asarray(exog)
            if arr.size:
                raise ValueError("exog was supplied but the model has no Regression component.")
        return None
    if exog is None:
        raise ValueError(f"The model requires exog with {model.n_exog} columns.")
    try:
        import pandas as pd

        if isinstance(exog, pd.DataFrame):
            names: list[str] = []
            all_named = True
            for component in model.components:
                if isinstance(component, Regression):
                    if component.feature_names is None:
                        all_named = False
                        break
                    names.extend(component.feature_names)
            arr = exog.loc[:, names].to_numpy(dtype=float) if all_named else exog.to_numpy(dtype=float)
        else:
            arr = np.asarray(exog, dtype=float)
    except ImportError:
        arr = np.asarray(exog, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape != (int(n_time), int(model.n_exog)):
        raise ValueError(
            f"exog must have shape ({n_time}, {model.n_exog}); got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("exog must contain only finite values.")
    return arr


@dataclass(frozen=True)
class DisturbancePath:
    """Initial state and standardized process disturbances."""

    x0: Array
    z: Array


@dataclass
class CompiledModel:
    model: Model
    transition: Array
    loading: Array
    initial_mean: Array
    initial_cov: Array
    state_names: tuple[str, ...]
    noise_names: tuple[str, ...]
    component_slices: dict[str, slice]
    regression_slices: dict[str, slice]
    y_scale: float
    observation_scale: float
    difference_scale: float
    n_time: int
    exog: Array | None = None

    @property
    def state_dim(self) -> int:
        return int(self.transition.shape[0])

    @property
    def noise_dim(self) -> int:
        return int(self.loading.shape[1])

    @property
    def family(self) -> str:
        return self.model.family

    def process_vector(self, params: dict[str, float]) -> Array:
        return np.asarray([float(params[f"sd.{name}"]) for name in self.noise_names])

    def transition_cov(self, params: dict[str, float]) -> Array:
        sd = self.process_vector(params)
        return self.loading @ np.diag(sd**2) @ self.loading.T

    def design(
        self,
        n_time: int | None = None,
        exog: Any = None,
        *,
        params: dict[str, float] | None = None,
    ) -> Array:
        n = self.n_time if n_time is None else int(n_time)
        if exog is None and n == self.n_time:
            xreg = self.exog
        else:
            xreg = _as_exog(exog, n, self.model)
        design = np.zeros((n, self.state_dim), dtype=float)
        for component in self.model.components:
            sl = self.component_slices[self._component_key(component)]
            if sl.start == sl.stop:
                continue
            if isinstance(component, (LocalLevel, LocalLinearTrend)):
                design[:, sl.start] = 1.0
            elif isinstance(component, DummySeasonal):
                design[:, sl.start] = 1.0
            elif isinstance(component, Regression):
                assert xreg is not None
                exog_sl = self.regression_slices[component.name]
                design[:, sl] = xreg[:, exog_sl]
        return design

    @staticmethod
    def _component_key(component: Any) -> str:
        if isinstance(component, LocalLevel):
            return "trend"
        if isinstance(component, LocalLinearTrend):
            return "trend"
        if isinstance(component, DummySeasonal):
            return "seasonal"
        if isinstance(component, Regression):
            return f"regression:{component.name}"
        raise TypeError(type(component).__name__)

    def eta(
        self,
        path: Array,
        exog: Any = None,
        *,
        params: dict[str, float] | None = None,
    ) -> Array:
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != self.state_dim:
            raise ValueError("path must have shape (T+1, state_dim).")
        h = self.design(path.shape[0] - 1, exog=exog)
        return np.einsum("tm,tm->t", h, path[1:])

    def to_disturbance(
        self,
        path: Array,
        params: dict[str, float],
        *,
        tolerance: float = 1e-7,
    ) -> DisturbancePath:
        path = np.asarray(path, dtype=float)
        if path.shape != (path.shape[0], self.state_dim) or path.shape[0] < 2:
            raise ValueError("path must have shape (T+1, state_dim).")
        transition_loading = self.loading * self.process_vector(params)[None, :]
        z = np.zeros((path.shape[0] - 1, self.noise_dim), dtype=float)
        for t in range(1, path.shape[0]):
            residual = path[t] - self.transition @ path[t - 1]
            if self.noise_dim == 0:
                if np.linalg.norm(residual) > tolerance * (1.0 + np.linalg.norm(path[t])):
                    raise ValueError("The path is not on the deterministic transition support.")
                continue
            disturbance, on_support = solve_affine_disturbance(
                transition_loading, residual, tolerance=tolerance
            )
            if not on_support:
                raise ValueError("The path is outside the structural innovation support.")
            z[t - 1] = disturbance
        return DisturbancePath(x0=path[0].copy(), z=z)

    def from_disturbance(self, value: DisturbancePath, params: dict[str, float]) -> Array:
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

    # Compatibility method names; both delegate to the disturbance strategy.
    def to_noncentered(self, path: Array, params: dict[str, float], **kwargs) -> DisturbancePath:
        return self.to_disturbance(path, params, **kwargs)

    def from_noncentered(self, value: DisturbancePath, params: dict[str, float]) -> Array:
        return self.from_disturbance(value, params)

    def project_path(self, path: Array, params: dict[str, float]) -> Array:
        """Remove floating-point drift from deterministic transition coordinates.

        Exact FFBS with a singular process covariance lives on an affine
        subspace. Backward covariance algebra can leave errors around machine
        precision in coordinates that are theoretically deterministic. This
        projection changes only those numerical remnants; it does not add
        process jitter or alter the declared model.
        """

        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != self.state_dim or path.shape[0] < 2:
            raise ValueError("path must have shape (T+1, state_dim).")
        output = np.zeros_like(path)
        output[0] = path[0]
        transition_loading = self.loading * self.process_vector(params)[None, :]
        for t in range(1, path.shape[0]):
            mean = self.transition @ output[t - 1]
            if self.noise_dim == 0:
                output[t] = mean
                continue
            disturbance, _ = solve_affine_disturbance(
                transition_loading, path[t] - mean
            )
            output[t] = mean + transition_loading @ disturbance
        return output

    def with_data(self, y: Array, exog: Any = None) -> "CompiledModel":
        return compile_model(self.model, y, exog=exog)


def _compile_scalar_model(model: Model, y: Array, exog: Any = None) -> CompiledModel:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if y_arr.size < 2:
        raise ValueError("At least two observations are required.")
    if not np.any(np.isfinite(y_arr)):
        raise ValueError("y contains no finite observations.")
    xreg = _as_exog(exog, y_arr.size, model)
    scale = robust_scale(y_arr)
    observation_scale, difference_scale = _data_reference_scales(y_arr, model.period)
    first = float(y_arr[np.flatnonzero(np.isfinite(y_arr))[0]])

    f_blocks: list[Array] = []
    r_blocks: list[Array] = []
    m0_blocks: list[Array] = []
    p0_blocks: list[Array] = []
    state_names: list[str] = []
    noise_names: list[str] = []
    component_slices: dict[str, slice] = {}
    regression_slices: dict[str, slice] = {}
    state_pos = 0
    exog_pos = 0

    for component in model.components:
        key = CompiledModel._component_key(component)
        dim = int(component.spec.state_dim)
        if dim == 0:
            component_slices[key] = slice(state_pos, state_pos)
            continue

        initial_params: dict[str, Any] = {}
        if isinstance(component, LocalLevel):
            if component.initial_mean is None:
                initial_params["m0_level"] = first
            if component.initial_sd is None:
                initial_params["v0_level"] = (5.0 * scale) ** 2
        elif isinstance(component, LocalLinearTrend):
            if component.initial_level is None:
                initial_params["m0_level"] = first
            if component.initial_level_sd is None:
                initial_params["v0_level"] = (5.0 * scale) ** 2
            if dim == 2 and component.initial_slope_sd is None:
                initial_params["v0_trend"] = (
                    scale / max(5.0, np.sqrt(y_arr.size))
                ) ** 2
        elif isinstance(component, DummySeasonal):
            if component.initial_sd is None:
                initial_params["v0_season"] = np.full(dim, scale**2)
        elif isinstance(component, Regression):
            if component.initial_sd is None:
                initial_params[f"v0_{component.name}"] = np.full(
                    dim, (2.0 * scale) ** 2
                )
            regression_slices[component.name] = slice(
                exog_pos,
                exog_pos + int(component.n_features),
            )
            exog_pos += int(component.n_features)
        else:
            raise TypeError(f"Unsupported component {type(component).__name__}.")

        unit_params = {
            f"sd.{noise_name}": 1.0
            for noise_name in component.spec.noise_names
        }
        f, r, q, offset = component.system_matrices(t=1, params=unit_params)
        f = np.asarray(f, dtype=float)
        r = np.asarray(r, dtype=float)
        q = np.asarray(q, dtype=float)
        offset = np.asarray(offset, dtype=float)
        if not np.allclose(offset, 0.0):
            raise ValueError("Compiled components must have zero transition offsets.")
        if q.shape != (component.spec.noise_dim, component.spec.noise_dim) or (
            q.size and not np.allclose(q, np.eye(component.spec.noise_dim))
        ):
            raise ValueError(
                "Component process scales must be represented by named SD parameters."
            )
        m0, p0 = component.initial_mean_var(initial_params)
        m0 = np.asarray(m0, dtype=float).reshape(dim)
        p0 = np.asarray(p0, dtype=float).reshape(dim)
        if np.any(p0 < 0.0):
            raise ValueError(f"Initial state variances for component '{key}' must be non-negative.")
        component_slices[key] = slice(state_pos, state_pos + f.shape[0])
        state_pos += f.shape[0]
        f_blocks.append(f)
        r_blocks.append(r)
        m0_blocks.append(m0)
        p0_blocks.append(p0)
        state_names.extend(component.spec.state_names)
        noise_names.extend(component.spec.noise_names)

    transition = block_diag(*f_blocks)
    loading = _rect_block_diag(r_blocks)
    initial_mean = np.concatenate(m0_blocks)
    initial_cov = np.diag(np.concatenate(p0_blocks))
    if len(noise_names) != len(set(noise_names)):
        raise ValueError("Process innovation names must be unique across components.")
    return CompiledModel(
        model=model,
        transition=transition,
        loading=loading,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
        state_names=tuple(state_names),
        noise_names=tuple(noise_names),
        component_slices=component_slices,
        regression_slices=regression_slices,
        y_scale=scale,
        observation_scale=observation_scale,
        difference_scale=difference_scale,
        n_time=y_arr.size,
        exog=xreg,
    )


def compile_model(model: Any, y: Any, exog: Any = None):
    """Compile either a univariate or a multi-series structural model.

    The public dispatcher is deliberately small: both paths return objects
    implementing the same transition, design, disturbance, and predictor
    contract consumed by inference backends.
    """

    from .multiseries import MultiSeriesModel

    if isinstance(model, MultiSeriesModel):
        from .multiseries_compiler import compile_multiseries_model

        return compile_multiseries_model(model, y, exog=exog)
    if not isinstance(model, Model):
        raise TypeError(
            "model must be a bucex.Model or bucex.MultiSeriesModel."
        )
    return _compile_scalar_model(model, y, exog=exog)


NonCenteredPath = DisturbancePath
