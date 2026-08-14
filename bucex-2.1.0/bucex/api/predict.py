"""Posterior predictive simulation and forecast summaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..diagnostics.scores import evaluate_ensemble


Array = np.ndarray


def _future_dates(dates: Array | None, horizon: int) -> Array:
    if dates is None:
        return np.arange(1, horizon + 1)
    values = np.asarray(dates)
    try:
        import pandas as pd

        parsed = pd.to_datetime(values)
        frequency = pd.infer_freq(parsed)
        if frequency is not None:
            return pd.date_range(parsed[-1], periods=horizon + 1, freq=frequency)[1:].to_numpy()
        if parsed.size >= 2:
            delta = parsed[-1] - parsed[-2]
            return np.asarray([parsed[-1] + (j + 1) * delta for j in range(horizon)])
    except Exception:
        pass
    if np.issubdtype(values.dtype, np.number) and values.size >= 2:
        step = values[-1] - values[-2]
        return values[-1] + step * np.arange(1, horizon + 1)
    return np.arange(1, horizon + 1)


@dataclass
class Forecast:
    observations: Array
    eta: Array
    states: Array
    parameters: dict[str, Array]
    dates: Array
    family: str
    tail: Any
    observation_model: Any
    transform_sign: Any = 1.0
    channel_names: tuple[str, ...] = ()

    @property
    def is_factor_forecast(self) -> bool:
        return bool(self.channel_names)

    def _channel_index(self, channel: str | None) -> int:
        if not self.is_factor_forecast:
            if channel is not None:
                raise ValueError("channel= is only valid for a factor forecast.")
            return 0
        if channel is None:
            raise ValueError(f"Choose channel from {self.channel_names}.")
        if channel not in self.channel_names:
            raise KeyError(f"Unknown channel '{channel}'. Available: {self.channel_names}")
        return self.channel_names.index(channel)

    @property
    def n_draws(self) -> int:
        return int(self.observations.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[1])

    def summary(self, level: float = 0.90, *, channel: str | None = None):
        if not 0.0 < float(level) < 1.0:
            raise ValueError("level must lie in (0, 1).")
        alpha = 1.0 - float(level)
        rows = []
        channels = (
            self.channel_names
            if self.is_factor_forecast and channel is None
            else (channel,) if self.is_factor_forecast else (None,)
        )
        for selected in channels:
            index = self._channel_index(selected) if self.is_factor_forecast else 0
            for h in range(self.horizon):
                obs = (
                    self.observations[:, h, index]
                    if self.is_factor_forecast
                    else self.observations[:, h]
                )
                eta = self.eta[:, h, index] if self.is_factor_forecast else self.eta[:, h]
                row = {
                    "time": self.dates[h],
                    "mean": float(np.mean(obs)),
                    "lower": float(np.quantile(obs, alpha / 2.0)),
                    "median": float(np.median(obs)),
                    "upper": float(np.quantile(obs, 1.0 - alpha / 2.0)),
                    "eta_lower": float(np.quantile(eta, alpha / 2.0)),
                    "eta_median": float(np.median(eta)),
                    "eta_upper": float(np.quantile(eta, 1.0 - alpha / 2.0)),
                }
                if selected is not None:
                    row["channel"] = selected
                rows.append(row)
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def tail_probability(self, threshold: float, *, channel: str | None = None) -> Array:
        index = self._channel_index(channel)
        observations = (
            self.observations[:, :, index]
            if self.is_factor_forecast
            else self.observations
        )
        tail = self.tail[index] if self.is_factor_forecast else self.tail
        if tail == "upper":
            return np.mean(observations > float(threshold), axis=0)
        return np.mean(observations < float(threshold), axis=0)

    def return_level(self, return_period: float, *, channel: str | None = None) -> Array:
        index = self._channel_index(channel)
        if self.is_factor_forecast:
            assert channel is not None
            observation_model = self.observation_model[channel]
            if getattr(observation_model, "name", None) != "gev":
                raise ValueError("Return levels require a GEV channel.")
            sign = float(np.asarray(self.transform_sign)[index])
            eta = self.eta[:, :, index]
            sigma = self.parameters[f"sigma.{channel}"][:, None]
            xi = self.parameters[f"xi.{channel}"][:, None]
        elif self.family != "gev":
            raise ValueError("Return levels require a GEV forecast.")
        else:
            observation_model = self.observation_model
            sign = float(self.transform_sign)
            eta = self.eta
            sigma = self.parameters["sigma"][:, None]
            xi = self.parameters["xi"][:, None]
        if float(return_period) <= 1.0:
            raise ValueError("return_period must exceed 1.")
        probability = 1.0 - 1.0 / float(return_period)
        eta_model = sign * eta
        level = observation_model.ppf(probability, eta_model, sigma=sigma, xi=xi)
        return sign * level

    def score(
        self,
        observed: Array,
        *,
        thresholds: list[float] | tuple[float, ...] = (),
        quantiles: list[float] | tuple[float, ...] = (0.9, 0.95, 0.99),
        channel: str | None = None,
    ):
        if self.is_factor_forecast and channel is None:
            observed_array = np.asarray(observed, dtype=float)
            if observed_array.shape != (self.horizon, len(self.channel_names)):
                raise ValueError(
                    "Factor observed values must have shape (horizon, n_channels), "
                    "or choose channel=."
                )
            tables = []
            for index, name in enumerate(self.channel_names):
                table = evaluate_ensemble(
                    self.observations[:, :, index],
                    observed_array[:, index],
                    thresholds=thresholds,
                    quantiles=quantiles,
                    tail=self.tail[index],
                )
                if hasattr(table, "assign"):
                    table = table.assign(channel=name)
                else:
                    for row in table:
                        row["channel"] = name
                tables.append(table)
            try:
                import pandas as pd

                return pd.concat(tables, ignore_index=True)
            except ImportError:
                return [row for table in tables for row in table]
        index = self._channel_index(channel)
        observed_array = np.asarray(observed, dtype=float).reshape(-1)
        if observed_array.size != self.horizon:
            raise ValueError("observed must have length equal to the forecast horizon.")
        ensemble = self.observations[:, :, index] if self.is_factor_forecast else self.observations
        tail = self.tail[index] if self.is_factor_forecast else self.tail
        return evaluate_ensemble(
            ensemble,
            observed_array,
            thresholds=thresholds,
            quantiles=quantiles,
            tail=tail,
        )

    def plot(
        self,
        *,
        level: float = 0.90,
        channel: str | None = None,
        ax=None,
        color: str = "C0",
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        if self.is_factor_forecast and channel is None:
            channel = self.channel_names[0]
        summary = self.summary(level=level, channel=channel)
        x = np.arange(self.horizon) if not hasattr(summary, "columns") else summary["time"]
        lower = np.asarray([row["lower"] for row in summary]) if not hasattr(summary, "columns") else summary["lower"].to_numpy()
        median = np.asarray([row["median"] for row in summary]) if not hasattr(summary, "columns") else summary["median"].to_numpy()
        upper = np.asarray([row["upper"] for row in summary]) if not hasattr(summary, "columns") else summary["upper"].to_numpy()
        ax.fill_between(x, lower, upper, color=color, alpha=0.2, label=f"{level:.0%} predictive interval")
        ax.plot(x, median, color=color, label="predictive median")
        ax.set_title(
            f"Posterior predictive forecast: {channel}"
            if channel is not None
            else "Posterior predictive forecast"
        )
        ax.legend()
        return ax


def posterior_predict(
    fit,
    horizon: int,
    *,
    exog_future=None,
    draws: int | None = None,
    seed: int | None = None,
    dates: Array | None = None,
) -> Forecast:
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    rng = np.random.default_rng(seed)
    total = fit.n_draws
    n_draws = total if draws is None else int(draws)
    if n_draws < 1:
        raise ValueError("draws must be positive.")
    indices = np.arange(total) if n_draws == total else rng.choice(total, size=n_draws, replace=n_draws > total)
    flat_states = fit.state_draws.reshape((total,) + fit.state_draws.shape[2:])
    if getattr(fit, "is_factor_model", False):
        required_parameters = [
            *(f"sd.{name}" for name in fit.compiled.noise_names),
            *fit.compiled.observation_parameter_names,
            *fit.compiled.loading_names,
        ]
        missing = [name for name in required_parameters if name not in fit.parameter_draws]
        if missing:
            raise ValueError(f"Fit is missing forecast parameters: {missing}")
        parameter_values = {
            name: fit.parameter(name)[indices] for name in required_parameters
        }
        n_channels = len(fit.channel_names)
        state_paths = np.zeros((n_draws, horizon, fit.compiled.state_dim))
        eta_model = np.zeros((n_draws, horizon, n_channels))
        observations_model = np.zeros_like(eta_model)
        for draw, posterior_index in enumerate(indices):
            state = flat_states[posterior_index, -1].copy()
            params = {name: float(values[draw]) for name, values in parameter_values.items()}
            process_sd = fit.compiled.process_vector(params)
            design = fit.compiled.design(horizon, exog=exog_future, params=params)
            for h in range(horizon):
                innovation = fit.compiled.loading @ (
                    process_sd * rng.normal(size=fit.compiled.noise_dim)
                )
                state = fit.compiled.transition @ state + innovation
                state_paths[draw, h] = state
                eta_model[draw, h] = design[h] @ state
                observations_model[draw, h] = fit.compiled.sample_observation(
                    eta_model[draw, h], params, rng
                )
        signs = np.asarray(fit.transform_sign, dtype=float)
        resolved_dates = _future_dates(fit.dates, horizon) if dates is None else np.asarray(dates)
        if np.asarray(resolved_dates).reshape(-1).size != horizon:
            raise ValueError("dates must have length equal to horizon.")
        return Forecast(
            observations=observations_model * signs[None, None, :],
            eta=eta_model * signs[None, None, :],
            states=state_paths,
            parameters=parameter_values,
            dates=resolved_dates,
            family=fit.family,
            tail=tuple(
                "lower" if sign < 0.0 else "upper" for sign in signs
            ),
            observation_model=fit.model.observations,
            transform_sign=signs,
            channel_names=fit.channel_names,
        )
    required_parameters = [
        *(f"sd.{name}" for name in fit.compiled.noise_names),
        "sigma",
        *(["xi"] if fit.family == "gev" else []),
    ]
    missing = [name for name in required_parameters if name not in fit.parameter_draws]
    if missing:
        raise ValueError(f"Fit is missing forecast parameters: {missing}")
    parameter_values = {
        name: fit.parameter(name)[indices]
        for name in required_parameters
    }
    design = fit.compiled.design(horizon, exog=exog_future)
    state_paths = np.zeros((n_draws, horizon, fit.compiled.state_dim))
    eta_model = np.zeros((n_draws, horizon))
    observations_model = np.zeros((n_draws, horizon))

    for draw, posterior_index in enumerate(indices):
        state = flat_states[posterior_index, -1].copy()
        params = {name: float(values[draw]) for name, values in parameter_values.items()}
        process_sd = fit.compiled.process_vector(params)
        for h in range(horizon):
            innovation = fit.compiled.loading @ (process_sd * rng.normal(size=fit.compiled.noise_dim))
            state = fit.compiled.transition @ state + innovation
            state_paths[draw, h] = state
            eta = float(design[h] @ state)
            eta_model[draw, h] = eta
            observations_model[draw, h] = float(
                fit.model.observation.sample(
                    eta=eta,
                    sigma=params["sigma"],
                    xi=params.get("xi"),
                    rng=rng,
                )
            )

    sign = float(fit.transform_sign)
    resolved_dates = _future_dates(fit.dates, horizon) if dates is None else np.asarray(dates)
    if np.asarray(resolved_dates).reshape(-1).size != horizon:
        raise ValueError("dates must have length equal to horizon.")
    return Forecast(
        observations=sign * observations_model,
        eta=sign * eta_model,
        states=state_paths,
        parameters=parameter_values,
        dates=resolved_dates,
        family=fit.family,
        tail="lower" if sign < 0.0 else "upper",
        observation_model=fit.model.observation,
        transform_sign=sign,
    )
