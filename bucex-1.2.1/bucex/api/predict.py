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
    tail: str
    observation_model: Any
    transform_sign: float = 1.0

    @property
    def n_draws(self) -> int:
        return int(self.observations.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[1])

    def summary(self, level: float = 0.90):
        if not 0.0 < float(level) < 1.0:
            raise ValueError("level must lie in (0, 1).")
        alpha = 1.0 - float(level)
        rows = []
        for h in range(self.horizon):
            obs = self.observations[:, h]
            eta = self.eta[:, h]
            rows.append(
                {
                    "time": self.dates[h],
                    "mean": float(np.mean(obs)),
                    "lower": float(np.quantile(obs, alpha / 2.0)),
                    "median": float(np.median(obs)),
                    "upper": float(np.quantile(obs, 1.0 - alpha / 2.0)),
                    "eta_lower": float(np.quantile(eta, alpha / 2.0)),
                    "eta_median": float(np.median(eta)),
                    "eta_upper": float(np.quantile(eta, 1.0 - alpha / 2.0)),
                }
            )
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def tail_probability(self, threshold: float) -> Array:
        if self.tail == "upper":
            return np.mean(self.observations > float(threshold), axis=0)
        return np.mean(self.observations < float(threshold), axis=0)

    def return_level(self, return_period: float) -> Array:
        if self.family != "gev":
            raise ValueError("Return levels require a GEV forecast.")
        if float(return_period) <= 1.0:
            raise ValueError("return_period must exceed 1.")
        probability = 1.0 - 1.0 / float(return_period)
        eta_model = self.transform_sign * self.eta
        sigma = self.parameters["sigma"][:, None]
        xi = self.parameters["xi"][:, None]
        level = self.observation_model.ppf(probability, eta_model, sigma=sigma, xi=xi)
        return self.transform_sign * level

    def score(
        self,
        observed: Array,
        *,
        thresholds: list[float] | tuple[float, ...] = (),
        quantiles: list[float] | tuple[float, ...] = (0.9, 0.95, 0.99),
    ):
        observed = np.asarray(observed, dtype=float).reshape(-1)
        if observed.size != self.horizon:
            raise ValueError("observed must have length equal to the forecast horizon.")
        return evaluate_ensemble(
            self.observations,
            observed,
            thresholds=thresholds,
            quantiles=quantiles,
            tail=self.tail,
        )

    def plot(self, *, level: float = 0.90, ax=None, color: str = "C0"):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        summary = self.summary(level=level)
        x = np.arange(self.horizon) if not hasattr(summary, "columns") else summary["time"]
        lower = np.asarray([row["lower"] for row in summary]) if not hasattr(summary, "columns") else summary["lower"].to_numpy()
        median = np.asarray([row["median"] for row in summary]) if not hasattr(summary, "columns") else summary["median"].to_numpy()
        upper = np.asarray([row["upper"] for row in summary]) if not hasattr(summary, "columns") else summary["upper"].to_numpy()
        ax.fill_between(x, lower, upper, color=color, alpha=0.2, label=f"{level:.0%} predictive interval")
        ax.plot(x, median, color=color, label="predictive median")
        ax.set_title("Posterior predictive forecast")
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
