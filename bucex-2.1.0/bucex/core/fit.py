"""User-facing fit and parallel-analysis result objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..inference.plan import InferencePlan
from ..models.compiler import CompiledModel
from ..models.structural import Model


Array = np.ndarray


@dataclass
class FitResult:
    model: Any
    compiled: Any
    priors: Any
    y: Array
    state_draws: Array
    parameter_draws: dict[str, Array]
    log_posterior: Array
    plan: InferencePlan
    sampler_diagnostics: dict[str, Any] = field(default_factory=dict)
    exog: Any = None
    dates: Array | None = None
    series_name: str | None = None
    transform_sign: Any = 1.0
    schema_version: str = "2.1"
    initial_values: dict[str, Any] = field(default_factory=dict)
    auxiliary_draws: dict[str, Array] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.y, dtype=float)
        if self.is_factor_model:
            expected_channels = len(self.compiled.channel_names)
            if values.ndim != 2 or values.shape[1] != expected_channels:
                raise ValueError(
                    f"Factor observations must have shape (T, {expected_channels})."
                )
            self.y = values
            signs = np.asarray(self.transform_sign, dtype=float).reshape(-1)
            if signs.size != expected_channels or not np.all(np.isin(signs, (-1.0, 1.0))):
                raise ValueError("Factor transform_sign must contain one +/-1 value per channel.")
            self.transform_sign = signs
        else:
            self.y = values.reshape(-1)
            self.transform_sign = float(np.asarray(self.transform_sign, dtype=float))
        self.state_draws = np.asarray(self.state_draws, dtype=float)
        self.log_posterior = np.asarray(self.log_posterior, dtype=float)
        if self.state_draws.ndim != 4:
            raise ValueError("state_draws must have shape (chains, draws, T+1, state_dim).")
        if self.log_posterior.shape != self.state_draws.shape[:2]:
            raise ValueError("log_posterior must have shape (chains, draws).")
        for key, values in self.parameter_draws.items():
            if np.asarray(values).shape[:2] != self.state_draws.shape[:2]:
                raise ValueError(f"parameter draw '{key}' has incompatible chain/draw dimensions.")
        for key, values in self.auxiliary_draws.items():
            values = np.asarray(values)
            if values.ndim >= 2 and values.shape[:2] != self.state_draws.shape[:2]:
                raise ValueError(f"auxiliary draw '{key}' has incompatible chain/draw dimensions.")
        if self.state_draws.shape[2] != self.n_time + 1:
            raise ValueError("state_draws must contain T+1 states for T observations.")
        if self.state_draws.shape[3] != self.compiled.state_dim:
            raise ValueError("state_draws has the wrong state dimension.")
        if self.dates is not None and np.asarray(self.dates).reshape(-1).size != self.n_time:
            raise ValueError("dates must have length T.")

    @property
    def is_factor_model(self) -> bool:
        return hasattr(self.compiled, "channel_names")

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(self.compiled.channel_names) if self.is_factor_model else ()

    @property
    def n_chains(self) -> int:
        return int(self.state_draws.shape[0])

    @property
    def draws_per_chain(self) -> int:
        return int(self.state_draws.shape[1])

    @property
    def n_draws(self) -> int:
        return self.n_chains * self.draws_per_chain

    @property
    def n_time(self) -> int:
        return int(self.y.shape[0])

    @property
    def state_names(self) -> tuple[str, ...]:
        return self.compiled.state_names

    @property
    def family(self) -> str:
        return self.model.family

    @property
    def obs(self):
        """Resolved observation model retained with the fit."""

        return self.model.observations if self.is_factor_model else self.model.observation

    @property
    def obs_name(self) -> str:
        return self.family

    @property
    def observed(self) -> Array:
        """Observations on their original orientation (not the minima transform)."""

        if self.is_factor_model:
            return self.y * np.asarray(self.transform_sign)[None, :]
        return float(self.transform_sign) * self.y

    @property
    def config(self) -> dict[str, Any]:
        """Stored MCMC, particle, and Laplace configuration."""

        return {
            key: self.sampler_diagnostics.get(key, {})
            for key in ("mcmc", "particles", "laplace")
        }

    @property
    def methods(self) -> dict[str, Any]:
        return {
            "engine": self.plan.engine,
            "parameterization": self.plan.parameterization,
            "asis": bool(self.plan.asis),
        }

    @property
    def meta(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "engine": self.plan.engine,
            "parameterization": self.plan.parameterization,
            "asis": self.plan.asis,
            "targets_exact_posterior": self.plan.targets_exact_posterior,
            "approximation": self.plan.approximation,
            "n_chains": self.n_chains,
            "draws_per_chain": self.draws_per_chain,
            "series_name": self.series_name,
            "transform_sign": (
                np.asarray(self.transform_sign).tolist()
                if self.is_factor_model
                else float(self.transform_sign)
            ),
            "prior_profile": getattr(self.priors, "profile", "custom"),
            "period": self.model.period,
            "continuous_spike_slab": any(
                name.startswith("slab.") for name in self.parameter_draws
            ),
            **self.metadata,
        }

    def parameter(self, name: str, *, combine_chains: bool = True) -> Array:
        if name not in self.parameter_draws:
            raise KeyError(f"Unknown parameter '{name}'. Available: {sorted(self.parameter_draws)}")
        values = np.asarray(self.parameter_draws[name])
        return values.reshape((-1,) + values.shape[2:]) if combine_chains else values

    def state(self, name: str, *, combine_chains: bool = True, include_initial: bool = False) -> Array:
        if name not in self.state_names:
            raise KeyError(f"Unknown state '{name}'. Available: {self.state_names}")
        start = 0 if include_initial else 1
        values = self.state_draws[:, :, start:, self.state_names.index(name)]
        return values.reshape((-1, values.shape[-1])) if combine_chains else values

    def state_original(self, name: str, *, combine_chains: bool = True, include_initial: bool = False) -> Array:
        """Return a state contribution on the original response orientation."""

        if self.is_factor_model:
            return self.state(
                name,
                combine_chains=combine_chains,
                include_initial=include_initial,
            )
        return float(self.transform_sign) * self.state(
            name,
            combine_chains=combine_chains,
            include_initial=include_initial,
        )

    def eta_draws(self, *, combine_chains: bool = True, original_scale: bool = False) -> Array:
        if self.is_factor_model:
            values = np.zeros(
                (
                    self.n_chains,
                    self.draws_per_chain,
                    self.n_time,
                    len(self.channel_names),
                ),
                dtype=float,
            )
            channel_index = {
                name: index for index, name in enumerate(self.channel_names)
            }
            paths = self.state_draws[:, :, 1:]
            for block in self.compiled.blocks:
                local_design = block.compiled.design(self.n_time)
                contribution = np.einsum(
                    "cdtm,tm->cdt",
                    paths[..., block.state_slice],
                    local_design,
                )
                if block.kind == "channel":
                    values[..., channel_index[block.name]] += contribution
                    continue
                factor = self.model.factor(block.name)
                for channel in self.channel_names:
                    specification = factor.loading_for(channel)
                    key = f"loading.{block.name}.{channel}"
                    if key in self.parameter_draws:
                        loading = np.asarray(self.parameter_draws[key], dtype=float)
                    else:
                        loading = np.full(
                            (self.n_chains, self.draws_per_chain),
                            float(specification.value),
                        )
                    values[..., channel_index[channel]] += (
                        loading[:, :, None] * contribution
                    )
            if original_scale:
                values = values * np.asarray(self.transform_sign)[None, None, None, :]
            if combine_chains:
                return values.reshape((-1, self.n_time, len(self.channel_names)))
            return values
        design = self.compiled.design()
        values = np.einsum("cdtm,tm->cdt", self.state_draws[:, :, 1:], design)
        if original_scale:
            values = float(self.transform_sign) * values
        return values.reshape((-1, self.n_time)) if combine_chains else values

    def channel_eta_draws(
        self,
        channel: str,
        *,
        combine_chains: bool = True,
        original_scale: bool = True,
    ) -> Array:
        if not self.is_factor_model:
            raise ValueError("channel_eta_draws is available only for factor models.")
        if channel not in self.channel_names:
            raise KeyError(f"Unknown channel '{channel}'. Available: {self.channel_names}")
        values = self.eta_draws(
            combine_chains=combine_chains,
            original_scale=original_scale,
        )
        return values[..., self.channel_names.index(channel)]

    def loading_draws(
        self,
        factor: str,
        channel: str,
        *,
        original_scale: bool = False,
    ) -> Array:
        if not self.is_factor_model:
            raise ValueError("loading_draws is available only for factor models.")
        if channel not in self.channel_names:
            raise KeyError(f"Unknown channel '{channel}'. Available: {self.channel_names}")
        key = f"loading.{factor}.{channel}"
        if key in self.parameter_draws:
            values = self.parameter(key)
        else:
            spec = self.model.factor(factor).loading_for(channel)
            values = np.full(self.n_draws, float(spec.value))
        if original_scale:
            values = values * float(
                np.asarray(self.transform_sign)[self.channel_names.index(channel)]
            )
        return values

    def factor_draws(
        self,
        factor: str,
        *,
        state: str = "level",
        combine_chains: bool = True,
        include_initial: bool = False,
    ) -> Array:
        if not self.is_factor_model:
            raise ValueError("factor_draws is available only for factor models.")
        return self.state(
            f"factor.{factor}.{state}",
            combine_chains=combine_chains,
            include_initial=include_initial,
        )

    def _factor_name(self, factor: str | None = None) -> str:
        if not self.is_factor_model:
            raise ValueError("This method is available only for factor models.")
        names = tuple(self.compiled.factor_names)
        if factor is None:
            if len(names) != 1:
                raise ValueError(f"Choose factor from {names}.")
            return names[0]
        if factor not in names:
            raise KeyError(f"Unknown factor '{factor}'. Available: {names}")
        return factor

    def factor(
        self,
        name: str | None = None,
        *,
        state: str = "level",
        combine_chains: bool = True,
        include_initial: bool = False,
    ) -> Array:
        """Compact alias for :meth:`factor_draws`.

        For a one-factor fit, ``name`` may be omitted.
        """

        return self.factor_draws(
            self._factor_name(name),
            state=state,
            combine_chains=combine_chains,
            include_initial=include_initial,
        )

    def reconstructed_state(
        self,
        channel: str,
        *,
        combine_chains: bool = True,
        original_scale: bool = True,
    ) -> Array:
        """Posterior draws of a channel's complete reconstructed predictor."""

        return self.channel_eta_draws(
            channel,
            combine_chains=combine_chains,
            original_scale=original_scale,
        )

    # Compatibility with the earlier PosteriorBundle API.
    def mu_draws(self, *, original_scale: bool = True) -> Array:
        return self.eta_draws(combine_chains=True, original_scale=original_scale)

    @property
    def draws_states(self) -> Array:
        return self.state_draws.reshape((-1,) + self.state_draws.shape[2:])

    @property
    def draws_static(self) -> dict[str, Array]:
        output = {name: self.parameter(name) for name in self.parameter_draws}
        # Compact aliases used by the 0.3 research scripts.  The canonical v1
        # names remain ``sd.<process>`` and are the only names stored on disk.
        aliases = {"level": "level", "slope": "trend", "seasonal": "season"}
        for process, legacy in aliases.items():
            key = f"sd.{process}"
            if key in output:
                output.setdefault(f"s_{legacy}", output[key])
                output.setdefault(f"q_{legacy}", output[key] ** 2)
        return output

    @property
    def draws_aux(self) -> dict[str, Array]:
        """Algorithm-specific draws, separate from semantic model states."""

        output = {name: np.asarray(value) for name, value in self.auxiliary_draws.items()}
        for name, value in self.sampler_diagnostics.get("draw_metrics", {}).items():
            output.setdefault(name, np.asarray(value))
        return output

    @property
    def initial_params(self) -> dict[str, Any]:
        """Compatibility alias for the recorded per-chain initial values."""

        return self.initial_values

    @property
    def logpost(self) -> Array:
        return self.log_posterior.reshape(-1)

    @property
    def acceptance(self) -> dict[str, float]:
        values = self.sampler_diagnostics.get("acceptance", {})
        return {name: float(np.nanmean(rate)) for name, rate in values.items()}

    def process_sd_draws(self) -> dict[str, Array]:
        return {name: self.parameter(f"sd.{name}") for name in self.compiled.noise_names}

    def posterior_summary(
        self,
        values: Array,
        *,
        credible_interval: float = 0.90,
        axis: int = 0,
    ) -> dict[str, Array]:
        alpha = 1.0 - float(credible_interval)
        low, median, high = np.quantile(values, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0], axis=axis)
        return {"lower": low, "median": median, "upper": high}

    def static_summary(self, credible_interval: float = 0.90):
        rows: dict[str, dict[str, float]] = {}
        for name in self.parameter_draws:
            values = self.parameter(name)
            if values.ndim != 1:
                continue
            summary = self.posterior_summary(values, credible_interval=credible_interval)
            rows[name] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                **{key: float(value) for key, value in summary.items()},
            }
        return rows

    def inclusion_probabilities(self) -> dict[str, float]:
        output = {}
        for name in self.compiled.noise_names:
            key = f"slab.{name}"
            if key in self.parameter_draws:
                output[name] = float(np.mean(self.parameter(key)))
        fs_keys = {
            "level": "state_level",
            "slope": "state_trend",
            "seasonal": "state_season",
        }
        for name, key in fs_keys.items():
            if key in self.parameter_draws:
                output[name] = float(np.mean(self.parameter(key) == 2))
        if not output:
            raise ValueError("This fit has no spike-and-slab process priors.")
        return output

    def component_probabilities(self):
        """Posterior spike/slab probabilities for continuous selection.

        Unlike the prototype's dimension-changing SSVS table, these are
        probabilities for a near-zero continuous spike versus a wider dynamic
        slab.  Exact fixed components are declared with :class:`FixedSD`.
        """

        fs_keys = {
            "level": "state_level",
            "slope": "state_trend",
            "seasonal": "state_season",
        }
        if any(key in self.parameter_draws for key in fs_keys.values()):
            rows = []
            for name, key in fs_keys.items():
                if key not in self.parameter_draws:
                    continue
                values = np.asarray(self.parameter(key), dtype=int)
                rows.append(
                    {
                        "process": name,
                        "zero": float(np.mean(values == 0)),
                        "fixed": float(np.mean(values == 1)),
                        "dynamic": float(np.mean(values == 2)),
                    }
                )
        else:
            probabilities = self.inclusion_probabilities()
            rows = [
                {"process": name, "spike": 1.0 - value, "slab": value}
                for name, value in probabilities.items()
            ]
        try:
            import pandas as pd

            return pd.DataFrame(rows).set_index("process")
        except ImportError:
            return rows

    def component_transition_summary(self):
        """Switching diagnostics for continuous spike/slab indicators."""

        rows = []
        for name in self.compiled.noise_names:
            key = f"slab.{name}"
            if key not in self.parameter_draws:
                continue
            values = np.asarray(self.parameter(key), dtype=int).reshape(-1)
            switches = int(np.sum(values[1:] != values[:-1])) if values.size > 1 else 0
            rows.append(
                {
                    "process": name,
                    "n_draws": int(values.size),
                    "n_switches": switches,
                    "switch_rate": float(switches / max(values.size - 1, 1)),
                    "first_state": "slab" if values[0] else "spike",
                    "last_state": "slab" if values[-1] else "spike",
                }
            )
        for name, key in {
            "level": "state_level",
            "slope": "state_trend",
            "seasonal": "state_season",
        }.items():
            if key not in self.parameter_draws:
                continue
            values = np.asarray(self.parameter(key), dtype=int).reshape(-1)
            switches = int(np.sum(values[1:] != values[:-1])) if values.size > 1 else 0
            labels = {0: "zero", 1: "fixed", 2: "dynamic"}
            rows.append(
                {
                    "process": name,
                    "n_draws": int(values.size),
                    "n_switches": switches,
                    "switch_rate": float(switches / max(values.size - 1, 1)),
                    "first_state": labels[int(values[0])],
                    "last_state": labels[int(values[-1])],
                }
            )
        if not rows:
            raise ValueError("This fit has no spike-and-slab process priors.")
        try:
            import pandas as pd

            return pd.DataFrame(rows).set_index("process")
        except ImportError:
            return rows

    def structural_model_probabilities(self):
        """Joint probabilities of continuous spike/slab allocations."""

        fs_keys = {
            "level": "state_level",
            "slope": "state_trend",
            "seasonal": "state_season",
        }
        if any(key in self.parameter_draws for key in fs_keys.values()):
            names = [name for name, key in fs_keys.items() if key in self.parameter_draws]
            values = np.column_stack(
                [np.asarray(self.parameter(fs_keys[name]), dtype=int) for name in names]
            )
            labels = {0: "zero", 1: "fixed", 2: "dynamic"}
            unique, counts = np.unique(values, axis=0, return_counts=True)
            rows = []
            for allocation, count in zip(unique, counts):
                row = {
                    name: labels[int(indicator)]
                    for name, indicator in zip(names, allocation)
                }
                row["probability"] = float(count / values.shape[0])
                rows.append(row)
            try:
                import pandas as pd

                return pd.DataFrame(rows).sort_values(
                    "probability", ascending=False
                ).reset_index(drop=True)
            except ImportError:
                return sorted(rows, key=lambda row: row["probability"], reverse=True)

        names = [
            name for name in self.compiled.noise_names
            if f"slab.{name}" in self.parameter_draws
        ]
        if not names:
            raise ValueError("This fit has no spike-and-slab process priors.")
        values = np.column_stack(
            [np.asarray(self.parameter(f"slab.{name}"), dtype=int) for name in names]
        )
        unique, counts = np.unique(values, axis=0, return_counts=True)
        rows = []
        for allocation, count in zip(unique, counts):
            row = {
                name: "slab" if indicator else "spike"
                for name, indicator in zip(names, allocation)
            }
            row["probability"] = float(count / values.shape[0])
            rows.append(row)
        try:
            import pandas as pd

            return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)
        except ImportError:
            return sorted(rows, key=lambda row: row["probability"], reverse=True)

    def most_probable_structure(self) -> dict[str, Any]:
        table = self.structural_model_probabilities()
        if hasattr(table, "iloc"):
            return dict(table.iloc[0])
        return dict(table[0])

    def _observation_context(self, channel: str | None = None):
        if self.is_factor_model:
            if channel is None:
                raise ValueError(f"Choose channel from {self.channel_names}.")
            if channel not in self.channel_names:
                raise KeyError(f"Unknown channel '{channel}'. Available: {self.channel_names}")
            index = self.channel_names.index(channel)
            specification = self.model.channel(channel)
            return {
                "channel": channel,
                "family": specification.family,
                "observation": specification.observation,
                "sign": float(np.asarray(self.transform_sign)[index]),
                "eta": self.channel_eta_draws(
                    channel, combine_chains=True, original_scale=False
                ),
                "sigma": self.parameter(f"sigma.{channel}")[:, None],
                "xi": (
                    self.parameter(f"xi.{channel}")[:, None]
                    if specification.family == "gev"
                    else None
                ),
            }
        if channel is not None:
            raise ValueError("channel= is only valid for factor-model results.")
        return {
            "channel": None,
            "family": self.family,
            "observation": self.model.observation,
            "sign": float(self.transform_sign),
            "eta": self.eta_draws(),
            "sigma": self.parameter("sigma")[:, None],
            "xi": self.parameter("xi")[:, None] if self.family == "gev" else None,
        }

    def endpoint_draws(
        self,
        *,
        original_scale: bool = True,
        channel: str | None = None,
    ) -> Array:
        context = self._observation_context(channel)
        if context["family"] != "gev":
            raise ValueError("Endpoints are defined only for GEV fits.")
        eta = context["eta"]
        sigma = context["sigma"]
        xi = context["xi"]
        adjustment = np.full(np.broadcast_shapes(eta.shape, sigma.shape, xi.shape), np.inf)
        np.divide(-sigma, xi, out=adjustment, where=xi < 0.0)
        endpoint = np.where(xi < 0.0, eta + adjustment, np.inf)
        return context["sign"] * endpoint if original_scale else endpoint

    @property
    def time(self) -> Array:
        return np.arange(1, self.n_time + 1) if self.dates is None else np.asarray(self.dates)

    def _annual_groups(self) -> tuple[list[Array], Array]:
        if self.dates is not None:
            try:
                import pandas as pd

                years = np.asarray(pd.to_datetime(self.dates).year)
                unique = np.unique(years)
                return [np.flatnonzero(years == year) for year in unique], unique
            except Exception:
                pass
        period = int(self.model.period or 1)
        groups = [
            np.arange(start, min(start + period, self.n_time))
            for start in range(0, self.n_time, period)
        ]
        return groups, np.arange(1, len(groups) + 1)

    def exceedance_probability_draws(
        self,
        threshold: float,
        *,
        annual: bool = False,
        return_labels: bool = True,
        channel: str | None = None,
    ) -> Array | tuple[Array, Array]:
        context = self._observation_context(channel)
        threshold_model = context["sign"] * float(threshold)
        cdf = context["observation"].cdf(
            threshold_model,
            context["eta"],
            sigma=context["sigma"],
            xi=context["xi"],
        )
        probability = np.clip(1.0 - cdf, 0.0, 1.0)
        labels = self.time
        if annual:
            groups, labels = self._annual_groups()
            probability = np.column_stack(
                [1.0 - np.prod(1.0 - probability[:, group], axis=1) for group in groups]
            )
        return (probability, labels) if return_labels else probability

    def return_period_draws(
        self,
        threshold: float,
        *,
        annual: bool = True,
        minimum_probability: float = 1e-12,
        return_labels: bool = True,
        min_probability: float | None = None,
        channel: str | None = None,
    ) -> Array | tuple[Array, Array]:
        if min_probability is not None:
            minimum_probability = float(min_probability)
        probability, labels = self.exceedance_probability_draws(
            threshold, annual=annual, return_labels=True, channel=channel
        )
        values = 1.0 / np.maximum(probability, float(minimum_probability))
        return (values, labels) if return_labels else values

    def return_level_draws(
        self,
        return_period: float,
        *,
        channel: str | None = None,
    ) -> Array:
        context = self._observation_context(channel)
        if context["family"] != "gev":
            raise ValueError("Return levels are defined only for GEV fits.")
        if float(return_period) <= 1.0:
            raise ValueError("return_period must exceed 1.")
        probability = 1.0 - 1.0 / float(return_period)
        level = context["observation"].ppf(
            probability,
            context["eta"],
            sigma=context["sigma"],
            xi=context["xi"],
        )
        return context["sign"] * level

    def event_label(self, threshold: float, *, channel: str | None = None) -> str:
        context = self._observation_context(channel)
        operator = "<" if context["sign"] < 0.0 else ">"
        label = channel if channel is not None else (self.series_name or "Y")
        return f"P({label} {operator} {float(threshold):g})"

    def level_rate_draws(
        self,
        start_year: int,
        end_year: int,
        *,
        scale: str | float = "decade",
    ) -> Array:
        """Finite-change rate of the latent level between calendar years."""

        if self.dates is None:
            raise ValueError("level_rate_draws requires calendar dates.")
        if int(end_year) <= int(start_year):
            raise ValueError("Require end_year > start_year.")
        import pandas as pd

        years = np.asarray(pd.to_datetime(self.dates).year, dtype=int)
        start = np.flatnonzero(years == int(start_year))
        end = np.flatnonzero(years == int(end_year))
        if start.size == 0 or end.size == 0:
            raise ValueError(
                f"Requested years are unavailable; fitted range is {years.min()}-{years.max()}."
            )
        level = self.state_original("level")
        annual_rate = (
            np.mean(level[:, end], axis=1) - np.mean(level[:, start], axis=1)
        ) / float(end_year - start_year)
        if isinstance(scale, (int, float)):
            multiplier = float(scale)
        else:
            key = str(scale).lower()
            multipliers = {
                "year": 1.0,
                "annual": 1.0,
                "decade": 10.0,
                "decadal": 10.0,
                "century": 100.0,
                "centennial": 100.0,
            }
            if key not in multipliers:
                raise ValueError("scale must be year, decade, century, or a numeric multiplier.")
            multiplier = multipliers[key]
        return multiplier * annual_rate

    def factor_rate_draws(
        self,
        factor: str,
        start_year: int,
        end_year: int,
        *,
        scale: str | float = "decade",
    ) -> Array:
        """Finite-change rate of a shared factor's level state."""

        if not self.is_factor_model:
            raise ValueError("factor_rate_draws is available only for factor models.")
        return self._calendar_rate(
            self.factor_draws(factor, state="level"),
            start_year,
            end_year,
            scale=scale,
        )

    def factor_rate_summary(
        self,
        factor: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        *,
        scale: str | float = "decade",
        credible_interval: float = 0.90,
    ) -> dict[str, float | int | str]:
        """Summarize the finite-change rate of a shared factor.

        With one factor, its name may be omitted.  If the years are omitted,
        the first and last fitted calendar years are used.
        """

        resolved = self._factor_name(factor)
        if self.dates is None:
            raise ValueError("factor_rate_summary requires fitted calendar dates.")
        import pandas as pd

        years = np.asarray(pd.to_datetime(self.dates).year, dtype=int)
        start = int(years.min()) if start_year is None else int(start_year)
        end = int(years.max()) if end_year is None else int(end_year)
        values = self.factor_rate_draws(resolved, start, end, scale=scale)
        summary = self.posterior_summary(
            values, credible_interval=credible_interval
        )
        return {
            "factor": resolved,
            "start_year": start,
            "end_year": end,
            "lower": float(summary["lower"]),
            "median": float(summary["median"]),
            "upper": float(summary["upper"]),
            "probability_positive": float(np.mean(values > 0.0)),
        }

    def loading_probability(
        self,
        factor: str | None,
        channel: str,
        *,
        threshold: float = 1.0,
        above: bool = True,
        original_scale: bool = True,
    ) -> float:
        """Posterior probability that a loading is above/below a threshold."""

        resolved = self._factor_name(factor)
        values = self.loading_draws(
            resolved, channel, original_scale=original_scale
        )
        comparison = values > float(threshold) if above else values < float(threshold)
        return float(np.mean(comparison))

    def factor_probabilities(
        self,
        factor: str | None = None,
        *,
        start_year: int | None = None,
        end_year: int | None = None,
        scale: str | float = "decade",
        loading_threshold: float = 1.0,
        original_scale: bool = True,
    ) -> dict[str, Any]:
        """Rate and loading probabilities for a one-factor interpretation."""

        resolved = self._factor_name(factor)
        rate = self.factor_rate_summary(
            resolved,
            start_year,
            end_year,
            scale=scale,
        )
        loadings = {
            channel: self.loading_draws(
                resolved, channel, original_scale=original_scale
            )
            for channel in self.channel_names
        }
        return {
            "factor": resolved,
            "start_year": rate["start_year"],
            "end_year": rate["end_year"],
            "factor_rate_positive": rate["probability_positive"],
            "loading_threshold": float(loading_threshold),
            "loading_above_threshold": {
                channel: float(np.mean(values > float(loading_threshold)))
                for channel, values in loadings.items()
            },
            "loading_positive": {
                channel: float(np.mean(values > 0.0))
                for channel, values in loadings.items()
            },
        }

    def channel_rate_draws(
        self,
        channel: str,
        start_year: int,
        end_year: int,
        *,
        scale: str | float = "decade",
    ) -> Array:
        """Finite-change rate of a channel's full latent predictor."""

        if not self.is_factor_model:
            raise ValueError("channel_rate_draws is available only for factor models.")
        return self._calendar_rate(
            self.channel_eta_draws(channel, original_scale=True),
            start_year,
            end_year,
            scale=scale,
        )

    def _calendar_rate(
        self,
        values: Array,
        start_year: int,
        end_year: int,
        *,
        scale: str | float,
    ) -> Array:
        if self.dates is None:
            raise ValueError("Calendar rate methods require fitted dates.")
        if int(end_year) <= int(start_year):
            raise ValueError("Require end_year > start_year.")
        import pandas as pd

        years = np.asarray(pd.to_datetime(self.dates).year, dtype=int)
        start = np.flatnonzero(years == int(start_year))
        end = np.flatnonzero(years == int(end_year))
        if start.size == 0 or end.size == 0:
            raise ValueError(
                f"Requested years are unavailable; fitted range is {years.min()}-{years.max()}."
            )
        annual_rate = (
            np.mean(values[:, end], axis=1) - np.mean(values[:, start], axis=1)
        ) / float(end_year - start_year)
        if isinstance(scale, (int, float)):
            multiplier = float(scale)
        else:
            multipliers = {
                "year": 1.0,
                "annual": 1.0,
                "decade": 10.0,
                "decadal": 10.0,
                "century": 100.0,
                "centennial": 100.0,
            }
            key = str(scale).lower()
            if key not in multipliers:
                raise ValueError("scale must be year, decade, century, or numeric.")
            multiplier = multipliers[key]
        return multiplier * annual_rate

    def period_rate_draws(
        self,
        periods: Mapping[str, tuple[int, int]],
        *,
        scale: str | float = "decade",
    ) -> dict[str, Array]:
        return {
            str(name): self.level_rate_draws(start, end, scale=scale)
            for name, (start, end) in periods.items()
        }

    def period_rate_summary(
        self,
        periods: Mapping[str, tuple[int, int]],
        *,
        scale: str | float = "decade",
        credible_interval: float = 0.90,
    ):
        rows = []
        for name, values in self.period_rate_draws(periods, scale=scale).items():
            summary = self.posterior_summary(values, credible_interval=credible_interval)
            rows.append(
                {
                    "period": name,
                    "lower": float(summary["lower"]),
                    "median": float(summary["median"]),
                    "upper": float(summary["upper"]),
                    "probability_positive": float(np.mean(values > 0.0)),
                }
            )
        try:
            import pandas as pd

            return pd.DataFrame(rows).set_index("period")
        except ImportError:
            return rows

    def rate_contrast_draws(
        self,
        recent: tuple[int, int],
        reference: tuple[int, int],
        *,
        scale: str | float = "decade",
    ) -> Array:
        return self.level_rate_draws(*recent, scale=scale) - self.level_rate_draws(
            *reference, scale=scale
        )

    def rate_contrast_summary(
        self,
        recent: tuple[int, int],
        reference: tuple[int, int],
        *,
        scale: str | float = "decade",
        credible_interval: float = 0.90,
    ) -> dict[str, float]:
        values = self.rate_contrast_draws(recent, reference, scale=scale)
        summary = self.posterior_summary(values, credible_interval=credible_interval)
        return {
            "lower": float(summary["lower"]),
            "median": float(summary["median"]),
            "upper": float(summary["upper"]),
            "probability_positive": float(np.mean(values > 0.0)),
        }

    def forecast(self, horizon: int, **kwargs):
        from ..api.predict import posterior_predict

        return posterior_predict(self, horizon, **kwargs)

    def diagnostics(self):
        from ..diagnostics.posterior import fit_diagnostics

        return fit_diagnostics(self)

    def plot(self, kind: str = "state", *, type: str | None = None, **kwargs):
        from ..plotting import plot_fit

        return plot_fit(self, kind=kind if type is None else type, **kwargs)

    def save(self, path: str | Path) -> None:
        from ..io import save_fit

        save_fit(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FitResult":
        from ..io import load_fit

        return load_fit(path)

    def summary_dict(self) -> dict[str, Any]:
        return {**self.meta, "state_names": self.state_names, "parameters": self.static_summary()}


@dataclass
class BulkTailFit:
    bulk: FitResult
    tail: FitResult
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: {
            "joint_likelihood": False,
            "posterior_draws_paired": False,
            "interpretation": "parallel independent bulk and tail fits",
        }
    )

    def __post_init__(self) -> None:
        if self.bulk.family != "gaussian" or self.tail.family != "gev":
            raise ValueError("BulkTailFit requires a Gaussian bulk fit and a GEV tail fit.")
        if self.bulk.n_time != self.tail.n_time:
            raise ValueError("Bulk and tail fits must be aligned and have equal length.")

    def forecast(self, horizon: int, **kwargs) -> dict[str, Any]:
        seed = kwargs.pop("seed", None)
        sequence = np.random.SeedSequence(seed)
        bulk_seed, tail_seed = [int(item.generate_state(1)[0]) for item in sequence.spawn(2)]
        return {
            "bulk": self.bulk.forecast(horizon, seed=bulk_seed, **kwargs),
            "tail": self.tail.forecast(horizon, seed=tail_seed, **kwargs),
        }

    def process_sd_summary(self, credible_interval: float = 0.90):
        rows = []
        for role, fit in (("bulk", self.bulk), ("tail", self.tail)):
            for name, values in fit.process_sd_draws().items():
                summary = fit.posterior_summary(values, credible_interval=credible_interval)
                rows.append(
                    {
                        "role": role,
                        "process": name,
                        **{key: float(value) for key, value in summary.items()},
                    }
                )
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def plot(self, kind: str = "states", **kwargs):
        from ..plotting import plot_bulk_tail

        return plot_bulk_tail(self, kind=kind, **kwargs)


# Prototype compatibility name.
PosteriorBundle = FitResult


def _same_data(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_same_data(left[key], right[key]) for key in left)
    return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))


def combine_fits(fits: Iterable[FitResult]) -> FitResult:
    """Combine independently run compatible fits along the chain dimension."""

    items = list(fits)
    if not items:
        raise ValueError("At least one fit is required.")
    first = items[0]
    for index, fit in enumerate(items[1:], start=2):
        if fit.model != first.model or fit.priors != first.priors or fit.plan != first.plan:
            raise ValueError(f"Fit {index} has a different model, prior, or inference plan.")
        if fit.draws_per_chain != first.draws_per_chain:
            raise ValueError("All fits must have the same number of retained draws per chain.")
        if not np.array_equal(fit.y, first.y, equal_nan=True):
            raise ValueError(f"Fit {index} uses different observations.")
        if not _same_data(fit.exog, first.exog):
            raise ValueError(f"Fit {index} uses different exogenous values.")
        if (fit.dates is None) != (first.dates is None) or (
            fit.dates is not None and not np.array_equal(fit.dates, first.dates)
        ):
            raise ValueError(f"Fit {index} uses different dates.")
        if set(fit.parameter_draws) != set(first.parameter_draws):
            raise ValueError(f"Fit {index} stores different parameters.")

    diagnostics = {
        key: value
        for key, value in first.sampler_diagnostics.items()
        if key not in {"acceptance", "final_proposal_steps", "draw_metrics", "mcmc"}
    }
    for group in ("acceptance", "final_proposal_steps", "draw_metrics"):
        keys = set(first.sampler_diagnostics.get(group, {}))
        if any(set(fit.sampler_diagnostics.get(group, {})) != keys for fit in items):
            raise ValueError(f"Fits have incompatible {group} diagnostics.")
        diagnostics[group] = {
            name: np.concatenate(
                [np.asarray(fit.sampler_diagnostics[group][name]) for fit in items],
                axis=0,
            )
            for name in keys
        }
    mcmc = dict(first.sampler_diagnostics.get("mcmc", {}))
    mcmc["chains"] = int(sum(fit.n_chains for fit in items))
    mcmc["combined_independent_runs"] = len(items)
    mcmc["seeds"] = [fit.sampler_diagnostics.get("mcmc", {}).get("seed") for fit in items]
    diagnostics["mcmc"] = mcmc

    auxiliary_keys = set(first.auxiliary_draws)
    if any(set(fit.auxiliary_draws) != auxiliary_keys for fit in items):
        raise ValueError("Fits store different auxiliary draws.")
    initial_parameters = []
    initial_indicators = []
    for fit in items:
        initial_parameters.extend(fit.initial_values.get("parameters_by_chain", []))
        initial_indicators.extend(fit.initial_values.get("indicators_by_chain", []))
    initial_values = dict(first.initial_values)
    initial_values["parameters_by_chain"] = initial_parameters
    initial_values["indicators_by_chain"] = initial_indicators

    return FitResult(
        model=first.model,
        compiled=first.compiled,
        priors=first.priors,
        y=first.y.copy(),
        exog=None if first.exog is None else np.asarray(first.exog).copy(),
        dates=None if first.dates is None else np.asarray(first.dates).copy(),
        series_name=first.series_name,
        transform_sign=first.transform_sign,
        state_draws=np.concatenate([fit.state_draws for fit in items], axis=0),
        parameter_draws={
            name: np.concatenate([fit.parameter_draws[name] for fit in items], axis=0)
            for name in first.parameter_draws
        },
        log_posterior=np.concatenate([fit.log_posterior for fit in items], axis=0),
        plan=first.plan,
        sampler_diagnostics=diagnostics,
        schema_version=first.schema_version,
        initial_values=initial_values,
        auxiliary_draws={
            name: np.concatenate([fit.auxiliary_draws[name] for fit in items], axis=0)
            for name in auxiliary_keys
        },
        metadata=dict(first.metadata),
    )
