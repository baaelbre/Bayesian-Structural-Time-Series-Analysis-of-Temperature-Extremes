"""User-facing fit and parallel-analysis result objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .compiler import CompiledModel
from .model import Model
from .plan import InferencePlan
from .priors import Priors


Array = np.ndarray


@dataclass
class FitResult:
    model: Model
    compiled: CompiledModel
    priors: Priors
    y: Array
    state_draws: Array
    parameter_draws: dict[str, Array]
    log_posterior: Array
    plan: InferencePlan
    sampler_diagnostics: dict[str, Any] = field(default_factory=dict)
    exog: Array | None = None
    dates: Array | None = None
    series_name: str | None = None
    transform_sign: float = 1.0
    schema_version: str = "1.0"
    initial_values: dict[str, Any] = field(default_factory=dict)
    auxiliary_draws: dict[str, Array] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y, dtype=float).reshape(-1)
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
        if self.state_draws.shape[2] != self.y.size + 1:
            raise ValueError("state_draws must contain T+1 states for T observations.")
        if self.state_draws.shape[3] != self.compiled.state_dim:
            raise ValueError("state_draws has the wrong state dimension.")
        if self.dates is not None and np.asarray(self.dates).reshape(-1).size != self.y.size:
            raise ValueError("dates must have length T.")

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
        return int(self.y.size)

    @property
    def state_names(self) -> tuple[str, ...]:
        return self.compiled.state_names

    @property
    def family(self) -> str:
        return self.model.family

    @property
    def obs(self):
        """Resolved observation model retained with the fit."""

        return self.model.observation

    @property
    def obs_name(self) -> str:
        return self.family

    @property
    def observed(self) -> Array:
        """Observations on their original orientation (not the minima transform)."""

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
            "transform_sign": float(self.transform_sign),
            "prior_profile": self.priors.profile,
            "period": self.model.period,
            "continuous_spike_slab": any(
                name.startswith("slab.") for name in self.parameter_draws
            ),
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

        return float(self.transform_sign) * self.state(
            name,
            combine_chains=combine_chains,
            include_initial=include_initial,
        )

    def eta_draws(self, *, combine_chains: bool = True, original_scale: bool = False) -> Array:
        design = self.compiled.design()
        values = np.einsum("cdtm,tm->cdt", self.state_draws[:, :, 1:], design)
        if original_scale:
            values = float(self.transform_sign) * values
        return values.reshape((-1, self.n_time)) if combine_chains else values

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
                output[f"s_{legacy}"] = output[key]
                output[f"q_{legacy}"] = output[key] ** 2
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
        if not output:
            raise ValueError("This fit has no spike-and-slab process priors.")
        return output

    def component_probabilities(self):
        """Posterior spike/slab probabilities for continuous selection.

        Unlike the prototype's dimension-changing SSVS table, these are
        probabilities for a near-zero continuous spike versus a wider dynamic
        slab.  Exact fixed components are declared with :class:`FixedSD`.
        """

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
        if not rows:
            raise ValueError("This fit has no spike-and-slab process priors.")
        try:
            import pandas as pd

            return pd.DataFrame(rows).set_index("process")
        except ImportError:
            return rows

    def structural_model_probabilities(self):
        """Joint probabilities of continuous spike/slab allocations."""

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

    def endpoint_draws(self, *, original_scale: bool = True) -> Array:
        if self.family != "gev":
            raise ValueError("Endpoints are defined only for GEV fits.")
        eta = self.eta_draws()
        sigma = self.parameter("sigma")[:, None]
        xi = self.parameter("xi")[:, None]
        adjustment = np.full(np.broadcast_shapes(eta.shape, sigma.shape, xi.shape), np.inf)
        np.divide(-sigma, xi, out=adjustment, where=xi < 0.0)
        endpoint = np.where(xi < 0.0, eta + adjustment, np.inf)
        return float(self.transform_sign) * endpoint if original_scale else endpoint

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
        return_labels: bool = False,
    ) -> Array | tuple[Array, Array]:
        eta = self.eta_draws()
        sigma = self.parameter("sigma")[:, None]
        xi = self.parameter("xi")[:, None] if self.family == "gev" else None
        threshold_model = float(self.transform_sign) * float(threshold)
        cdf = self.model.observation.cdf(threshold_model, eta, sigma=sigma, xi=xi)
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
        annual: bool = False,
        minimum_probability: float = 1e-12,
        return_labels: bool = False,
    ) -> Array | tuple[Array, Array]:
        probability, labels = self.exceedance_probability_draws(
            threshold, annual=annual, return_labels=True
        )
        values = 1.0 / np.maximum(probability, float(minimum_probability))
        return (values, labels) if return_labels else values

    def return_level_draws(self, return_period: float) -> Array:
        if self.family != "gev":
            raise ValueError("Return levels are defined only for GEV fits.")
        if float(return_period) <= 1.0:
            raise ValueError("return_period must exceed 1.")
        probability = 1.0 - 1.0 / float(return_period)
        eta = self.eta_draws()
        sigma = self.parameter("sigma")[:, None]
        xi = self.parameter("xi")[:, None]
        level = self.model.observation.ppf(probability, eta, sigma=sigma, xi=xi)
        return float(self.transform_sign) * level

    def event_label(self, threshold: float) -> str:
        operator = "<" if self.transform_sign < 0.0 else ">"
        return f"P({self.series_name or 'Y'} {operator} {float(threshold):g})"

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
        from .forecast import posterior_predict

        return posterior_predict(self, horizon, **kwargs)

    def diagnostics(self):
        from .diagnostics import fit_diagnostics

        return fit_diagnostics(self)

    def plot(self, kind: str = "state", *, type: str | None = None, **kwargs):
        from .plots import plot_fit

        return plot_fit(self, kind=kind if type is None else type, **kwargs)

    def save(self, path: str | Path) -> None:
        from .io import save_fit

        save_fit(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FitResult":
        from .io import load_fit

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
        from .plots import plot_bulk_tail

        return plot_bulk_tail(self, kind=kind, **kwargs)


# Prototype compatibility name.
PosteriorBundle = FitResult


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
        if (fit.exog is None) != (first.exog is None) or (
            fit.exog is not None and not np.array_equal(fit.exog, first.exog)
        ):
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
    )
