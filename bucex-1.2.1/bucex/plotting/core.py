"""Data-agnostic posterior, risk, and prior-versus-posterior plots."""
from __future__ import annotations

import numpy as np

from ..priors.process import FixedSD


def _prior_density(prior, grid):
    return np.asarray([np.exp(prior.logpdf(float(value))) for value in grid])


def _interval(values, credible_interval: float, *, axis: int = 0):
    if not 0.0 < float(credible_interval) < 1.0:
        raise ValueError("credible_interval must lie in (0, 1).")
    alpha = 1.0 - float(credible_interval)
    return np.nanquantile(
        np.asarray(values, dtype=float),
        [alpha / 2.0, 0.5, 1.0 - alpha / 2.0],
        axis=axis,
    )


def _time(fit):
    return np.arange(fit.n_time) if fit.dates is None else fit.dates


def _structural_prior_samples(fit, name: str, size: int, rng) -> np.ndarray:
    """Draw the implied process SD under an FS signed-scale prior."""

    priors = fit.priors
    component = {"level": "level", "slope": "trend", "seasonal": "season"}[name]
    if getattr(priors, "pc", None) is not None:
        rate = (
            priors.pc.standardized_rate_for(component)
            / priors.pc.coefficient_scale_for(component)
        )
        return rng.exponential(scale=1.0 / rate, size=size)
    if getattr(priors, "horseshoe", None) is not None:
        prior = priors.horseshoe
        local = np.abs(rng.standard_cauchy(size=size))
        global_scale = np.abs(rng.standard_cauchy(size=size)) * prior.global_scale
        slab2 = 1.0 / rng.gamma(
            shape=0.5 * prior.slab_df,
            scale=2.0 / (prior.slab_df * prior.slab_scale**2),
            size=size,
        )
        regularized = slab2 * local**2 / (slab2 + global_scale**2 * local**2)
        variance = (
            prior.coefficient_scale_for(component) ** 2
            * global_scale**2
            * regularized
        )
        return np.abs(rng.normal(scale=np.sqrt(variance)))
    if getattr(priors, "lasso", None) is not None:
        prior = priors.lasso
        if prior.componentwise:
            lambda2 = rng.gamma(
                prior.a_for(component),
                scale=1.0 / prior.b_for(component),
                size=size,
            )
        else:
            lambda2 = rng.gamma(
                prior.a_lambda,
                scale=1.0 / prior.b_lambda,
                size=size,
            )
        tau = rng.exponential(scale=2.0 / lambda2)
        variance_scale = float(prior.fixed_variance)
        if prior.variance_mode == "observation":
            variance_scale = (
                priors.sigma2.b / (priors.sigma2.a - 1.0)
                if priors.sigma2.a > 1.0
                else 1.0
            )
        scale = prior.coefficient_scale_for(component)
        return np.abs(rng.normal(scale=np.sqrt(variance_scale * scale**2 * tau)))
    if getattr(priors, "ssvs", None) is not None:
        prior = priors.ssvs
        probability = {
            "level": prior.level_dynamic_probability,
            "trend": prior.trend_probabilities[2],
            "season": prior.season_probabilities[2],
        }[component]
        active = rng.random(size) < probability
        values = np.zeros(size)
        values[active] = np.abs(
            rng.normal(
                scale=prior.innovation_slab_sd[component],
                size=int(active.sum()),
            )
        )
        return values
    prior = getattr(priors, {"level": "s_level", "trend": "s_trend", "season": "s_season"}[component])
    return np.abs(rng.normal(loc=prior.mean, scale=prior.sd, size=size))


def plot_process_sds(
    fit,
    *,
    bins: int = 35,
    credible_interval: float = 0.90,
    prior_draws: int = 5000,
    figsize=None,
):
    """Overlay each process-SD prior with its marginal posterior."""

    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    names = list(fit.compiled.noise_names)
    if not names:
        raise ValueError("The model has no stochastic process standard deviations.")
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=figsize or (7.5, max(2.5, 2.4 * len(names))),
        squeeze=False,
    )
    for axis, name in zip(axes[:, 0], names):
        posterior = fit.parameter(f"sd.{name}")
        process_priors = getattr(fit.priors, "process", None)
        prior = None if process_priors is None else process_priors[name]
        grid_upper = max(
            float(np.quantile(posterior, 0.995)),
            float(np.max(posterior)),
            1e-10,
        )
        if prior is None:
            prior_sample = _structural_prior_samples(
                fit,
                name,
                int(prior_draws),
                np.random.default_rng(140),
            )
            grid_upper = max(grid_upper, float(np.quantile(prior_sample, 0.995)))
        elif not isinstance(prior, FixedSD):
            prior_sample = np.asarray(prior.sample(np.random.default_rng(140), size=prior_draws))
            grid_upper = max(grid_upper, float(np.quantile(prior_sample, 0.995)))
        grid = np.linspace(0.0, grid_upper * 1.05, 400)
        if prior is None:
            axis.hist(
                prior_sample,
                bins=bins,
                range=(0.0, grid[-1]),
                density=True,
                histtype="step",
                color="0.35",
                linestyle="--",
                label="prior",
            )
        elif isinstance(prior, FixedSD):
            axis.axvline(prior.value, color="0.35", linestyle="--", label="prior")
        else:
            axis.plot(
                grid,
                _prior_density(prior, grid),
                color="0.35",
                linestyle="--",
                label="prior",
            )
        if np.std(posterior) > 1e-12 and np.unique(posterior).size > 2:
            density = gaussian_kde(posterior)
            axis.plot(grid, density(grid), color="C0", label="posterior")
            axis.fill_between(grid, 0.0, density(grid), color="C0", alpha=0.18)
        else:
            axis.axvline(float(np.mean(posterior)), color="C0", label="posterior")
        lower, median, upper = _interval(posterior, credible_interval)
        axis.axvspan(lower, upper, color="C0", alpha=0.08)
        axis.axvline(median, color="C0", linewidth=1.0)
        axis.set_title(f"Process SD: {name}")
        axis.set_xlabel("innovation standard deviation")
        axis.set_ylabel("density")
        axis.legend()
    figure.tight_layout()
    return figure, axes[:, 0]


def plot_state(
    fit,
    *,
    state: str = "level",
    credible_interval: float = 0.90,
    ax=None,
    color: str = "C0",
):
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 4))
    else:
        figure = ax.figure
    values = fit.state_original(state)
    lower, median, upper = _interval(values, credible_interval)
    x = _time(fit)
    observed = fit.transform_sign * fit.y
    ax.scatter(x, observed, s=9, color="0.55", alpha=0.55, label="observed")
    ax.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=0.2,
        label=f"{credible_interval:.0%} credible interval",
    )
    ax.plot(x, median, color=color, label=f"{state} median")
    ax.set_title(fit.series_name or f"Posterior {state}")
    ax.legend()
    return figure, ax


def plot_level_slope(fit, *, credible_interval: float = 0.90, figsize=(9, 7)):
    import matplotlib.pyplot as plt

    if "slope" not in fit.state_names:
        raise ValueError("The model has no slope state.")
    figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    plot_state(fit, state="level", credible_interval=credible_interval, ax=axes[0])
    slope = fit.state_original("slope")
    lower, median, upper = _interval(slope, credible_interval)
    x = _time(fit)
    axes[1].fill_between(x, lower, upper, color="C1", alpha=0.2)
    axes[1].plot(x, median, color="C1")
    axes[1].axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    axes[1].set_title("Latent slope per observation interval")
    figure.tight_layout()
    return figure, axes


def plot_endpoint(fit, *, credible_interval: float = 0.90, ax=None, color="C3"):
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 4))
    else:
        figure = ax.figure
    values = np.asarray(fit.endpoint_draws(original_scale=True), dtype=float)
    values[~np.isfinite(values)] = np.nan
    if np.all(np.isnan(values)):
        raise ValueError("No posterior draw has a finite GEV endpoint.")
    lower, median, upper = _interval(values, credible_interval)
    x = _time(fit)
    ax.fill_between(x, lower, upper, color=color, alpha=0.2)
    ax.plot(x, median, color=color)
    direction = "lower" if fit.transform_sign < 0.0 else "upper"
    ax.set_title(f"Finite GEV {direction} endpoint")
    ax.set_ylabel("endpoint")
    return figure, ax


def plot_risk(
    fit,
    *,
    threshold: float,
    kind: str,
    annual: bool = True,
    credible_interval: float = 0.90,
    max_return_period: float | None = None,
    ax=None,
    color="C3",
):
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 4))
    else:
        figure = ax.figure
    if kind == "exceedance":
        values, labels = fit.exceedance_probability_draws(
            threshold, annual=annual, return_labels=True
        )
        title = fit.event_label(threshold)
        ylabel = "annual event probability" if annual else "event probability"
    elif kind == "return_period":
        values, labels = fit.return_period_draws(
            threshold, annual=annual, return_labels=True
        )
        if max_return_period is not None:
            values = np.minimum(values, float(max_return_period))
        title = f"Return period for {fit.event_label(threshold)[2:-1]}"
        ylabel = "years" if annual else "observation intervals"
        ax.set_yscale("log")
    else:
        raise ValueError("kind must be exceedance or return_period.")
    lower, median, upper = _interval(values, credible_interval)
    ax.fill_between(labels, lower, upper, color=color, alpha=0.2)
    ax.plot(labels, median, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return figure, ax


def plot_component_probabilities(fit, *, ax=None):
    import matplotlib.pyplot as plt

    table = fit.component_probabilities()
    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 3.5))
    else:
        figure = ax.figure
    names = list(table.index) if hasattr(table, "index") else [row["process"] for row in table]
    positions = np.arange(len(names))
    if hasattr(table, "columns") and "dynamic" in table.columns:
        bottom = np.zeros(len(names))
        for label, color in (("zero", "0.78"), ("fixed", "C1"), ("dynamic", "C0")):
            values = table[label].to_numpy()
            ax.bar(positions, values, bottom=bottom, label=label, color=color)
            bottom += values
    else:
        slab = table["slab"].to_numpy() if hasattr(table, "columns") else np.asarray([row["slab"] for row in table])
        ax.bar(positions, 1.0 - slab, label="continuous spike", color="0.75")
        ax.bar(positions, slab, bottom=1.0 - slab, label="dynamic slab", color="C0")
    ax.set_xticks(positions, names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("posterior probability")
    ax.set_title("Process innovation spike/slab allocation")
    ax.legend()
    return figure, ax


def plot_fit(fit, kind: str = "state", **kwargs):
    key = str(kind).lower().replace("-", "_")
    if key in {"process_sd", "process_sds", "prior_posterior_sd"}:
        return plot_process_sds(fit, **kwargs)
    if key in {"state", "level", "slope"}:
        if key in {"level", "slope"} and "state" not in kwargs:
            kwargs["state"] = key
        return plot_state(fit, **kwargs)
    if key in {"level_slope", "states"}:
        return plot_level_slope(fit, **kwargs)
    if key == "endpoint":
        return plot_endpoint(fit, **kwargs)
    if key in {"exceedance", "return_period"}:
        return plot_risk(fit, kind=key, **kwargs)
    if key in {"component_probabilities", "inclusion_probabilities"}:
        return plot_component_probabilities(fit, **kwargs)
    raise ValueError(
        "kind must be state, level, slope, level_slope, process_sd, endpoint, "
        "exceedance, return_period, or component_probabilities."
    )


def plot_bulk_tail(
    pair,
    *,
    kind: str = "states",
    credible_interval: float = 0.90,
    figsize=(10, 7),
):
    import matplotlib.pyplot as plt

    if kind not in {"states", "level"}:
        raise ValueError("BulkTailFit currently supports kind='states'.")
    figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    plot_state(pair.bulk, state="level", credible_interval=credible_interval, ax=axes[0])
    axes[0].set_title("Bulk (Gaussian)")
    plot_state(pair.tail, state="level", credible_interval=credible_interval, ax=axes[1])
    axes[1].set_title("Tail location (GEV)")
    figure.suptitle("Parallel bulk and tail fits (independent posteriors)")
    figure.tight_layout()
    return figure, axes


def plot_collection(collection, *, kind: str = "level", credible_interval: float = 0.90, figsize=None):
    """Plot one state for each fit in a Uccle collection."""

    import matplotlib.pyplot as plt

    state = "slope" if str(kind).lower() == "slope" else "level"
    names = list(collection.fits)
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=figsize or (9, max(3, 2.5 * len(names))),
        squeeze=False,
        sharex=True,
    )
    for axis, name in zip(axes[:, 0], names):
        plot_state(
            collection.fits[name],
            state=state,
            credible_interval=credible_interval,
            ax=axis,
        )
        axis.set_title(name)
    figure.tight_layout()
    return figure, axes[:, 0]


def plot(value, type: str = "state", **kwargs):
    """Compatibility spelling for :func:`plot_fit`."""

    if hasattr(value, "fits"):
        return plot_collection(value, kind=type, **kwargs)
    return plot_fit(value, kind=type, **kwargs)
