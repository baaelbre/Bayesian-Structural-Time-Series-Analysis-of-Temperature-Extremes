"""Data-agnostic posterior, risk, and prior-versus-posterior plots."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..priors.process import FixedSD


def _prior_density(prior, grid):
    return np.asarray([np.exp(prior.logpdf(float(value))) for value in grid])


def _save_result(result, save) -> None:
    """Save the figure in a plotting result using one consistent API."""

    if save is None:
        return
    options = {}
    if isinstance(save, dict):
        options = dict(save)
        try:
            path = options.pop("path")
        except KeyError as exc:
            raise ValueError("A save mapping requires a 'path' entry.") from exc
    else:
        path = save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = result[0] if isinstance(result, tuple) else result
    if not hasattr(figure, "savefig"):
        raise TypeError("The selected plot did not return a Matplotlib figure.")
    options.setdefault("bbox_inches", "tight")
    figure.savefig(path, **options)


def _triple_gamma_absolute_density(prior, component: str, grid) -> np.ndarray:
    """Analytic |signed scale| density for fixed, unregularized triple gamma."""

    from scipy.special import betaln, gammaln, hyperu

    values = np.asarray(grid, dtype=float)
    scale = float(prior.coefficient_scale_for(component))
    x = np.maximum(values / scale, np.finfo(float).tiny)
    a = float(prior.spike_shape)
    c = float(prior.tail_shape)
    phi = float(prior.global_scale)
    log_constant = (
        np.log(2.0)
        + gammaln(c + 0.5)
        - 0.5 * np.log(2.0 * np.pi * phi)
        - betaln(a, c)
        - np.log(scale)
    )
    density = np.exp(log_constant) * hyperu(
        c + 0.5,
        1.5 - a,
        x**2 / (2.0 * phi),
    )
    density = np.asarray(density, dtype=float)
    if density.size > 1 and not np.isfinite(density[0]):
        density[0] = density[1]
    return density


def _structural_prior_density(fit, name: str, grid):
    """Return an analytic structural-SD density where one is available."""

    from scipy.stats import norm

    priors = fit.priors
    component = {"level": "level", "slope": "trend", "seasonal": "season"}.get(
        name, name
    )
    if getattr(priors, "pc", None) is not None:
        rate = (
            priors.pc.standardized_rate_for(component)
            / priors.pc.coefficient_scale_for(component)
        )
        return rate * np.exp(-rate * np.asarray(grid)), "prior (analytic PC)", 0.0
    triple_gamma = getattr(priors, "triple_gamma", None)
    selected_tg = set(getattr(priors, "triple_gamma_processes", ()))
    if triple_gamma is not None and (
        not selected_tg or component in selected_tg
    ):
        if (
            not triple_gamma.regularized
            and not triple_gamma.learn_global
            and not triple_gamma.learn_shapes
        ):
            return (
                _triple_gamma_absolute_density(triple_gamma, component, grid),
                "prior (analytic triple gamma)",
                0.0,
            )
        return None
    if getattr(priors, "ssvs", None) is not None:
        prior = priors.ssvs
        probability = {
            "level": prior.level_dynamic_probability,
            "trend": prior.trend_probabilities[2],
            "season": prior.season_probabilities[2],
        }[component]
        scale = float(prior.innovation_slab_sd[component])
        density = 2.0 * norm.pdf(np.asarray(grid), loc=0.0, scale=scale)
        return density, "prior slab (analytic half-normal)", 1.0 - probability
    key = {"level": "s_level", "trend": "s_trend", "season": "s_season"}.get(
        component
    )
    prior = None if key is None else getattr(priors, key, None)
    if prior is not None:
        values = np.asarray(grid)
        density = norm.pdf(values, loc=prior.mean, scale=prior.sd)
        density += norm.pdf(-values, loc=prior.mean, scale=prior.sd)
        return density, "prior (analytic folded normal)", 0.0
    return None


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
    component = {"level": "level", "slope": "trend", "seasonal": "season"}.get(
        name, name
    )
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
    if getattr(priors, "triple_gamma", None) is not None:
        prior = priors.triple_gamma
        if prior.learn_shapes:
            a = 0.5 * rng.beta(*prior.spike_shape_prior, size=size)
            c = 0.5 * rng.beta(*prior.tail_shape_prior, size=size)
        else:
            a = np.full(size, prior.spike_shape)
            c = np.full(size, prior.tail_shape)
        numerator = rng.gamma(a, scale=1.0)
        denominator = rng.gamma(c, scale=1.0)
        if prior.learn_global:
            global_scale = (
                rng.gamma(c, scale=1.0)
                / np.maximum(rng.gamma(a, scale=1.0), 1e-300)
            )
        else:
            global_scale = np.full(size, prior.global_scale)
        variance = global_scale * numerator / np.maximum(denominator, 1e-300)
        if prior.regularized:
            slab2 = 1.0 / rng.gamma(
                shape=0.5 * prior.slab_df,
                scale=2.0 / (prior.slab_df * prior.slab_scale**2),
                size=size,
            )
            variance = slab2 * variance / (slab2 + variance)
        variance *= prior.coefficient_scale_for(component) ** 2
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
    truths=None,
    title: str | None = None,
    figsize=None,
    xmax=None,
):
    """Overlay each process-SD prior with its marginal posterior.

    Heavy-tailed global-local priors can have astronomically large upper
    quantiles. Their default display is capped at five coefficient reference
    scales (or 1.5 times the posterior range, whichever is larger) so the
    scientifically relevant spike is visible. Pass a positive numeric
    ``xmax`` or a mapping keyed by process name to override that display limit.
    The density itself is not truncated or renormalized.
    """

    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    names = list(fit.compiled.noise_names)
    truths = {} if truths is None else dict(truths)
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
        hierarchical_processes = set(
            getattr(fit.priors, "horseshoe_processes", ())
        ) | set(getattr(fit.priors, "triple_gamma_processes", ()))
        if name in hierarchical_processes:
            # The process mapping is only a baseline/calibration carrier for a
            # factor hierarchy; the hierarchy is the actual active prior.
            prior = None
        grid_upper = max(
            float(np.quantile(posterior, 0.995)),
            float(np.max(posterior)),
            1e-10,
        )
        prior_tail_clipped = False
        if prior is None:
            prior_sample = _structural_prior_samples(
                fit,
                name,
                int(prior_draws),
                np.random.default_rng(140),
            )
            prior_sample = np.asarray(prior_sample, dtype=float)
            prior_sample = prior_sample[np.isfinite(prior_sample)]
            if prior_sample.size:
                prior_upper = float(np.quantile(prior_sample, 0.995))
                hierarchy = (
                    getattr(fit.priors, "horseshoe", None)
                    or getattr(fit.priors, "triple_gamma", None)
                )
                if hierarchy is not None:
                    component = {
                        "level": "level",
                        "slope": "trend",
                        "seasonal": "season",
                    }.get(name, name)
                    reference = float(
                        hierarchy.coefficient_scale_for(component)
                    )
                    display_cap = max(1.5 * grid_upper, 5.0 * reference)
                    prior_tail_clipped = prior_upper > display_cap
                    prior_upper = min(prior_upper, display_cap)
                grid_upper = max(grid_upper, prior_upper)
        elif not isinstance(prior, FixedSD):
            prior_sample = np.asarray(prior.sample(np.random.default_rng(140), size=prior_draws))
            grid_upper = max(grid_upper, float(np.quantile(prior_sample, 0.995)))
        if xmax is not None:
            selected_xmax = (
                xmax.get(name, xmax.get(f"sd.{name}"))
                if isinstance(xmax, dict)
                else xmax
            )
            if selected_xmax is not None:
                if float(selected_xmax) <= 0.0:
                    raise ValueError("Every process-SD xmax must be positive.")
                grid_upper = float(selected_xmax)
                prior_tail_clipped = False
        grid = np.linspace(0.0, grid_upper * 1.05, 400)
        structural_ssvs = bool(
            prior is None and getattr(fit.priors, "ssvs", None) is not None
        )
        if prior is None:
            analytic = _structural_prior_density(fit, name, grid)
            if analytic is not None:
                prior_density, prior_label, prior_zero_mass = analytic
                label = prior_label
                if prior_zero_mass > 0.0:
                    label += f"; P(SD=0)={prior_zero_mass:.2f}"
                if prior_tail_clipped:
                    label += "; heavy tail continues"
                axis.plot(
                    grid,
                    prior_density,
                    color="0.35",
                    linestyle="--",
                    label=label,
                )
            else:
                positive_prior = prior_sample[prior_sample > 0.0]
                prior_zero_mass = float(np.mean(prior_sample == 0.0))
                plotted_prior = (
                    positive_prior
                    if structural_ssvs and positive_prior.size
                    else prior_sample
                )
                if plotted_prior.size > 2 and np.std(plotted_prior) > 1e-14:
                    prior_kde = gaussian_kde(plotted_prior)
                    label = "prior (smooth Monte Carlo)"
                    if prior_zero_mass > 0.0:
                        label += f"; P(SD=0)={prior_zero_mass:.2f}"
                    if prior_tail_clipped:
                        label += "; heavy tail continues"
                    axis.plot(
                        grid,
                        prior_kde(grid),
                        color="0.35",
                        linestyle="--",
                        label=label,
                    )
                elif plotted_prior.size:
                    axis.axvline(
                        float(np.mean(plotted_prior)),
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
        posterior_zero_mass = float(np.mean(posterior == 0.0))
        positive_posterior = posterior[posterior > 0.0]
        density_values = (
            positive_posterior
            if posterior_zero_mass > 0.0 and positive_posterior.size > 2
            else posterior
        )
        if np.std(density_values) > 1e-12 and np.unique(density_values).size > 2:
            density = gaussian_kde(density_values)
            axis.plot(
                grid,
                density(grid),
                color="C0",
                label=("posterior slab" if posterior_zero_mass > 0.0 else "posterior"),
            )
            axis.fill_between(grid, 0.0, density(grid), color="C0", alpha=0.18)
        else:
            axis.axvline(float(np.mean(posterior)), color="C0", label="posterior")
        if posterior_zero_mass > 0.0:
            axis.axvline(
                0.0,
                color="C0",
                linewidth=2.0,
                label=f"posterior P(SD=0)={posterior_zero_mass:.2f}",
            )
        lower, median, upper = _interval(posterior, credible_interval)
        axis.axvspan(lower, upper, color="C0", alpha=0.08)
        axis.axvline(median, color="C0", linewidth=1.0)
        truth_name = f"sd.{name}"
        if truth_name in truths:
            axis.axvline(
                float(truths[truth_name]),
                color="black",
                linewidth=1.1,
                linestyle="--",
                label="truth",
            )
        axis.set_title(f"Process SD: {name}")
        axis.set_xlabel("innovation standard deviation")
        axis.set_ylabel("density")
        axis.legend()
    if title is not None:
        figure.suptitle(str(title), y=0.995)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    else:
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


def plot_predictor(
    fit,
    *,
    credible_interval: float = 0.90,
    ax=None,
    color: str = "C3",
):
    """Plot observations against the complete univariate latent predictor."""

    import matplotlib.pyplot as plt

    if fit.is_factor_model:
        raise ValueError("Use plot_channel_predictor for a factor model.")
    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 4))
    else:
        figure = ax.figure
    values = fit.eta_draws(original_scale=True)
    lower, median, upper = _interval(values, credible_interval)
    x = _time(fit)
    ax.scatter(x, fit.observed, s=9, color="0.55", alpha=0.55, label="observed")
    ax.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=0.2,
        label=f"{credible_interval:.0%} credible interval",
    )
    ax.plot(x, median, color=color, label="complete latent predictor")
    ax.set_title(fit.series_name or "Posterior predictor")
    ax.legend()
    return figure, ax


def plot_factor_state(
    fit,
    *,
    factor: str,
    state: str = "level",
    credible_interval: float = 0.90,
    ax=None,
    color: str = "C0",
):
    """Plot one semantic state from a shared dynamic factor."""

    import matplotlib.pyplot as plt

    if not fit.is_factor_model:
        raise ValueError("plot_factor_state requires a factor-model fit.")
    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 4))
    else:
        figure = ax.figure
    values = fit.factor_draws(factor, state=state)
    lower, median, upper = _interval(values, credible_interval)
    x = _time(fit)
    ax.fill_between(x, lower, upper, color=color, alpha=0.2)
    ax.plot(x, median, color=color)
    ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    ax.set_title(f"Shared factor: {factor}.{state}")
    return figure, ax


def plot_channel_predictor(
    fit,
    *,
    channel: str,
    credible_interval: float = 0.90,
    ax=None,
    color: str = "C0",
):
    """Plot observed data against a channel's full latent predictor."""

    import matplotlib.pyplot as plt

    if not fit.is_factor_model:
        raise ValueError("plot_channel_predictor requires a factor-model fit.")
    if channel not in fit.channel_names:
        raise KeyError(f"Unknown channel '{channel}'. Available: {fit.channel_names}")
    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 4))
    else:
        figure = ax.figure
    index = fit.channel_names.index(channel)
    values = fit.channel_eta_draws(channel, original_scale=True)
    lower, median, upper = _interval(values, credible_interval)
    x = _time(fit)
    ax.scatter(x, fit.observed[:, index], s=9, color="0.55", alpha=0.55, label="observed")
    ax.fill_between(x, lower, upper, color=color, alpha=0.2, label="credible interval")
    ax.plot(x, median, color=color, label="latent predictor")
    ax.set_title(channel)
    ax.legend()
    return figure, ax


def plot_factor_decomposition(
    fit,
    *,
    factor: str | None = None,
    channels=None,
    baseline=None,
    credible_interval: float = 0.90,
    truth=None,
    figsize=None,
):
    """Plot complete, shared and idiosyncratic paths with credible bands."""

    import matplotlib.pyplot as plt

    if not fit.is_factor_model:
        raise ValueError("plot_factor_decomposition requires a factor-model fit.")
    selected = list(fit.channel_names if channels is None else channels)
    unknown = sorted(set(selected) - set(fit.channel_names))
    if unknown:
        raise KeyError(f"Unknown channels {unknown}; available={fit.channel_names}.")
    figure, axes = plt.subplots(
        len(selected),
        3,
        figsize=figsize or (16, max(3.0, 3.0 * len(selected))),
        squeeze=False,
        sharex=True,
    )
    x = _time(fit)
    truth = {} if truth is None else truth
    truth_colors = {"predictor": "black", "shared": "black", "deviation": "0.35"}
    colors = {"predictor": "C3", "shared": "C0", "deviation": "C1"}
    titles = {
        "predictor": "complete predictor",
        "shared": "loading x common factor",
        "deviation": "idiosyncratic deviation",
    }
    for row, channel in enumerate(selected):
        decomposition = fit.channel_decomposition(
            channel,
            factor,
            baseline=baseline,
        )
        for column, component in enumerate(("predictor", "shared", "deviation")):
            axis = axes[row, column]
            values = decomposition[component]
            lower, median, upper = _interval(values, credible_interval)
            if component == "predictor":
                index = fit.channel_names.index(channel)
                axis.scatter(
                    x,
                    fit.observed[:, index],
                    s=8,
                    color="0.55",
                    alpha=0.35,
                    label="observed",
                )
            axis.fill_between(
                x,
                lower,
                upper,
                color=colors[component],
                alpha=0.18,
                label=f"posterior {credible_interval:.0%} interval",
            )
            axis.plot(x, median, color=colors[component], label="posterior median")
            component_truth = truth.get(component, {})
            if channel in component_truth:
                axis.plot(
                    x,
                    np.asarray(component_truth[channel]),
                    color=truth_colors[component],
                    linestyle="--" if component == "deviation" else "-",
                    label="truth",
                )
            axis.axhline(0.0, color="0.5", linewidth=0.7, alpha=0.6)
            axis.set_title(f"{channel}: {titles[component]}")
            if row == 0:
                axis.legend(fontsize=8)
    figure.tight_layout()
    return figure, axes


def plot_parameter_densities(
    fit,
    *,
    parameters,
    truths=None,
    credible_interval: float = 0.90,
    figsize=None,
):
    """Posterior scalar densities with optional simulation truths."""

    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    names = [parameters] if isinstance(parameters, str) else list(parameters)
    if not names:
        raise ValueError("Choose at least one parameter.")
    truths = {} if truths is None else truths
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=figsize or (7.5, max(2.5, 2.4 * len(names))),
        squeeze=False,
    )
    for axis, name in zip(axes[:, 0], names):
        values = np.asarray(fit.parameter(name), dtype=float).reshape(-1)
        lower, median, upper = _interval(values, credible_interval)
        spread = max(float(np.std(values)), abs(float(median)) * 1e-3, 1e-10)
        grid = np.linspace(
            min(float(np.min(values)), float(lower)) - 0.25 * spread,
            max(float(np.max(values)), float(upper)) + 0.25 * spread,
            400,
        )
        if np.unique(values).size > 2 and np.std(values) > 1e-12:
            density = gaussian_kde(values)
            axis.plot(grid, density(grid), color="C0", label="posterior")
            axis.fill_between(grid, 0.0, density(grid), color="C0", alpha=0.18)
        else:
            axis.axvline(float(np.mean(values)), color="C0", label="posterior (fixed)")
        axis.axvspan(lower, upper, color="C0", alpha=0.08)
        axis.axvline(median, color="C0", linewidth=1.0, linestyle="--")
        if name in truths:
            axis.axvline(float(truths[name]), color="black", label="truth")
        axis.set_title(name)
        axis.set_ylabel("density")
        axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel("parameter value")
    figure.tight_layout()
    return figure, axes[:, 0]


def plot_process_sd_traces(
    fit,
    *,
    parameters=None,
    truths=None,
    figsize=None,
):
    """Chain-specific traces for every innovation standard deviation."""

    import matplotlib.pyplot as plt

    names = (
        [f"sd.{name}" for name in fit.compiled.noise_names]
        if parameters is None
        else ([parameters] if isinstance(parameters, str) else list(parameters))
    )
    truths = {} if truths is None else truths
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=figsize or (10, max(2.5, 2.1 * len(names))),
        squeeze=False,
        sharex=True,
    )
    for axis, name in zip(axes[:, 0], names):
        values = np.asarray(fit.parameter(name, combine_chains=False), dtype=float)
        for chain in range(values.shape[0]):
            axis.plot(values[chain], linewidth=0.8, alpha=0.8, label=f"chain {chain + 1}")
        if name in truths:
            axis.axhline(float(truths[name]), color="black", linestyle="--", label="truth")
        axis.set_ylabel(name)
    axes[0, 0].legend(ncol=min(fit.n_chains + int(bool(truths)), 5), fontsize=8)
    axes[-1, 0].set_xlabel("retained draw")
    figure.tight_layout()
    return figure, axes[:, 0]


def plot_parameter_acfs(
    fit,
    *,
    parameters=None,
    max_lag: int = 50,
    reference_band: bool = True,
    figsize=None,
):
    """Chain-specific autocorrelation functions for scalar parameters.

    The chains are never concatenated before computing an ACF.  This avoids a
    false discontinuity at chain boundaries and makes persistent chains easy
    to spot.  The optional band is the usual ``+-1.96/sqrt(n)`` white-noise
    reference, not a posterior credible interval.
    """

    import matplotlib.pyplot as plt

    if parameters is None:
        prefixes = ("sd.", "sigma", "xi", "loading.")
        names = [
            name
            for name, values in fit.parameter_draws.items()
            if np.asarray(values).ndim == 2 and name.startswith(prefixes)
        ]
    else:
        names = [parameters] if isinstance(parameters, str) else list(parameters)
    if not names:
        raise ValueError("Choose at least one scalar parameter for the ACF plot.")
    max_lag = int(max_lag)
    if max_lag < 1:
        raise ValueError("max_lag must be at least one.")
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=figsize or (9, max(2.5, 2.2 * len(names))),
        squeeze=False,
        sharex=True,
    )
    for axis, name in zip(axes[:, 0], names):
        values = np.asarray(
            fit.parameter(name, combine_chains=False), dtype=float
        )
        lag_count = min(max_lag, values.shape[1] - 1)
        lags = np.arange(lag_count + 1)
        for chain_index, chain in enumerate(values):
            centered = chain - np.mean(chain)
            variance = float(centered @ centered)
            if variance <= 0.0:
                acf = np.full(lag_count + 1, np.nan)
                acf[0] = 1.0
            else:
                size = 1 << (2 * chain.size - 1).bit_length()
                spectrum = np.fft.rfft(centered, n=size)
                covariance = np.fft.irfft(
                    spectrum * np.conjugate(spectrum), n=size
                )[: lag_count + 1]
                acf = covariance / covariance[0]
            axis.plot(
                lags,
                acf,
                linewidth=1.0,
                label=f"chain {chain_index + 1}",
            )
        if reference_band:
            band = 1.96 / np.sqrt(max(values.shape[1], 1))
            axis.axhspan(-band, band, color="0.5", alpha=0.12)
        axis.axhline(0.0, color="0.4", linewidth=0.7)
        axis.set_ylim(-1.0, 1.05)
        axis.set_ylabel(name)
    axes[0, 0].legend(ncol=min(fit.n_chains, 5), fontsize=8)
    axes[-1, 0].set_xlabel("lag (retained draws)")
    figure.tight_layout()
    return figure, axes[:, 0]


def plot_loading_deviation_joint(
    fit,
    *,
    channel: str,
    factor: str | None = None,
    summary: str = "factor_projection",
    baseline=None,
    truth=None,
    ax=None,
):
    """Joint posterior revealing loading--deviation compensation."""

    import matplotlib.pyplot as plt

    paired = fit.loading_deviation_draws(
        channel,
        factor,
        summary=summary,
        baseline=baseline,
    )
    loading = paired["loading"]
    deviation = paired["deviation_summary"]
    correlation = fit.loading_deviation_correlation(
        channel,
        factor,
        summary=summary,
        baseline=baseline,
    )
    if ax is None:
        figure, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        figure = ax.figure
    ax.scatter(loading, deviation, s=10, alpha=0.25, color="C0")
    if truth is not None:
        ax.scatter(
            [float(truth[0])],
            [float(truth[1])],
            marker="*",
            s=130,
            color="black",
            label="truth",
        )
        ax.legend()
    ax.set_xlabel("factor loading")
    ax.set_ylabel(str(summary).replace("_", " "))
    suffix = "undefined (fixed loading)" if not np.isfinite(correlation) else f"{correlation:.3f}"
    ax.set_title(f"{channel}: loading vs deviation; posterior corr = {suffix}")
    return figure, ax


def plot_loading_deviation_correlations(
    fit,
    *,
    factor: str | None = None,
    channels=None,
    summaries=("factor_projection", "final_change"),
    baseline=None,
    ax=None,
):
    """Compare loading--deviation posterior correlations across channels."""

    import matplotlib.pyplot as plt

    selected = list(fit.channel_names if channels is None else channels)
    summaries = list(summaries)
    values = np.asarray(
        [
            [
                fit.loading_deviation_correlation(
                    channel,
                    factor,
                    summary=summary,
                    baseline=baseline,
                )
                for channel in selected
            ]
            for summary in summaries
        ],
        dtype=float,
    )
    if ax is None:
        figure, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(selected)), 4.5))
    else:
        figure = ax.figure
    positions = np.arange(len(selected), dtype=float)
    width = 0.8 / max(len(summaries), 1)
    for index, summary in enumerate(summaries):
        offset = (index - 0.5 * (len(summaries) - 1)) * width
        ax.bar(
            positions + offset,
            values[index],
            width=width,
            label=str(summary).replace("_", " "),
        )
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.axhline(0.7, color="0.6", linewidth=0.7, linestyle="--")
    ax.axhline(-0.7, color="0.6", linewidth=0.7, linestyle="--")
    ax.set_xticks(positions, selected)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("posterior correlation")
    ax.set_title("Loading--idiosyncratic-deviation confounding")
    ax.legend(fontsize=8)
    figure.tight_layout()
    return figure, ax


def plot_idiosyncratic_innovations(
    fit,
    *,
    channel: str,
    factor: str | None = None,
    credible_interval: float = 0.90,
    truth=None,
    ax=None,
):
    """Plot posterior channel-specific innovations ``Delta alpha[t]``."""

    import matplotlib.pyplot as plt

    values = fit.idiosyncratic_innovation_draws(channel, factor)
    lower, median, upper = _interval(values, credible_interval)
    x = _time(fit)[1:]
    if ax is None:
        figure, ax = plt.subplots(figsize=(10, 4))
    else:
        figure = ax.figure
    ax.fill_between(x, lower, upper, color="C1", alpha=0.18)
    ax.plot(x, median, color="C1", label="posterior median innovation")
    if truth is not None:
        truth_values = np.asarray(truth, dtype=float).reshape(-1)
        if truth_values.size == fit.n_time:
            truth_values = np.diff(truth_values)
        if truth_values.size != fit.n_time - 1:
            raise ValueError("truth must have length n_time or n_time - 1.")
        ax.plot(x, truth_values, color="black", linewidth=0.9, label="truth")
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_title(f"{channel}: idiosyncratic innovations")
    ax.set_ylabel("Delta alpha")
    ax.legend(fontsize=8)
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
    save = kwargs.pop("save", None)

    def finish(result):
        _save_result(result, save)
        return result

    key = str(kind).lower().replace("-", "_")
    if getattr(fit, "is_factor_model", False):
        if key in {"factor", "shared_factor", "factor_state"}:
            return finish(plot_factor_state(fit, **kwargs))
        if key in {"state", "channel", "channel_predictor", "fit"}:
            if "channel" not in kwargs:
                kwargs["channel"] = fit.channel_names[0]
            return finish(plot_channel_predictor(fit, **kwargs))
        if key in {"process_sd", "process_sds", "prior_posterior_sd"}:
            return finish(plot_process_sds(fit, **kwargs))
        if key in {"decomposition", "factor_decomposition"}:
            return finish(plot_factor_decomposition(fit, **kwargs))
        if key in {"parameter_density", "parameter_densities", "densities"}:
            return finish(plot_parameter_densities(fit, **kwargs))
        if key in {"trace", "traces", "process_sd_traces"}:
            return finish(plot_process_sd_traces(fit, **kwargs))
        if key in {"acf", "acfs", "autocorrelation", "autocorrelations"}:
            return finish(plot_parameter_acfs(fit, **kwargs))
        if key in {"loading_deviation", "loading_deviation_joint"}:
            return finish(plot_loading_deviation_joint(fit, **kwargs))
        if key in {"identification", "loading_deviation_correlations"}:
            return finish(plot_loading_deviation_correlations(fit, **kwargs))
        if key in {"idiosyncratic_innovations", "idio_innovations"}:
            return finish(plot_idiosyncratic_innovations(fit, **kwargs))
        raise ValueError(
            "Factor-model kind must be channel, factor, process_sd, "
            "factor_decomposition, parameter_density, traces, acf, "
            "loading_deviation, identification, or idiosyncratic_innovations."
        )
    if key in {"process_sd", "process_sds", "prior_posterior_sd"}:
        return finish(plot_process_sds(fit, **kwargs))
    if key in {"predictor", "eta", "fit"}:
        return finish(plot_predictor(fit, **kwargs))
    if key in {"parameter_density", "parameter_densities", "densities"}:
        return finish(plot_parameter_densities(fit, **kwargs))
    if key in {"trace", "traces", "process_sd_traces"}:
        return finish(plot_process_sd_traces(fit, **kwargs))
    if key in {"acf", "acfs", "autocorrelation", "autocorrelations"}:
        return finish(plot_parameter_acfs(fit, **kwargs))
    if key in {"state", "level", "slope"}:
        if key in {"level", "slope"} and "state" not in kwargs:
            kwargs["state"] = key
        return finish(plot_state(fit, **kwargs))
    if key in {"level_slope", "states"}:
        return finish(plot_level_slope(fit, **kwargs))
    if key == "endpoint":
        return finish(plot_endpoint(fit, **kwargs))
    if key in {"exceedance", "return_period"}:
        return finish(plot_risk(fit, kind=key, **kwargs))
    if key in {"component_probabilities", "inclusion_probabilities"}:
        return finish(plot_component_probabilities(fit, **kwargs))
    raise ValueError(
        "kind must be state, level, slope, level_slope, predictor, process_sd, "
        "parameter_density, traces, acf, endpoint, exceedance, return_period, or "
        "component_probabilities."
    )


def plot_bulk_tail(
    pair,
    *,
    kind: str = "states",
    credible_interval: float = 0.90,
    figsize=(10, 7),
    save=None,
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
    result = (figure, axes)
    _save_result(result, save)
    return result


def plot_collection(
    collection,
    *,
    kind: str = "level",
    credible_interval: float = 0.90,
    figsize=None,
    save=None,
):
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
    result = (figure, axes[:, 0])
    _save_result(result, save)
    return result


def plot(value, type: str = "state", **kwargs):
    """Compatibility spelling for :func:`plot_fit`."""

    if hasattr(value, "fits"):
        return plot_collection(value, kind=type, **kwargs)
    return plot_fit(value, kind=type, **kwargs)
