from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np


def _mpl():
    import matplotlib.pyplot as plt

    return plt


def _series_color(name: Optional[str]) -> str:
    if name and name.startswith("TX"):
        return "#c84c4c"
    if name and name.startswith("TN"):
        return "#3f78a8"
    return "#4c6f8c"


def _clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)


def _band(ax, x, draws, *, credible_interval: float, color: str, label: Optional[str] = None):
    alpha = 1.0 - credible_interval
    low, med, high = np.nanquantile(draws, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0], axis=0)
    ax.fill_between(x, low, high, color=color, alpha=0.20, linewidth=0)
    ax.plot(x, med, color=color, linewidth=1.6, label=label)
    return low, med, high


def _slope_multiplier(fit, slope_scale: str | float) -> tuple[float, str]:
    if isinstance(slope_scale, (int, float)):
        return float(slope_scale), "scaled time unit"
    value = slope_scale.lower()
    if value in {"raw", "month", "monthly"}:
        return 1.0, "°C per month"
    if value in {"year", "annual"}:
        return 12.0, "°C per year"
    if value in {"decade", "decadal"}:
        return 120.0, "°C per decade"
    raise ValueError("slope_scale must be 'raw', 'year', 'decade', or a numeric multiplier.")


def _plot_state(
    fit,
    state: str,
    *,
    ax=None,
    credible_interval: float = 0.90,
    color: Optional[str] = None,
    slope_scale: str | float = "decade",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 3.4))
    color = color or _series_color(fit.series_name)
    draws = fit.state_draws(state, original_scale=True)
    if state == "beta":
        multiplier, unit = _slope_multiplier(fit, slope_scale)
        draws = multiplier * draws
        ylabel = ylabel or unit
    else:
        ylabel = ylabel or "°C"
    _band(ax, fit.time, draws, credible_interval=credible_interval, color=color)
    ax.set_title(title or f"{fit.series_name or ''} {state}".strip())
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year" if fit.dates is not None else "Time")
    _clean_axis(ax)
    return ax


def _plot_level_slope(
    fit,
    *,
    credible_interval: float = 0.90,
    color: Optional[str] = None,
    slope_scale: str | float = "decade",
    figsize=(10.0, 3.6),
):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    _plot_state(
        fit,
        "alpha",
        ax=axes[0],
        credible_interval=credible_interval,
        color=color,
        title=f"{fit.series_name or ''} level $\\alpha_t$".strip(),
    )
    _plot_state(
        fit,
        "beta",
        ax=axes[1],
        credible_interval=credible_interval,
        color=color,
        slope_scale=slope_scale,
        title=f"{fit.series_name or ''} slope $\\beta_t$".strip(),
    )
    return fig, axes


def _as_thresholds(threshold: float | Iterable[float]) -> list[float]:
    if np.isscalar(threshold):
        return [float(threshold)]
    return [float(v) for v in threshold]


def _plot_risk(
    fit,
    *,
    threshold: float | Iterable[float],
    kind: str,
    annual: bool,
    credible_interval: float,
    ax=None,
    colors: Optional[Iterable[str]] = None,
    max_return_period: Optional[float] = 1e4,
    log_y: Optional[bool] = None,
):
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 3.7))
    thresholds = _as_thresholds(threshold)
    palette = list(colors) if colors is not None else []
    defaults = [_series_color(fit.series_name), "#6f5d8f", "#6a8f68", "#b07a3e"]

    for i, value in enumerate(thresholds):
        color = palette[i] if i < len(palette) else defaults[i % len(defaults)]
        if kind == "exceedance":
            draws, x = fit.exceedance_probability_draws(value, annual=annual)
            label = fit.event_label(value)
        else:
            draws, x = fit.return_period_draws(value, annual=annual)
            if max_return_period is not None:
                draws = np.minimum(draws, float(max_return_period))
            label = f"{value:g} °C"
        _band(ax, x, draws, credible_interval=credible_interval, color=color, label=label)

    if kind == "exceedance":
        ax.set_ylabel("Annual exceedance probability" if annual else "Exceedance probability")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{fit.series_name or ''} tail risk".strip())
    else:
        ax.set_ylabel("Annual return period (years)" if annual else "Block return period")
        ax.set_title(f"{fit.series_name or ''} return periods".strip())
        use_log = True if log_y is None else bool(log_y)
        if use_log:
            ax.set_yscale("log")
    ax.set_xlabel("Year" if fit.dates is not None else "Time")
    if len(thresholds) > 1 or kind == "exceedance":
        ax.legend(frameon=False)
    _clean_axis(ax)
    return ax


def _plot_endpoint(
    fit,
    *,
    ax=None,
    credible_interval: float = 0.90,
    color: Optional[str] = None,
    threshold: Optional[float | Iterable[float]] = None,
):
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 3.7))
    color = color or _series_color(fit.series_name)
    draws = fit.endpoint_draws(original_scale=True)
    _band(ax, fit.time, draws, credible_interval=credible_interval, color=color)
    if threshold is not None:
        for value in _as_thresholds(threshold):
            ax.axhline(value, color="0.35", linestyle="--", linewidth=0.9)
    lower = float(fit.transform_sign) < 0
    ax.set_ylabel("Lower endpoint (°C)" if lower else "Upper endpoint (°C)")
    ax.set_title(f"{fit.series_name or ''} GEV endpoint".strip())
    ax.set_xlabel("Year" if fit.dates is not None else "Time")
    _clean_axis(ax)
    return ax


def _plot_component_probabilities(fit, *, ax=None):
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 3.2))
    table = fit.component_probabilities()
    components = list(table.index)
    left = np.zeros(len(components), dtype=float)
    colors = {"zero": "#d7d7d7", "fixed": "#84a9c0", "dynamic": "#d97961"}
    for label in ("zero", "fixed", "dynamic"):
        values = table[label].to_numpy(dtype=float)
        ax.barh(components, values, left=left, label=label.capitalize(), color=colors[label])
        left += values
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Posterior probability")
    ax.set_title(f"{fit.series_name or ''} component structure".strip())
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    _clean_axis(ax)
    return ax


def _fs_prior_sd_samples(fit, component: str, size: int, rng) -> np.ndarray:
    priors = getattr(fit, "priors", None)
    if priors is None:
        raise ValueError("The fit does not retain its resolved priors.")
    key = {"level": "s_level", "trend": "s_trend", "season": "s_season"}[component]
    if getattr(priors, "pc", None) is not None:
        rate = priors.pc.standardized_rate_for(component) / priors.pc.coefficient_scale_for(component)
        return rng.exponential(scale=1.0 / rate, size=size)
    if getattr(priors, "horseshoe", None) is not None:
        hp = priors.horseshoe
        local = np.abs(rng.standard_cauchy(size=size))
        global_scale = np.abs(rng.standard_cauchy(size=size)) * hp.global_scale
        shape = 0.5 * hp.slab_df
        scale = 0.5 * hp.slab_df * hp.slab_scale**2
        slab2 = 1.0 / rng.gamma(shape=shape, scale=1.0 / scale, size=size)
        local2 = local**2
        global2 = global_scale**2
        regularized = slab2 * local2 / (slab2 + global2 * local2)
        variance = hp.coefficient_scale_for(component) ** 2 * global2 * regularized
        return np.abs(rng.normal(scale=np.sqrt(variance)))
    if getattr(priors, "lasso", None) is not None:
        lp = priors.lasso
        if getattr(lp, "componentwise", False):
            lambda2 = rng.gamma(
                shape=lp.a_for(component),
                scale=1.0 / lp.b_for(component),
                size=size,
            )
        else:
            lambda2 = rng.gamma(
                shape=lp.a_lambda, scale=1.0 / lp.b_lambda, size=size
            )
        tau = rng.exponential(scale=2.0 / lambda2)
        variance_scale = float(lp.fixed_variance)
        if lp.variance_mode == "observation":
            sigma2_prior = priors.sigma2
            variance_scale = (
                sigma2_prior.b / (sigma2_prior.a - 1.0)
                if sigma2_prior.a > 1.0
                else 1.0
            )
        coefficient_scale = float(lp.coefficient_scale_for(component))
        return np.abs(
            rng.normal(scale=np.sqrt(variance_scale * coefficient_scale**2 * tau))
        )
    if getattr(priors, "ssvs", None) is not None:
        ssvs = priors.ssvs
        dynamic_probability = (
            ssvs.level_dynamic_probability
            if component == "level"
            else (
                ssvs.trend_probabilities[2]
                if component == "trend"
                else ssvs.season_probabilities[2]
            )
        )
        active = rng.random(size) < dynamic_probability
        values = np.zeros(size)
        values[active] = np.abs(
            rng.normal(scale=ssvs.innovation_slab_sd[component], size=int(active.sum()))
        )
        return values
    prior = getattr(priors, key)
    return np.abs(rng.normal(loc=prior.mean, scale=prior.sd, size=size))


def _plot_process_sd(fit, *, bins: int = 40, figsize=None, prior_draws: int = 20000):
    plt = _mpl()
    components = [
        component
        for component, key in (
            ("level", "s_level"),
            ("trend", "s_trend"),
            ("season", "s_season"),
        )
        if key in fit.draws_static
    ]
    if not components:
        raise ValueError("The fit contains no signed FS process-scale draws.")
    fig, axes = plt.subplots(
        1,
        len(components),
        figsize=figsize or (3.6 * len(components), 3.1),
        squeeze=False,
        constrained_layout=True,
    )
    rng = np.random.default_rng(101)
    for axis, component in zip(axes.flat, components):
        posterior = np.abs(np.asarray(fit.draws_static[f"s_{component}"], dtype=float))
        prior = _fs_prior_sd_samples(fit, component, int(prior_draws), rng)
        upper = float(np.quantile(np.r_[prior, posterior], 0.995))
        edges = np.linspace(0.0, max(upper, np.finfo(float).eps), int(bins) + 1)
        axis.hist(prior, bins=edges, density=True, histtype="step", color="0.35", label="Prior")
        axis.hist(posterior, bins=edges, density=True, alpha=0.35, color="#4c78a8", label="Posterior")
        axis.set_title(f"{component.capitalize()} process SD")
        axis.set_xlabel("Standard deviation")
        _clean_axis(axis)
    axes.flat[0].set_ylabel("Density")
    axes.flat[-1].legend(frameon=False)
    return fig, axes.squeeze()


def _interval_summary(values, credible_interval: float):
    alpha = 1.0 - float(credible_interval)
    return np.quantile(values, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0])


def _plot_period_rates(
    fit,
    *,
    periods,
    credible_interval: float = 0.90,
    ax=None,
    color: Optional[str] = None,
):
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 3.4))
    color = color or _series_color(fit.series_name)
    draws = fit.period_rate_draws(periods, scale="decade")
    labels = list(draws)
    positions = np.arange(len(labels))
    for position, label in zip(positions, labels):
        low, med, high = _interval_summary(draws[label], credible_interval)
        inner_low, _, inner_high = _interval_summary(draws[label], 0.50)
        ax.plot([low, high], [position, position], color=color, linewidth=1.1)
        ax.plot([inner_low, inner_high], [position, position], color=color, linewidth=4.0)
        ax.plot(med, position, marker="o", color=color, markersize=5)
    ax.axvline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Level change (°C per decade)")
    ax.set_title(f"{fit.series_name or ''} period-average change".strip())
    _clean_axis(ax)
    return ax


def _plot_collection_rate_acceleration(
    collection,
    *,
    recent=(1980, 2022),
    reference=(1950, 1979),
    credible_interval: float = 0.90,
    figsize=(7.2, 4.2),
):
    plt = _mpl()
    order = [
        name
        for name in ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
        if name in collection.fits
    ]
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    positions = np.arange(len(order))
    for position, name in zip(positions, order):
        fit = collection.fits[name]
        values = fit.rate_contrast_draws(recent, reference, scale="decade")
        low, med, high = _interval_summary(values, credible_interval)
        inner_low, _, inner_high = _interval_summary(values, 0.50)
        color = _series_color(name)
        ax.plot([low, high], [position, position], color=color, linewidth=1.1)
        ax.plot([inner_low, inner_high], [position, position], color=color, linewidth=4.0)
        ax.plot(med, position, marker="o", color=color, markersize=5)
        probability = float(np.mean(values > 0.0))
        ax.text(
            high,
            position,
            f"  P={probability:.2f}",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax.axvline(0.0, color="0.35", linewidth=0.9, linestyle="--")
    ax.set_yticks(positions, order)
    ax.invert_yaxis()
    ax.set_xlabel("Recent minus mid-century rate (°C per decade)")
    ax.set_title(
        f"Acceleration: {recent[0]}–{recent[1]} versus "
        f"{reference[0]}–{reference[1]}"
    )
    _clean_axis(ax)
    return fig, ax


def _plot_collection(collection, *, type: str, credible_interval: float = 0.90, **kwargs):
    plt = _mpl()
    fits = collection.fits
    order = [name for name in ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn") if name in fits]
    if type in {"rate_acceleration", "acceleration", "rate_contrast"}:
        return _plot_collection_rate_acceleration(
            collection, credible_interval=credible_interval, **kwargs
        )
    if type in {"level_slope", "states"}:
        fig, axes = plt.subplots(
            len(order),
            2,
            figsize=kwargs.pop("figsize", (10.5, 2.45 * len(order))),
            squeeze=False,
            constrained_layout=True,
        )
        for row, name in enumerate(order):
            _plot_state(
                fits[name],
                "alpha",
                ax=axes[row, 0],
                credible_interval=credible_interval,
                title=f"{name} level $\\alpha_t$",
            )
            _plot_state(
                fits[name],
                "beta",
                ax=axes[row, 1],
                credible_interval=credible_interval,
                slope_scale=kwargs.get("slope_scale", "decade"),
                title=f"{name} slope $\\beta_t$",
            )
        return fig, axes

    if type not in {"level", "slope"}:
        raise ValueError("Collections support type='level', 'slope', or 'level_slope'.")
    state = "alpha" if type == "level" else "beta"
    ncol = 2
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=kwargs.pop("figsize", (10.0, 3.0 * nrow)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, name in zip(axes.flat, order):
        _plot_state(
            fits[name],
            state,
            ax=ax,
            credible_interval=credible_interval,
            slope_scale=kwargs.get("slope_scale", "decade"),
            title=name,
        )
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    return fig, axes


def plot(obj: Any, type: str = "level", **kwargs):
    """Plot fitted structural trajectories or tail-risk summaries.

    Parameters
    ----------
    obj:
        A :class:`PosteriorBundle` or :class:`UccleFitCollection`.
    type:
        ``'level'``, ``'slope'``, ``'level_slope'``, ``'exceedance'``,
        ``'return_period'``, ``'endpoint'``, ``'period_rates'`` or ``'component_probabilities'``.
    """
    type = type.lower().replace("-", "_").replace(" ", "_")
    credible_interval = float(kwargs.pop("credible_interval", 0.90))

    if hasattr(obj, "fits"):
        return _plot_collection(obj, type=type, credible_interval=credible_interval, **kwargs)

    if type in {"level", "alpha"}:
        ax = _plot_state(obj, "alpha", credible_interval=credible_interval, **kwargs)
        return ax.figure, ax
    if type in {"slope", "beta"}:
        ax = _plot_state(obj, "beta", credible_interval=credible_interval, **kwargs)
        return ax.figure, ax
    if type in {"level_slope", "states"}:
        return _plot_level_slope(obj, credible_interval=credible_interval, **kwargs)
    if type in {"exceedance", "exceedance_probability", "risk"}:
        if "threshold" not in kwargs:
            raise ValueError("plot(type='exceedance') requires threshold=...")
        ax = _plot_risk(
            obj,
            kind="exceedance",
            annual=bool(kwargs.pop("annual", False)),
            credible_interval=credible_interval,
            **kwargs,
        )
        return ax.figure, ax
    if type in {"return_period", "return_periods"}:
        if "threshold" not in kwargs:
            raise ValueError("plot(type='return_period') requires threshold=...")
        ax = _plot_risk(
            obj,
            kind="return_period",
            annual=bool(kwargs.pop("annual", True)),
            credible_interval=credible_interval,
            **kwargs,
        )
        return ax.figure, ax
    if type in {"endpoint", "gev_endpoint"}:
        ax = _plot_endpoint(obj, credible_interval=credible_interval, **kwargs)
        return ax.figure, ax
    if type in {"period_rates", "finite_rates", "level_rates"}:
        if "periods" not in kwargs:
            raise ValueError("plot(type='period_rates') requires periods={label: (start, end)}.")
        ax = _plot_period_rates(
            obj, credible_interval=credible_interval, **kwargs
        )
        return ax.figure, ax
    if type in {"component_probabilities", "components", "structure"}:
        ax = _plot_component_probabilities(obj, **kwargs)
        return ax.figure, ax
    if type in {"process_sd", "process_sds", "prior_posterior_sd", "shrinkage"}:
        return _plot_process_sd(obj, **kwargs)
    raise ValueError(
        "Unknown plot type. Use level, slope, level_slope, exceedance, "
        "return_period, endpoint, period_rates, process_sd, or component_probabilities."
    )
