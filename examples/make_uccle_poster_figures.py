#!/usr/bin/env python3
"""Create poster-ready Uccle figures from pooled bucex fits.

Main layout requested:
1. Top panel: all six posterior level trajectories in a 2x3 grid.
2. Middle panel: four extreme-index slope plots (2x2) plus a thin vertical
   acceleration summary on the right.
3. Bottom panel: TXx record-risk panel (return periods + moving endpoint).

The script saves each panel separately and also writes a combined preview.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bucex import PosteriorBundle

SCRIPT_VERSION = "poster-layout-v2-2026-07-20"

SERIES = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
LEVEL_LAYOUT = (("TXm", "TXx", "TXn"), ("TNm", "TNx", "TNn"))
EXTREME_SERIES = ("TXx", "TXn", "TNx", "TNn")
ACCEL_ORDER = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")

REFERENCE_PERIOD = (1950, 1979)
RECENT_PERIOD = (1980, 2022)

NAVY = "#082B57"
TEXT = "#172033"
GRID = "#D7DEE8"
TX_RED = "#CF4A45"
TX_RED_DARK = "#A91518"
TX_RED_LIGHT = "#E88983"
TN_BLUE = "#3E79A9"

TICK_YEARS = [1900, 1960, 2020]

SERIES_LABELS = {
    "TXm": "TXm — mean daytime",
    "TNm": "TNm — mean nighttime",
    "TXx": "TXx — hottest day",
    "TXn": "TXn — coldest daytime max",
    "TNx": "TNx — warmest night",
    "TNn": "TNn — coldest night",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create poster-ready Uccle figures.")
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--credible-interval", type=float, default=0.90)
    parser.add_argument("--inner-credible-interval", type=float, default=0.50)
    parser.add_argument("--risk-start-year", type=int, default=1970)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
    )
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": NAVY,
            "axes.linewidth": 0.8,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def series_color(name: str) -> str:
    return TX_RED if name.startswith("TX") else TN_BLUE


def style_axis(ax, *, zero_line: bool = False, grid_x: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NAVY)
    ax.spines["bottom"].set_color(NAVY)
    ax.grid(axis="y", color=GRID, alpha=0.58, linewidth=0.65)
    if grid_x:
        ax.grid(axis="x", color=GRID, alpha=0.50, linewidth=0.60)
    ax.set_axisbelow(True)
    if zero_line:
        ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--", zorder=1)


def save_figure(fig, out_dir: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03, "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(out_dir / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def load_fits(fit_dir: Path) -> dict[str, PosteriorBundle]:
    fits: dict[str, PosteriorBundle] = {}
    missing: list[str] = []
    for name in SERIES:
        candidates = sorted(fit_dir.glob(f"{name}_bucex_v*.pkl"))
        if not candidates:
            missing.append(name)
            continue
        path = candidates[-1]
        fit = PosteriorBundle.load(path)
        if fit.draws_states is None:
            raise ValueError(f"{path} does not contain posterior state draws.")
        if fit.dates is None:
            raise ValueError(f"{path} does not contain dates.")
        fits[name] = fit
    if missing:
        raise FileNotFoundError("Missing pooled fit files for: " + ", ".join(missing))
    return fits


def dates_for_fit(fit: PosteriorBundle) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(fit.dates)))


def posterior_interval(draws: np.ndarray, credible_interval: float, axis: int = 0):
    alpha = 1.0 - credible_interval
    q = np.nanquantile(np.asarray(draws, dtype=float), [alpha / 2.0, 0.5, 1.0 - alpha / 2.0], axis=axis)
    return q[0], q[1], q[2]


def add_band(ax, x, draws: np.ndarray, *, color: str, credible_interval: float, linewidth: float = 2.0, alpha: float = 0.20, label: str | None = None):
    lower, median, upper = posterior_interval(draws, credible_interval)
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, median, color=color, linewidth=linewidth, label=label)
    return lower, median, upper


def set_year_ticks(ax) -> None:
    ax.set_xticks(pd.to_datetime([f"{year}-01-01" for year in TICK_YEARS]))
    ax.set_xticklabels([str(year) for year in TICK_YEARS])


def annual_means_from_monthly(draws: np.ndarray, dates: pd.DatetimeIndex):
    years = np.asarray(dates.year)
    unique_years = np.unique(years)
    out = np.full((draws.shape[0], len(unique_years)), np.nan)
    for j, year in enumerate(unique_years):
        idx = np.flatnonzero(years == year)
        out[:, j] = np.mean(draws[:, idx], axis=1)
    return out, unique_years


def level_rate_draws(fit: PosteriorBundle, period: tuple[int, int]) -> np.ndarray:
    draws = fit.state_draws("alpha", original_scale=True)
    annual, years = annual_means_from_monthly(draws, dates_for_fit(fit))
    start, end = period
    keep = (years >= start) & (years <= end)
    if np.sum(keep) < 2:
        raise ValueError(f"{fit.series_name} has too few years in period {period}.")
    subset = annual[:, keep]
    n_years = years[keep][-1] - years[keep][0]
    return 10.0 * (subset[:, -1] - subset[:, 0]) / float(n_years)


def acceleration_draws(fit: PosteriorBundle) -> np.ndarray:
    return level_rate_draws(fit, RECENT_PERIOD) - level_rate_draws(fit, REFERENCE_PERIOD)


def annual_endpoint_draws(fit: PosteriorBundle):
    endpoint = np.asarray(fit.endpoint_draws(original_scale=True), dtype=float)
    finite_probability = float(np.mean(np.isfinite(endpoint)))
    endpoint = np.where(np.isfinite(endpoint), endpoint, np.nan)
    dates = dates_for_fit(fit)
    years = np.asarray(dates.year)
    unique_years = np.unique(years)
    annual = np.full((endpoint.shape[0], len(unique_years)), np.nan)
    for j, year in enumerate(unique_years):
        idx = np.flatnonzero(years == year)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            annual[:, j] = np.nanmax(endpoint[:, idx], axis=1)
    return annual, unique_years, finite_probability


def plot_level_axis(ax, fit: PosteriorBundle, credible_interval: float, *, title: str, show_xlabel: bool, show_ylabel: bool) -> None:
    add_band(
        ax,
        dates_for_fit(fit),
        fit.state_draws("alpha", original_scale=True),
        color=series_color(fit.series_name or ""),
        credible_interval=credible_interval,
        linewidth=2.1,
        alpha=0.20,
    )
    ax.set_title(title, color=NAVY, pad=5)
    if show_xlabel:
        ax.set_xlabel("Year")
    if show_ylabel:
        ax.set_ylabel("Temperature (°C)")
    set_year_ticks(ax)
    style_axis(ax)


def plot_slope_axis(ax, fit: PosteriorBundle, credible_interval: float, *, title: str, ylim: tuple[float, float], show_xlabel: bool, show_ylabel: bool) -> None:
    beta_draws = 120.0 * fit.state_draws("beta", original_scale=True)
    add_band(
        ax,
        dates_for_fit(fit),
        beta_draws,
        color=series_color(fit.series_name or ""),
        credible_interval=credible_interval,
        linewidth=2.0,
        alpha=0.20,
    )
    ax.set_ylim(*ylim)
    ax.set_title(title, color=NAVY, pad=5)
    if show_xlabel:
        ax.set_xlabel("Year")
    if show_ylabel:
        ax.set_ylabel("°C per decade")
    set_year_ticks(ax)
    style_axis(ax, zero_line=True)


def make_all_levels(fits: dict[str, PosteriorBundle], credible_interval: float):
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.8), constrained_layout=True)
    for i, row in enumerate(LEVEL_LAYOUT):
        for j, name in enumerate(row):
            plot_level_axis(
                axes[i, j],
                fits[name],
                credible_interval,
                title=SERIES_LABELS[name],
                show_xlabel=(i == 1),
                show_ylabel=(j == 0),
            )
    fig.suptitle("Location level trajectories", fontsize=16, fontweight="bold", color=NAVY)
    return fig


def extreme_slope_ylim(fits: dict[str, PosteriorBundle]) -> tuple[float, float]:
    arrays = []
    for name in EXTREME_SERIES:
        arrays.append(120.0 * fits[name].state_draws("beta", original_scale=True).ravel())
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    lo = float(np.quantile(values, 0.01))
    hi = float(np.quantile(values, 0.99))
    span = max(hi - lo, 0.2)
    lo -= 0.08 * span
    hi += 0.08 * span
    return lo, hi


def plot_acceleration_axis(ax, fits: dict[str, PosteriorBundle], outer_interval: float, inner_interval: float) -> None:
    order = list(ACCEL_ORDER)
    y = np.arange(len(order))
    draws_all = [acceleration_draws(fits[name]) for name in order]
    for i, (name, draws) in enumerate(zip(order, draws_all)):
        lo90, med, hi90 = posterior_interval(draws, outer_interval)
        lo50, _, hi50 = posterior_interval(draws, inner_interval)
        color = series_color(name)
        ax.hlines(i, lo90, hi90, color=color, linewidth=1.5, alpha=0.85)
        ax.hlines(i, lo50, hi50, color=color, linewidth=5.0, alpha=0.95)
        ax.plot(med, i, marker="o", color=color, markersize=6.2)
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("Recent − mid-century\n°C per decade")
    ax.set_title("Acceleration", color=NAVY, pad=6)
    ax.grid(axis="x", color=GRID, alpha=0.65, linewidth=0.65)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(NAVY)


def make_middle_panel(fits: dict[str, PosteriorBundle], credible_interval: float, inner_interval: float):
    fig = plt.figure(figsize=(10.8, 6.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.85])
    slope_order = (("TXx", "TXn"), ("TNx", "TNn"))
    ylim = extreme_slope_ylim(fits)
    for i, row in enumerate(slope_order):
        for j, name in enumerate(row):
            ax = fig.add_subplot(gs[i, j])
            plot_slope_axis(
                ax,
                fits[name],
                credible_interval,
                title=name,
                ylim=ylim,
                show_xlabel=(i == 1),
                show_ylabel=(j == 0),
            )
    ax_acc = fig.add_subplot(gs[:, 2])
    plot_acceleration_axis(ax_acc, fits, outer_interval=credible_interval, inner_interval=inner_interval)
    fig.suptitle("Warming rates", fontsize=16, fontweight="bold", color=NAVY)
    return fig


def make_txx_risk(fit: PosteriorBundle, credible_interval: float, risk_start_year: int):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)

    thresholds = (
        (36.8, TX_RED_LIGHT, "1947 record: 36.8 °C"),
        (39.7, TX_RED_DARK, "2019 record: 39.7 °C"),
    )
    for threshold, color, label in thresholds:
        draws, years = fit.return_period_draws(threshold, annual=True)
        years = np.asarray(years)
        keep = years >= risk_start_year
        draws = np.minimum(draws[:, keep], 10_000.0)
        add_band(
            axes[0],
            years[keep],
            draws,
            color=color,
            credible_interval=credible_interval,
            label=label,
            linewidth=2.1,
            alpha=0.18,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylim(1.0, 10_000.0)
    axes[0].set_title("How quickly do records become plausible?", color=NAVY)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Annual return period (years)")
    axes[0].legend(loc="upper right")
    style_axis(axes[0])

    endpoint, years, p_finite = annual_endpoint_draws(fit)
    keep = years >= risk_start_year
    add_band(
        axes[1],
        years[keep],
        endpoint[:, keep],
        color=TX_RED,
        credible_interval=credible_interval,
        linewidth=2.1,
        alpha=0.20,
    )
    axes[1].axhline(36.8, color=TX_RED_LIGHT, linestyle="--", linewidth=1.0)
    axes[1].axhline(39.7, color=TX_RED_DARK, linestyle="--", linewidth=1.0)
    axes[1].set_title("The finite upper endpoint moves", color=NAVY)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Annual upper endpoint (°C)")
    style_axis(axes[1])

    if p_finite < 0.995:
        warnings.warn("Some TXx endpoint draws are infinite; the plotted interval uses finite draws only.")

    return fig


def make_preview(fits: dict[str, PosteriorBundle], credible_interval: float, inner_interval: float, risk_start_year: int):
    fig = plt.figure(figsize=(11.4, 15.0), constrained_layout=True)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.2, 0.9])

    # Top: six level plots.
    gs_top = outer[0].subgridspec(2, 3)
    for i, row in enumerate(LEVEL_LAYOUT):
        for j, name in enumerate(row):
            ax = fig.add_subplot(gs_top[i, j])
            plot_level_axis(
                ax,
                fits[name],
                credible_interval,
                title=name,
                show_xlabel=(i == 1),
                show_ylabel=(j == 0),
            )

    # Middle: four extreme slopes + acceleration.
    gs_mid = outer[1].subgridspec(2, 3, width_ratios=[1.0, 1.0, 0.85])
    slope_order = (("TXx", "TXn"), ("TNx", "TNn"))
    ylim = extreme_slope_ylim(fits)
    for i, row in enumerate(slope_order):
        for j, name in enumerate(row):
            ax = fig.add_subplot(gs_mid[i, j])
            plot_slope_axis(
                ax,
                fits[name],
                credible_interval,
                title=name,
                ylim=ylim,
                show_xlabel=(i == 1),
                show_ylabel=(j == 0),
            )
    ax_acc = fig.add_subplot(gs_mid[:, 2])
    plot_acceleration_axis(ax_acc, fits, outer_interval=credible_interval, inner_interval=inner_interval)

    # Bottom: risk panel.
    gs_bot = outer[2].subgridspec(1, 2)
    fit = fits["TXx"]
    ax0 = fig.add_subplot(gs_bot[0, 0])
    ax1 = fig.add_subplot(gs_bot[0, 1])
    thresholds = (
        (36.8, TX_RED_LIGHT, "1947 record: 36.8 °C"),
        (39.7, TX_RED_DARK, "2019 record: 39.7 °C"),
    )
    for threshold, color, label in thresholds:
        draws, years = fit.return_period_draws(threshold, annual=True)
        years = np.asarray(years)
        keep = years >= risk_start_year
        add_band(ax0, years[keep], np.minimum(draws[:, keep], 10_000.0), color=color, credible_interval=credible_interval, linewidth=1.9, alpha=0.18, label=label)
    ax0.set_yscale("log")
    ax0.set_ylim(1.0, 10_000.0)
    ax0.set_title("How quickly do records become plausible?", color=NAVY)
    ax0.set_xlabel("Year")
    ax0.set_ylabel("Annual return period (years)")
    ax0.legend(loc="upper right")
    style_axis(ax0)

    endpoint, years, _ = annual_endpoint_draws(fit)
    keep = years >= risk_start_year
    add_band(ax1, years[keep], endpoint[:, keep], color=TX_RED, credible_interval=credible_interval, linewidth=1.9, alpha=0.20)
    ax1.axhline(36.8, color=TX_RED_LIGHT, linestyle="--", linewidth=0.9)
    ax1.axhline(39.7, color=TX_RED_DARK, linestyle="--", linewidth=0.9)
    ax1.set_title("The finite upper endpoint moves", color=NAVY)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual upper endpoint (°C)")
    style_axis(ax1)

    fig.suptitle("Tracking evolving temperature risk with structural time series", fontsize=18, fontweight="bold", color=NAVY)
    return fig


def acceleration_summary_table(fits: dict[str, PosteriorBundle], credible_interval: float) -> pd.DataFrame:
    rows = []
    for name in ACCEL_ORDER:
        draws = acceleration_draws(fits[name])
        lower, median, upper = posterior_interval(draws, credible_interval)
        rows.append(
            {
                "series": name,
                "reference_start": REFERENCE_PERIOD[0],
                "reference_end": REFERENCE_PERIOD[1],
                "recent_start": RECENT_PERIOD[0],
                "recent_end": RECENT_PERIOD[1],
                "median_difference_C_per_decade": float(median),
                "lower_difference_C_per_decade": float(lower),
                "upper_difference_C_per_decade": float(upper),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.credible_interval < 1.0:
        raise ValueError("--credible-interval must lie strictly between 0 and 1.")
    if not 0.0 < args.inner_credible_interval < args.credible_interval:
        raise ValueError("--inner-credible-interval must be positive and smaller than --credible-interval.")

    set_style()
    fits = load_fits(args.fit_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using {SCRIPT_VERSION}")
    print(f"Loaded pooled fits from: {args.fit_dir}")

    figures = {
        "poster_all_levels": make_all_levels(fits, args.credible_interval),
        "poster_middle_panel": make_middle_panel(fits, args.credible_interval, args.inner_credible_interval),
        "poster_txx_risk": make_txx_risk(fits["TXx"], args.credible_interval, args.risk_start_year),
        "poster_results_preview": make_preview(fits, args.credible_interval, args.inner_credible_interval, args.risk_start_year),
    }

    for stem, fig in figures.items():
        save_figure(fig, args.out_dir, stem, formats=args.formats, dpi=args.dpi)
        print(f"Saved {stem}")

    acceleration_summary_table(fits, args.credible_interval).to_csv(
        args.out_dir / "poster_acceleration_summary.csv", index=False
    )
    print(f"Poster outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
