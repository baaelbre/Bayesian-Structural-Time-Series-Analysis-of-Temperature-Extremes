#!/usr/bin/env python3
"""Create poster-ready Uccle figures from six saved bucex fits.

This script does not fit any models. It loads the six PosteriorBundle files
written by the HPC jobs and creates separate vector/raster figures that can be
placed directly in PowerPoint, Illustrator, Inkscape, or LaTeX.

Expected input files
--------------------
    <fit-dir>/TXm_bucex_v0.3.3.pkl
    <fit-dir>/TNm_bucex_v0.3.3.pkl
    <fit-dir>/TXx_bucex_v0.3.3.pkl
    <fit-dir>/TXn_bucex_v0.3.3.pkl
    <fit-dir>/TNx_bucex_v0.3.3.pkl
    <fit-dir>/TNn_bucex_v0.3.3.pkl

Example
-------
    python examples/make_uccle_poster_figures.py \
        --fit-dir results/uccle_v033_regularized/fits \
        --out-dir results/uccle_v033_regularized/poster

Main outputs
------------
    poster_bulk_levels.*
    poster_extreme_levels.*
    poster_txx_risk.*
    poster_acceleration.*
    poster_period_rates.*
    poster_results_preview.*
    poster_period_rate_summary.csv
    poster_acceleration_summary.csv

The finite-period level rate calculation is performed draw by draw. Thus, every
box or interval represents posterior uncertainty in the average local slope of
a period; it does not pool correlated monthly states as if they were separate
observations.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bucex import PosteriorBundle


SERIES = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
EXTREME_SERIES = ("TXx", "TXn", "TNx", "TNn")

# Edit these once if you prefer climatological normals instead.
PERIODS: Mapping[str, tuple[int, int]] = {
    "Early\n1892–1949": (1892, 1949),
    "Mid-century\n1950–1979": (1950, 1979),
    "Recent\n1980–2022": (1980, 2022),
}
REFERENCE_PERIOD = (1950, 1979)
RECENT_PERIOD = (1980, 2022)

# Poster palette.
NAVY = "#082B57"
TX_RED = "#CF4A45"
TX_RED_DARK = "#A91518"
TX_RED_LIGHT = "#E88983"
TN_BLUE = "#3E79A9"
GRID = "#D7DEE8"
TEXT = "#172033"

SERIES_LABELS = {
    "TXm": "TXm — mean daytime temperature",
    "TNm": "TNm — mean nighttime temperature",
    "TXx": "TXx — hottest day",
    "TXn": "TXn — coldest daytime maximum",
    "TNx": "TNx — warmest night",
    "TNn": "TNn — coldest night",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create poster-ready figures from six saved Uccle bucex fits."
    )
    parser.add_argument(
        "--fit-dir",
        type=Path,
        default=Path("results/uccle_v033_regularized/fits"),
        help="Directory containing the six PosteriorBundle pickle files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/uccle_v033_regularized/poster"),
        help="Directory in which the poster figures are written.",
    )
    parser.add_argument(
        "--credible-interval",
        type=float,
        default=0.90,
        help="Outer posterior credible interval. Default: 0.90.",
    )
    parser.add_argument(
        "--inner-credible-interval",
        type=float,
        default=0.50,
        help="Inner interval used in acceleration plots. Default: 0.50.",
    )
    parser.add_argument(
        "--risk-start-year",
        type=int,
        default=1970,
        help="First year shown in the TXx risk panels. Default: 1970.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Resolution for PNG output. Default: 400.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
        help="Figure formats to save. Default: png pdf svg.",
    )
    return parser.parse_args()


def set_poster_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
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


def style_axis(ax, *, zero_line: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NAVY)
    ax.spines["bottom"].set_color(NAVY)
    ax.grid(axis="y", color=GRID, alpha=0.58, linewidth=0.65)
    ax.set_axisbelow(True)
    if zero_line:
        ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--", zorder=1)


def load_fits(fit_dir: Path) -> dict[str, PosteriorBundle]:
    fits: dict[str, PosteriorBundle] = {}
    missing: list[Path] = []

    for name in SERIES:
        preferred = fit_dir / f"{name}_bucex_v0.3.3.pkl"
        candidates = [preferred] + sorted(
            fit_dir.glob(f"{name}_bucex_v*.pkl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        path = next((item for item in candidates if item.exists()), preferred)
        if not path.exists():
            missing.append(preferred)
            continue
        fit = PosteriorBundle.load(path)
        if fit.draws_states is None:
            raise ValueError(f"{path} does not contain posterior state draws.")
        if fit.dates is None:
            raise ValueError(f"{path} does not contain dates.")
        fits[name] = fit

    if missing:
        files = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing Uccle fit files:\n{files}")

    return fits


def dates_for_fit(fit: PosteriorBundle) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(fit.dates)))
    if len(dates) != fit.n_time:
        raise ValueError(
            f"Date length ({len(dates)}) does not equal fitted time length "
            f"({fit.n_time}) for {fit.series_name}."
        )
    return dates


def posterior_interval(
    draws: np.ndarray,
    credible_interval: float,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = 1.0 - credible_interval
    return tuple(
        np.nanquantile(
            np.asarray(draws, dtype=float),
            [alpha / 2.0, 0.5, 1.0 - alpha / 2.0],
            axis=axis,
        )
    )


def add_band(
    ax,
    x,
    draws: np.ndarray,
    *,
    color: str,
    credible_interval: float,
    label: str | None = None,
    linewidth: float = 2.0,
    alpha: float = 0.20,
    zorder: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, median, upper = posterior_interval(draws, credible_interval)
    ax.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=zorder - 1,
    )
    ax.plot(x, median, color=color, linewidth=linewidth, label=label, zorder=zorder)
    return lower, median, upper


def save_figure(
    fig,
    out_dir: Path,
    stem: str,
    *,
    formats: Iterable[str],
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.03,
            "facecolor": "white",
        }
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(out_dir / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def plot_level_on_axis(
    ax,
    fit: PosteriorBundle,
    *,
    credible_interval: float,
    title: str,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    dates = dates_for_fit(fit)
    draws = fit.state_draws("alpha", original_scale=True)
    color = series_color(fit.series_name or "")
    add_band(
        ax,
        dates,
        draws,
        color=color,
        credible_interval=credible_interval,
        linewidth=2.0,
    )
    ax.set_title(title, color=NAVY, pad=5)
    if show_xlabel:
        ax.set_xlabel("Year")
    if show_ylabel:
        ax.set_ylabel("Temperature (°C)")
    style_axis(ax)


def make_bulk_levels(
    fits: dict[str, PosteriorBundle],
    *,
    credible_interval: float,
):
    fig, ax = plt.subplots(figsize=(7.4, 3.45), constrained_layout=True)

    for name in ("TXm", "TNm"):
        fit = fits[name]
        add_band(
            ax,
            dates_for_fit(fit),
            fit.state_draws("alpha", original_scale=True),
            color=series_color(name),
            credible_interval=credible_interval,
            label=name,
            linewidth=2.2,
            alpha=0.18,
        )

    ax.set_title("Bulk temperature: posterior level trajectories", color=NAVY)
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper left", ncol=2)
    style_axis(ax)
    return fig


def make_extreme_levels(
    fits: dict[str, PosteriorBundle],
    *,
    credible_interval: float,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.4, 5.4),
        sharex=True,
        constrained_layout=True,
    )

    for ax, name in zip(axes.flat, EXTREME_SERIES):
        plot_level_on_axis(
            ax,
            fits[name],
            credible_interval=credible_interval,
            title=SERIES_LABELS[name],
            show_xlabel=ax in axes[-1, :],
            show_ylabel=ax in axes[:, 0],
        )

    fig.suptitle(
        "Tail-specific change: structural DGEV level trajectories",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )
    return fig


def annual_endpoint_draws(
    fit: PosteriorBundle,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return annual TXx upper-endpoint draws.

    The endpoint of an annual maximum over monthly block maxima is the maximum
    of the twelve monthly endpoints. This removes the seasonal zig-zag and is
    the natural annual endpoint to place beside annual return periods.
    """
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


def make_txx_risk(
    fit: PosteriorBundle,
    *,
    credible_interval: float,
    risk_start_year: int,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.55),
        constrained_layout=True,
    )

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
        axes[1].text(
            0.02,
            0.03,
            f"P(finite endpoint) = {p_finite:.3f}",
            transform=axes[1].transAxes,
            fontsize=7.5,
            color="0.35",
        )
        warnings.warn(
            "Some TXx posterior draws have xi >= 0, so their endpoint is infinite. "
            "The plotted endpoint intervals use finite draws only."
        )

    fig.suptitle(
        "Tracking evolving hot-day risk",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )
    return fig


def period_rate_draws(
    fit: PosteriorBundle,
    periods: Mapping[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Finite-period rates derived from posterior level changes."""
    return fit.period_rate_draws(dict(periods), scale="decade")


def acceleration_draws(fit: PosteriorBundle) -> np.ndarray:
    return fit.rate_contrast_draws(
        recent=RECENT_PERIOD,
        reference=REFERENCE_PERIOD,
        scale="decade",
    )


def make_acceleration(
    fits: dict[str, PosteriorBundle],
    *,
    outer_interval: float,
    inner_interval: float,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)

    order = list(reversed(SERIES))
    y = np.arange(len(order))
    all_draws: list[np.ndarray] = []

    for i, name in enumerate(order):
        draws = acceleration_draws(fits[name])
        all_draws.append(draws)
        lo90, med, hi90 = posterior_interval(draws, outer_interval)
        lo50, _, hi50 = posterior_interval(draws, inner_interval)
        color = series_color(name)

        ax.hlines(i, lo90, hi90, color=color, linewidth=1.6, alpha=0.75, zorder=2)
        ax.hlines(i, lo50, hi50, color=color, linewidth=5.0, alpha=0.95, zorder=3)
        ax.plot(med, i, marker="o", color=color, markersize=6.5, zorder=4)

    all_values = np.concatenate(all_draws)
    finite_values = all_values[np.isfinite(all_values)]
    left = float(np.quantile(finite_values, 0.005))
    right = float(np.quantile(finite_values, 0.995))
    span = max(right - left, 0.1)
    annotation_x = right + 0.10 * span
    ax.set_xlim(left - 0.10 * span, right + 0.42 * span)

    for i, (name, draws) in enumerate(zip(order, all_draws)):
        probability = float(np.mean(draws > 0.0))
        ax.text(
            annotation_x,
            i,
            f"P(acceleration) = {probability:.2f}",
            va="center",
            ha="left",
            fontsize=8.0,
            color=TEXT,
        )

    ax.axvline(0.0, color="0.30", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(
        "Change in fitted level rate (°C per decade)\n"
        "1980–2022 minus 1950–1979"
    )
    ax.set_title("Has warming accelerated?", color=NAVY)
    ax.grid(axis="x", color=GRID, alpha=0.65, linewidth=0.65)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(NAVY)
    return fig


def make_period_rates(
    fits: dict[str, PosteriorBundle],
):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.0, 5.8),
        sharey=True,
        constrained_layout=True,
    )

    labels = list(PERIODS.keys())

    for ax, name in zip(axes.flat, SERIES):
        draws = period_rate_draws(fits[name], PERIODS)
        values = [draws[label] for label in labels]
        color = series_color(name)

        boxes = ax.boxplot(
            values,
            positions=np.arange(1, len(labels) + 1),
            widths=0.58,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.6},
            whiskerprops={"color": color, "linewidth": 1.0},
            capprops={"color": color, "linewidth": 1.0},
            boxprops={"edgecolor": color, "linewidth": 1.0},
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.78)

        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(name, color=NAVY)
        style_axis(ax)

    axes[0, 0].set_ylabel("Fitted level change (°C per decade)")
    axes[1, 0].set_ylabel("Fitted level change (°C per decade)")
    fig.suptitle(
        "Posterior finite-period warming rates",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )
    return fig


def period_rate_summary_table(
    fits: dict[str, PosteriorBundle],
    *,
    credible_interval: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for name in SERIES:
        draws_by_period = period_rate_draws(fits[name], PERIODS)
        for label, (start, end) in PERIODS.items():
            draws = draws_by_period[label]
            lower, median, upper = posterior_interval(draws, credible_interval)
            rows.append(
                {
                    "series": name,
                    "period": label.replace("\n", " "),
                    "start_year": start,
                    "end_year": end,
                    "median_C_per_decade": float(median),
                    "lower_C_per_decade": float(lower),
                    "upper_C_per_decade": float(upper),
                    "probability_positive": float(np.mean(draws > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def acceleration_summary_table(
    fits: dict[str, PosteriorBundle],
    *,
    credible_interval: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for name in SERIES:
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
                "probability_acceleration": float(np.mean(draws > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def make_results_preview(
    fits: dict[str, PosteriorBundle],
    *,
    credible_interval: float,
    inner_interval: float,
    risk_start_year: int,
):
    """Create a compact preview; use the separate files for final poster layout."""
    fig = plt.figure(figsize=(12.0, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(1.05, 1.25, 1.05))

    # Bulk overlay.
    ax_bulk = fig.add_subplot(grid[0, :])
    for name in ("TXm", "TNm"):
        fit = fits[name]
        add_band(
            ax_bulk,
            dates_for_fit(fit),
            fit.state_draws("alpha", original_scale=True),
            color=series_color(name),
            credible_interval=credible_interval,
            label=name,
            linewidth=2.2,
            alpha=0.18,
        )
    ax_bulk.set_title("A. Compare bulk warming", color=NAVY, loc="left")
    ax_bulk.set_xlabel("Year")
    ax_bulk.set_ylabel("Temperature (°C)")
    ax_bulk.legend(loc="upper left", ncol=2)
    style_axis(ax_bulk)

    # Four tail levels.
    for column, name in enumerate(EXTREME_SERIES):
        ax = fig.add_subplot(grid[1, column])
        panel_title = "" if column == 0 else name
        plot_level_on_axis(
            ax,
            fits[name],
            credible_interval=credible_interval,
            title=panel_title,
            show_xlabel=True,
            show_ylabel=(column == 0),
        )
        if column == 0:
            ax.set_title("B. Tail change\nTXx", color=NAVY, loc="left")

    # Return periods.
    ax_risk = fig.add_subplot(grid[2, 0:2])
    fit = fits["TXx"]
    for threshold, color, label in (
        (36.8, TX_RED_LIGHT, "36.8 °C"),
        (39.7, TX_RED_DARK, "39.7 °C"),
    ):
        draws, years = fit.return_period_draws(threshold, annual=True)
        years = np.asarray(years)
        keep = years >= risk_start_year
        add_band(
            ax_risk,
            years[keep],
            np.minimum(draws[:, keep], 10_000.0),
            color=color,
            credible_interval=credible_interval,
            label=label,
            linewidth=1.9,
            alpha=0.18,
        )
    ax_risk.set_yscale("log")
    ax_risk.set_ylim(1.0, 10_000.0)
    ax_risk.set_title("C. Evolving record risk", color=NAVY, loc="left")
    ax_risk.set_xlabel("Year")
    ax_risk.set_ylabel("Annual return period")
    ax_risk.legend(loc="upper right")
    style_axis(ax_risk)

    # Annual endpoint.
    ax_endpoint = fig.add_subplot(grid[2, 2])
    endpoint, years, _ = annual_endpoint_draws(fit)
    keep = years >= risk_start_year
    add_band(
        ax_endpoint,
        years[keep],
        endpoint[:, keep],
        color=TX_RED,
        credible_interval=credible_interval,
        linewidth=1.9,
        alpha=0.20,
    )
    ax_endpoint.axhline(39.7, color=TX_RED_DARK, linestyle="--", linewidth=0.9)
    ax_endpoint.set_title("Moving endpoint", color=NAVY)
    ax_endpoint.set_xlabel("Year")
    ax_endpoint.set_ylabel("°C")
    style_axis(ax_endpoint)

    # Acceleration summary.
    ax_acc = fig.add_subplot(grid[2, 3])
    order = list(reversed(SERIES))
    all_draws = [acceleration_draws(fits[name]) for name in order]
    for i, (name, draws) in enumerate(zip(order, all_draws)):
        lo90, med, hi90 = posterior_interval(draws, credible_interval)
        lo50, _, hi50 = posterior_interval(draws, inner_interval)
        color = series_color(name)
        ax_acc.hlines(i, lo90, hi90, color=color, linewidth=1.3)
        ax_acc.hlines(i, lo50, hi50, color=color, linewidth=4.0)
        ax_acc.plot(med, i, "o", color=color, markersize=4.5)
    ax_acc.axvline(0.0, color="0.35", linestyle="--", linewidth=0.9)
    ax_acc.set_yticks(np.arange(len(order)))
    ax_acc.set_yticklabels(order, fontsize=7.0)
    ax_acc.set_title("Acceleration", color=NAVY)
    ax_acc.set_xlabel("Recent − mid-century\n°C per decade", fontsize=8)
    ax_acc.grid(axis="x", color=GRID, alpha=0.6, linewidth=0.6)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)
    ax_acc.spines["left"].set_visible(False)

    fig.suptitle(
        "Tracking evolving temperature risk with structural time series",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
    )
    return fig


def main() -> None:
    args = parse_args()

    if not 0.0 < args.credible_interval < 1.0:
        raise ValueError("--credible-interval must lie strictly between 0 and 1.")
    if not 0.0 < args.inner_credible_interval < args.credible_interval:
        raise ValueError(
            "--inner-credible-interval must be positive and smaller than "
            "--credible-interval."
        )

    set_poster_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fits = load_fits(args.fit_dir)

    print(f"Loaded six fitted models from: {args.fit_dir}")

    figures = {
        "poster_bulk_levels": make_bulk_levels(
            fits,
            credible_interval=args.credible_interval,
        ),
        "poster_extreme_levels": make_extreme_levels(
            fits,
            credible_interval=args.credible_interval,
        ),
        "poster_txx_risk": make_txx_risk(
            fits["TXx"],
            credible_interval=args.credible_interval,
            risk_start_year=args.risk_start_year,
        ),
        "poster_acceleration": make_acceleration(
            fits,
            outer_interval=args.credible_interval,
            inner_interval=args.inner_credible_interval,
        ),
        "poster_period_rates": make_period_rates(fits),
        "poster_results_preview": make_results_preview(
            fits,
            credible_interval=args.credible_interval,
            inner_interval=args.inner_credible_interval,
            risk_start_year=args.risk_start_year,
        ),
    }

    for stem, fig in figures.items():
        save_figure(
            fig,
            args.out_dir,
            stem,
            formats=args.formats,
            dpi=args.dpi,
        )
        print(f"Saved {stem}")

    period_table = period_rate_summary_table(
        fits,
        credible_interval=args.credible_interval,
    )
    period_table.to_csv(
        args.out_dir / "poster_period_rate_summary.csv",
        index=False,
    )

    acceleration_table = acceleration_summary_table(
        fits,
        credible_interval=args.credible_interval,
    )
    acceleration_table.to_csv(
        args.out_dir / "poster_acceleration_summary.csv",
        index=False,
    )

    print(f"Poster outputs written to: {args.out_dir}")
    print("Use the separate SVG/PDF files for the final PowerPoint composition.")


if __name__ == "__main__":
    main()
