"""Tables and figures for the focused COMPSTAT presentation workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..core import FitResult


COLORS = {
    "navy": "#123B4A",
    "teal": "#1D7F7A",
    "coral": "#D96C4F",
    "gold": "#E9B949",
    "blue": "#3B6FB6",
    "grey": "#7A8589",
    "light": "#E8F0F2",
}
STATE_LABELS = {0: "zero", 1: "fixed", 2: "dynamic"}


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def presentation_style() -> None:
    """Apply one restrained, high-contrast style to every workflow figure."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["navy"],
            "axes.labelcolor": COLORS["navy"],
            "axes.titlecolor": COLORS["navy"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "xtick.color": COLORS["navy"],
            "ytick.color": COLORS["navy"],
            "grid.color": "#D9E1E3",
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(
    figure,
    base: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 180,
) -> list[Path]:
    base_path = Path(base)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for extension in formats:
        target = base_path.with_suffix(f".{extension}")
        figure.savefig(target, dpi=dpi)
        artifacts.append(target)
    return artifacts


def _write_table(table: Any, path: str | Path, *, index: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    frame.to_csv(target, index=index)
    return target


def _safe_table(callable_):
    try:
        return callable_()
    except (KeyError, ValueError):
        return None


def fit_summary(fit: FitResult, *, source: str | Path | None = None) -> dict[str, Any]:
    diagnostics = fit.diagnostics()
    return {
        "source": None if source is None else str(source),
        "series": fit.series_name,
        "n_time": fit.n_time,
        "n_chains": fit.n_chains,
        "draws_per_chain": fit.draws_per_chain,
        "plan": fit.plan.to_dict(),
        "prior_profile": fit.meta.get("prior_profile"),
        "targets_exact_posterior": fit.plan.targets_exact_posterior,
        "warm_start_source": fit.meta.get("warm_start_source_engine"),
        "engine_diagnostics": diagnostics.get("engine", {}),
    }


def posterior_trajectory_table(
    fit: FitResult,
    *,
    credible_interval: float = 0.90,
) -> pd.DataFrame:
    values = np.asarray(fit.eta_draws(original_scale=True), dtype=float)
    alpha = 1.0 - float(credible_interval)
    lower, median, upper = np.quantile(
        values, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0], axis=0
    )
    date = (
        pd.to_datetime(fit.dates)
        if fit.dates is not None
        else np.arange(1, fit.n_time + 1)
    )
    return pd.DataFrame(
        {
            "date": date,
            "observed": np.asarray(fit.observed, dtype=float),
            "lower": lower,
            "median": median,
            "upper": upper,
        }
    )


def selection_recovery_table(
    fit: FitResult,
    truth: Mapping[str, Any] | None = None,
) -> pd.DataFrame | None:
    table = _safe_table(fit.component_probabilities)
    if table is None:
        return None
    frame = table.reset_index()
    if truth is None:
        return frame
    structural = dict(truth.get("structural_truth", truth))
    truth_codes = []
    correct = []
    for _, row in frame.iterrows():
        process = str(row["process"])
        code = int(structural[process])
        truth_codes.append(code)
        correct.append(float(row[STATE_LABELS[code]]))
    frame["truth_code"] = truth_codes
    frame["truth_state"] = [STATE_LABELS[value] for value in truth_codes]
    frame["probability_true_state"] = correct
    return frame


def export_fit_results(
    fit: FitResult,
    output_dir: str | Path,
    *,
    source: str | Path | None = None,
    truth: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Export the same semantic tables for simulated and observed fits."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    summary = target / "fit_summary.json"
    summary.write_text(
        json.dumps(fit_summary(fit, source=source), indent=2, default=_json_default),
        encoding="utf-8",
    )
    artifacts.append(summary)

    static = pd.DataFrame.from_dict(fit.static_summary(), orient="index")
    static.index.name = "parameter"
    artifacts.append(_write_table(static, target / "parameters.csv"))
    artifacts.append(
        _write_table(fit.diagnostics()["parameters"], target / "diagnostics.csv")
    )
    artifacts.append(
        _write_table(
            posterior_trajectory_table(fit),
            target / "posterior_trajectory.csv",
            index=False,
        )
    )
    engine = pd.DataFrame(
        [
            {"metric": name, "value": value}
            for name, value in fit.diagnostics().get("engine", {}).items()
        ]
    )
    artifacts.append(
        _write_table(engine, target / "algorithm_diagnostics.csv", index=False)
    )

    selection = selection_recovery_table(fit, truth)
    if selection is not None:
        artifacts.append(
            _write_table(selection, target / "selection_probabilities.csv", index=False)
        )
    models = _safe_table(fit.structural_model_probabilities)
    if models is not None:
        artifacts.append(_write_table(models, target / "structural_models.csv"))
    switching = _safe_table(fit.component_transition_summary)
    if switching is not None:
        artifacts.append(_write_table(switching, target / "selection_switching.csv"))
    return artifacts


def plot_uccle_record_figures(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 180,
) -> list[Path]:
    """Descriptive record figures used before any model is introduced."""

    import matplotlib.pyplot as plt

    presentation_style()
    values = frame.copy()
    if "date" in values:
        values["date"] = pd.to_datetime(values["date"])
        values = values.set_index("date")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    figure, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    for axis, name in zip(axes.ravel(), ("TXx", "TXn", "TNx", "TNn")):
        series = values[name]
        axis.scatter(series.index, series, s=4, alpha=0.28, color=COLORS["grey"])
        axis.plot(
            series.rolling(60, center=True, min_periods=24).median(),
            color=COLORS["teal"],
            linewidth=1.7,
            label="5-year rolling median",
        )
        axis.set_title(name)
        axis.set_ylabel("°C")
        axis.grid(axis="y")
    axes[0, 0].legend(loc="upper left")
    figure.suptitle("Uccle monthly temperature-extreme record", color=COLORS["navy"], weight="bold")
    figure.tight_layout()
    artifacts.extend(_save_figure(figure, target / "00_uccle_extremes", formats=formats, dpi=dpi))
    plt.close(figure)

    txx = values["TXx"]
    annual = txx.resample("YS").max()
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.scatter(txx.index, txx, s=5, alpha=0.22, color=COLORS["grey"], label="monthly TXx")
    axis.plot(annual.index, annual, color=COLORS["coral"], linewidth=0.9, alpha=0.75, label="annual maximum")
    axis.plot(
        annual.rolling(10, center=True, min_periods=5).mean(),
        color=COLORS["navy"],
        linewidth=2.4,
        label="10-year mean of annual maxima",
    )
    axis.axhline(40.0, color=COLORS["gold"], linestyle="--", linewidth=1.2, label="40 °C")
    axis.set_title("Evolution of monthly maximum daily temperature (TXx)")
    axis.set_ylabel("°C")
    axis.grid(axis="y")
    axis.legend(ncol=4, loc="upper left")
    figure.tight_layout()
    artifacts.extend(_save_figure(figure, target / "01_txx_evolution", formats=formats, dpi=dpi))
    plt.close(figure)

    years = np.asarray(txx.index.year)
    first_start = int(years.min())
    first_end = first_start + 29
    last_end = int(years.max())
    last_start = last_end - 29
    early = txx[(txx.index.year >= first_start) & (txx.index.year <= first_end)]
    late = txx[(txx.index.year >= last_start) & (txx.index.year <= last_end)]
    early_month = early.groupby(early.index.month).mean()
    late_month = late.groupby(late.index.month).mean()
    month = np.arange(1, 13)
    figure, axis = plt.subplots(figsize=(9, 4.6))
    axis.plot(month, early_month, marker="o", color=COLORS["grey"], label=f"{first_start}–{first_end}")
    axis.plot(month, late_month, marker="o", color=COLORS["coral"], label=f"{last_start}–{last_end}")
    axis.fill_between(month, early_month, late_month, color=COLORS["coral"], alpha=0.12)
    axis.set_xticks(month, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    axis.set_ylabel("mean monthly TXx (°C)")
    axis.set_title("The annual TXx cycle has shifted, not disappeared")
    axis.grid(axis="y")
    axis.legend()
    figure.tight_layout()
    artifacts.extend(_save_figure(figure, target / "02_txx_cycle_shift", formats=formats, dpi=dpi))
    plt.close(figure)
    return artifacts


def plot_tail_simulations(
    tables: Mapping[str, pd.DataFrame],
    truths: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 180,
) -> list[Path]:
    """Show each tail-class simulation separately, then compare densities."""

    import matplotlib.pyplot as plt
    from scipy.stats import genextreme

    presentation_style()
    target = Path(output_dir)
    order = ("bounded_tail", "gumbel_tail", "heavy_tail")
    artifacts: list[Path] = []

    for number, name in enumerate(order, start=1):
        table = tables[name]
        truth = truths[name]
        time = table["time"]
        figure, axis = plt.subplots(figsize=(11, 4.4), layout="constrained")
        axis.scatter(
            time,
            table["y"],
            s=9,
            alpha=0.36,
            color=COLORS["grey"],
            label="simulated maximum",
        )
        axis.plot(
            time,
            table["eta"],
            color=COLORS["teal"],
            linewidth=2.2,
            label="latent location",
        )
        axis.set_title(
            f"{truth['title']}: xi={truth['xi']:+.2f}, "
            f"sigma={truth['sigma']:.2f}"
        )
        axis.set_ylabel("block maximum")
        axis.set_xlabel("block")
        axis.grid(axis="y")
        axis.legend(loc="upper left")
        artifacts.extend(
            _save_figure(
                figure,
                target / f"10_tail_{number:02d}_{name}",
                formats=formats,
                dpi=dpi,
            )
        )
        plt.close(figure)

    ranges = [
        genextreme.ppf(
            (0.001, 0.995),
            c=-float(truths[name]["xi"]),
            loc=0.0,
            scale=float(truths[name]["sigma"]),
        )
        for name in order
    ]
    lower = min(float(values[0]) for values in ranges)
    upper = max(float(values[1]) for values in ranges)
    padding = 0.05 * (upper - lower)
    x = np.linspace(lower - padding, upper + padding, 900)
    figure, axis = plt.subplots(figsize=(9, 4.6), layout="constrained")
    palette = (COLORS["blue"], COLORS["teal"], COLORS["coral"])
    for name, color in zip(order, palette):
        truth = truths[name]
        xi, sigma = float(truth["xi"]), float(truth["sigma"])
        density = genextreme.pdf(x, c=-xi, loc=0.0, scale=sigma)
        axis.plot(x, density, color=color, linewidth=2.2, label=f"xi={xi:+.2f}: {truth['title']}")
        if xi < 0.0:
            endpoint = -sigma / xi
            axis.axvline(endpoint, color=color, linestyle="--", linewidth=1.0)
    axis.set_xlabel("value relative to location")
    axis.set_ylabel("density")
    axis.set_title("The shape parameter changes the tail, not the latent signal")
    axis.grid(axis="y")
    axis.legend()
    artifacts.extend(
        _save_figure(
            figure,
            target / "11_gev_shape_comparison",
            formats=formats,
            dpi=dpi,
        )
    )
    plt.close(figure)
    return artifacts


def plot_scale_simulations(
    tables: Mapping[str, pd.DataFrame],
    truths: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 180,
) -> list[Path]:
    """Show each scale simulation separately, then compare GEV densities."""

    import matplotlib.pyplot as plt
    from scipy.stats import genextreme

    presentation_style()
    target = Path(output_dir)
    order = ("low_scale", "reference_scale", "high_scale")
    artifacts: list[Path] = []

    for number, name in enumerate(order, start=1):
        table = tables[name]
        truth = truths[name]
        time = table["time"]
        figure, axis = plt.subplots(figsize=(11, 4.4), layout="constrained")
        axis.scatter(
            time,
            table["y"],
            s=9,
            alpha=0.36,
            color=COLORS["grey"],
            label="simulated maximum",
        )
        axis.plot(
            time,
            table["eta"],
            color=COLORS["navy"],
            linewidth=2.2,
            label="latent location",
        )
        axis.set_title(
            f"{truth['title']}: sigma={truth['sigma']:.2f}, "
            f"xi={truth['xi']:+.2f}"
        )
        axis.set_xlabel("block")
        axis.set_ylabel("block maximum")
        axis.grid(axis="y")
        axis.legend(loc="upper left")
        artifacts.extend(
            _save_figure(
                figure,
                target / f"12_scale_{number:02d}_{name}",
                formats=formats,
                dpi=dpi,
            )
        )
        plt.close(figure)

    ranges = [
        genextreme.ppf(
            (0.001, 0.995),
            c=-float(truths[name]["xi"]),
            loc=0.0,
            scale=float(truths[name]["sigma"]),
        )
        for name in order
    ]
    lower = min(float(values[0]) for values in ranges)
    upper = max(float(values[1]) for values in ranges)
    padding = 0.05 * (upper - lower)
    x = np.linspace(lower - padding, upper + padding, 900)
    figure, axis = plt.subplots(figsize=(9, 4.6), layout="constrained")
    palette = (COLORS["blue"], COLORS["teal"], COLORS["coral"])
    for name, color in zip(order, palette):
        truth = truths[name]
        xi, sigma = float(truth["xi"]), float(truth["sigma"])
        density = genextreme.pdf(x, c=-xi, loc=0.0, scale=sigma)
        axis.plot(
            x,
            density,
            color=color,
            linewidth=2.2,
            label=f"sigma={sigma:.2f}",
        )
        if xi < 0.0:
            axis.axvline(-sigma / xi, color=color, linestyle="--", linewidth=1.0)
    axis.set_xlabel("value relative to location")
    axis.set_ylabel("density")
    scale_xi = {float(truths[name]["xi"]) for name in order}
    axis.set_title(
        "The scale changes dispersion and the bounded-tail endpoint"
        if all(value < 0.0 for value in scale_xi)
        else "The scale changes dispersion at a fixed GEV shape"
    )
    axis.grid(axis="y")
    axis.legend()
    artifacts.extend(
        _save_figure(
            figure,
            target / "13_gev_scale_comparison",
            formats=formats,
            dpi=dpi,
        )
    )
    plt.close(figure)
    return artifacts


def plot_structural_simulations(
    tables: Mapping[str, pd.DataFrame],
    truths: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 180,
) -> list[Path]:
    """Give every structural series its own plot and decomposition figure."""

    import matplotlib.pyplot as plt

    presentation_style()
    target = Path(output_dir)
    names = list(tables)
    artifacts: list[Path] = []
    component_colors = (COLORS["teal"], COLORS["coral"], COLORS["blue"])

    for number, name in enumerate(names, start=1):
        table, truth = tables[name], truths[name]
        time = table["time"]
        labels = truth["structural_truth"]
        state = ", ".join(
            f"{key}={STATE_LABELS[int(value)]}" for key, value in labels.items()
        )

        figure, axis = plt.subplots(figsize=(11, 4.4), layout="constrained")
        axis.scatter(
            time,
            table["y"],
            s=8,
            alpha=0.28,
            color=COLORS["grey"],
            label="simulated maximum",
        )
        axis.plot(
            time,
            table["eta"],
            color=COLORS["navy"],
            linewidth=2.1,
            label="true predictor",
        )
        axis.set_title(
            f"{truth['title']}\n"
            f"{state}; period={truth['period']}, "
            f"sigma={truth['sigma']:.2f}, xi={truth['xi']:+.2f}",
            loc="left",
        )
        axis.set_xlabel("block")
        axis.set_ylabel("block maximum")
        axis.grid(axis="y")
        axis.legend(loc="upper left")
        artifacts.extend(
            _save_figure(
                figure,
                target / f"20_{number:02d}_{name}_series",
                formats=formats,
                dpi=dpi,
            )
        )
        plt.close(figure)

        figure, axes = plt.subplots(
            3,
            1,
            figsize=(11, 7.2),
            sharex=True,
            layout="constrained",
        )
        for axis, component, color in zip(
            axes,
            ("level", "slope", "seasonal"),
            component_colors,
        ):
            state_name = STATE_LABELS[int(labels[component])]
            axis.plot(time, table[component], color=color, linewidth=1.8)
            axis.set_ylabel(component)
            axis.set_title(f"{component}: {state_name}", loc="left", fontsize=10.5)
            axis.grid(axis="y")
        axes[-1].set_xlabel("block")
        figure.suptitle(
            f"{truth['title']}: true unobserved components",
            color=COLORS["navy"],
            weight="bold",
        )
        artifacts.extend(
            _save_figure(
                figure,
                target / f"21_{number:02d}_{name}_decomposition",
                formats=formats,
                dpi=dpi,
            )
        )
        plt.close(figure)
    return artifacts


def plot_fit_results(
    fit: FitResult,
    output_dir: str | Path,
    *,
    truth_table: pd.DataFrame | None = None,
    truth: Mapping[str, Any] | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 180,
    diagnostics: bool = False,
) -> list[Path]:
    """Generate one uniform figure set for every saved fit."""

    import matplotlib.pyplot as plt

    presentation_style()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    artifacts: list[Path] = []

    trajectory = posterior_trajectory_table(fit)
    figure, axis = plt.subplots(figsize=(10.5, 4.2))
    x = pd.to_datetime(trajectory["date"]) if fit.dates is not None else trajectory["date"]
    axis.scatter(x, trajectory["observed"], s=6, alpha=0.25, color=COLORS["grey"], label="observed")
    axis.fill_between(x, trajectory["lower"], trajectory["upper"], color=COLORS["teal"], alpha=0.18, label="90% interval")
    axis.plot(x, trajectory["median"], color=COLORS["teal"], linewidth=2.0, label="posterior median")
    if truth_table is not None:
        if fit.dates is not None and "date" in truth_table:
            truth_x = pd.to_datetime(truth_table["date"])
        elif "time" in truth_table:
            truth_x = truth_table["time"]
        else:
            truth_x = np.arange(1, len(truth_table) + 1)
        axis.plot(truth_x, truth_table["eta"], color=COLORS["navy"], linestyle="--", linewidth=1.4, label="true predictor")
    axis.set_title(f"{fit.series_name}: posterior latent predictor ({fit.plan.engine})")
    axis.set_ylabel("GEV location / °C")
    axis.grid(axis="y")
    axis.legend(ncol=4)
    figure.tight_layout()
    artifacts.extend(_save_figure(figure, target / "posterior_trajectory", formats=formats, dpi=dpi))
    plt.close(figure)

    if _safe_table(fit.component_probabilities) is not None:
        figure, _ = fit.plot("component_probabilities")
        figure.suptitle(f"{fit.series_name}: structural selection ({fit.plan.engine})", y=1.02, color=COLORS["navy"], weight="bold")
        artifacts.extend(_save_figure(figure, target / "selection_probabilities", formats=formats, dpi=dpi))
        plt.close(figure)

    parameter_truth = {} if truth is None else dict(truth.get("parameter_truth", {}))
    figure, _ = fit.plot("process_sd", truths=parameter_truth)
    figure.suptitle(f"{fit.series_name}: prior to posterior ({fit.plan.engine})", y=1.01, color=COLORS["navy"], weight="bold")
    artifacts.extend(_save_figure(figure, target / "prior_to_posterior_process_sd", formats=formats, dpi=dpi))
    plt.close(figure)

    if fit.family == "gev":
        figure, _ = fit.plot("parameter_density", parameters=("sigma", "xi"), truths=parameter_truth)
        artifacts.extend(_save_figure(figure, target / "gev_parameters", formats=formats, dpi=dpi))
        plt.close(figure)
        if truth_table is None:
            figure, _ = fit.plot("endpoint")
            artifacts.extend(_save_figure(figure, target / "endpoint", formats=formats, dpi=dpi))
            plt.close(figure)

    if diagnostics:
        figure, _ = fit.plot("traces")
        artifacts.extend(_save_figure(figure, target / "traces", formats=formats, dpi=dpi))
        plt.close(figure)
        figure, _ = fit.plot("acf", max_lag=50)
        artifacts.extend(_save_figure(figure, target / "acf", formats=formats, dpi=dpi))
        plt.close(figure)
    return artifacts


def collect_selection_probabilities(
    fits: Mapping[tuple[str, str], FitResult],
    truths: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    rows = []
    for (engine, name), fit in fits.items():
        truth = None if truths is None else truths.get(name)
        table = selection_recovery_table(fit, truth)
        if table is None:
            continue
        table.insert(0, "name", name)
        table.insert(0, "engine", engine)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_selection_recovery(
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 180,
) -> list[Path]:
    """Heat map of posterior probability assigned to each simulated truth."""

    import matplotlib.pyplot as plt

    if table.empty or "probability_true_state" not in table:
        return []
    presentation_style()
    engines = [name for name in ("laplace", "pgas") if name in set(table["engine"])]
    if not engines:
        return []
    scenarios = list(dict.fromkeys(table["name"]))
    components = ("level", "slope", "seasonal")
    figure, axes = plt.subplots(
        1,
        len(engines),
        figsize=(5.7 * len(engines) + 1.5, max(4.5, 0.55 * len(scenarios) + 1.8)),
        squeeze=False,
    )
    image = None
    for panel, (axis, engine) in enumerate(zip(axes[0], engines)):
        selected = table[table["engine"] == engine]
        matrix = np.full((len(scenarios), len(components)), np.nan)
        for i, scenario in enumerate(scenarios):
            for j, component in enumerate(components):
                row = selected[(selected["name"] == scenario) & (selected["process"] == component)]
                if not row.empty:
                    matrix[i, j] = float(row.iloc[0]["probability_true_state"])
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] > 0.58 else COLORS["navy"])
        axis.set_xticks(range(len(components)), components)
        axis.set_yticks(range(len(scenarios)))
        axis.set_yticklabels(
            [value.replace("_", " ") for value in scenarios]
            if panel == 0
            else []
        )
        axis.set_title(engine.upper())
    color_axis = figure.add_axes((0.91, 0.18, 0.018, 0.64))
    figure.colorbar(
        image,
        cax=color_axis,
        label="posterior probability of true state",
    )
    figure.suptitle("Structural-selection recovery", color=COLORS["navy"], weight="bold")
    figure.subplots_adjust(
        left=0.25,
        right=0.88,
        top=0.84,
        bottom=0.12,
        wspace=0.16,
    )
    artifacts = _save_figure(figure, Path(output_dir) / "30_selection_recovery", formats=formats, dpi=dpi)
    plt.close(figure)
    return artifacts


def plot_uccle_selection_comparison(
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 180,
) -> list[Path]:
    """Compare Laplace and PGAS component probabilities for four extremes."""

    import matplotlib.pyplot as plt

    if table.empty:
        return []
    presentation_style()
    series = list(dict.fromkeys(table["name"]))
    components = ("level", "slope", "seasonal")
    engines = [value for value in ("laplace", "pgas") if value in set(table["engine"])]
    figure, axes = plt.subplots(len(series), len(components), figsize=(12, 2.4 * len(series)), squeeze=False)
    state_colors = {"zero": "#C8D0D2", "fixed": COLORS["gold"], "dynamic": COLORS["teal"]}
    for row, name in enumerate(series):
        for column, component in enumerate(components):
            axis = axes[row, column]
            selected = table[(table["name"] == name) & (table["process"] == component)]
            x = np.arange(len(engines))
            bottom = np.zeros(len(engines))
            for state in ("zero", "fixed", "dynamic"):
                values = np.asarray([
                    float(selected[selected["engine"] == engine].iloc[0][state])
                    if not selected[selected["engine"] == engine].empty else 0.0
                    for engine in engines
                ])
                axis.bar(x, values, bottom=bottom, color=state_colors[state], label=state)
                bottom += values
            axis.set_xticks(x, [value.upper() for value in engines])
            axis.set_ylim(0.0, 1.0)
            if row == 0:
                axis.set_title(component)
            if column == 0:
                axis.set_ylabel(name)
    axes[0, -1].legend(ncol=3, bbox_to_anchor=(1.0, 1.38), loc="upper right")
    figure.suptitle("Uccle structural probabilities: approximation versus PGAS", color=COLORS["navy"], weight="bold")
    figure.tight_layout()
    artifacts = _save_figure(figure, Path(output_dir) / "40_uccle_selection_comparison", formats=formats, dpi=dpi)
    plt.close(figure)
    return artifacts


__all__ = [
    "COLORS",
    "collect_selection_probabilities",
    "export_fit_results",
    "fit_summary",
    "plot_fit_results",
    "plot_scale_simulations",
    "plot_selection_recovery",
    "plot_structural_simulations",
    "plot_tail_simulations",
    "plot_uccle_record_figures",
    "plot_uccle_selection_comparison",
    "posterior_trajectory_table",
    "presentation_style",
    "selection_recovery_table",
]
