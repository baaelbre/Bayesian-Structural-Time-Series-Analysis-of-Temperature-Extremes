"""Tidy tables and figures generated from saved :class:`FitResult` objects."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..core import FitResult


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


def _write_table(table: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    frame.to_csv(path, index=True)
    return path


def _safe_table(callable_):
    try:
        return callable_()
    except ValueError:
        return None


def _full_period_rates(fit: FitResult) -> pd.DataFrame | None:
    if fit.dates is None:
        return None
    years = np.asarray(pd.to_datetime(fit.dates).year, dtype=int)
    start, end = int(years.min()), int(years.max())
    if start == end:
        return None
    if fit.is_multiseries_model:
        return pd.DataFrame(
            [fit.channel_rate_summary(channel, start, end) for channel in fit.channel_names]
        ).set_index("channel")
    return fit.period_rate_summary({"fitted_period": (start, end)})


def _txx_risk_table(
    fit: FitResult,
    *,
    thresholds: tuple[float, ...] = (30.0, 35.0, 40.0),
    credible_interval: float = 0.90,
) -> pd.DataFrame | None:
    channel = None
    if fit.is_multiseries_model:
        if "TXx" not in fit.channel_names:
            return None
        channel = "TXx"
    elif fit.family != "gev" or fit.series_name != "TXx":
        return None
    alpha = 1.0 - credible_interval
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        probability, labels = fit.exceedance_probability_draws(
            threshold, annual=True, return_labels=True, channel=channel
        )
        low, median, high = np.quantile(
            probability, [alpha / 2.0, 0.5, 1.0 - alpha / 2.0], axis=0
        )
        for label, lo, med, hi in zip(labels, low, median, high):
            rows.append(
                {
                    "threshold_c": threshold,
                    "year": int(label),
                    "lower": float(lo),
                    "median": float(med),
                    "upper": float(hi),
                }
            )
    return pd.DataFrame(rows).set_index(["threshold_c", "year"])


def fit_summary(fit: FitResult, *, source: str | Path | None = None) -> dict[str, Any]:
    """Return the compact machine-readable contract for one fit."""

    diagnostics = fit.diagnostics()
    restored = int(fit.meta.get("restored_iterations", 0) or 0)
    return {
        "source": None if source is None else str(source),
        "series": fit.series_name,
        "channels": list(fit.channel_names),
        "n_time": fit.n_time,
        "n_chains": fit.n_chains,
        "draws_per_chain": fit.draws_per_chain,
        "plan": fit.plan.to_dict(),
        "prior_profile": fit.meta.get("prior_profile"),
        "hierarchy_pool": fit.meta.get("hierarchy_pool"),
        "model_space": fit.meta.get("trend_model_space"),
        "targets_exact_posterior": fit.plan.targets_exact_posterior,
        "restored_iterations": restored,
        "engine_diagnostics": diagnostics.get("engine", {}),
    }


def export_fit_results(
    fit: FitResult,
    output_dir: str | Path,
    *,
    source: str | Path | None = None,
) -> list[Path]:
    """Write a complete set of tidy, analysis-ready tables for one fit."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    summary_path = target / "fit_summary.json"
    summary_path.write_text(
        json.dumps(fit_summary(fit, source=source), indent=2, default=_json_default),
        encoding="utf-8",
    )
    artifacts.append(summary_path)

    static = pd.DataFrame.from_dict(fit.static_summary(), orient="index")
    static.index.name = "parameter"
    artifacts.append(_write_table(static, target / "parameters.csv"))
    artifacts.append(
        _write_table(fit.diagnostics()["parameters"], target / "diagnostics.csv")
    )

    optional = {
        "component_probabilities.csv": _safe_table(fit.component_probabilities),
        "structural_models.csv": _safe_table(fit.structural_model_probabilities),
        "allocation_switching.csv": _safe_table(fit.component_transition_summary),
        "hierarchical_probabilities.csv": _safe_table(fit.hierarchical_probabilities),
        "hierarchical_slabs.csv": _safe_table(fit.hierarchical_slab_summary),
        "rates.csv": _full_period_rates(fit),
        "txx_annual_exceedance.csv": _txx_risk_table(fit),
    }
    for filename, table in optional.items():
        if table is not None:
            artifacts.append(_write_table(table, target / filename))
    return artifacts


def plot_fit_results(
    fit: FitResult,
    output_dir: str | Path,
    *,
    diagnostics: bool = False,
) -> list[Path]:
    """Generate the standard presentation figures from an existing fit."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    mpl_config = target.parent / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    artifacts: list[Path] = []

    def draw(kind: str, filename: str, **kwargs) -> None:
        path = target / filename
        fit.plot(kind, save=path, **kwargs)
        artifacts.append(path)

    if fit.is_multiseries_model:
        for channel in fit.channel_names:
            draw("channel", f"channel_{channel}.png", channel=channel)
        draw("component_probabilities", "component_probabilities.png")
        draw("hierarchy", "hierarchy.png")
        draw("process_sd", "process_sds.png")
    else:
        draw("predictor", "predictor.png")
        draw("level_slope", "level_slope.png")
        draw("process_sd", "process_sds.png")
        if _safe_table(fit.component_probabilities) is not None:
            draw("component_probabilities", "component_probabilities.png")
        if fit.family == "gev":
            draw("endpoint", "endpoint.png")
    if diagnostics:
        draw("traces", "traces.png")
        draw("acf", "acf.png", max_lag=50)
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass
    return artifacts
