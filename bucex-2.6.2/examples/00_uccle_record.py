"""Describe the complete Uccle record before fitting a model.

Run from the package root with ``python examples/00_uccle_record.py``.
All settings intended for editing are collected directly below.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (SOURCE_ROOT / "bucex").is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


# Results. BUCEX_RUN_ID may supply one shared timestamp to several HPC jobs.
# The directory name contains only the settings useful for browsing results;
# run_config.json records the complete configuration.
RESULTS_ROOT = Path(os.environ.get("BUCEX_RESULTS_ROOT", "results"))
SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = os.environ.get("BUCEX_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OVERWRITE = os.environ.get("BUCEX_OVERWRITE", "0").lower() in {"1", "true", "yes"}

# Data and smoothing.
DATA_DIR: Path | None = (
    Path(os.environ["BUCEX_DATA_DIR"]) if "BUCEX_DATA_DIR" in os.environ else None
)  # None uses the Uccle data shipped with bucex.
START = os.environ.get("BUCEX_START", "1892-01-01")
END: str | None = os.environ.get("BUCEX_END") or None
SERIES = ("TXx", "TXn", "TNx", "TNn")
LOESS_FRACTION_MONTHLY = 0.16
LOESS_FRACTION_ANNUAL = 0.25
LOESS_ROBUST_ITERATIONS = 2

# Graphics.
FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
COLORS = {"navy": "#123B4A", "teal": "#1D7F7A", "grey": "#7A8589"}

START_TAG = START.removesuffix("-01-01")
END_TAG = (END or "latest").removesuffix("-12-31")
RUN_SIGNATURE = f"y{START_TAG}-{END_TAG}"
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


def main() -> None:
    values = [
        bx.load_uccle_series(name, DATA_DIR, start=START, end=END)
        for name in SERIES
    ]
    uccle = pd.concat(values, axis=1, join="inner")
    if list(uccle.columns) != list(SERIES) or uccle.isna().any().any():
        raise ValueError("The four Uccle series are not completely aligned.")
    if uccle.index.min() > pd.Timestamp(START):
        raise ValueError(f"The available record starts at {uccle.index.min()}, not {START}.")

    table_dir = OUTPUT_DIR / "tables"
    figure_dir = OUTPUT_DIR / "figures"
    data_path = table_dir / "uccle_extremes.csv"
    summary_path = table_dir / "uccle_integrity.csv"
    config_path = OUTPUT_DIR / "run_config.json"
    if not OVERWRITE:
        existing = [path for path in (config_path, data_path, summary_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite {existing[0]}; set BUCEX_OVERWRITE=1 to rerun."
            )
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "script": SCRIPT_NAME,
        "created_at": datetime.now().astimezone().isoformat(),
        "run_timestamp": RUN_TIMESTAMP,
        "run_signature": RUN_SIGNATURE,
        "output_directory": str(OUTPUT_DIR),
        "bucex_version": bx.__version__,
        "data": {
            "data_dir": None if DATA_DIR is None else str(DATA_DIR),
            "requested_start": START,
            "requested_end": END,
            "actual_start": str(uccle.index.min()),
            "actual_end": str(uccle.index.max()),
            "series": list(SERIES),
        },
        "loess": {
            "monthly_fraction": LOESS_FRACTION_MONTHLY,
            "annual_fraction": LOESS_FRACTION_ANNUAL,
            "robust_iterations": LOESS_ROBUST_ITERATIONS,
        },
        "figures": {"formats": list(FIGURE_FORMATS), "dpi": FIGURE_DPI},
    }
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    uccle.rename_axis("date").reset_index().to_csv(data_path, index=False)
    pd.DataFrame(
        [
            {
                "series": name,
                "description": bx.UCCLE_INFO[name]["description"],
                "tail": bx.UCCLE_INFO[name]["tail"],
                "n": int(uccle[name].size),
                "start": uccle[name].index.min(),
                "end": uccle[name].index.max(),
                "minimum": float(uccle[name].min()),
                "maximum": float(uccle[name].max()),
                "mean": float(uccle[name].mean()),
                "sd": float(uccle[name].std()),
            }
            for name in SERIES
        ]
    ).to_csv(summary_path, index=False)

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["navy"],
            "axes.labelcolor": COLORS["navy"],
            "axes.titlecolor": COLORS["navy"],
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "xtick.color": COLORS["navy"],
            "ytick.color": COLORS["navy"],
        }
    )

    # Convert dates to elapsed days only for the numerical LOESS calculation;
    # the original datetime index remains on the horizontal axis.
    x_monthly = (uccle.index - uccle.index[0]).days.to_numpy(float)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    for axis, name in zip(axes.ravel(), SERIES):
        series = uccle[name]
        smooth = bx.loess_smooth(
            x_monthly,
            series.to_numpy(float),
            fraction=LOESS_FRACTION_MONTHLY,
            robust_iterations=LOESS_ROBUST_ITERATIONS,
        )
        axis.scatter(series.index, series, s=4, alpha=0.25, color=COLORS["grey"])
        axis.plot(series.index, smooth, color=COLORS["teal"], linewidth=1.8, label="LOESS")
        axis.set_title(name)
        axis.set_ylabel("°C")
        axis.grid(axis="y", alpha=0.35)
    axes[0, 0].legend(loc="upper left")
    figure.suptitle("Uccle monthly temperature extremes, 1892 onward", color=COLORS["navy"], weight="bold")
    figure.tight_layout()
    for extension in FIGURE_FORMATS:
        figure.savefig(figure_dir / f"00_uccle_extremes.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)

    annual_txx = uccle["TXx"].resample("YE").max()
    x_annual = (annual_txx.index - annual_txx.index[0]).days.to_numpy(float)
    annual_smooth = bx.loess_smooth(
        x_annual,
        annual_txx.to_numpy(float),
        fraction=LOESS_FRACTION_ANNUAL,
        robust_iterations=LOESS_ROBUST_ITERATIONS,
    )
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.scatter(annual_txx.index, annual_txx, s=18, alpha=0.45, color=COLORS["grey"], label="annual TXx")
    axis.plot(annual_txx.index, annual_smooth, color=COLORS["teal"], linewidth=2.4, label="LOESS")
    axis.set_title("Evolution of annual maximum temperature (TXx)")
    axis.set_ylabel("°C")
    axis.grid(axis="y", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    for extension in FIGURE_FORMATS:
        figure.savefig(figure_dir / f"01_txx_evolution.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)

    # A direct comparison of the first and last 30 years makes changes in the
    # within-year pattern visible before introducing a seasonal state model.
    years = sorted(uccle.index.year.unique())
    early_years = years[:30]
    late_years = years[-30:]
    early = uccle.loc[uccle.index.year.astype(int).isin(early_years), "TXx"].groupby(lambda date: date.month).mean()
    late = uccle.loc[uccle.index.year.astype(int).isin(late_years), "TXx"].groupby(lambda date: date.month).mean()
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.plot(range(1, 13), early, marker="o", color=COLORS["grey"], label=f"{early_years[0]}–{early_years[-1]}")
    axis.plot(range(1, 13), late, marker="o", color=COLORS["teal"], label=f"{late_years[0]}–{late_years[-1]}")
    axis.set_xticks(range(1, 13), ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
    axis.set_title("TXx seasonal cycle: early and recent climates")
    axis.set_ylabel("mean monthly TXx / °C")
    axis.grid(axis="y", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    for extension in FIGURE_FORMATS:
        figure.savefig(figure_dir / f"02_txx_seasonal_cycle.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)

    print(f"Uccle outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
