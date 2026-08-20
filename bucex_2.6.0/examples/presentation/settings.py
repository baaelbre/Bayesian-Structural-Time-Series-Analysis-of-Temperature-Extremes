"""Shared, environment-configurable settings for all presentation scripts."""
from __future__ import annotations

import os
from pathlib import Path

import bucex as bx


def _optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value in {None, ""} else int(value)


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


PROFILE = os.environ.get("BUCEX_PROFILE", "pilot")
OUTPUT_DIR = Path(os.environ.get("BUCEX_OUTPUT_DIR", "results/presentation"))
DATA_DIR = os.environ.get("BUCEX_DATA_DIR")
OVERWRITE = _flag("BUCEX_OVERWRITE")
DIAGNOSTICS = _flag("BUCEX_DIAGNOSTICS")
SCENARIO = os.environ.get("BUCEX_SCENARIO") or None
SERIES = os.environ.get("BUCEX_SERIES") or None
FORMATS = tuple(
    item.strip()
    for item in os.environ.get("BUCEX_FORMATS", "pdf,png").split(",")
    if item.strip()
)


def workflow() -> bx.PresentationWorkflow:
    config = bx.PresentationConfig.for_profile(
        PROFILE,
        output_dir=OUTPUT_DIR,
        data_dir=DATA_DIR,
        seed=int(os.environ.get("BUCEX_SEED", "26000")),
        progress=_flag("BUCEX_PROGRESS", True),
        draws=_optional_int("BUCEX_DRAWS"),
        warmup=_optional_int("BUCEX_WARMUP"),
        chains=_optional_int("BUCEX_CHAINS"),
        particles=_optional_int("BUCEX_PARTICLES"),
        simulation_months=_optional_int("BUCEX_SIMULATION_MONTHS"),
        figure_formats=FORMATS,
        figure_dpi=int(os.environ.get("BUCEX_FIGURE_DPI", "180")),
    )
    return bx.PresentationWorkflow(config)

