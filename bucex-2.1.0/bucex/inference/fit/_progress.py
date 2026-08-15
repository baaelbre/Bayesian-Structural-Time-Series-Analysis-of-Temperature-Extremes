"""Uniform, dependency-free progress display for every MCMC kernel."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def progress_interval(total: int, requested: int | None = None) -> int:
    """Return a common reporting interval (about twenty updates per chain)."""

    if requested is not None:
        if int(requested) < 1:
            raise ValueError("A progress interval must be positive.")
        return int(requested)
    return max(1, int(total) // 20)


def should_report_progress(
    completed: int,
    *,
    total: int,
    warmup: int,
    every: int,
) -> bool:
    """Report at a common cadence and exactly at phase/final boundaries."""

    return bool(
        int(completed) % int(every) == 0
        or (int(warmup) > 0 and int(completed) == int(warmup))
        or int(completed) == int(total)
    )


def _format_scalar(name: str, value: float) -> str:
    value = float(value)
    lower = str(name).lower()
    if lower.startswith("sigma") or lower.startswith("xi"):
        return f"{value:.4f}"
    if lower.startswith("q_") or lower.startswith("q[") or lower.startswith("sd"):
        return f"{value:.3g}"
    if "ess" in lower:
        return f"{value:.1f}"
    if "change" in lower or "fraction" in lower:
        return f"{value:.2f}"
    return f"{value:.3g}"


def _format_parameter(name: str, value: Any) -> str | None:
    """Format scalar or already compact grouped current-parameter values."""

    if isinstance(value, str):
        return f"{name}={value}"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return f"{name}={_format_scalar(name, numeric)}"


def compact_group(values: Mapping[str, float], *, precision: int = 3) -> str:
    """Format named values as one stable, compact progress field."""

    pieces = []
    for name, value in values.items():
        numeric = float(value)
        if np.isfinite(numeric):
            pieces.append(f"{name}:{numeric:.{int(precision)}g}")
    return "(" + ",".join(pieces) + ")"


def univariate_progress_parameters(
    params_state: Mapping[str, Any],
    params_observation: Mapping[str, Any],
    *,
    horseshoe_state: Mapping[str, Any] | None = None,
    triple_gamma_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Current scientific parameters in the same order for Gaussian and GEV."""

    output: dict[str, Any] = {}
    if "sigma" in params_observation:
        output["sigma"] = params_observation["sigma"]
    if "xi" in params_observation:
        output["xi"] = params_observation["xi"]
    for label, key in (
        ("Q_level", "q_level"),
        ("Q_trend", "q_trend"),
        ("Q_season", "q_season"),
    ):
        if key in params_state:
            output[label] = params_state[key]
    if horseshoe_state:
        if horseshoe_state.get("global") is not None:
            output["hs_global"] = horseshoe_state["global"]
        if horseshoe_state.get("slab2") is not None:
            output["hs_slab"] = np.sqrt(float(horseshoe_state["slab2"]))
    if triple_gamma_state:
        output["tg_phi"] = triple_gamma_state.get("global")
        output["tg_a"] = triple_gamma_state.get("a")
        output["tg_c"] = triple_gamma_state.get("c")
        if triple_gamma_state.get("slab2") is not None:
            output["tg_slab"] = np.sqrt(
                float(triple_gamma_state["slab2"])
            )
    return output


def factor_progress_parameters(
    compiled: Any,
    params: Mapping[str, Any],
    *,
    horseshoe_state: Mapping[str, Any] | None = None,
    triple_gamma_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact current-parameter groups for potentially large factor models."""

    sigmas = {
        channel: params[f"sigma.{channel}"]
        for channel in compiled.channel_names
        if f"sigma.{channel}" in params
    }
    xis = {
        channel: params[f"xi.{channel}"]
        for channel in compiled.channel_names
        if f"xi.{channel}" in params
    }
    factor_q: dict[str, float] = {}
    idiosyncratic_q: dict[str, float] = {}
    seasonal_q: dict[str, float] = {}
    other_channel_q: dict[str, float] = {}
    for process in compiled.noise_names:
        key = f"sd.{process}"
        if key not in params:
            continue
        label = process.removeprefix("factor.").removeprefix("channel.")
        value = float(params[key]) ** 2
        if process.startswith("factor."):
            factor_q[label] = value
        elif process.endswith(".level"):
            idiosyncratic_q[label] = value
        elif process.endswith(".seasonal"):
            seasonal_q[label] = value
        else:
            other_channel_q[label] = value
    loadings = {
        key.removeprefix("loading."): params[key]
        for key in compiled.estimated_loading_names
        if key in params
    }
    output: dict[str, Any] = {}
    if sigmas:
        output["sigma"] = compact_group(sigmas, precision=3)
    if xis:
        output["xi"] = compact_group(xis, precision=3)
    if factor_q:
        output["Q_factor"] = compact_group(factor_q, precision=3)
    if idiosyncratic_q:
        output["Q_idio"] = compact_group(idiosyncratic_q, precision=3)
    if seasonal_q:
        output["Q_season"] = compact_group(seasonal_q, precision=3)
    if other_channel_q:
        output["Q_channel"] = compact_group(other_channel_q, precision=3)
    if loadings:
        output["loading"] = compact_group(loadings, precision=3)
    if horseshoe_state:
        if horseshoe_state.get("global") is not None:
            output["hs_global"] = horseshoe_state["global"]
        if horseshoe_state.get("slab2") is not None:
            output["hs_slab"] = np.sqrt(float(horseshoe_state["slab2"]))
    if triple_gamma_state:
        output["tg_phi"] = triple_gamma_state.get("global")
        output["tg_a"] = triple_gamma_state.get("a")
        output["tg_c"] = triple_gamma_state.get("c")
        if triple_gamma_state.get("slab2") is not None:
            output["tg_slab"] = np.sqrt(
                float(triple_gamma_state["slab2"])
            )
    return output


def mcmc_progress_line(
    *,
    label: str,
    engine: str,
    chain: int,
    chains: int,
    completed: int,
    total: int,
    warmup: int,
    saved: int,
    draws: int,
    elapsed: float,
    parameters: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    particles: int | None = None,
    details: tuple[str, ...] = (),
) -> str:
    """Format one informative, log-friendly MCMC progress line."""

    completed = int(completed)
    total = max(int(total), 1)
    warmup = max(int(warmup), 0)
    fraction = min(max(completed / total, 0.0), 1.0)
    width = 20
    filled = min(int(fraction * width), width)
    bar = "#" * filled + "-" * (width - filled)

    if warmup > 0 and completed <= warmup:
        phase = "warmup"
        phase_done = completed
        phase_total = warmup
    else:
        phase = "sampling"
        phase_done = max(completed - warmup, 0)
        phase_total = max(total - warmup, 1)

    rate = completed / max(float(elapsed), 1e-12)
    eta = (total - completed) / max(rate, 1e-12)
    pieces = [
        f"[{label} | {str(engine).upper()} | chain {chain}/{chains}]",
        f"[{bar}] {100.0 * fraction:5.1f}%",
        f"it {completed}/{total}",
        f"{phase} {phase_done}/{phase_total}",
        f"saved {saved}/{draws}",
    ]

    for name, value in ({} if parameters is None else parameters).items():
        formatted = _format_parameter(name, value)
        if formatted is not None:
            pieces.append(formatted)

    values = {} if metrics is None else metrics
    if str(engine).lower() == "pgas":
        ess = float(values.get("particle_min_ess", np.nan))
        ancestors = float(values.get("particle_mean_unique_ancestors", np.nan))
        if np.isfinite(ess):
            denominator = "" if particles is None else f"/{int(particles)}"
            pieces.append(f"particle_min_ess={ess:.1f}{denominator}")
        if np.isfinite(ancestors):
            pieces.append(f"ancestors={ancestors:.1f}")
        path_change = float(values.get("particle_changed_fraction", np.nan))
        if np.isfinite(path_change):
            pieces.append(f"path_change={path_change:.2f}")
    elif str(engine).lower() == "laplace":
        iterations = float(values.get("laplace_iterations", np.nan))
        converged = float(values.get("laplace_converged", np.nan))
        if np.isfinite(iterations):
            pieces.append(f"laplace_it={iterations:.0f}")
        if np.isfinite(converged):
            pieces.append(f"converged={int(bool(converged))}")

    pieces.extend(str(value) for value in details if str(value))
    pieces.extend((f"elapsed {_duration(elapsed)}", f"ETA {_duration(eta)}"))
    return " | ".join(pieces)


__all__ = [
    "compact_group",
    "factor_progress_parameters",
    "mcmc_progress_line",
    "progress_interval",
    "should_report_progress",
    "univariate_progress_parameters",
]
