"""Small dependency-free progress display for MCMC kernels."""
from __future__ import annotations

from collections.abc import Mapping

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
    metrics: Mapping[str, float] | None = None,
    particles: int | None = None,
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
        f"{phase} {phase_done}/{phase_total}",
        f"saved {saved}/{draws}",
    ]

    values = {} if metrics is None else metrics
    if str(engine).lower() == "pgas":
        ess = float(values.get("particle_min_ess", np.nan))
        ancestors = float(values.get("particle_mean_unique_ancestors", np.nan))
        if np.isfinite(ess):
            denominator = "" if particles is None else f"/{int(particles)}"
            pieces.append(f"ESSmin {ess:.1f}{denominator}")
        if np.isfinite(ancestors):
            pieces.append(f"ancestors {ancestors:.1f}")
    elif str(engine).lower() == "laplace":
        iterations = float(values.get("laplace_iterations", np.nan))
        converged = float(values.get("laplace_converged", np.nan))
        if np.isfinite(iterations):
            pieces.append(f"Laplace iters {iterations:.0f}")
        if np.isfinite(converged):
            pieces.append(f"converged {bool(converged)}")

    pieces.extend((f"elapsed {_duration(elapsed)}", f"ETA {_duration(eta)}"))
    return " | ".join(pieces)


__all__ = ["mcmc_progress_line"]
