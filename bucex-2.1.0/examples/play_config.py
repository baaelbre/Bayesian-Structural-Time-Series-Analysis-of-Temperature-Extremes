"""Shared knobs for the numbered bucex play scripts.

Environment variables let you change run length without editing a script:

    BUCEX_QUICK=0 BUCEX_DRAWS=2000 BUCEX_WARMUP=2000 \
    BUCEX_CHAINS=4 BUCEX_PARTICLES=1024 python examples/05_factor_gaussian.py
"""
from __future__ import annotations

import os

import bucex as bx


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


QUICK = _boolean("BUCEX_QUICK", True)
PROGRESS = _boolean("BUCEX_PROGRESS", True)
SHOW = _boolean("BUCEX_SHOW", True)


def _integer(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(default if value is None else value)


def mcmc(
    seed: int,
    *,
    quick_draws: int = 100,
    quick_warmup: int = 100,
    full_draws: int = 2_000,
    full_warmup: int = 2_000,
) -> bx.MCMC:
    """Return one consistent MCMC configuration for a play script."""

    draws = _integer("BUCEX_DRAWS", quick_draws if QUICK else full_draws)
    warmup = _integer(
        "BUCEX_WARMUP", quick_warmup if QUICK else full_warmup
    )
    chains = _integer("BUCEX_CHAINS", 2 if QUICK else 4)
    supplied_every = os.getenv("BUCEX_PROGRESS_EVERY")
    return bx.MCMC(
        draws=draws,
        warmup=warmup,
        chains=chains,
        seed=int(seed),
        progress=PROGRESS,
        progress_every=(None if supplied_every is None else int(supplied_every)),
    )


def particles() -> bx.Particles:
    """Return a quick or publication-scale PGAS configuration."""

    default = 96 if QUICK else 1_024
    return bx.Particles(
        n=_integer("BUCEX_PARTICLES", default),
        proposal=os.getenv("BUCEX_PARTICLE_PROPOSAL", "guided"),
    )


def describe() -> str:
    mode = "quick exploratory" if QUICK else "long"
    return (
        f"{mode} run; progress={PROGRESS}; figures={SHOW}. "
        "Environment variables in examples/play_config.py override the defaults."
    )


def show_figures() -> None:
    """Display all figures unless BUCEX_SHOW=0."""

    if SHOW:
        import matplotlib.pyplot as plt

        plt.show()


__all__ = [
    "PROGRESS",
    "QUICK",
    "SHOW",
    "describe",
    "mcmc",
    "particles",
    "show_figures",
]
