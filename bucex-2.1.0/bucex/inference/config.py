"""Inference configuration shared by every parameterization."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCMC:
    """Retained draws, warmup and chain configuration."""

    draws: int = 1000
    warmup: int = 1000
    thin: int = 1
    chains: int = 4
    seed: int | None = None
    progress: bool = False
    progress_every: int | None = None
    adapt: bool = True

    def __post_init__(self) -> None:
        if int(self.draws) < 1 or int(self.warmup) < 0 or int(self.thin) < 1:
            raise ValueError("draws and thin must be positive; warmup must be non-negative.")
        if int(self.chains) < 1:
            raise ValueError("chains must be positive.")
        if self.progress_every is not None and int(self.progress_every) < 1:
            raise ValueError("progress_every must be positive when supplied.")

    @property
    def iterations(self) -> int:
        return int(self.warmup) + 1 + (int(self.draws) - 1) * int(self.thin)

    # FS kernels use the original compact names internally.
    @property
    def n_iter(self) -> int:
        return self.iterations

    @property
    def burn(self) -> int:
        return int(self.warmup)


@dataclass(frozen=True)
class GibbsConfig:
    """Deprecated 0.3 spelling, translated once at the public API boundary."""

    n_iter: int = 2000
    burn: int = 1000
    thin: int = 1
    seed: int | None = None
    progress: bool = False
    progress_every: int | None = None

    def __post_init__(self) -> None:
        if int(self.n_iter) <= int(self.burn) or int(self.burn) < 0:
            raise ValueError("Require n_iter > burn >= 0.")
        if int(self.thin) < 1:
            raise ValueError("thin must be positive.")

    def to_mcmc(self, *, chains: int = 1) -> MCMC:
        return MCMC(
            draws=len(range(int(self.burn), int(self.n_iter), int(self.thin))),
            warmup=int(self.burn),
            thin=int(self.thin),
            chains=int(chains),
            seed=self.seed,
            progress=bool(self.progress),
            progress_every=self.progress_every,
        )


@dataclass(frozen=True)
class Laplace:
    max_iterations: int = 30
    tolerance: float = 1e-5
    curvature_floor: float = 1e-6
    maximum_variance: float = 1e8
    draw_attempts: int = 30

    def __post_init__(self) -> None:
        if int(self.max_iterations) < 1 or float(self.tolerance) <= 0.0:
            raise ValueError("Laplace iterations and tolerance must be positive.")


@dataclass(frozen=True)
class Particles:
    n: int = 256
    ess_threshold: float = 0.5
    resampling: str = "systematic"
    proposal: str = "guided"

    def __post_init__(self) -> None:
        if int(self.n) < 2:
            raise ValueError("Particles.n must be at least 2.")
        if not 0.0 < float(self.ess_threshold) <= 1.0:
            raise ValueError("ess_threshold must lie in (0, 1].")
        if self.resampling not in {"systematic", "multinomial"}:
            raise ValueError("resampling must be systematic or multinomial.")
        if self.proposal not in {"bootstrap", "guided"}:
            raise ValueError("proposal must be bootstrap or guided.")
