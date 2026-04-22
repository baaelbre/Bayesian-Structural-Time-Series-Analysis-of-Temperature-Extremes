from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class InverseGammaPrior:
    """
    Inverse-gamma prior IG(a, b) with density proportional to

        x^(-a-1) exp(-b / x),   x > 0
    """
    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0.0:
            raise ValueError("InverseGammaPrior.a must be > 0.")
        if self.b <= 0.0:
            raise ValueError("InverseGammaPrior.b must be > 0.")


@dataclass(frozen=True)
class NormalPrior:
    """
    Scalar Gaussian prior N(mean, sd^2).
    """
    mean: float
    sd: float

    def __post_init__(self) -> None:
        if self.sd <= 0.0:
            raise ValueError("NormalPrior.sd must be > 0.")


@dataclass(frozen=True)
class DiagonalNormalPrior:
    """
    Independent Gaussian prior for a vector.
    """
    mean: Sequence[float]
    sd: Sequence[float]

    def mean_array(self) -> np.ndarray:
        return np.asarray(self.mean, dtype=float)

    def sd_array(self) -> np.ndarray:
        return np.asarray(self.sd, dtype=float)

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        sd = np.asarray(self.sd, dtype=float)
        if mean.ndim != 1 or sd.ndim != 1:
            raise ValueError("DiagonalNormalPrior.mean and .sd must be 1D.")
        if mean.shape != sd.shape:
            raise ValueError("DiagonalNormalPrior.mean and .sd must have the same shape.")
        if np.any(sd <= 0.0):
            raise ValueError("All entries of DiagonalNormalPrior.sd must be > 0.")


@dataclass(frozen=True)
class InitialStatePriors:
    """
    Priors for initial-state hyperparameters.

    These are priors on the parameters used by model.initial_state(params_state).
    """
    m0_level: Optional[NormalPrior] = None
    v0_level: Optional[InverseGammaPrior] = None

    m0_trend: Optional[NormalPrior] = None
    v0_trend: Optional[InverseGammaPrior] = None

    m0_season: Optional[DiagonalNormalPrior] = None
    v0_season: Optional[InverseGammaPrior] = None


@dataclass(frozen=True)
class CenteredGaussianPriors:
    sigma2: InverseGammaPrior
    q_level: Optional[InverseGammaPrior] = None
    q_trend: Optional[InverseGammaPrior] = None
    q_season: Optional[InverseGammaPrior] = None
    initial: InitialStatePriors = field(default_factory=InitialStatePriors)

    def active_q_priors(self) -> dict[str, InverseGammaPrior]:
        out: dict[str, InverseGammaPrior] = {}
        if self.q_level is not None:
            out["q_level"] = self.q_level
        if self.q_trend is not None:
            out["q_trend"] = self.q_trend
        if self.q_season is not None:
            out["q_season"] = self.q_season
        return out


@dataclass(frozen=True)
class CenteredGEVPriors:
    log_sigma: NormalPrior
    xi: NormalPrior
    xi_max_abs: float = 0.45

    q_level: Optional[InverseGammaPrior] = None
    q_trend: Optional[InverseGammaPrior] = None
    q_season: Optional[InverseGammaPrior] = None
    initial: InitialStatePriors = field(default_factory=InitialStatePriors)

    def __post_init__(self) -> None:
        if self.xi_max_abs <= 0.0:
            raise ValueError("CenteredGEVPriors.xi_max_abs must be > 0.")

    def active_q_priors(self) -> dict[str, InverseGammaPrior]:
        out: dict[str, InverseGammaPrior] = {}
        if self.q_level is not None:
            out["q_level"] = self.q_level
        if self.q_trend is not None:
            out["q_trend"] = self.q_trend
        if self.q_season is not None:
            out["q_season"] = self.q_season
        return out


@dataclass(frozen=True)
class NonCenteredGaussianPriors:
    sigma2: InverseGammaPrior
    alpha0: NormalPrior
    beta0: NormalPrior
    s_level: NormalPrior
    gamma0_season: Optional[DiagonalNormalPrior] = None
    s_trend: Optional[NormalPrior] = None
    s_season: Optional[NormalPrior] = None


@dataclass(frozen=True)
class NonCenteredGEVPriors:
    log_sigma: NormalPrior
    xi: NormalPrior
    alpha0: NormalPrior
    beta0: NormalPrior
    s_level: NormalPrior
    gamma0_season: Optional[DiagonalNormalPrior] = None
    s_trend: Optional[NormalPrior] = None
    s_season: Optional[NormalPrior] = None
    xi_max_abs: float = 0.45

    def __post_init__(self) -> None:
        if self.xi_max_abs <= 0.0:
            raise ValueError("NonCenteredGEVPriors.xi_max_abs must be > 0.")
