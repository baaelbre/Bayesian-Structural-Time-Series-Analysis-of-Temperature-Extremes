from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True)
class InverseGammaPrior:
    """Inverse-gamma prior ``IG(a, b)``.

    The density is proportional to ``x**(-a-1) * exp(-b/x)`` for ``x > 0``.
    ``b`` is therefore the inverse-scale/rate-like parameter used throughout the
    manuscript code.
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0.0:
            raise ValueError("InverseGammaPrior.a must be > 0.")
        if self.b <= 0.0:
            raise ValueError("InverseGammaPrior.b must be > 0.")


@dataclass(frozen=True)
class GammaPrior:
    """Gamma prior in shape-rate form."""

    shape: float
    rate: float

    def __post_init__(self) -> None:
        if self.shape <= 0.0:
            raise ValueError("GammaPrior.shape must be > 0.")
        if self.rate <= 0.0:
            raise ValueError("GammaPrior.rate must be > 0.")


@dataclass(frozen=True)
class UniformPrior:
    """Continuous uniform prior on ``[lower, upper]``."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError("UniformPrior requires lower < upper.")


@dataclass(frozen=True)
class NormalPrior:
    """Scalar Gaussian prior ``N(mean, sd**2)``."""

    mean: float
    sd: float

    def __post_init__(self) -> None:
        if self.sd <= 0.0:
            raise ValueError("NormalPrior.sd must be > 0.")


@dataclass(frozen=True)
class DiagonalNormalPrior:
    """Independent Gaussian prior for a vector."""

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
class BayesianLassoPrior:
    """Hierarchical Bayesian-lasso prior for signed innovation scales.

    For each active structural block ``k`` the non-centred sampler uses

    ``s_k | tau_k ~ N(0, variance_scale * tau_k)``

    ``tau_k | lambda2 ~ Exp(lambda2 / 2)``

    ``lambda2 ~ Gamma(a_lambda, b_lambda)``  (shape-rate).

    ``variance_mode='observation'`` reproduces the Gaussian manuscript code,
    where ``variance_scale = sigma**2``. ``variance_mode='fixed'`` is used by
    the DGEV code, where the pseudo-Gaussian regression has no natural residual
    variance and the manuscript implementation used ``fixed_variance=1``.
    """

    a_lambda: float = 1.0
    b_lambda: float = 1.0
    initial_tau: float = 1.0
    initial_lambda2: float = 1.0
    variance_mode: str = "fixed"
    fixed_variance: float = 1.0

    def __post_init__(self) -> None:
        if self.a_lambda <= 0.0 or self.b_lambda <= 0.0:
            raise ValueError("BayesianLassoPrior Gamma hyperparameters must be > 0.")
        if self.initial_tau <= 0.0 or self.initial_lambda2 <= 0.0:
            raise ValueError("BayesianLassoPrior initial values must be > 0.")
        if self.variance_mode not in {"fixed", "observation"}:
            raise ValueError("variance_mode must be 'fixed' or 'observation'.")
        if self.fixed_variance <= 0.0:
            raise ValueError("fixed_variance must be > 0.")

    def variance_scale(self, sigma2: float | None = None) -> float:
        if self.variance_mode == "observation":
            if sigma2 is None or sigma2 <= 0.0:
                raise ValueError("A positive sigma2 is required for variance_mode='observation'.")
            return float(sigma2)
        return float(self.fixed_variance)


@dataclass(frozen=True)
class InitialStatePriors:
    """Priors for centred initial-state hyperparameters."""

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
        return {
            key: value
            for key, value in {
                "q_level": self.q_level,
                "q_trend": self.q_trend,
                "q_season": self.q_season,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class CenteredGEVPriors:
    # Kept for backwards compatibility with v0.1 centred fits.
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
        return {
            key: value
            for key, value in {
                "q_level": self.q_level,
                "q_trend": self.q_trend,
                "q_season": self.q_season,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class NonCenteredGaussianPriors:
    """Priors for the non-centred structural Gaussian model."""

    sigma2: InverseGammaPrior
    alpha0: NormalPrior
    beta0: NormalPrior
    s_level: Optional[NormalPrior] = None
    gamma0_season: Optional[DiagonalNormalPrior] = None
    s_trend: Optional[NormalPrior] = None
    s_season: Optional[NormalPrior] = None
    lasso: Optional[BayesianLassoPrior] = None

    def __post_init__(self) -> None:
        if self.lasso is None and self.s_level is None:
            raise ValueError("Provide either a Bayesian lasso prior or s_level NormalPrior.")


XiPrior = Union[NormalPrior, UniformPrior]


@dataclass(frozen=True)
class NonCenteredGEVPriors:
    """Priors for the non-centred structural DGEV model.

    v0.2 uses the manuscript prior by default: ``sigma**2 ~ IG(a,b)`` and
    ``xi ~ Uniform(lower, upper)``. The old v0.1 log-normal/normal formulation
    remains accepted for backwards compatibility.
    """

    alpha0: NormalPrior
    beta0: NormalPrior
    xi: XiPrior
    sigma2: Optional[InverseGammaPrior] = None
    log_sigma: Optional[NormalPrior] = None
    s_level: Optional[NormalPrior] = None
    gamma0_season: Optional[DiagonalNormalPrior] = None
    s_trend: Optional[NormalPrior] = None
    s_season: Optional[NormalPrior] = None
    lasso: Optional[BayesianLassoPrior] = None
    xi_max_abs: float = 0.5

    def __post_init__(self) -> None:
        if self.sigma2 is None and self.log_sigma is None:
            raise ValueError("Provide sigma2=InverseGammaPrior(...) or log_sigma=NormalPrior(...).")
        if self.xi_max_abs <= 0.0:
            raise ValueError("NonCenteredGEVPriors.xi_max_abs must be > 0.")
        if self.lasso is None and self.s_level is None:
            raise ValueError("Provide either a Bayesian lasso prior or s_level NormalPrior.")


# ---------------------------------------------------------------------------
# Manuscript profiles
# ---------------------------------------------------------------------------

def manuscript_gaussian_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
) -> NonCenteredGaussianPriors:
    """Return the prior profile used by the Uccle Gaussian analysis."""

    k = period - 1
    return NonCenteredGaussianPriors(
        sigma2=InverseGammaPrior(a=2.0, b=1.0),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, np.sqrt(10.0)),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k),
            sd=np.full(k, np.sqrt(5.0)),
        ),
        lasso=BayesianLassoPrior(
            a_lambda=1.0,
            b_lambda=1.0,
            variance_mode="observation",
        ),
    )


def manuscript_gev_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
) -> NonCenteredGEVPriors:
    """Return the prior profile used by the Uccle DGEV analysis."""

    k = period - 1
    return NonCenteredGEVPriors(
        sigma2=InverseGammaPrior(a=2.0, b=2.0),
        xi=UniformPrior(-0.5, 0.5),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, np.sqrt(10.0)),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k),
            sd=np.full(k, np.sqrt(5.0)),
        ),
        lasso=BayesianLassoPrior(
            a_lambda=1.0,
            b_lambda=1.0,
            variance_mode="fixed",
            fixed_variance=1.0,
        ),
    )
