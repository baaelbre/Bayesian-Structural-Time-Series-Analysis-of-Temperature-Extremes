from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

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

    @property
    def componentwise(self) -> bool:
        return False

    def coefficient_scale_for(self, component: str) -> float:
        return 1.0


@dataclass(frozen=True)
class ComponentwiseBayesianLassoPrior:
    """Component-specific Bayesian-lasso prior for signed innovation scales.

    For component ``k`` in ``{level, trend, season}`` the hierarchy is

    ``s_k | tau_k ~ N(0, variance_scale * c_k**2 * tau_k)``

    ``tau_k | lambda2_k ~ Exp(lambda2_k / 2)``

    ``lambda2_k ~ Gamma(a_k, b_k)``  (shape-rate).

    The coefficient scales ``c_k`` put the three structural innovations on
    interpretable, component-specific scales. This is important because a
    monthly slope innovation is typically orders of magnitude smaller than a
    level or seasonal innovation.
    """

    a_lambda: Mapping[str, float] = field(
        default_factory=lambda: {"level": 2.0, "trend": 2.0, "season": 2.0}
    )
    b_lambda: Mapping[str, float] = field(
        default_factory=lambda: {"level": 1.0, "trend": 1.0, "season": 1.0}
    )
    initial_tau: Mapping[str, float] = field(
        default_factory=lambda: {"level": 1.0, "trend": 1.0, "season": 1.0}
    )
    initial_lambda2: Mapping[str, float] = field(
        default_factory=lambda: {"level": 1.0, "trend": 1.0, "season": 1.0}
    )
    coefficient_scale: Mapping[str, float] = field(
        default_factory=lambda: {"level": 0.03, "trend": 0.0002, "season": 0.03}
    )
    variance_mode: str = "fixed"
    fixed_variance: float = 1.0

    def __post_init__(self) -> None:
        required = {"level", "trend", "season"}
        for name, values in (
            ("a_lambda", self.a_lambda),
            ("b_lambda", self.b_lambda),
            ("initial_tau", self.initial_tau),
            ("initial_lambda2", self.initial_lambda2),
            ("coefficient_scale", self.coefficient_scale),
        ):
            missing = required - set(values)
            if missing:
                raise ValueError(f"{name} is missing components: {sorted(missing)}")
            if any(float(values[key]) <= 0.0 for key in required):
                raise ValueError(f"All {name} values must be positive.")
        if self.variance_mode not in {"fixed", "observation"}:
            raise ValueError("variance_mode must be 'fixed' or 'observation'.")
        if self.fixed_variance <= 0.0:
            raise ValueError("fixed_variance must be > 0.")

    @property
    def componentwise(self) -> bool:
        return True

    def variance_scale(self, sigma2: float | None = None) -> float:
        if self.variance_mode == "observation":
            if sigma2 is None or sigma2 <= 0.0:
                raise ValueError("A positive sigma2 is required for variance_mode='observation'.")
            return float(sigma2)
        return float(self.fixed_variance)

    def coefficient_scale_for(self, component: str) -> float:
        return float(self.coefficient_scale[component])

    def a_for(self, component: str) -> float:
        return float(self.a_lambda[component])

    def b_for(self, component: str) -> float:
        return float(self.b_lambda[component])

    def initial_tau_for(self, component: str) -> float:
        return float(self.initial_tau[component])

    def initial_lambda2_for(self, component: str) -> float:
        return float(self.initial_lambda2[component])


@dataclass(frozen=True)
class SSVSPrior:
    """Exact structural spike-and-slab prior for non-centred models.

    The package enumerates the complete structural model space. The level is
    always present and is either fixed (``s_level = 0``) or dynamic. Trend and
    seasonality can be zero, fixed, or dynamic. Active signed innovation scales
    receive Gaussian slab priors; inactive coefficients are exactly zero.

    ``trend_probabilities`` and ``season_probabilities`` are ordered as
    ``(zero, fixed, dynamic)``.
    """

    innovation_slab_sd: Mapping[str, float] = field(
        default_factory=lambda: {
            "level": 0.03,
            "trend": 0.0002,
            "season": 0.03,
        }
    )
    level_dynamic_probability: float = 0.5
    trend_probabilities: Sequence[float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    season_probabilities: Sequence[float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    def __post_init__(self) -> None:
        required = {"level", "trend", "season"}
        supplied = set(self.innovation_slab_sd)
        missing = required - supplied
        if missing:
            raise ValueError(f"Missing innovation slab scales for: {sorted(missing)}")
        if any(float(self.innovation_slab_sd[key]) <= 0.0 for key in required):
            raise ValueError("All innovation_slab_sd values must be positive.")
        if not 0.0 <= float(self.level_dynamic_probability) <= 1.0:
            raise ValueError("level_dynamic_probability must lie in [0, 1].")
        for name, values in (
            ("trend_probabilities", self.trend_probabilities),
            ("season_probabilities", self.season_probabilities),
        ):
            values = np.asarray(values, dtype=float)
            if values.shape != (3,):
                raise ValueError(f"{name} must contain (zero, fixed, dynamic).")
            if np.any(values < 0.0) or not np.isclose(values.sum(), 1.0):
                raise ValueError(f"{name} entries must be non-negative and sum to one.")

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
    lasso: Optional[Union[BayesianLassoPrior, ComponentwiseBayesianLassoPrior]] = None
    ssvs: Optional[SSVSPrior] = None

    def __post_init__(self) -> None:
        strategies = int(self.lasso is not None) + int(self.ssvs is not None) + int(self.s_level is not None)
        if strategies != 1:
            raise ValueError(
                "Choose exactly one innovation prior: normal, Bayesian lasso, or SSVS."
            )


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
    lasso: Optional[Union[BayesianLassoPrior, ComponentwiseBayesianLassoPrior]] = None
    ssvs: Optional[SSVSPrior] = None
    xi_max_abs: float = 0.5

    def __post_init__(self) -> None:
        if self.sigma2 is None and self.log_sigma is None:
            raise ValueError("Provide sigma2=InverseGammaPrior(...) or log_sigma=NormalPrior(...).")
        if self.xi_max_abs <= 0.0:
            raise ValueError("NonCenteredGEVPriors.xi_max_abs must be > 0.")
        strategies = int(self.lasso is not None) + int(self.ssvs is not None) + int(self.s_level is not None)
        if strategies != 1:
            raise ValueError(
                "Choose exactly one innovation prior: normal, Bayesian lasso, or SSVS."
            )


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


# ---------------------------------------------------------------------------
# v0.3.2 normal and structural SSVS profiles
# ---------------------------------------------------------------------------

def normal_gaussian_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    level_sd: float = 0.03,
    trend_sd: float = 0.0002,
    season_sd: float = 0.03,
) -> NonCenteredGaussianPriors:
    """Scale-aware Gaussian priors for monthly structural temperature models."""

    k = period - 1
    return NonCenteredGaussianPriors(
        sigma2=InverseGammaPrior(a=2.0, b=1.0),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        s_level=NormalPrior(0.0, level_sd),
        s_trend=NormalPrior(0.0, trend_sd),
        s_season=NormalPrior(0.0, season_sd),
    )


def normal_gev_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    level_sd: float = 0.03,
    trend_sd: float = 0.0002,
    season_sd: float = 0.03,
) -> NonCenteredGEVPriors:
    """Scale-aware DGEV priors for monthly structural temperature models."""

    k = period - 1
    return NonCenteredGEVPriors(
        sigma2=InverseGammaPrior(a=2.0, b=2.0),
        xi=UniformPrior(-0.5, 0.5),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        s_level=NormalPrior(0.0, level_sd),
        s_trend=NormalPrior(0.0, trend_sd),
        s_season=NormalPrior(0.0, season_sd),
    )


def ssvs_gaussian_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    ssvs: Optional[SSVSPrior] = None,
) -> NonCenteredGaussianPriors:
    """Gaussian structural SSVS prior with exact zero/fixed/dynamic states."""

    k = period - 1
    return NonCenteredGaussianPriors(
        sigma2=InverseGammaPrior(a=2.0, b=1.0),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        ssvs=SSVSPrior() if ssvs is None else ssvs,
    )


def ssvs_gev_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    ssvs: Optional[SSVSPrior] = None,
) -> NonCenteredGEVPriors:
    """DGEV structural SSVS prior using Laplace pseudo-observations."""

    k = period - 1
    return NonCenteredGEVPriors(
        sigma2=InverseGammaPrior(a=2.0, b=2.0),
        xi=UniformPrior(-0.5, 0.5),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        ssvs=SSVSPrior() if ssvs is None else ssvs,
    )


# ---------------------------------------------------------------------------
# v0.3.2 component-wise regularized lasso profiles
# ---------------------------------------------------------------------------

def regularized_gaussian_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    coefficient_scale: Optional[Mapping[str, float]] = None,
) -> NonCenteredGaussianPriors:
    """Component-wise lasso calibrated for monthly temperature series.

    Unlike the manuscript profile, the level, slope and seasonal innovation
    scales have separate shrinkage parameters and interpretable coefficient
    scales. The Gaussian observation variance is not used to rescale these
    structural priors.
    """

    k = period - 1
    scales = (
        {"level": 0.03, "trend": 0.0002, "season": 0.03}
        if coefficient_scale is None
        else dict(coefficient_scale)
    )
    return NonCenteredGaussianPriors(
        sigma2=InverseGammaPrior(a=2.0, b=1.0),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        lasso=ComponentwiseBayesianLassoPrior(
            coefficient_scale=scales,
            variance_mode="fixed",
            fixed_variance=1.0,
        ),
    )


def regularized_gev_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    coefficient_scale: Optional[Mapping[str, float]] = None,
) -> NonCenteredGEVPriors:
    """Component-wise scale-aware lasso for monthly DGEV models."""

    k = period - 1
    scales = (
        {"level": 0.03, "trend": 0.0002, "season": 0.03}
        if coefficient_scale is None
        else dict(coefficient_scale)
    )
    return NonCenteredGEVPriors(
        sigma2=InverseGammaPrior(a=2.0, b=2.0),
        xi=UniformPrior(-0.5, 0.5),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        lasso=ComponentwiseBayesianLassoPrior(
            coefficient_scale=scales,
            variance_mode="fixed",
            fixed_variance=1.0,
        ),
    )
