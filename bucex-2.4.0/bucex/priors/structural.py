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

    def logpdf(self, value: float) -> float:
        if self.lower <= float(value) <= self.upper:
            return float(-np.log(self.upper - self.lower))
        return -np.inf

    def sample(self, rng, size=None):
        return rng.uniform(self.lower, self.upper, size=size)

    def initial(self) -> float:
        return 0.5 * (float(self.lower) + float(self.upper))

    def to_dict(self) -> dict[str, float | str]:
        return {
            "type": "uniform",
            "lower": float(self.lower),
            "upper": float(self.upper),
        }


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
class RegularizedHorseshoePrior:
    """Regularized horseshoe prior for innovation-scale coefficients.

    The structural innovation standard deviations are sampled as *signed*
    coefficients in the Fruehwirth--Schnatter regression.  Components have
    different physical units, so the prior is placed on standardized
    coefficients ``u_k = s_k / coefficient_scale[k]``:

    ``u_k | lambda_k, tau, c ~ N(0, tau**2 * lambda_tilde_k**2)``

    ``lambda_tilde_k**2 = c**2 * lambda_k**2 /
                          (c**2 + tau**2 * lambda_k**2)``

    ``lambda_k ~ half-Cauchy(0, 1)`` and
    ``tau ~ half-Cauchy(0, global_scale)``.  The regularizing slab uses
    ``c**2 ~ InvGamma(slab_df / 2, slab_df * slab_scale**2 / 2)``.

    Conditional on ``lambda_k``, ``tau`` and ``c`` the coefficients retain a
    Gaussian prior.  The Frühwirth--Schnatter sampler uses signed coefficients;
    centered and disturbance samplers use the corresponding half-Normal law
    for positive standard deviations.  The normalization difference is
    constant in the hierarchy, so both representations share the same latent
    local, global and slab scales.
    """

    coefficient_scale: Mapping[str, float] = field(
        default_factory=lambda: {
            "level": 0.03,
            "trend": 0.0002,
            "season": 0.03,
        }
    )
    global_scale: float = 0.25
    slab_scale: float = 2.0
    slab_df: float = 4.0
    initial_local: float = 1.0
    initial_global: Optional[float] = None
    initial_slab2: Optional[float] = None

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.coefficient_scale)
        if not names:
            raise ValueError("coefficient_scale must contain at least one component.")
        if any(float(self.coefficient_scale[key]) <= 0.0 for key in names):
            raise ValueError("All regularized-horseshoe coefficient scales must be positive.")
        for name, value in (
            ("global_scale", self.global_scale),
            ("slab_scale", self.slab_scale),
            ("slab_df", self.slab_df),
            ("initial_local", self.initial_local),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"RegularizedHorseshoePrior.{name} must be positive.")
        if self.initial_global is not None and float(self.initial_global) <= 0.0:
            raise ValueError("initial_global must be positive when supplied.")
        if self.initial_slab2 is not None and float(self.initial_slab2) <= 0.0:
            raise ValueError("initial_slab2 must be positive when supplied.")

    def coefficient_scale_for(self, component: str) -> float:
        return float(self.coefficient_scale[component])

    def initial_global_value(self) -> float:
        return float(
            self.global_scale if self.initial_global is None else self.initial_global
        )

    def initial_slab2_value(self) -> float:
        return float(
            self.slab_scale**2
            if self.initial_slab2 is None
            else self.initial_slab2
        )

    def conditional_variance(
        self,
        component: str,
        *,
        local: float,
        global_scale: float,
        slab2: float,
    ) -> float:
        local2 = max(float(local) ** 2, 1e-24)
        global2 = max(float(global_scale) ** 2, 1e-24)
        slab2 = max(float(slab2), 1e-24)
        regularized_local2 = slab2 * local2 / (slab2 + global2 * local2)
        coefficient_scale = self.coefficient_scale_for(component)
        return float(coefficient_scale**2 * global2 * regularized_local2)


@dataclass(frozen=True)
class TripleGammaPrior:
    """Triple-gamma prior for signed innovation-scale coefficients.

    This is the normal-gamma-gamma representation in Cadonna,
    Frühwirth-Schnatter and Knaus (2020).  For the standardized signed scale
    ``u_k = s_k / coefficient_scale[k]`` it uses

    ``u_k | r_k, d_k, phi ~ N(0, phi * r_k / d_k)``,

    ``r_k ~ Gamma(a, 1)`` and ``d_k ~ Gamma(c, 1)``.

    Consequently ``r_k / d_k`` is beta-prime distributed.  When
    ``learn_global=True``, the paper's model-size calibration is used,
    ``phi ~ BetaPrime(c, a)``.  The local shrinkage factor is
    ``rho_k = 1 / (1 + phi * r_k / d_k)``: values near one suppress a process
    innovation, while values near zero leave it dynamic.

    ``a=c=1/2`` gives the horseshoe member of the family.  The Bayesian lasso,
    double gamma, normal-gamma, folded-t/half-t and Gaussian limits follow from
    the limiting cases described in the paper.  Those identities concern the
    *unregularized* hierarchy; setting ``regularized=True`` adds the optional
    finite-variance slab

    ``v_reg = slab2 * v / (slab2 + v)``, ``v = phi * r_k / d_k``.

    Shape learning is deliberately optional.  With only three structural
    innovations, fixed scientifically chosen ``a`` and ``c`` are usually more
    stable.  If enabled, ``2a`` and ``2c`` receive independent beta priors and
    are therefore restricted to ``(0, 1)``, exactly as in the paper.
    """

    coefficient_scale: Mapping[str, float] = field(
        default_factory=lambda: {
            "level": 0.03,
            "trend": 0.0002,
            "season": 0.03,
        }
    )
    spike_shape: float = 0.10
    tail_shape: float = 0.10
    global_scale: float = 1.0
    learn_global: bool = True
    learn_shapes: bool = False
    spike_shape_prior: Sequence[float] = (6.0, 6.0)
    tail_shape_prior: Sequence[float] = (6.0, 6.0)
    regularized: bool = False
    slab_scale: float = 2.0
    slab_df: float = 4.0
    initial_numerator: float = 1.0
    initial_denominator: float = 1.0
    initial_slab2: Optional[float] = None

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.coefficient_scale)
        if not names:
            raise ValueError("coefficient_scale must contain at least one component.")
        if any(float(self.coefficient_scale[key]) <= 0.0 for key in names):
            raise ValueError("All triple-gamma coefficient scales must be positive.")
        for name, value in (
            ("spike_shape", self.spike_shape),
            ("tail_shape", self.tail_shape),
            ("global_scale", self.global_scale),
            ("slab_scale", self.slab_scale),
            ("slab_df", self.slab_df),
            ("initial_numerator", self.initial_numerator),
            ("initial_denominator", self.initial_denominator),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"TripleGammaPrior.{name} must be positive.")
        if self.learn_shapes and (
            float(self.spike_shape) >= 0.5 or float(self.tail_shape) >= 0.5
        ):
            raise ValueError(
                "learn_shapes=True requires initial spike_shape and tail_shape in (0, 0.5)."
            )
        for name, values in (
            ("spike_shape_prior", self.spike_shape_prior),
            ("tail_shape_prior", self.tail_shape_prior),
        ):
            values = np.asarray(values, dtype=float)
            if values.shape != (2,) or np.any(values <= 0.0):
                raise ValueError(f"{name} must contain two positive beta shapes.")
        if self.initial_slab2 is not None and float(self.initial_slab2) <= 0.0:
            raise ValueError("initial_slab2 must be positive when supplied.")

    @property
    def a(self) -> float:
        return float(self.spike_shape)

    @property
    def c(self) -> float:
        return float(self.tail_shape)

    def coefficient_scale_for(self, component: str) -> float:
        return float(self.coefficient_scale[component])

    def initial_slab2_value(self) -> float:
        return float(
            self.slab_scale**2
            if self.initial_slab2 is None
            else self.initial_slab2
        )

    def standardized_variance(
        self,
        *,
        numerator: float,
        denominator: float,
        global_scale: float,
        slab2: Optional[float] = None,
    ) -> float:
        variance = (
            max(float(global_scale), 1e-24)
            * max(float(numerator), 1e-24)
            / max(float(denominator), 1e-24)
        )
        if self.regularized:
            if slab2 is None:
                raise ValueError("A slab2 value is required by regularized triple gamma.")
            slab2 = max(float(slab2), 1e-24)
            variance = slab2 * variance / (slab2 + variance)
        return float(variance)

    def conditional_variance(
        self,
        component: str,
        *,
        numerator: float,
        denominator: float,
        global_scale: float,
        slab2: Optional[float] = None,
    ) -> float:
        scale = self.coefficient_scale_for(component)
        return float(
            scale**2
            * self.standardized_variance(
                numerator=numerator,
                denominator=denominator,
                global_scale=global_scale,
                slab2=slab2,
            )
        )

    def shrinkage_factor(
        self,
        *,
        numerator: float,
        denominator: float,
        global_scale: float,
    ) -> float:
        base_variance = (
            max(float(global_scale), 1e-24)
            * max(float(numerator), 1e-24)
            / max(float(denominator), 1e-24)
        )
        return float(1.0 / (1.0 + base_variance))


@dataclass(frozen=True)
class PCInnovationPrior:
    """PC prior on structural innovation standard deviations.

    For each signed FS scale ``s_k``, ``P(|s_k| > upper_k) = alpha_k``.
    This is the symmetric exponential density with rate
    ``-log(alpha_k) / upper_k``.  The sampler uses its exact normal--exponential
    mixture, keeping the Gaussian FS update conditionally conjugate.
    """

    upper: Mapping[str, float] = field(
        default_factory=lambda: {
            "level": 0.03,
            "trend": 0.0002,
            "season": 0.03,
        }
    )
    alpha: Union[float, Mapping[str, float]] = 0.05
    initial_tau: float = 1.0

    def __post_init__(self) -> None:
        required = {"level", "trend", "season"}
        missing = required - set(self.upper)
        if missing:
            raise ValueError(f"PCInnovationPrior.upper is missing: {sorted(missing)}")
        if any(float(self.upper[key]) <= 0.0 for key in required):
            raise ValueError("All PC prior upper scales must be positive.")
        values = (
            [float(self.alpha)]
            if not isinstance(self.alpha, Mapping)
            else [float(self.alpha[key]) for key in required]
        )
        if any(not 0.0 < value < 1.0 for value in values):
            raise ValueError("PC prior alpha values must lie in (0, 1).")
        if float(self.initial_tau) <= 0.0:
            raise ValueError("PCInnovationPrior.initial_tau must be positive.")

    def alpha_for(self, component: str) -> float:
        if isinstance(self.alpha, Mapping):
            return float(self.alpha[component])
        return float(self.alpha)

    def coefficient_scale_for(self, component: str) -> float:
        return float(self.upper[component])

    def standardized_rate_for(self, component: str) -> float:
        return float(-np.log(self.alpha_for(component)))


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
    trend_model_probabilities: Mapping[str, float] | None = None

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
        if self.trend_model_probabilities is not None:
            required_models = {
                "linear_trend",
                "rw1_drift",
                "rw2_smooth_trend",
                "local_linear_trend",
            }
            supplied_models = set(self.trend_model_probabilities)
            if supplied_models != required_models:
                raise ValueError(
                    "trend_model_probabilities must contain exactly "
                    f"{sorted(required_models)}."
                )
            model_probabilities = np.asarray(
                [self.trend_model_probabilities[name] for name in sorted(required_models)],
                dtype=float,
            )
            if (
                np.any(~np.isfinite(model_probabilities))
                or np.any(model_probabilities < 0.0)
                or not np.isclose(model_probabilities.sum(), 1.0)
            ):
                raise ValueError(
                    "trend_model_probabilities must be non-negative and sum to one."
                )

@dataclass(frozen=True)
class FSGaussianPriors:
    """Priors for the non-centred structural Gaussian model."""

    sigma2: InverseGammaPrior
    alpha0: NormalPrior
    beta0: NormalPrior
    s_level: Optional[NormalPrior] = None
    gamma0_season: Optional[DiagonalNormalPrior] = None
    s_trend: Optional[NormalPrior] = None
    s_season: Optional[NormalPrior] = None
    lasso: Optional[Union[BayesianLassoPrior, ComponentwiseBayesianLassoPrior]] = None
    horseshoe: Optional[RegularizedHorseshoePrior] = None
    triple_gamma: Optional[TripleGammaPrior] = None
    pc: Optional[PCInnovationPrior] = None
    ssvs: Optional[SSVSPrior] = None

    def __post_init__(self) -> None:
        strategies = (
            int(self.lasso is not None)
            + int(self.horseshoe is not None)
            + int(self.triple_gamma is not None)
            + int(self.pc is not None)
            + int(self.ssvs is not None)
            + int(self.s_level is not None)
        )
        if strategies != 1:
            raise ValueError(
                "Choose exactly one innovation prior: normal, Bayesian lasso, "
                "regularized horseshoe, triple gamma, PC prior, or SSVS."
            )
        if self.horseshoe is not None:
            missing = {"level", "trend", "season"} - set(
                self.horseshoe.coefficient_scale
            )
            if missing:
                raise ValueError(
                    "A univariate FS regularized horseshoe requires coefficient "
                    f"scales for level, trend, and season; missing={sorted(missing)}."
                )
        if self.triple_gamma is not None:
            missing = {"level", "trend", "season"} - set(
                self.triple_gamma.coefficient_scale
            )
            if missing:
                raise ValueError(
                    "A univariate FS triple gamma requires coefficient scales "
                    f"for level, trend, and season; missing={sorted(missing)}."
                )

    @property
    def profile(self) -> str:
        if self.lasso is not None:
            return "regularized_lasso" if self.lasso.componentwise else "manuscript_lasso"
        if self.horseshoe is not None:
            return "regularized_horseshoe"
        if self.triple_gamma is not None:
            return (
                "regularized_triple_gamma"
                if self.triple_gamma.regularized
                else "triple_gamma"
            )
        if self.pc is not None:
            return "pc"
        if self.ssvs is not None:
            return "ssvs"
        return "normal"


XiPrior = Union[NormalPrior, UniformPrior]


@dataclass(frozen=True)
class FSGEVPriors:
    """Priors for the FS structural DGEV model.

    Both an inverse-gamma prior on ``sigma**2`` and a Normal prior on
    ``log(sigma)`` are supported explicitly.
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
    horseshoe: Optional[RegularizedHorseshoePrior] = None
    triple_gamma: Optional[TripleGammaPrior] = None
    pc: Optional[PCInnovationPrior] = None
    ssvs: Optional[SSVSPrior] = None
    xi_max_abs: float = 0.5

    def __post_init__(self) -> None:
        if self.sigma2 is None and self.log_sigma is None:
            raise ValueError("Provide sigma2=InverseGammaPrior(...) or log_sigma=NormalPrior(...).")
        if self.xi_max_abs <= 0.0:
            raise ValueError("FSGEVPriors.xi_max_abs must be > 0.")
        strategies = (
            int(self.lasso is not None)
            + int(self.horseshoe is not None)
            + int(self.triple_gamma is not None)
            + int(self.pc is not None)
            + int(self.ssvs is not None)
            + int(self.s_level is not None)
        )
        if strategies != 1:
            raise ValueError(
                "Choose exactly one innovation prior: normal, Bayesian lasso, "
                "regularized horseshoe, triple gamma, PC prior, or SSVS."
            )
        if self.horseshoe is not None:
            missing = {"level", "trend", "season"} - set(
                self.horseshoe.coefficient_scale
            )
            if missing:
                raise ValueError(
                    "A univariate FS regularized horseshoe requires coefficient "
                    f"scales for level, trend, and season; missing={sorted(missing)}."
                )
        if self.triple_gamma is not None:
            missing = {"level", "trend", "season"} - set(
                self.triple_gamma.coefficient_scale
            )
            if missing:
                raise ValueError(
                    "A univariate FS triple gamma requires coefficient scales "
                    f"for level, trend, and season; missing={sorted(missing)}."
                )

    @property
    def profile(self) -> str:
        if self.lasso is not None:
            return "regularized_lasso" if self.lasso.componentwise else "manuscript_lasso"
        if self.horseshoe is not None:
            return "regularized_horseshoe"
        if self.triple_gamma is not None:
            return (
                "regularized_triple_gamma"
                if self.triple_gamma.regularized
                else "triple_gamma"
            )
        if self.pc is not None:
            return "pc"
        if self.ssvs is not None:
            return "ssvs"
        return "normal"


# ---------------------------------------------------------------------------
# Manuscript profiles
# ---------------------------------------------------------------------------

def manuscript_gaussian_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
) -> FSGaussianPriors:
    """Return the prior profile used by the Uccle Gaussian analysis."""

    k = period - 1
    return FSGaussianPriors(
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
) -> FSGEVPriors:
    """Return the prior profile used by the Uccle DGEV analysis."""

    k = period - 1
    return FSGEVPriors(
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
# Normal and structural SSVS profiles
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
) -> FSGaussianPriors:
    """Scale-aware Gaussian priors for monthly structural temperature models."""

    k = period - 1
    return FSGaussianPriors(
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
) -> FSGEVPriors:
    """Scale-aware DGEV priors for monthly structural temperature models."""

    k = period - 1
    return FSGEVPriors(
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


def _resolve_ssvs_prior(
    ssvs: Optional[SSVSPrior],
    *,
    innovation_slab_sd: Optional[Mapping[str, float]],
    level_dynamic_probability: Optional[float],
    trend_probabilities: Optional[Sequence[float]],
    season_probabilities: Optional[Sequence[float]],
) -> SSVSPrior:
    """Resolve the object and convenience-keyword SSVS APIs consistently."""

    settings = {
        "innovation_slab_sd": innovation_slab_sd,
        "level_dynamic_probability": level_dynamic_probability,
        "trend_probabilities": trend_probabilities,
        "season_probabilities": season_probabilities,
    }
    supplied = {name: value for name, value in settings.items() if value is not None}
    if ssvs is not None and supplied:
        names = ", ".join(sorted(supplied))
        raise ValueError(
            "Pass either ssvs=SSVSPrior(...) or direct SSVS settings, not both; "
            f"direct settings supplied: {names}."
        )
    if ssvs is not None:
        return ssvs
    return SSVSPrior(**supplied)


def ssvs_gaussian_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    ssvs: Optional[SSVSPrior] = None,
    innovation_slab_sd: Optional[Mapping[str, float]] = None,
    level_dynamic_probability: Optional[float] = None,
    trend_probabilities: Optional[Sequence[float]] = None,
    season_probabilities: Optional[Sequence[float]] = None,
) -> FSGaussianPriors:
    """Gaussian structural SSVS prior with exact structural states.

    The SSVS settings may be supplied either as an explicit :class:`SSVSPrior`
    through ``ssvs=`` or as the readable convenience keywords exposed here.
    Do not mix the two forms in one call.
    """

    k = period - 1
    resolved_ssvs = _resolve_ssvs_prior(
        ssvs,
        innovation_slab_sd=innovation_slab_sd,
        level_dynamic_probability=level_dynamic_probability,
        trend_probabilities=trend_probabilities,
        season_probabilities=season_probabilities,
    )
    return FSGaussianPriors(
        sigma2=InverseGammaPrior(a=2.0, b=1.0),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        ssvs=resolved_ssvs,
    )


def ssvs_gev_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    ssvs: Optional[SSVSPrior] = None,
    innovation_slab_sd: Optional[Mapping[str, float]] = None,
    level_dynamic_probability: Optional[float] = None,
    trend_probabilities: Optional[Sequence[float]] = None,
    season_probabilities: Optional[Sequence[float]] = None,
) -> FSGEVPriors:
    """DGEV structural SSVS prior with exact zero/fixed/dynamic states.

    The SSVS settings may be supplied either as an explicit :class:`SSVSPrior`
    through ``ssvs=`` or as the readable convenience keywords exposed here.
    Do not mix the two forms in one call.
    """

    k = period - 1
    resolved_ssvs = _resolve_ssvs_prior(
        ssvs,
        innovation_slab_sd=innovation_slab_sd,
        level_dynamic_probability=level_dynamic_probability,
        trend_probabilities=trend_probabilities,
        season_probabilities=season_probabilities,
    )
    return FSGEVPriors(
        sigma2=InverseGammaPrior(a=2.0, b=2.0),
        xi=UniformPrior(-0.5, 0.5),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        ssvs=resolved_ssvs,
    )


# ---------------------------------------------------------------------------
# Componentwise regularized-lasso profiles
# ---------------------------------------------------------------------------

def regularized_gaussian_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    coefficient_scale: Optional[Mapping[str, float]] = None,
) -> FSGaussianPriors:
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
    return FSGaussianPriors(
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
) -> FSGEVPriors:
    """Component-wise scale-aware lasso for monthly DGEV models."""

    k = period - 1
    scales = (
        {"level": 0.03, "trend": 0.0002, "season": 0.03}
        if coefficient_scale is None
        else dict(coefficient_scale)
    )
    return FSGEVPriors(
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


# ---------------------------------------------------------------------------
# Regularized-horseshoe profiles
# ---------------------------------------------------------------------------

def regularized_horseshoe_gaussian_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    coefficient_scale: Optional[Mapping[str, float]] = None,
    global_scale: float = 0.25,
    slab_scale: float = 2.0,
    slab_df: float = 4.0,
) -> FSGaussianPriors:
    """Scale-aware regularized horseshoe for monthly Gaussian models."""

    k = period - 1
    scales = (
        {"level": 0.03, "trend": 0.0002, "season": 0.03}
        if coefficient_scale is None
        else dict(coefficient_scale)
    )
    return FSGaussianPriors(
        sigma2=InverseGammaPrior(a=2.0, b=1.0),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        horseshoe=RegularizedHorseshoePrior(
            coefficient_scale=scales,
            global_scale=global_scale,
            slab_scale=slab_scale,
            slab_df=slab_df,
        ),
    )


def regularized_horseshoe_gev_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    coefficient_scale: Optional[Mapping[str, float]] = None,
    global_scale: float = 0.25,
    slab_scale: float = 2.0,
    slab_df: float = 4.0,
) -> FSGEVPriors:
    """Scale-aware regularized horseshoe for monthly DGEV models."""

    k = period - 1
    scales = (
        {"level": 0.03, "trend": 0.0002, "season": 0.03}
        if coefficient_scale is None
        else dict(coefficient_scale)
    )
    return FSGEVPriors(
        sigma2=InverseGammaPrior(a=2.0, b=2.0),
        xi=UniformPrior(-0.5, 0.5),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        horseshoe=RegularizedHorseshoePrior(
            coefficient_scale=scales,
            global_scale=global_scale,
            slab_scale=slab_scale,
            slab_df=slab_df,
        ),
    )


def triple_gamma_gaussian_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    coefficient_scale: Optional[Mapping[str, float]] = None,
    spike_shape: float = 0.10,
    tail_shape: float = 0.10,
    global_scale: float = 1.0,
    learn_global: bool = True,
    learn_shapes: bool = False,
    regularized: bool = False,
    slab_scale: float = 2.0,
    slab_df: float = 4.0,
) -> FSGaussianPriors:
    """Paper-calibrated triple-gamma prior for a Gaussian FS model.

    Set ``regularized=True`` to cap the standardized local variance with a
    finite inverse-gamma slab.  Shape learning is off by default because a
    univariate structural model has only three innovation scales.
    """

    k = period - 1
    scales = (
        {"level": 0.03, "trend": 0.0002, "season": 0.03}
        if coefficient_scale is None
        else dict(coefficient_scale)
    )
    return FSGaussianPriors(
        sigma2=InverseGammaPrior(a=2.0, b=1.0),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        triple_gamma=TripleGammaPrior(
            coefficient_scale=scales,
            spike_shape=spike_shape,
            tail_shape=tail_shape,
            global_scale=global_scale,
            learn_global=learn_global,
            learn_shapes=learn_shapes,
            regularized=regularized,
            slab_scale=slab_scale,
            slab_df=slab_df,
        ),
    )


def triple_gamma_gev_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    coefficient_scale: Optional[Mapping[str, float]] = None,
    spike_shape: float = 0.10,
    tail_shape: float = 0.10,
    global_scale: float = 1.0,
    learn_global: bool = True,
    learn_shapes: bool = False,
    regularized: bool = False,
    slab_scale: float = 2.0,
    slab_df: float = 4.0,
) -> FSGEVPriors:
    """Paper-calibrated triple-gamma prior for a DGEV FS model."""

    k = period - 1
    scales = (
        {"level": 0.03, "trend": 0.0002, "season": 0.03}
        if coefficient_scale is None
        else dict(coefficient_scale)
    )
    return FSGEVPriors(
        sigma2=InverseGammaPrior(a=2.0, b=2.0),
        xi=UniformPrior(-0.5, 0.5),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        triple_gamma=TripleGammaPrior(
            coefficient_scale=scales,
            spike_shape=spike_shape,
            tail_shape=tail_shape,
            global_scale=global_scale,
            learn_global=learn_global,
            learn_shapes=learn_shapes,
            regularized=regularized,
            slab_scale=slab_scale,
            slab_df=slab_df,
        ),
    )


def regularized_triple_gamma_gaussian_priors(
    period: int = 12, **kwargs
) -> FSGaussianPriors:
    """Triple gamma with the optional finite-variance regularizing slab."""

    return triple_gamma_gaussian_priors(period=period, regularized=True, **kwargs)


def regularized_triple_gamma_gev_priors(
    period: int = 12, **kwargs
) -> FSGEVPriors:
    """DGEV triple gamma with the optional finite-variance slab."""

    return triple_gamma_gev_priors(period=period, regularized=True, **kwargs)


def pc_gaussian_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    upper: Optional[Mapping[str, float]] = None,
    alpha: Union[float, Mapping[str, float]] = 0.05,
) -> FSGaussianPriors:
    """Interpretable PC shrinkage profile for the Gaussian FS sampler."""
    k = period - 1
    bounds = (
        {"level": 0.03, "trend": 0.0002, "season": 0.03}
        if upper is None
        else dict(upper)
    )
    return FSGaussianPriors(
        sigma2=InverseGammaPrior(a=2.0, b=1.0),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        pc=PCInnovationPrior(upper=bounds, alpha=alpha),
    )


def pc_gev_priors(
    period: int = 12,
    *,
    alpha_mean: float = 0.0,
    beta_mean: float = 0.0,
    beta_sd: float = 0.005,
    upper: Optional[Mapping[str, float]] = None,
    alpha: Union[float, Mapping[str, float]] = 0.05,
) -> FSGEVPriors:
    """Interpretable PC shrinkage profile for the DGEV FS sampler."""
    k = period - 1
    bounds = (
        {"level": 0.03, "trend": 0.0002, "season": 0.03}
        if upper is None
        else dict(upper)
    )
    return FSGEVPriors(
        sigma2=InverseGammaPrior(a=2.0, b=2.0),
        xi=UniformPrior(-0.5, 0.5),
        alpha0=NormalPrior(alpha_mean, np.sqrt(10.0)),
        beta0=NormalPrior(beta_mean, beta_sd),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(k), sd=np.full(k, np.sqrt(5.0))
        ),
        pc=PCInnovationPrior(upper=bounds, alpha=alpha),
    )


# Compatibility names for custom prior objects created with pre-1.2 code.
NonCenteredGaussianPriors = FSGaussianPriors
NonCenteredGEVPriors = FSGEVPriors
