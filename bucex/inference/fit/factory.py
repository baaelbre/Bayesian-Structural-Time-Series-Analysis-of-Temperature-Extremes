from __future__ import annotations

from ...models.base import StateSpaceModel
from .base import GibbsConfig
from .centered_gaussian import CenteredGaussianGibbs
from .centered_gev import CenteredGEVGibbs
from .noncentered_gaussian import NonCenteredGaussianGibbs
from .noncentered_gev import NonCenteredGEVGibbs
from .priors import (
    CenteredGaussianPriors,
    CenteredGEVPriors,
    NonCenteredGaussianPriors,
    NonCenteredGEVPriors,
)


def make_gaussian_gibbs(
    model: StateSpaceModel,
    priors: CenteredGaussianPriors | NonCenteredGaussianPriors,
    config: GibbsConfig = GibbsConfig(),
    *,
    parameterization: str = "centered",
) -> CenteredGaussianGibbs | NonCenteredGaussianGibbs:
    if parameterization == "centered":
        if not isinstance(priors, CenteredGaussianPriors):
            raise TypeError("Centered Gaussian sampler requires CenteredGaussianPriors.")
        return CenteredGaussianGibbs(model=model, priors=priors, config=config)
    if parameterization == "noncentered":
        if not isinstance(priors, NonCenteredGaussianPriors):
            raise TypeError("Non-centred Gaussian sampler requires NonCenteredGaussianPriors.")
        return NonCenteredGaussianGibbs(model=model, priors=priors, config=config)
    raise ValueError("parameterization must be 'centered' or 'noncentered'.")


def make_gev_gibbs(
    model: StateSpaceModel,
    priors: CenteredGEVPriors | NonCenteredGEVPriors,
    config: GibbsConfig = GibbsConfig(),
    *,
    parameterization: str = "centered",
    step_log_sigma: float = 0.08,
    step_u_xi: float = 0.08,
) -> CenteredGEVGibbs | NonCenteredGEVGibbs:
    if parameterization == "centered":
        if not isinstance(priors, CenteredGEVPriors):
            raise TypeError("Centered GEV sampler requires CenteredGEVPriors.")
        return CenteredGEVGibbs(
            model=model,
            priors=priors,
            config=config,
            step_log_sigma=step_log_sigma,
            step_u_xi=step_u_xi,
        )
    if parameterization == "noncentered":
        if not isinstance(priors, NonCenteredGEVPriors):
            raise TypeError("Non-centred GEV sampler requires NonCenteredGEVPriors.")
        return NonCenteredGEVGibbs(
            model=model,
            priors=priors,
            config=config,
            step_log_sigma=step_log_sigma,
            step_u_xi=step_u_xi,
        )
    raise ValueError("parameterization must be 'centered' or 'noncentered'.")