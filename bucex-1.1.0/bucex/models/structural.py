from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..components.base import Array, Component, ParamDict
from ..components.compose import compose_components, compose_initial_state, compose_system_only
from .base import LinearDesign, LinearGaussianSystem, StateSpaceModel


@dataclass
class StructuralSSM(StateSpaceModel):
    """Structural state-space model built by composing component blocks."""

    components: Sequence[Component]
    obs: Any
    eta_name: str = "mu"

    def __post_init__(self) -> None:
        self._state_names = tuple(name for comp in self.components for name in comp.spec.state_names)
        self._m = len(self._state_names)

    @property
    def state_dim(self) -> int:
        return self._m

    @property
    def eta_dim(self) -> int:
        return 1

    @property
    def state_names(self) -> Tuple[str, ...]:
        return self._state_names

    def initial_state(self, params_state: ParamDict) -> Tuple[Array, Array]:
        return compose_initial_state(self.components, params_state)

    def system(self, t: int, params_state: ParamDict) -> LinearGaussianSystem:
        T, R, Q, c, _ = compose_system_only(self.components, t=t, params=params_state)
        return LinearGaussianSystem(T=T, R=R, Q=Q, c=c)

    def design(self, t: int, params_state: ParamDict, exog_t: Optional[Array] = None) -> LinearDesign:
        mats = compose_components(self.components, t=t, params=params_state, exog_t=exog_t)
        return LinearDesign(Z=mats.Z, d=mats.d)

    def obs_params(
        self,
        t: int,
        x_t: Array,
        eta_t: Array,
        params_obs: ParamDict,
        exog_t: Optional[Array] = None,
    ) -> dict[str, Any]:
        out = dict(params_obs)
        out[self.eta_name] = float(np.atleast_1d(eta_t)[0])
        return out


StructuralModel = StructuralSSM
