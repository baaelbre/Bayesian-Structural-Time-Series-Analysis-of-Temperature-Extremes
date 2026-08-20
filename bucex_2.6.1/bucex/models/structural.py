"""One declarative and executable structural-model representation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..components import (
    DummySeasonal,
    LocalLevel,
    LocalLinearTrend,
    Regression,
    component_from_dict,
)
from ..components.base import Array, Component, ParamDict
from ..components.compose import compose_components, compose_initial_state, compose_system_only
from ..observation import GEV, Gaussian, Observation, observation_from_dict
from .base import LinearDesign, LinearGaussianSystem, StateSpaceModel


@dataclass(init=False)
class Model(StateSpaceModel):
    """A structural state-space model shared by every inference strategy.

    ``observation=`` is canonical; ``obs=`` is a compatibility spelling that
    resolves to the same field.
    """

    observation: Observation
    components: tuple[Component, ...]
    name: str | None
    eta_name: str

    def __init__(
        self,
        observation: Observation | None = None,
        components: Sequence[Component] = (),
        name: str | None = None,
        *,
        obs: Observation | None = None,
        eta_name: str = "mu",
    ) -> None:
        if observation is not None and obs is not None:
            raise ValueError("Give observation= or obs=, not both.")
        resolved = observation if observation is not None else obs
        if not isinstance(resolved, (Gaussian, GEV)):
            raise TypeError("observation must be Gaussian() or GEV().")
        self.observation = resolved
        self.components = tuple(components)
        self.name = name
        self.eta_name = str(eta_name)
        self._validate()

    def _validate(self) -> None:
        if not self.components:
            raise ValueError("A model requires at least one structural component.")
        trend_like = sum(
            isinstance(component, (LocalLevel, LocalLinearTrend))
            for component in self.components
        )
        if trend_like != 1:
            raise ValueError("Specify exactly one LocalLevel or LocalLinearTrend component.")
        if sum(isinstance(component, DummySeasonal) for component in self.components) > 1:
            raise ValueError("At most one DummySeasonal component is supported.")
        regression_names = [
            component.name for component in self.components if isinstance(component, Regression)
        ]
        if len(regression_names) != len(set(regression_names)):
            raise ValueError("Regression component names must be unique.")
        names = self.state_names
        if len(names) != len(set(names)):
            raise ValueError("State names must be unique across components.")

    @property
    def obs(self) -> Observation:
        """Compatibility alias for the canonical observation family."""

        return self.observation

    @property
    def family(self) -> str:
        return self.observation.name

    @property
    def period(self) -> int | None:
        for component in self.components:
            if isinstance(component, DummySeasonal) and component.mode != "off":
                return int(component.period)
        return None

    @property
    def n_exog(self) -> int:
        return sum(
            int(component.n_features)
            for component in self.components
            if isinstance(component, Regression) and component.mode != "off"
        )

    @property
    def state_dim(self) -> int:
        return sum(int(component.spec.state_dim) for component in self.components)

    @property
    def eta_dim(self) -> int:
        return 1

    @property
    def state_names(self) -> Tuple[str, ...]:
        return tuple(
            name
            for component in self.components
            for name in component.spec.state_names
        )

    @property
    def noise_names(self) -> Tuple[str, ...]:
        return tuple(
            name
            for component in self.components
            for name in component.spec.noise_names
        )

    def initial_state(self, params_state: ParamDict) -> Tuple[Array, Array]:
        return compose_initial_state(self.components, params_state)

    def system(self, t: int, params_state: ParamDict) -> LinearGaussianSystem:
        transition, loading, covariance, offset, _ = compose_system_only(
            self.components,
            t=t,
            params=params_state,
        )
        return LinearGaussianSystem(
            T=transition,
            R=loading,
            Q=covariance,
            c=offset,
        )

    def design(
        self,
        t: int,
        params_state: ParamDict,
        exog_t: Optional[Array] = None,
    ) -> LinearDesign:
        matrices = compose_components(
            self.components,
            t=t,
            params=params_state,
            exog_t=exog_t,
        )
        return LinearDesign(Z=matrices.Z, d=matrices.d)

    def obs_params(
        self,
        t: int,
        x_t: Array,
        eta_t: Array,
        params_obs: ParamDict,
        exog_t: Optional[Array] = None,
    ) -> dict[str, Any]:
        output = dict(params_obs)
        output[self.eta_name] = float(np.atleast_1d(eta_t)[0])
        return output

    def with_observation(self, observation: Observation) -> "Model":
        return Model(observation, self.components, self.name, eta_name=self.eta_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "components": [component.to_dict() for component in self.components],
            "name": self.name,
            "eta_name": self.eta_name,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Model":
        return cls(
            observation=observation_from_dict(value["observation"]),
            components=[component_from_dict(item) for item in value["components"]],
            name=value.get("name"),
            eta_name=value.get("eta_name", "mu"),
        )


def structural_model(
    family: str = "gaussian",
    *,
    trend: str = "local_linear",
    period: int | None = None,
    xi_bounds: tuple[float, float] = (-0.5, 0.5),
) -> Model:
    family_key = str(family).lower()
    if family_key == "gaussian":
        observation: Observation = Gaussian()
    elif family_key == "gev":
        observation = GEV(xi_bounds=xi_bounds)
    else:
        raise ValueError("family must be 'gaussian' or 'gev'.")
    if trend == "local_level":
        components: list[Component] = [LocalLevel()]
    elif trend == "local_linear":
        components = [LocalLinearTrend()]
    else:
        raise ValueError("trend must be 'local_level' or 'local_linear'.")
    if period is not None:
        components.append(DummySeasonal(period=int(period)))
    return Model(observation=observation, components=components)


StructuralSSM = Model
StructuralModel = Model
