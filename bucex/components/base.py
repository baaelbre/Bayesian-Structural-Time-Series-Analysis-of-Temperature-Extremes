from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

Mode = str  # "dynamic" | "static" | "off"
Array = np.ndarray
ParamDict = Dict[str, Any]


@dataclass(frozen=True)
class ComponentSpec:
    """Metadata for one structural component block."""

    name: str
    mode: Mode
    state_dim: int
    noise_dim: int
    state_names: Tuple[str, ...] = ()


class Component(Protocol):
    """Linear-Gaussian structural component.

    A component contributes a block to the latent state evolution

        x_t = T x_{t-1} + c + R eps_t,
        eps_t ~ N(0, Q),

    and a contribution to the scalar linear predictor

        eta_t = Z x_t + d.

    `exog_t` is optional and allows regression-like components to inject
    time-varying covariates into the design without changing the rest of the
    package architecture.
    """

    spec: ComponentSpec

    def initial_mean_var(self, params: ParamDict) -> Tuple[Array, Array]:
        """Return (m0, v0diag) for the component state."""

    def system_matrices(self, t: int, params: ParamDict) -> Tuple[Array, Array, Array, Array]:
        """Return (T, R, Q, c) for the component state transition."""

    def design_matrices(
        self,
        t: int,
        params: ParamDict,
        exog_t: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Return (Z, d) for the component contribution to eta_t."""
