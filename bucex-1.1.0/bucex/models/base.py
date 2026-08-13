from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

Array = np.ndarray
ParamDict = Dict[str, Any]


@dataclass(frozen=True)
class LinearGaussianSystem:
    """
    Linear-Gaussian latent state evolution at time t:

        x_t = T_t x_{t-1} + c_t + R_t eps_t,
        eps_t ~ N(0, Q_t)

    Hence the state innovation covariance is:

        Var(x_t | x_{t-1}) = R_t Q_t R_t^T

    Shapes
    ------
    T : (m, m)
        State transition matrix.
    R : (m, r)
        Innovation loading / selection matrix.
    Q : (r, r)
        Covariance matrix of the innovation eps_t.
    c : (m,)
        Deterministic offset / drift term.
    """
    T: Array
    R: Array
    Q: Array
    c: Array


@dataclass(frozen=True)
class LinearDesign:
    """
    Linear predictor mapping at time t:

        eta_t = Z_t x_t + d_t

    Shapes
    ------
    Z : (p, m)
        Design / observation loading matrix from state to linear predictor.
    d : (p,)
        Deterministic offset term in the linear predictor.
    """
    Z: Array
    d: Array


@runtime_checkable
class StateSpaceModel(Protocol):
    """
    Minimal model interface used by simulation, filtering, smoothing,
    FFBS, and later Bayesian fitting.

    Conventions
    -----------
    - x_t is the latent state, dimension m
    - eta_t is the linear predictor, dimension p
    - the observation model maps eta_t (plus possibly other parameters)
      to p(y_t | x_t) or p(y_t | eta_t)

    Design philosophy
    -----------------
    This protocol defines the *generative structure* of the model.
    Filtering, smoothing, FFBS, Gibbs, PMMH, etc. belong in the
    inference layer and should consume this interface.
    """

    # ------------------------------------------------------------------
    # Basic dimensions / names
    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        """Dimension m of the latent state x_t."""
        ...

    @property
    def eta_dim(self) -> int:
        """Dimension p of the linear predictor eta_t."""
        ...

    @property
    def state_names(self) -> Tuple[str, ...]:
        """Names of the latent state coordinates."""
        ...

    # ------------------------------------------------------------------
    # Observation model object
    # ------------------------------------------------------------------
    obs: Any
    """
    Observation model instance, e.g. GaussianObs(), GEVObs(), etc.

    The object is expected to expose methods such as:
      - logpdf(...)
      - sample(...)
      - optionally grad_eta(...), hess_eta(...)

    depending on the backend used in inference.
    """

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    def initial_state(self, params_state: ParamDict) -> Tuple[Array, Array]:
        """
        Return prior mean and covariance for the initial latent state x_0:

            x_0 ~ N(m0, P0)

        Returns
        -------
        m0 : (m,)
            Mean vector of the initial state.
        P0 : (m, m)
            Covariance matrix of the initial state.
        """
        ...

    # ------------------------------------------------------------------
    # State evolution
    # ------------------------------------------------------------------
    def system(self, t: int, params_state: ParamDict) -> LinearGaussianSystem:
        """
        Return the state evolution objects at time t.

        Parameters
        ----------
        t : int
            Time index, conventionally 1-based in simulation/filter code.
        params_state : dict
            State/process parameters.

        Returns
        -------
        LinearGaussianSystem
            The matrices (T_t, R_t, Q_t, c_t).
        """
        ...

    # ------------------------------------------------------------------
    # Design / linear predictor
    # ------------------------------------------------------------------
    def design(
        self,
        t: int,
        params_state: ParamDict,
        exog_t: Optional[Array] = None,
    ) -> LinearDesign:
        """
        Return the linear predictor mapping at time t.

        Parameters
        ----------
        t : int
            Time index, conventionally 1-based.
        params_state : dict
            State/process parameters.
        exog_t : array, optional
            Exogenous regressors at time t, if relevant.

        Returns
        -------
        LinearDesign
            The matrices (Z_t, d_t) defining eta_t = Z_t x_t + d_t.
        """
        ...

    # ------------------------------------------------------------------
    # Observation parameter mapping
    # ------------------------------------------------------------------
    def obs_params(
        self,
        t: int,
        x_t: Array,
        eta_t: Array,
        params_obs: ParamDict,
        exog_t: Optional[Array] = None,
    ) -> Dict[str, Any]:
        """
        Map the latent state / linear predictor to the parameter dictionary
        expected by the observation model.

        Typical examples
        ----------------
        Gaussian:
            params_obs = {"sigma": 1.0}
            -> {"mu": eta_t, "sigma": 1.0}

        GEV:
            params_obs = {"sigma": 1.2, "xi": -0.2}
            -> {"mu": eta_t, "sigma": 1.2, "xi": -0.2}

        Returns
        -------
        dict
            Dictionary of observation parameters to be passed to
            model.obs.logpdf(...) or model.obs.sample(...).
        """
        ...