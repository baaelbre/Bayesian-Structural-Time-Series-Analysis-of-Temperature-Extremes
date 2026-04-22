from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

Array = np.ndarray
ParamDict = Dict[str, Any]


@dataclass
class GibbsConfig:
    """
    Generic Gibbs / MH-within-Gibbs sampler configuration.
    """
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 1
    seed: Optional[int] = None
    progress: bool = True
    progress_every: int = 100


@dataclass
class GibbsState:
    """
    Mutable container for the current MCMC state.

    Attributes
    ----------
    x
        Centered latent trajectory, shape (T+1, m), if applicable.
    z
        Non-centered latent trajectory, shape (T+1, m_z), if applicable.
    params_state
        Current state/process parameter dictionary.
    params_obs
        Current observation parameter dictionary.
    aux
        Auxiliary latent variables / local scales / cached objects.
    """
    x: Optional[Array] = None
    z: Optional[Array] = None
    params_state: ParamDict = field(default_factory=dict)
    params_obs: ParamDict = field(default_factory=dict)
    aux: ParamDict = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """
    Generic optimization / MAP configuration.
    """
    max_iter: int = 500
    tol: float = 1e-6
    verbose: bool = True
    seed: Optional[int] = None


class PosteriorFitter:
    """
    Lightweight protocol-style base class for fitters.

    Concrete classes should implement a `fit(...)` method and typically return
    a PosteriorBundle from `core.results`.
    """

    def fit(self, *args, **kwargs):
        raise NotImplementedError