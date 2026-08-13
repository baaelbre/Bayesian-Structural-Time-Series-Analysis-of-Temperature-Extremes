from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import numpy as np

from ...core.results import FilterResult, SmootherResult, StateSample

Array = np.ndarray
ParamDict = Dict[str, Any]


class FilterBackend(Protocol):
    """
    Protocol for conditional state filtering backends.

    A filter returns p(x_t | y_1:t, theta) summaries, conditional on fixed parameters.
    """

    def filter(
        self,
        y: Array,
        model: Any,
        params_state: ParamDict,
        params_obs: ParamDict,
        exog: Optional[Array] = None,
    ) -> FilterResult: ...


class SmootherBackend(Protocol):
    """
    Protocol for conditional state smoothing backends.

    A smoother returns p(x_t | y_1:T, theta) summaries, conditional on fixed parameters.
    """

    def smooth(
        self,
        y: Array,
        model: Any,
        params_state: ParamDict,
        params_obs: ParamDict,
        exog: Optional[Array] = None,
    ) -> SmootherResult: ...


class StateSamplerBackend(Protocol):
    """
    Protocol for conditional state trajectory samplers.

    A state sampler returns one sampled path from p(x_0:T | y_1:T, theta),
    conditional on fixed parameters.
    """

    def sample_states(
        self,
        y: Array,
        model: Any,
        params_state: ParamDict,
        params_obs: ParamDict,
        exog: Optional[Array] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> StateSample: ...