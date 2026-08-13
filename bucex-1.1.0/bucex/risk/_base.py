from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import genextreme, norm


def _obs_name(obs: Any) -> str | None:
    spec = getattr(obs, "spec", None)
    return getattr(spec, "name", None) if spec is not None else None


def cdf_from_obs(y: np.ndarray | float, eta: np.ndarray | float, obs: Any, params: dict[str, Any]):
    name = _obs_name(obs)
    y = np.asarray(y, dtype=float)
    eta = np.asarray(eta, dtype=float)

    if name == "gaussian":
        sigma = float(params["sigma"])
        return norm.cdf(y, loc=eta, scale=sigma)

    if name == "gev":
        sigma = float(params["sigma"])
        xi = float(params["xi"])
        return genextreme.cdf(y, c=-xi, loc=eta, scale=sigma)

    raise NotImplementedError(f"CDF helper not implemented for obs='{name}'.")


def ppf_from_obs(q: np.ndarray | float, eta: np.ndarray | float, obs: Any, params: dict[str, Any]):
    name = _obs_name(obs)
    q = np.asarray(q, dtype=float)
    eta = np.asarray(eta, dtype=float)

    if name == "gaussian":
        sigma = float(params["sigma"])
        return norm.ppf(q, loc=eta, scale=sigma)

    if name == "gev":
        sigma = float(params["sigma"])
        xi = float(params["xi"])
        return genextreme.ppf(q, c=-xi, loc=eta, scale=sigma)

    raise NotImplementedError(f"PPF helper not implemented for obs='{name}'.")
