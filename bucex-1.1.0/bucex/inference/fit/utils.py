from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...models.base import StateSpaceModel
from .priors import InitialStatePriors

Array = np.ndarray
ParamDict = Dict[str, Any]


def sample_inverse_gamma(
    a: float,
    b: float,
    rng: np.random.Generator,
) -> float:
    if a <= 0.0:
        raise ValueError("a must be > 0.")
    if b <= 0.0:
        raise ValueError("b must be > 0.")
    g = rng.gamma(shape=a, scale=1.0 / b)
    return float(1.0 / g)


def extract_component_blocks(model: StateSpaceModel) -> Dict[str, slice]:
    names = list(model.state_names)
    out: Dict[str, slice] = {}

    if "alpha" in names:
        i = names.index("alpha")
        out["alpha"] = slice(i, i + 1)

    if "beta" in names:
        i = names.index("beta")
        out["beta"] = slice(i, i + 1)

    g_idx = [i for i, nm in enumerate(names) if nm.startswith("g")]
    if g_idx:
        out["seasonal"] = slice(min(g_idx), max(g_idx) + 1)

    return out


def one_step_state_residuals(
    x: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    exog: Optional[Array] = None,
) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must have shape (T+1, m).")

    Tn, m = x.shape[0] - 1, x.shape[1]

    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != Tn:
            raise ValueError("exog must have shape (T, k) matching x path length T+1.")

    resid = np.zeros((Tn, m), dtype=float)

    for t in range(1, Tn + 1):
        sys = model.system(t=t, params_state=params_state)
        T_t = np.asarray(sys.T, dtype=float)
        c_t = np.asarray(sys.c, dtype=float).reshape(m)
        resid[t - 1] = x[t] - (T_t @ x[t - 1] + c_t)

    return resid


def innovation_residuals(
    x: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    exog: Optional[Array] = None,
) -> Dict[str, Array]:
    r = one_step_state_residuals(
        x=x,
        model=model,
        params_state=params_state,
        exog=exog,
    )

    blocks = extract_component_blocks(model)
    out: Dict[str, Array] = {}

    if "alpha" in blocks:
        out["q_level"] = r[:, blocks["alpha"]].reshape(-1)

    if "beta" in blocks:
        out["q_trend"] = r[:, blocks["beta"]].reshape(-1)

    if "seasonal" in blocks:
        seasonal_block = r[:, blocks["seasonal"]]
        out["q_season"] = seasonal_block[:, 0].reshape(-1)

    return out


def _sample_normal_posterior(
    x: float,
    obs_var: float,
    prior_mean: float,
    prior_sd: float,
    rng: np.random.Generator,
) -> float:
    prior_var = prior_sd * prior_sd
    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    post_mean = post_var * (prior_mean / prior_var + x / obs_var)
    return float(rng.normal(loc=post_mean, scale=np.sqrt(post_var)))


def update_initial_state_hyperparams(
    x: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    priors: InitialStatePriors,
    rng: np.random.Generator,
) -> ParamDict:
    """
    Gibbs update of initial-state hyperparameters using the sampled x_0.

    Notes
    -----
    - m0_* are updated conditionally on current v0_*
    - v0_* are updated conditionally on current m0_*
    - for seasonality, updates are done coordinate-wise with independent priors
    """
    x = np.asarray(x, dtype=float)
    x0 = x[0]
    blocks = extract_component_blocks(model)

    out = dict(params_state)

    # ------------------------------------------------------------
    # level
    # ------------------------------------------------------------
    if "alpha" in blocks:
        x0_alpha = float(x0[blocks["alpha"]][0])

        if priors.m0_level is not None:
            v0 = float(out["v0_level"])
            out["m0_level"] = _sample_normal_posterior(
                x=x0_alpha,
                obs_var=v0,
                prior_mean=priors.m0_level.mean,
                prior_sd=priors.m0_level.sd,
                rng=rng,
            )

        if priors.v0_level is not None:
            m0 = float(out["m0_level"])
            a_post = priors.v0_level.a + 0.5
            b_post = priors.v0_level.b + 0.5 * (x0_alpha - m0) ** 2
            out["v0_level"] = sample_inverse_gamma(a_post, b_post, rng)

    # ------------------------------------------------------------
    # trend
    # ------------------------------------------------------------
    if "beta" in blocks:
        x0_beta = float(x0[blocks["beta"]][0])

        if priors.m0_trend is not None:
            v0 = float(out["v0_trend"])
            out["m0_trend"] = _sample_normal_posterior(
                x=x0_beta,
                obs_var=v0,
                prior_mean=priors.m0_trend.mean,
                prior_sd=priors.m0_trend.sd,
                rng=rng,
            )

        if priors.v0_trend is not None:
            m0 = float(out["m0_trend"])
            a_post = priors.v0_trend.a + 0.5
            b_post = priors.v0_trend.b + 0.5 * (x0_beta - m0) ** 2
            out["v0_trend"] = sample_inverse_gamma(a_post, b_post, rng)

    # ------------------------------------------------------------
    # seasonality
    # ------------------------------------------------------------
    if "seasonal" in blocks:
        x0_season = np.asarray(x0[blocks["seasonal"]], dtype=float).reshape(-1)
        K = x0_season.size

        if "m0_season" not in out:
            out["m0_season"] = np.zeros(K, dtype=float)
        if "v0_season" not in out:
            out["v0_season"] = np.ones(K, dtype=float)

        m0_season = np.asarray(out["m0_season"], dtype=float).reshape(-1)
        v0_season = np.asarray(out["v0_season"], dtype=float).reshape(-1)

        if m0_season.size != K or v0_season.size != K:
            raise ValueError("m0_season and v0_season must match the seasonal state dimension.")

        if priors.m0_season is not None:
            prior_mean = priors.m0_season.mean_array()
            prior_sd = priors.m0_season.sd_array()
            if prior_mean.size != K or prior_sd.size != K:
                raise ValueError("InitialStatePriors.m0_season must match the seasonal dimension.")

            for j in range(K):
                m0_season[j] = _sample_normal_posterior(
                    x=float(x0_season[j]),
                    obs_var=float(v0_season[j]),
                    prior_mean=float(prior_mean[j]),
                    prior_sd=float(prior_sd[j]),
                    rng=rng,
                )

            out["m0_season"] = m0_season.copy()

        if priors.v0_season is not None:
            new_v0 = np.zeros(K, dtype=float)
            for j in range(K):
                a_post = priors.v0_season.a + 0.5
                b_post = priors.v0_season.b + 0.5 * (float(x0_season[j]) - float(m0_season[j])) ** 2
                new_v0[j] = sample_inverse_gamma(a_post, b_post, rng)

            out["v0_season"] = new_v0

    return out