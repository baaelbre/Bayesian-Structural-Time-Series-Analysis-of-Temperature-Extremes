from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...core.results import PosteriorBundle
from ...models.base import StateSpaceModel
from ..dispatch import sample_states
from .base import GibbsConfig
from .priors import CenteredGaussianPriors, InverseGammaPrior
from .utils import (
    innovation_residuals,
    sample_inverse_gamma,
    update_initial_state_hyperparams,
)

Array = np.ndarray
ParamDict = Dict[str, Any]


def _as_1d_y(y: Array) -> Array:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    raise ValueError(
        "CenteredGaussianGibbs currently supports univariate Gaussian observations only."
    )


def _reconstruct_eta(
    x: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    exog: Optional[Array] = None,
) -> Array:
    """
    Recompute eta_t = Z_t x_t + d_t for t=1,...,T.

    Parameters
    ----------
    x : array
        State path of shape (T+1, m).
    model : StateSpaceModel
    params_state : dict
    exog : array, optional
        Shape (T, k) if present.

    Returns
    -------
    eta : array
        Shape (T,).
    """
    x = np.asarray(x, dtype=float)
    Tn = x.shape[0] - 1

    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != Tn:
            raise ValueError("exog must have shape (T, k).")

    eta = np.zeros(Tn, dtype=float)

    for t in range(1, Tn + 1):
        exog_t = None if exog is None else exog[t - 1]
        des = model.design(t=t, params_state=params_state, exog_t=exog_t)
        eta_t = des.Z @ x[t] + des.d
        eta[t - 1] = float(np.atleast_1d(eta_t)[0])

    return eta


def _sigma2_update(
    y: Array,
    x: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    prior: InverseGammaPrior,
    rng: np.random.Generator,
    exog: Optional[Array] = None,
) -> float:
    """
    Conjugate update for Gaussian observation variance sigma^2.
    """
    eta = _reconstruct_eta(x, model, params_state, exog=exog)
    resid = y - eta

    a_post = prior.a + 0.5 * y.size
    b_post = prior.b + 0.5 * float(resid @ resid)

    return sample_inverse_gamma(a_post, b_post, rng)


def _variance_update_from_residuals(
    resid: Array,
    prior: InverseGammaPrior,
    rng: np.random.Generator,
) -> float:
    """
    Conjugate inverse-gamma update from Gaussian innovation residuals.
    """
    resid = np.asarray(resid, dtype=float)

    a_post = prior.a + 0.5 * resid.size
    b_post = prior.b + 0.5 * float(resid @ resid)

    return sample_inverse_gamma(a_post, b_post, rng)


class CenteredGaussianGibbs:
    """
    Gibbs sampler for Gaussian structural state-space models in centered parametrization.

    Current version
    ---------------
    Updates:
      1. latent states x_{0:T} via a conditional state sampler
      2. initial-state hyperparameters via x_0
      3. Gaussian observation variance sigma^2 via conjugate inverse-gamma
      4. selected process innovation variances q_* via conjugate inverse-gamma
         using centered innovation residuals extracted from the sampled state path

    Current limitations
    -------------------
    - univariate Gaussian observations only
    - recognized process variance keys are currently:
          q_level, q_trend, q_season
      as returned by `innovation_residuals(...)`
    - with your current component implementation, q_* are interpreted as
      innovation variances, not standard deviations
    """

    def __init__(
        self,
        model: StateSpaceModel,
        priors: CenteredGaussianPriors,
        config: GibbsConfig = GibbsConfig(),
    ) -> None:
        self.model = model
        self.priors = priors
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def fit(
        self,
        y: Array,
        init_params_state: ParamDict,
        init_params_obs: ParamDict,
        exog: Optional[Array] = None,
        state_method: str = "ffbs",
        state_kwargs: Optional[dict[str, Any]] = None,
    ) -> PosteriorBundle:
        """
        Run centered Gaussian Gibbs sampling.

        Parameters
        ----------
        y : array
            Univariate observations, shape (T,) or (T,1).
        init_params_state : dict
            Initial values for state/process parameters.
        init_params_obs : dict
            Initial values for observation parameters. Must contain 'sigma' or 'sigma2'.
        exog : array, optional
            Exogenous regressors, shape (T, k).
        state_method : str
            Passed to `sample_states(...)`. Typical choices are:
              - "ffbs"
              - "particle"
              - "auto"
        state_kwargs : dict, optional
            Extra keyword arguments forwarded to `sample_states(...)`, for example:
              {
                  "particle_method": "bootstrap",
                  "particle_n_particles": 2000,
              }

        Returns
        -------
        PosteriorBundle
        """
        y1 = _as_1d_y(y)
        Tn = y1.size
        m = self.model.state_dim

        params_state = dict(init_params_state)
        params_obs = dict(init_params_obs)
        state_kwargs = {} if state_kwargs is None else dict(state_kwargs)

        if "sigma2" in params_obs and "sigma" not in params_obs:
            params_obs["sigma"] = float(np.sqrt(float(params_obs["sigma2"])))
        if "sigma" not in params_obs:
            raise KeyError("init_params_obs must contain 'sigma' or 'sigma2'.")

        n_iter = self.config.n_iter
        burn = self.config.burn
        thin = self.config.thin

        save_iters = list(range(burn, n_iter, thin))
        n_keep = len(save_iters)
        save_set = set(save_iters)

        season_dim = sum(str(nm).startswith("g") for nm in self.model.state_names)

        draws_states = np.zeros((n_keep, Tn + 1, m), dtype=float)

        draws_params_state: Dict[str, np.ndarray] = {
            "q_level": np.full(n_keep, np.nan, dtype=float),
            "q_trend": np.full(n_keep, np.nan, dtype=float),
            "q_season": np.full(n_keep, np.nan, dtype=float),
            "m0_level": np.full(n_keep, np.nan, dtype=float),
            "v0_level": np.full(n_keep, np.nan, dtype=float),
            "m0_trend": np.full(n_keep, np.nan, dtype=float),
            "v0_trend": np.full(n_keep, np.nan, dtype=float),
        }

        if season_dim > 0:
            draws_params_state["m0_season"] = np.full((n_keep, season_dim), np.nan, dtype=float)
            draws_params_state["v0_season"] = np.full((n_keep, season_dim), np.nan, dtype=float)

        draws_params_obs: Dict[str, np.ndarray] = {
            "sigma": np.zeros(n_keep, dtype=float),
            "sigma2": np.zeros(n_keep, dtype=float),
        }

        logpost = np.full(n_keep, np.nan, dtype=float)
        keep_idx = 0

        progress_every = (
            self.config.progress_every
            if self.config.progress_every > 0
            else max(1, n_iter // 50)
        )

        for it in range(n_iter):
            # ----------------------------------------------------------
            # 1) x | y, theta
            # ----------------------------------------------------------
            state_draw = sample_states(
                y=y1,
                model=self.model,
                params_state=params_state,
                params_obs=params_obs,
                exog=exog,
                rng=self.rng,
                method=state_method,
                **state_kwargs,
            )
            x_path = state_draw.x

            # ----------------------------------------------------------
            # 1b) initial-state hyperparameters | x_0
            # ----------------------------------------------------------
            params_state = update_initial_state_hyperparams(
                x=x_path,
                model=self.model,
                params_state=params_state,
                priors=self.priors.initial,
                rng=self.rng,
            )

            # ----------------------------------------------------------
            # 2) sigma^2 | x, y
            # ----------------------------------------------------------
            sigma2 = _sigma2_update(
                y=y1,
                x=x_path,
                model=self.model,
                params_state=params_state,
                prior=self.priors.sigma2,
                rng=self.rng,
                exog=exog,
            )
            params_obs["sigma2"] = float(sigma2)
            params_obs["sigma"] = float(np.sqrt(sigma2))

            # ----------------------------------------------------------
            # 3) q_* | x
            # ----------------------------------------------------------
            innov = innovation_residuals(
                x=x_path,
                model=self.model,
                params_state=params_state,
                exog=exog,
            )

            if (
                "q_level" in params_state
                and self.priors.q_level is not None
                and "q_level" in innov
            ):
                params_state["q_level"] = _variance_update_from_residuals(
                    innov["q_level"],
                    self.priors.q_level,
                    self.rng,
                )

            if (
                "q_trend" in params_state
                and self.priors.q_trend is not None
                and "q_trend" in innov
            ):
                params_state["q_trend"] = _variance_update_from_residuals(
                    innov["q_trend"],
                    self.priors.q_trend,
                    self.rng,
                )

            if (
                "q_season" in params_state
                and self.priors.q_season is not None
                and "q_season" in innov
            ):
                params_state["q_season"] = _variance_update_from_residuals(
                    innov["q_season"],
                    self.priors.q_season,
                    self.rng,
                )

            # ----------------------------------------------------------
            # Progress
            # ----------------------------------------------------------
            if self.config.progress and (((it + 1) % progress_every == 0) or (it == n_iter - 1)):
                ql = params_state.get("q_level", np.nan)
                qt = params_state.get("q_trend", np.nan)
                qs = params_state.get("q_season", np.nan)
                m0l = params_state.get("m0_level", np.nan)
                v0l = params_state.get("v0_level", np.nan)

                print(
                    f"[it {it+1}/{n_iter}] "
                    f"sigma={params_obs['sigma']:.4f} "
                    f"q_level={float(ql):.4g} "
                    f"q_trend={float(qt):.4g} "
                    f"q_season={float(qs):.4g} "
                    f"m0_level={float(m0l):.4g} "
                    f"v0_level={float(v0l):.4g}"
                )

            # ----------------------------------------------------------
            # Store
            # ----------------------------------------------------------
            if it in save_set:
                draws_states[keep_idx] = x_path

                draws_params_obs["sigma"][keep_idx] = float(params_obs["sigma"])
                draws_params_obs["sigma2"][keep_idx] = float(params_obs["sigma2"])

                for key in (
                    "q_level",
                    "q_trend",
                    "q_season",
                    "m0_level",
                    "v0_level",
                    "m0_trend",
                    "v0_trend",
                ):
                    if key in params_state and key in draws_params_state:
                        draws_params_state[key][keep_idx] = float(params_state[key])

                if "m0_season" in params_state and "m0_season" in draws_params_state:
                    draws_params_state["m0_season"][keep_idx] = np.asarray(
                        params_state["m0_season"],
                        dtype=float,
                    )

                if "v0_season" in params_state and "v0_season" in draws_params_state:
                    draws_params_state["v0_season"][keep_idx] = np.asarray(
                        params_state["v0_season"],
                        dtype=float,
                    )

                fr = state_draw.filter_result
                if fr is not None:
                    logpost[keep_idx] = float(fr.loglik)

                keep_idx += 1

        draws_params_state = {
            k: v for k, v in draws_params_state.items()
            if not np.all(np.isnan(v))
        }

        return PosteriorBundle(
            draws_static={**draws_params_state, **draws_params_obs},
            draws_states=draws_states,
            logpost=logpost,
            acceptance={},
            meta={
                "sampler": "centered_gaussian_gibbs",
                "n_iter": n_iter,
                "burn": burn,
                "thin": thin,
                "state_method": state_method,
                "state_kwargs": state_kwargs,
            },
        )