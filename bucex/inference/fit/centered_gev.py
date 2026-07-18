from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...core.results import PosteriorBundle
from ...models.base import StateSpaceModel
from ..dispatch import sample_states
from .base import GibbsConfig
from .priors import CenteredGEVPriors, InverseGammaPrior
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
    raise ValueError("CenteredGEVGibbs currently supports univariate observations only.")


def _reconstruct_eta(
    x: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    exog: Optional[Array] = None,
) -> Array:
    """
    Recompute eta_t = Z_t x_t + d_t for t=1,...,T.
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


def _normal_logpdf(x: float, mean: float, sd: float) -> float:
    z = (float(x) - float(mean)) / float(sd)
    return float(-0.5 * z * z - np.log(sd) - 0.5 * np.log(2.0 * np.pi))


def _xi_to_u(xi: float, xi_max_abs: float) -> float:
    x = float(xi) / float(xi_max_abs)
    x = np.clip(x, -0.999999, 0.999999)
    return float(np.arctanh(x))


def _u_to_xi(u: float, xi_max_abs: float) -> float:
    return float(xi_max_abs * np.tanh(float(u)))


def _log_abs_dxi_du(u: float, xi_max_abs: float) -> float:
    th = np.tanh(float(u))
    val = float(xi_max_abs) * (1.0 - th * th)
    return float(np.log(max(val, 1e-15)))


def _observation_loglik(
    y: Array,
    x: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    params_obs: ParamDict,
    exog: Optional[Array] = None,
) -> float:
    """
    Exact GEV observation log-likelihood conditional on a state path x.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    Tn = y.size

    if x.shape[0] != Tn + 1:
        raise ValueError("x must have shape (T+1, m).")

    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != Tn:
            raise ValueError("exog must have shape (T, k).")

    eta = _reconstruct_eta(x, model, params_state, exog=exog)

    ll = 0.0
    for t in range(1, Tn + 1):
        exog_t = None if exog is None else exog[t - 1]
        eta_t = float(eta[t - 1])

        op = model.obs_params(
            t=t,
            x_t=x[t],
            eta_t=np.asarray([eta_t]),
            params_obs=params_obs,
            exog_t=exog_t,
        )

        try:
            ll += float(model.obs.logpdf(y=float(y[t - 1]), eta=eta_t, params=op))
        except Exception:
            return -np.inf

    return float(ll)


def _transformed_obs_logpost(
    log_sigma: float,
    u_xi: float,
    y: Array,
    x: Array,
    model: StateSpaceModel,
    params_state: ParamDict,
    priors: CenteredGEVPriors,
    exog: Optional[Array] = None,
) -> float:
    """
    Log posterior in transformed coordinates:
      - log_sigma in R
      - u_xi in R, with xi = xi_max_abs * tanh(u_xi)
    """
    sigma = float(np.exp(log_sigma))
    xi = _u_to_xi(u_xi, priors.xi_max_abs)

    params_obs = {
        "sigma": sigma,
        "xi": xi,
    }

    ll = _observation_loglik(
        y=y,
        x=x,
        model=model,
        params_state=params_state,
        params_obs=params_obs,
        exog=exog,
    )
    if not np.isfinite(ll):
        return -np.inf

    lp = 0.0
    lp += _normal_logpdf(log_sigma, priors.log_sigma.mean, priors.log_sigma.sd)
    lp += _normal_logpdf(xi, priors.xi.mean, priors.xi.sd)
    lp += _log_abs_dxi_du(u_xi, priors.xi_max_abs)

    return float(ll + lp)


def _variance_update_from_residuals(
    resid: Array,
    prior: InverseGammaPrior,
    rng: np.random.Generator,
) -> float:
    resid = np.asarray(resid, dtype=float)
    a_post = prior.a + 0.5 * resid.size
    b_post = prior.b + 0.5 * float(resid @ resid)
    return sample_inverse_gamma(a_post, b_post, rng)


class CenteredGEVGibbs:
    """
    Gibbs sampler for structural state-space models with GEV observations.

    Current version
    ---------------
    Updates:
      1. latent states x_{0:T} via a conditional state sampler
      2. initial-state hyperparameters via x_0
      3. process innovation variances q_* via conjugate inverse-gamma
      4. observation parameters (sigma, xi) via random-walk MH in transformed coordinates

    Current limitations
    -------------------
    - univariate GEV observations only
    - scalar eta_t only
    - recognized q_* keys are currently:
          q_level, q_trend, q_season
    """

    def __init__(
        self,
        model: StateSpaceModel,
        priors: CenteredGEVPriors,
        config: GibbsConfig = GibbsConfig(),
        step_log_sigma: float = 0.08,
        step_u_xi: float = 0.08,
    ) -> None:
        self.model = model
        self.priors = priors
        self.config = config
        self.step_log_sigma = float(step_log_sigma)
        self.step_u_xi = float(step_u_xi)
        self.rng = np.random.default_rng(config.seed)

    def _mh_update_obs_params(
        self,
        y: Array,
        x: Array,
        params_state: ParamDict,
        params_obs: ParamDict,
        exog: Optional[Array] = None,
    ) -> tuple[ParamDict, bool]:
        """
        Random-walk MH update for (sigma, xi).
        """
        cur_sigma = float(params_obs["sigma"])
        cur_xi = float(params_obs["xi"])

        if cur_sigma <= 0.0:
            raise ValueError("Current sigma must be > 0 for GEV MH update.")

        cur_log_sigma = float(np.log(cur_sigma))
        cur_u_xi = _xi_to_u(cur_xi, self.priors.xi_max_abs)

        prop_log_sigma = cur_log_sigma + self.rng.normal(scale=self.step_log_sigma)
        prop_u_xi = cur_u_xi + self.rng.normal(scale=self.step_u_xi)

        logp_cur = _transformed_obs_logpost(
            log_sigma=cur_log_sigma,
            u_xi=cur_u_xi,
            y=y,
            x=x,
            model=self.model,
            params_state=params_state,
            priors=self.priors,
            exog=exog,
        )
        logp_prop = _transformed_obs_logpost(
            log_sigma=prop_log_sigma,
            u_xi=prop_u_xi,
            y=y,
            x=x,
            model=self.model,
            params_state=params_state,
            priors=self.priors,
            exog=exog,
        )

        log_alpha = logp_prop - logp_cur
        if np.log(self.rng.random()) < log_alpha:
            return {
                "sigma": float(np.exp(prop_log_sigma)),
                "xi": float(_u_to_xi(prop_u_xi, self.priors.xi_max_abs)),
            }, True

        return params_obs.copy(), False

    def fit(
        self,
        y: Array,
        init_params_state: ParamDict,
        init_params_obs: ParamDict,
        exog: Optional[Array] = None,
        state_method: str = "laplace",
        state_kwargs: Optional[dict[str, Any]] = None,
    ) -> PosteriorBundle:
        """
        Run centered GEV Gibbs sampling.

        Parameters
        ----------
        y : array
            Univariate observations, shape (T,) or (T,1).
        init_params_state : dict
            Initial values for state/process parameters.
        init_params_obs : dict
            Must contain 'sigma' and 'xi'.
        exog : array, optional
            Exogenous regressors, shape (T, k).
        state_method : str
            Typical choices are:
              - "laplace"
              - "particle"
              - "auto"
        state_kwargs : dict, optional
            Extra keyword arguments forwarded to `sample_states(...)`, for example:
              {
                  "particle_method": "auxiliary",
                  "particle_n_particles": 3000,
              }
        """
        y1 = _as_1d_y(y)
        Tn = y1.size
        m = self.model.state_dim

        params_state = dict(init_params_state)
        params_obs = dict(init_params_obs)
        state_kwargs = {} if state_kwargs is None else dict(state_kwargs)

        if "sigma" not in params_obs or "xi" not in params_obs:
            raise KeyError("init_params_obs must contain both 'sigma' and 'xi'.")

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
            "xi": np.zeros(n_keep, dtype=float),
        }

        logpost = np.full(n_keep, np.nan, dtype=float)
        keep_idx = 0
        accept_obs = 0

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
            # 2) q_* | x
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
            # 3) (sigma, xi) | x, y
            # ----------------------------------------------------------
            params_obs, accepted = self._mh_update_obs_params(
                y=y1,
                x=x_path,
                params_state=params_state,
                params_obs=params_obs,
                exog=exog,
            )
            accept_obs += int(accepted)

            # ----------------------------------------------------------
            # Progress
            # ----------------------------------------------------------
            if self.config.progress and (((it + 1) % progress_every == 0) or (it == n_iter - 1)):
                ql = params_state.get("q_level", np.nan)
                qt = params_state.get("q_trend", np.nan)
                qs = params_state.get("q_season", np.nan)
                m0l = params_state.get("m0_level", np.nan)
                v0l = params_state.get("v0_level", np.nan)

                cur_ll = _observation_loglik(
                    y=y1,
                    x=x_path,
                    model=self.model,
                    params_state=params_state,
                    params_obs=params_obs,
                    exog=exog,
                )

                print(
                    f"[it {it+1}/{n_iter}] "
                    f"sigma={params_obs['sigma']:.4f} "
                    f"xi={params_obs['xi']:.4f} "
                    f"q_level={float(ql):.4g} "
                    f"q_trend={float(qt):.4g} "
                    f"q_season={float(qs):.4g} "
                    f"m0_level={float(m0l):.4g} "
                    f"v0_level={float(v0l):.4g} "
                    f"loglik={cur_ll:.2f}"
                )

            # ----------------------------------------------------------
            # Store
            # ----------------------------------------------------------
            if it in save_set:
                draws_states[keep_idx] = x_path

                draws_params_obs["sigma"][keep_idx] = float(params_obs["sigma"])
                draws_params_obs["xi"][keep_idx] = float(params_obs["xi"])

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

                logpost[keep_idx] = _observation_loglik(
                    y=y1,
                    x=x_path,
                    model=self.model,
                    params_state=params_state,
                    params_obs=params_obs,
                    exog=exog,
                )

                keep_idx += 1

        draws_params_state = {
            k: v for k, v in draws_params_state.items()
            if not np.all(np.isnan(v))
        }

        return PosteriorBundle(
            draws_static={**draws_params_state, **draws_params_obs},
            draws_states=draws_states,
            logpost=logpost,
            acceptance={
                "obs_mh": accept_obs / n_iter,
            },
            meta={
                "sampler": "centered_gev_gibbs",
                "n_iter": n_iter,
                "burn": burn,
                "thin": thin,
                "state_method": state_method,
                "state_kwargs": state_kwargs,
                "step_log_sigma": self.step_log_sigma,
                "step_u_xi": self.step_u_xi,
            },
        )