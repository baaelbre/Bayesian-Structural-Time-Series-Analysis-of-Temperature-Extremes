from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...core.results import PosteriorBundle
from ...models.base import StateSpaceModel
from .base import GibbsConfig
from .noncentered_utils import (
    NCPLayout,
    canonicalize_ncp_params,
    ffbs_gaussian_1d,
    gaussian_theta_update,
    infer_ncp_layout,
    map_ncp_to_centered,
    measurement_vector,
    mu_from_ncp,
    random_sign_switches,
    build_ncp_system,
)
from .priors import InverseGammaPrior, NonCenteredGaussianPriors
from .utils import sample_inverse_gamma

Array = np.ndarray
ParamDict = Dict[str, Any]


def _as_1d_y(y: Array) -> Array:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    raise ValueError("NonCenteredGaussianGibbs currently supports univariate Gaussian observations only.")


def _sigma2_update_from_mu(y: Array, mu: Array, prior: InverseGammaPrior, rng: np.random.Generator) -> float:
    resid = np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)
    a_post = prior.a + 0.5 * resid.size
    b_post = prior.b + 0.5 * float(resid @ resid)
    return sample_inverse_gamma(a_post, b_post, rng)


class NonCenteredGaussianGibbs:
    def __init__(
        self,
        model: StateSpaceModel,
        priors: NonCenteredGaussianPriors,
        config: GibbsConfig = GibbsConfig(),
    ) -> None:
        self.model = model
        self.priors = priors
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.layout: NCPLayout = infer_ncp_layout(model)

    def fit(
        self,
        y: Array,
        init_params_state: ParamDict,
        init_params_obs: ParamDict,
        exog: Optional[Array] = None,
        state_method: str = "ffbs",
        state_kwargs: Optional[dict[str, Any]] = None,
    ) -> PosteriorBundle:
        y1 = _as_1d_y(y)
        Tn = y1.size
        m = self.model.state_dim

        if exog is not None:
            raise NotImplementedError("NonCenteredGaussianGibbs does not support exog yet.")
        if state_kwargs is None:
            state_kwargs = {}
        if state_method != "ffbs":
            raise NotImplementedError("NonCenteredGaussianGibbs currently supports only state_method='ffbs'.")

        params_state = canonicalize_ncp_params(init_params_state, self.layout)
        params_obs = dict(init_params_obs)
        if "sigma2" in params_obs and "sigma" not in params_obs:
            params_obs["sigma"] = float(np.sqrt(float(params_obs["sigma2"])))
        if "sigma" not in params_obs:
            raise KeyError("init_params_obs must contain 'sigma' or 'sigma2'.")
        params_obs["sigma2"] = float(params_obs["sigma"]) ** 2

        n_iter = self.config.n_iter
        burn = self.config.burn
        thin = self.config.thin
        save_iters = list(range(burn, n_iter, thin))
        n_keep = len(save_iters)
        save_set = set(save_iters)

        draws_states = np.zeros((n_keep, Tn + 1, m), dtype=float)
        draws_static: Dict[str, np.ndarray] = {
            "alpha0": np.zeros(n_keep, dtype=float),
            "s_level": np.zeros(n_keep, dtype=float),
            "q_level": np.zeros(n_keep, dtype=float),
            "sigma": np.zeros(n_keep, dtype=float),
            "sigma2": np.zeros(n_keep, dtype=float),
        }
        if self.layout.has_beta:
            draws_static["beta0"] = np.zeros(n_keep, dtype=float)
            draws_static["s_trend"] = np.zeros(n_keep, dtype=float)
            draws_static["q_trend"] = np.zeros(n_keep, dtype=float)
        if self.layout.season_dim > 0:
            draws_static["gamma0_season"] = np.zeros((n_keep, self.layout.season_dim), dtype=float)
            draws_static["s_season"] = np.zeros(n_keep, dtype=float)
            draws_static["q_season"] = np.zeros(n_keep, dtype=float)

        logpost = np.full(n_keep, np.nan, dtype=float)
        z_draws = np.zeros((n_keep, Tn + 1, self.layout.ncp_state_dim), dtype=float)

        G, Q = build_ncp_system(self.layout)
        keep_idx = 0
        progress_every = self.config.progress_every if self.config.progress_every > 0 else max(1, n_iter // 50)

        z_path = np.zeros((Tn + 1, self.layout.ncp_state_dim), dtype=float)
        for it in range(n_iter):
            y_center = y1 - (params_state["alpha0"] + params_state["beta0"] * np.arange(1, Tn + 1))
            if self.layout.season_dim > 0:
                from .noncentered_utils import baseline_mu_path
                y_center = y1 - baseline_mu_path(Tn, params_state, self.layout)
            H = measurement_vector(params_state, self.layout)
            z_path = ffbs_gaussian_1d(
                y=y_center,
                G=G,
                Q=Q,
                H=H,
                R=float(params_obs["sigma2"]),
                m0=np.zeros(self.layout.ncp_state_dim, dtype=float),
                C0=1e-6 * np.eye(self.layout.ncp_state_dim),
                rng=self.rng,
            )

            theta_draw = gaussian_theta_update(y=y1, z_path=z_path, sigma2=float(params_obs["sigma2"]), priors=self.priors, layout=self.layout, rng=self.rng)
            params_state.update(theta_draw)
            params_state["q_level"] = float(params_state["s_level"]) ** 2
            params_state["q_trend"] = float(params_state.get("s_trend", 0.0)) ** 2
            params_state["q_season"] = float(params_state.get("s_season", 0.0)) ** 2

            z_path, params_state = random_sign_switches(z_path, params_state, self.layout, self.rng)

            mu = mu_from_ncp(z_path, params_state, self.layout)
            sigma2 = _sigma2_update_from_mu(y1, mu, self.priors.sigma2, self.rng)
            params_obs["sigma2"] = float(sigma2)
            params_obs["sigma"] = float(np.sqrt(sigma2))
            x_path = map_ncp_to_centered(z_path, params_state, self.layout)

            if self.config.progress and (((it + 1) % progress_every == 0) or (it == n_iter - 1)):
                msg = f"[it {it+1}/{n_iter}] sigma={params_obs['sigma']:.4f} alpha0={params_state['alpha0']:.4g} s_level={params_state['s_level']:.4g}"
                if self.layout.has_beta:
                    msg += f" beta0={params_state['beta0']:.4g} s_trend={params_state['s_trend']:.4g}"
                if self.layout.season_dim > 0:
                    msg += f" s_season={params_state['s_season']:.4g}"
                print(msg)

            if it in save_set:
                draws_states[keep_idx] = x_path
                z_draws[keep_idx] = z_path
                draws_static["alpha0"][keep_idx] = float(params_state["alpha0"])
                if self.layout.has_beta:
                    draws_static["beta0"][keep_idx] = float(params_state["beta0"])
                draws_static["s_level"][keep_idx] = float(params_state["s_level"])
                draws_static["q_level"][keep_idx] = float(params_state["q_level"])
                draws_static["sigma"][keep_idx] = float(params_obs["sigma"])
                draws_static["sigma2"][keep_idx] = float(params_obs["sigma2"])
                if self.layout.has_beta:
                    draws_static["s_trend"][keep_idx] = float(params_state["s_trend"])
                    draws_static["q_trend"][keep_idx] = float(params_state["q_trend"])
                if self.layout.season_dim > 0:
                    draws_static["gamma0_season"][keep_idx] = np.asarray(params_state["gamma0_season"], dtype=float)
                    draws_static["s_season"][keep_idx] = float(params_state["s_season"])
                    draws_static["q_season"][keep_idx] = float(params_state["q_season"])
                resid = y1 - mu
                logpost[keep_idx] = float(-0.5 * Tn * np.log(2.0 * np.pi * params_obs["sigma2"]) - 0.5 * float(resid @ resid) / params_obs["sigma2"])
                keep_idx += 1

        return PosteriorBundle(
            draws_static=draws_static,
            draws_states=draws_states,
            logpost=logpost,
            acceptance={},
            meta={
                "sampler": "noncentered_gaussian_gibbs",
                "parameterization": "noncentered",
                "n_iter": n_iter,
                "burn": burn,
                "thin": thin,
                "state_method": state_method,
                "ncp_state_names": self.layout.ncp_state_names,
                "draws_states_ncp": z_draws,
            },
        )
