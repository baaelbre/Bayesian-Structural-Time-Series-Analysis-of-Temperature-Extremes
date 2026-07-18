from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...core.results import PosteriorBundle
from ...models.base import StateSpaceModel
from .base import GibbsConfig
from .noncentered_utils import (
    NCPLayout,
    build_ncp_system,
    canonicalize_ncp_params,
    ffbs_gaussian_1d,
    gaussian_theta_update,
    infer_ncp_layout,
    initialise_lasso,
    map_ncp_to_centered,
    measurement_vector,
    mu_from_ncp,
    random_sign_switches,
    update_lasso_scales,
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
    raise ValueError("NonCenteredGaussianGibbs supports univariate Gaussian observations only.")


def _sigma2_update_from_mu(
    y: Array,
    mu: Array,
    prior: InverseGammaPrior,
    rng: np.random.Generator,
    *,
    params_state: Optional[ParamDict] = None,
    tau: Optional[dict[str, float]] = None,
    lasso_uses_sigma2: bool = False,
) -> float:
    """Conjugate observation-variance update.

    When the Bayesian-lasso prior is scaled by ``sigma2``, the Gaussian priors
    ``s_k | tau_k, sigma2`` also contribute to this full conditional. Omitting
    those terms gives the wrong posterior for both the observation variance and
    the process scales.
    """
    resid = np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)
    ss = float(resid @ resid)
    n_extra = 0
    if lasso_uses_sigma2:
        if params_state is None or tau is None:
            raise ValueError("params_state and tau are required for the scaled lasso sigma2 update.")
        for block, key in (("level", "s_level"), ("trend", "s_trend"), ("season", "s_season")):
            if block in tau and key in params_state:
                ss += float(params_state[key]) ** 2 / max(float(tau[block]), 1e-12)
                n_extra += 1
    a_post = prior.a + 0.5 * (resid.size + n_extra)
    b_post = prior.b + 0.5 * ss
    return sample_inverse_gamma(a_post, b_post, rng)


class NonCenteredGaussianGibbs:
    """NCP FFBS sampler with optional hierarchical Bayesian lasso."""

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
            raise NotImplementedError("The v0.2 NCP Gaussian fitter does not support exog yet.")
        if state_method != "ffbs":
            raise NotImplementedError("The NCP Gaussian fitter supports state_method='ffbs'.")
        state_kwargs = {} if state_kwargs is None else dict(state_kwargs)

        params_state = canonicalize_ncp_params(init_params_state, self.layout)
        params_obs = dict(init_params_obs)
        if "sigma2" in params_obs and "sigma" not in params_obs:
            params_obs["sigma"] = float(np.sqrt(float(params_obs["sigma2"])))
        if "sigma" not in params_obs:
            raise KeyError("init_params_obs must contain 'sigma' or 'sigma2'.")
        params_obs["sigma2"] = float(params_obs["sigma"]) ** 2

        tau, lambda2 = initialise_lasso(self.priors, self.layout)

        n_iter, burn, thin = self.config.n_iter, self.config.burn, self.config.thin
        save_iters = list(range(burn, n_iter, thin))
        n_keep = len(save_iters)
        save_set = set(save_iters)

        draws_states = np.zeros((n_keep, Tn + 1, m), dtype=float)
        z_draws = np.zeros((n_keep, Tn + 1, self.layout.ncp_state_dim), dtype=float)
        draws_static: Dict[str, np.ndarray] = {
            "alpha0": np.zeros(n_keep),
            "s_level": np.zeros(n_keep),
            "q_level": np.zeros(n_keep),
            "sigma": np.zeros(n_keep),
            "sigma2": np.zeros(n_keep),
        }
        if self.layout.has_beta:
            draws_static.update(
                beta0=np.zeros(n_keep),
                s_trend=np.zeros(n_keep),
                q_trend=np.zeros(n_keep),
            )
        if self.layout.season_dim > 0:
            draws_static.update(
                gamma0_season=np.zeros((n_keep, self.layout.season_dim)),
                s_season=np.zeros(n_keep),
                q_season=np.zeros(n_keep),
            )
        if self.priors.lasso is not None:
            draws_static["lambda2"] = np.zeros(n_keep)
            for block in tau:
                draws_static[f"tau_{block}"] = np.zeros(n_keep)

        logpost = np.full(n_keep, np.nan)
        G, Q = build_ncp_system(self.layout)
        z_path = np.zeros((Tn + 1, self.layout.ncp_state_dim), dtype=float)
        keep_idx = 0
        progress_every = self.config.progress_every or max(1, n_iter // 50)

        for it in range(n_iter):
            from .noncentered_utils import baseline_mu_path

            offset = baseline_mu_path(Tn, params_state, self.layout)
            H = measurement_vector(params_state, self.layout)
            z_path = ffbs_gaussian_1d(
                y=y1 - offset,
                G=G,
                Q=Q,
                H=H,
                R=float(params_obs["sigma2"]),
                m0=np.zeros(self.layout.ncp_state_dim),
                C0=float(state_kwargs.get("C0_scale", 1e-6)) * np.eye(self.layout.ncp_state_dim),
                rng=self.rng,
            )

            lasso_var = (
                self.priors.lasso.variance_scale(float(params_obs["sigma2"]))
                if self.priors.lasso is not None
                else float(params_obs["sigma2"])
            )
            theta_draw = gaussian_theta_update(
                y=y1,
                z_path=z_path,
                sigma2=float(params_obs["sigma2"]),
                priors=self.priors,
                layout=self.layout,
                rng=self.rng,
                tau=tau or None,
                lasso_variance_scale=lasso_var,
            )
            params_state.update(theta_draw)
            z_path, params_state = random_sign_switches(z_path, params_state, self.layout, self.rng)

            if self.priors.lasso is not None:
                tau, lambda2 = update_lasso_scales(
                    params_state,
                    tau,
                    lambda2,
                    self.priors,
                    self.layout,
                    variance_scale=lasso_var,
                    rng=self.rng,
                )

            mu = mu_from_ncp(z_path, params_state, self.layout)
            sigma2 = _sigma2_update_from_mu(
                y1,
                mu,
                self.priors.sigma2,
                self.rng,
                params_state=params_state,
                tau=tau or None,
                lasso_uses_sigma2=(
                    self.priors.lasso is not None
                    and self.priors.lasso.variance_mode == "observation"
                ),
            )
            params_obs["sigma2"] = float(sigma2)
            params_obs["sigma"] = float(np.sqrt(sigma2))
            x_path = map_ncp_to_centered(z_path, params_state, self.layout)

            if self.config.progress and (((it + 1) % progress_every == 0) or it == n_iter - 1):
                msg = (
                    f"[it {it + 1}/{n_iter}] sigma={params_obs['sigma']:.4f} "
                    f"Q_level={params_state['q_level']:.3g}"
                )
                if self.layout.has_beta:
                    msg += f" Q_trend={params_state['q_trend']:.3g}"
                if self.layout.season_dim > 0:
                    msg += f" Q_season={params_state['q_season']:.3g}"
                if self.priors.lasso is not None:
                    msg += f" lambda2={lambda2:.3g}"
                print(msg)

            if it in save_set:
                draws_states[keep_idx] = x_path
                z_draws[keep_idx] = z_path
                for key in ("alpha0", "s_level", "q_level"):
                    draws_static[key][keep_idx] = float(params_state[key])
                draws_static["sigma"][keep_idx] = float(params_obs["sigma"])
                draws_static["sigma2"][keep_idx] = float(params_obs["sigma2"])
                if self.layout.has_beta:
                    for key in ("beta0", "s_trend", "q_trend"):
                        draws_static[key][keep_idx] = float(params_state[key])
                if self.layout.season_dim > 0:
                    draws_static["gamma0_season"][keep_idx] = params_state["gamma0_season"]
                    for key in ("s_season", "q_season"):
                        draws_static[key][keep_idx] = float(params_state[key])
                if self.priors.lasso is not None:
                    draws_static["lambda2"][keep_idx] = float(lambda2)
                    for block, value in tau.items():
                        draws_static[f"tau_{block}"][keep_idx] = float(value)
                resid = y1 - mu
                logpost[keep_idx] = float(
                    -0.5 * Tn * np.log(2.0 * np.pi * params_obs["sigma2"])
                    - 0.5 * float(resid @ resid) / params_obs["sigma2"]
                )
                keep_idx += 1

        return PosteriorBundle(
            draws_static=draws_static,
            draws_states=draws_states,
            logpost=logpost,
            acceptance={},
            meta={
                "sampler": "noncentered_gaussian_lasso_gibbs",
                "parameterization": "noncentered",
                "n_iter": n_iter,
                "burn": burn,
                "thin": thin,
                "state_method": state_method,
                "state_kwargs": state_kwargs,
                "ncp_state_names": self.layout.ncp_state_names,
                "draws_states_ncp": z_draws,
                "bayesian_lasso": self.priors.lasso is not None,
            },
        )
