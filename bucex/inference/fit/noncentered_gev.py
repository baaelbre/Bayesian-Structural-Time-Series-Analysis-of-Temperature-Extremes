from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...core.results import PosteriorBundle
from ...models.base import StateSpaceModel
from ..state.particle import ParticleConfig
from .base import GibbsConfig
from .noncentered_utils import (
    NCPLayout,
    build_ncp_system,
    canonicalize_ncp_params,
    ffbs_gaussian_1d_tvR,
    gev_theta_update,
    infer_ncp_layout,
    map_ncp_to_centered,
    measurement_vector,
    mu_from_ncp,
    ncp_particle_filter,
    ncp_particle_state_sample,
    random_sign_switches,
)
from .priors import NonCenteredGEVPriors

Array = np.ndarray
ParamDict = Dict[str, Any]


def _as_1d_y(y: Array) -> Array:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    raise ValueError("NonCenteredGEVGibbs currently supports univariate observations only.")


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


def _exact_gev_loglik(y: Array, mu: Array, model: StateSpaceModel, params_obs: ParamDict) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    ll = 0.0
    for yt, mut in zip(y, mu):
        try:
            ll += float(model.obs.logpdf(y=float(yt), eta=float(mut), params=params_obs))
        except Exception:
            return -np.inf
    return float(ll)




def _gev_support_ok(y: Array, mu: Array, model: StateSpaceModel, params_obs: ParamDict) -> bool:
    ll = _exact_gev_loglik(y, mu, model, params_obs)
    return bool(np.isfinite(ll))

def _transformed_obs_logpost(log_sigma: float, u_xi: float, y: Array, mu: Array, model: StateSpaceModel, priors: NonCenteredGEVPriors) -> float:
    sigma = float(np.exp(log_sigma))
    xi = _u_to_xi(u_xi, priors.xi_max_abs)
    ll = _exact_gev_loglik(y, mu, model, {"sigma": sigma, "xi": xi})
    if not np.isfinite(ll):
        return -np.inf
    lp = _normal_logpdf(log_sigma, priors.log_sigma.mean, priors.log_sigma.sd)
    lp += _normal_logpdf(xi, priors.xi.mean, priors.xi.sd)
    lp += _log_abs_dxi_du(u_xi, priors.xi_max_abs)
    return float(ll + lp)


def _laplace_pseudo_mu(y: Array, mu: Array, model: StateSpaceModel, params_obs: ParamDict, curvature_floor: float = 1e-8) -> tuple[Array, Array]:
    y = np.asarray(y, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Tn = y.size
    z_star = np.zeros(Tn, dtype=float)
    R_t = np.zeros(Tn, dtype=float)
    for t in range(Tn):
        grad = float(model.obs.grad_eta(float(y[t]), float(mu[t]), params_obs))
        hess = float(model.obs.hess_eta(float(y[t]), float(mu[t]), params_obs))
        info = max(-hess, float(curvature_floor))
        R_t[t] = 1.0 / info
        z_star[t] = float(mu[t]) + grad / info
    return z_star, R_t


class NonCenteredGEVGibbs:
    def __init__(
        self,
        model: StateSpaceModel,
        priors: NonCenteredGEVPriors,
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
        self.layout: NCPLayout = infer_ncp_layout(model)

    def _mh_update_obs_params(self, y: Array, mu: Array, params_obs: ParamDict) -> tuple[ParamDict, bool]:
        cur_sigma = float(params_obs["sigma"])
        cur_xi = float(params_obs["xi"])
        if cur_sigma <= 0.0:
            raise ValueError("Current sigma must be > 0 for GEV MH update.")
        cur_log_sigma = float(np.log(cur_sigma))
        cur_u_xi = _xi_to_u(cur_xi, self.priors.xi_max_abs)
        prop_log_sigma = cur_log_sigma + self.rng.normal(scale=self.step_log_sigma)
        prop_u_xi = cur_u_xi + self.rng.normal(scale=self.step_u_xi)
        logp_cur = _transformed_obs_logpost(cur_log_sigma, cur_u_xi, y, mu, self.model, self.priors)
        logp_prop = _transformed_obs_logpost(prop_log_sigma, prop_u_xi, y, mu, self.model, self.priors)
        if np.log(self.rng.random()) < (logp_prop - logp_cur):
            return {"sigma": float(np.exp(prop_log_sigma)), "xi": float(_u_to_xi(prop_u_xi, self.priors.xi_max_abs))}, True
        return dict(params_obs), False

    def fit(
        self,
        y: Array,
        init_params_state: ParamDict,
        init_params_obs: ParamDict,
        exog: Optional[Array] = None,
        state_method: str = "laplace",
        state_kwargs: Optional[dict[str, Any]] = None,
    ) -> PosteriorBundle:
        y1 = _as_1d_y(y)
        Tn = y1.size
        m = self.model.state_dim

        if exog is not None:
            raise NotImplementedError("NonCenteredGEVGibbs does not support exog yet.")
        if state_kwargs is None:
            state_kwargs = {}

        params_state = canonicalize_ncp_params(init_params_state, self.layout)
        params_obs = dict(init_params_obs)
        if "sigma" not in params_obs or "xi" not in params_obs:
            raise KeyError("init_params_obs must contain both 'sigma' and 'xi'.")

        z_path = np.zeros((Tn + 1, self.layout.ncp_state_dim), dtype=float)
        if not _gev_support_ok(y1, mu_from_ncp(z_path, params_state, self.layout), self.model, params_obs):
            raise ValueError(
                "Initial non-centred GEV state/parameter values violate the GEV support. "
                "Choose init_params_state/init_params_obs closer to the data."
            )

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
            "xi": np.zeros(n_keep, dtype=float),
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
        accept_obs = 0

        G, Q = build_ncp_system(self.layout)
        keep_idx = 0
        progress_every = self.config.progress_every if self.config.progress_every > 0 else max(1, n_iter // 50)
        max_state_tries = int(state_kwargs.get("max_state_tries", 20))

        cur_ll = _exact_gev_loglik(y1, mu_from_ncp(z_path, params_state, self.layout), self.model, params_obs)

        for it in range(n_iter):
            accepted = False

            for _ in range(max_state_tries):
                work_state = dict(params_state)
                work_obs = dict(params_obs)

                if state_method == "laplace":
                    mu_cur = mu_from_ncp(z_path, work_state, self.layout)
                    if not _gev_support_ok(y1, mu_cur, self.model, work_obs):
                        break
                    try:
                        z_star, R_t = _laplace_pseudo_mu(y1, mu_cur, self.model, work_obs)
                    except ValueError:
                        break
                    from .noncentered_utils import baseline_mu_path
                    offset = baseline_mu_path(Tn, work_state, self.layout)
                    y_ncp = z_star - offset
                    H = measurement_vector(work_state, self.layout)
                    cand_z = ffbs_gaussian_1d_tvR(
                        y=y_ncp,
                        G=G,
                        Q=Q,
                        H=H,
                        R_t=R_t,
                        m0=np.zeros(self.layout.ncp_state_dim, dtype=float),
                        C0=1e-6 * np.eye(self.layout.ncp_state_dim),
                        rng=self.rng,
                    )
                elif state_method == "particle":
                    particle_method = str(state_kwargs.get("particle_method", "bootstrap"))
                    particle_n_particles = int(state_kwargs.get("particle_n_particles", 1000))
                    particle_config = state_kwargs.get("particle_config", None)
                    if particle_config is None:
                        particle_config = ParticleConfig(n_particles=particle_n_particles, method=particle_method)
                    xs = ncp_particle_state_sample(
                        y=y1,
                        model=self.model,
                        params_state=work_state,
                        params_obs=work_obs,
                        layout=self.layout,
                        rng=self.rng,
                        n_particles=particle_n_particles,
                        method=particle_method,
                        config=particle_config,
                    )
                    cand_z = np.asarray(xs.meta["z_path"], dtype=float)
                    mu_cur = mu_from_ncp(cand_z, work_state, self.layout)
                    if not _gev_support_ok(y1, mu_cur, self.model, work_obs):
                        continue
                    try:
                        z_star, R_t = _laplace_pseudo_mu(y1, mu_cur, self.model, work_obs)
                    except ValueError:
                        continue
                else:
                    raise ValueError("state_method must be 'laplace' or 'particle'.")

                theta_draw = gev_theta_update(
                    z_pseudo=z_star,
                    R_t=R_t,
                    z_path=cand_z,
                    priors=self.priors,
                    layout=self.layout,
                    rng=self.rng,
                )
                cand_state = dict(work_state)
                cand_state.update(theta_draw)
                cand_state["q_level"] = float(cand_state["s_level"]) ** 2
                cand_state["q_trend"] = float(cand_state.get("s_trend", 0.0)) ** 2
                cand_state["q_season"] = float(cand_state.get("s_season", 0.0)) ** 2

                cand_z, cand_state = random_sign_switches(cand_z, cand_state, self.layout, self.rng)
                mu_exact = mu_from_ncp(cand_z, cand_state, self.layout)
                if not _gev_support_ok(y1, mu_exact, self.model, work_obs):
                    continue

                cand_obs, obs_acc = self._mh_update_obs_params(y1, mu_exact, work_obs)
                if not _gev_support_ok(y1, mu_exact, self.model, cand_obs):
                    continue

                z_path = cand_z
                params_state = cand_state
                params_obs = cand_obs
                accept_obs += int(obs_acc)
                cur_ll = _exact_gev_loglik(y1, mu_exact, self.model, params_obs)
                accepted = True
                break

            if not accepted:
                mu_exact = mu_from_ncp(z_path, params_state, self.layout)
                cur_ll = _exact_gev_loglik(y1, mu_exact, self.model, params_obs)

            x_path = map_ncp_to_centered(z_path, params_state, self.layout)

            if self.config.progress and (((it + 1) % progress_every == 0) or (it == n_iter - 1)):
                msg = f"[it {it+1}/{n_iter}] sigma={params_obs['sigma']:.4f} xi={params_obs['xi']:.4f} alpha0={params_state['alpha0']:.4g} s_level={params_state['s_level']:.4g} loglik={cur_ll:.2f}"
                if self.layout.has_beta:
                    msg += f" beta0={params_state['beta0']:.4g} s_trend={params_state['s_trend']:.4g}"
                if self.layout.season_dim > 0:
                    msg += f" s_season={params_state['s_season']:.4g}"
                if state_method == "particle":
                    msg += f" pf={state_kwargs.get('particle_method', 'bootstrap')}"
                if not accepted:
                    msg += " [restore/skip]"
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
                draws_static["xi"][keep_idx] = float(params_obs["xi"])
                if self.layout.has_beta:
                    draws_static["s_trend"][keep_idx] = float(params_state["s_trend"])
                    draws_static["q_trend"][keep_idx] = float(params_state["q_trend"])
                if self.layout.season_dim > 0:
                    draws_static["gamma0_season"][keep_idx] = np.asarray(params_state["gamma0_season"], dtype=float)
                    draws_static["s_season"][keep_idx] = float(params_state["s_season"])
                    draws_static["q_season"][keep_idx] = float(params_state["q_season"])
                logpost[keep_idx] = float(cur_ll)
                keep_idx += 1


        return PosteriorBundle(
            draws_static=draws_static,
            draws_states=draws_states,
            logpost=logpost,
            acceptance={"obs_mh": accept_obs / n_iter},
            meta={
                "sampler": "noncentered_gev_gibbs",
                "parameterization": "noncentered",
                "n_iter": n_iter,
                "burn": burn,
                "thin": thin,
                "state_method": state_method,
                "state_kwargs": state_kwargs,
                "ncp_state_names": self.layout.ncp_state_names,
                "draws_states_ncp": z_draws,
                "step_log_sigma": self.step_log_sigma,
                "step_u_xi": self.step_u_xi,
            },
        )
