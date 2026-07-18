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
    initialise_lasso,
    map_ncp_to_centered,
    measurement_vector,
    mu_from_ncp,
    ncp_particle_state_sample,
    random_sign_switches,
    update_lasso_scales,
)
from .priors import NonCenteredGEVPriors, NormalPrior, UniformPrior

Array = np.ndarray
ParamDict = Dict[str, Any]


def _as_1d_y(y: Array) -> Array:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    raise ValueError("NonCenteredGEVGibbs supports univariate observations only.")


def _normal_logpdf(x: float, mean: float, sd: float) -> float:
    z = (float(x) - float(mean)) / float(sd)
    return float(-0.5 * z * z - np.log(sd) - 0.5 * np.log(2.0 * np.pi))


def _exact_gev_loglik(
    y: Array,
    mu: Array,
    model: StateSpaceModel,
    params_obs: ParamDict,
) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    ll = 0.0
    for yt, mut in zip(y, mu):
        try:
            value = float(model.obs.logpdf(y=float(yt), eta=float(mut), params=params_obs))
        except Exception:
            return -np.inf
        if not np.isfinite(value):
            return -np.inf
        ll += value
    return float(ll)


def _gev_support_ok(y: Array, mu: Array, model: StateSpaceModel, params_obs: ParamDict) -> bool:
    return bool(np.isfinite(_exact_gev_loglik(y, mu, model, params_obs)))


def _laplace_pseudo_mu(
    y: Array,
    mu: Array,
    model: StateSpaceModel,
    params_obs: ParamDict,
    *,
    curvature_floor: float = 1e-10,
    shift_clip: float = 1e6,
) -> tuple[Array, Array]:
    y = np.asarray(y, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    z_star = np.zeros(y.size)
    R_t = np.zeros(y.size)
    for t in range(y.size):
        try:
            grad = float(model.obs.grad_eta(float(y[t]), float(mu[t]), params_obs))
            hess = float(model.obs.hess_eta(float(y[t]), float(mu[t]), params_obs))
        except Exception as exc:
            raise ValueError("Could not construct a valid GEV Laplace approximation.") from exc
        if not np.isfinite(grad) or abs(grad) > 1e6:
            grad = 0.0
        if not np.isfinite(hess) or hess >= -curvature_floor:
            hess = -curvature_floor
        info = max(-hess, curvature_floor)
        shift = np.clip(grad / info, -shift_clip, shift_clip)
        R_t[t] = np.clip(1.0 / info, 1e-12, 1e12)
        z_star[t] = float(mu[t] + shift)
    return z_star, R_t


class NonCenteredGEVGibbs:
    """Approximate DGEV Gibbs sampler with exact observation-parameter MH.

    The latent trajectory and the FS regression block use a local Laplace
    approximation. The ``sigma`` and ``xi`` updates always use the exact GEV
    likelihood and the priors supplied in :class:`NonCenteredGEVPriors`.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        priors: NonCenteredGEVPriors,
        config: GibbsConfig = GibbsConfig(),
        step_log_sigma: float = 0.05,
        step_xi: float = 0.05,
        step_u_xi: Optional[float] = None,
    ) -> None:
        self.model = model
        self.priors = priors
        self.config = config
        self.step_log_sigma = float(step_log_sigma)
        # step_u_xi is retained as a compatibility alias from v0.1.
        self.step_xi = float(step_xi if step_u_xi is None else step_u_xi)
        self.rng = np.random.default_rng(config.seed)
        self.layout: NCPLayout = infer_ncp_layout(model)

    def _log_sigma_prior(self, log_sigma: float) -> float:
        if self.priors.sigma2 is not None:
            # sigma^2 ~ IG(a,b), transformed to log(sigma). Constants omitted.
            a, b = self.priors.sigma2.a, self.priors.sigma2.b
            return float(-2.0 * a * log_sigma - b * np.exp(-2.0 * log_sigma))
        assert self.priors.log_sigma is not None
        return _normal_logpdf(log_sigma, self.priors.log_sigma.mean, self.priors.log_sigma.sd)

    def _xi_prior(self, xi: float) -> float:
        prior = self.priors.xi
        if isinstance(prior, UniformPrior):
            return 0.0 if prior.lower <= xi <= prior.upper else -np.inf
        if isinstance(prior, NormalPrior):
            if abs(xi) > self.priors.xi_max_abs:
                return -np.inf
            return _normal_logpdf(xi, prior.mean, prior.sd)
        raise TypeError("Unsupported xi prior.")

    def _mh_update_log_sigma(
        self,
        y: Array,
        mu: Array,
        params_obs: ParamDict,
    ) -> tuple[ParamDict, bool]:
        cur = float(np.log(float(params_obs["sigma"])))
        prop = cur + float(self.rng.normal(scale=self.step_log_sigma))
        cur_obs = dict(params_obs)
        prop_obs = dict(params_obs)
        prop_obs["sigma"] = float(np.exp(prop))
        ll_cur = _exact_gev_loglik(y, mu, self.model, cur_obs)
        ll_prop = _exact_gev_loglik(y, mu, self.model, prop_obs)
        if not np.isfinite(ll_prop):
            return cur_obs, False
        log_acc = ll_prop + self._log_sigma_prior(prop) - ll_cur - self._log_sigma_prior(cur)
        if np.log(self.rng.random()) < min(0.0, log_acc):
            return prop_obs, True
        return cur_obs, False

    def _mh_update_xi(
        self,
        y: Array,
        mu: Array,
        params_obs: ParamDict,
    ) -> tuple[ParamDict, bool]:
        cur = float(params_obs["xi"])
        prop = cur + float(self.rng.normal(scale=self.step_xi))
        lp_cur, lp_prop = self._xi_prior(cur), self._xi_prior(prop)
        if not np.isfinite(lp_prop):
            return dict(params_obs), False
        cur_obs = dict(params_obs)
        prop_obs = dict(params_obs)
        prop_obs["xi"] = prop
        ll_cur = _exact_gev_loglik(y, mu, self.model, cur_obs)
        ll_prop = _exact_gev_loglik(y, mu, self.model, prop_obs)
        if not np.isfinite(ll_prop):
            return cur_obs, False
        if np.log(self.rng.random()) < min(0.0, ll_prop + lp_prop - ll_cur - lp_cur):
            return prop_obs, True
        return cur_obs, False

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
        Tn, m = y1.size, self.model.state_dim
        if exog is not None:
            raise NotImplementedError("The v0.2 NCP DGEV fitter does not support exog yet.")
        state_kwargs = {} if state_kwargs is None else dict(state_kwargs)

        params_state = canonicalize_ncp_params(init_params_state, self.layout)
        params_obs = dict(init_params_obs)
        if "sigma" not in params_obs or "xi" not in params_obs:
            raise KeyError("init_params_obs must contain both 'sigma' and 'xi'.")
        tau, lambda2 = initialise_lasso(self.priors, self.layout)

        z_path = np.zeros((Tn + 1, self.layout.ncp_state_dim))
        if not _gev_support_ok(y1, mu_from_ncp(z_path, params_state, self.layout), self.model, params_obs):
            raise ValueError(
                "Initial DGEV values violate the GEV support. Use initial values closer to the data."
            )

        n_iter, burn, thin = self.config.n_iter, self.config.burn, self.config.thin
        save_iters = list(range(burn, n_iter, thin))
        save_set = set(save_iters)
        n_keep = len(save_iters)

        draws_states = np.zeros((n_keep, Tn + 1, m))
        z_draws = np.zeros((n_keep, Tn + 1, self.layout.ncp_state_dim))
        draws_static: Dict[str, np.ndarray] = {
            "alpha0": np.zeros(n_keep),
            "s_level": np.zeros(n_keep),
            "q_level": np.zeros(n_keep),
            "sigma": np.zeros(n_keep),
            "sigma2": np.zeros(n_keep),
            "xi": np.zeros(n_keep),
        }
        if self.layout.has_beta:
            draws_static.update(beta0=np.zeros(n_keep), s_trend=np.zeros(n_keep), q_trend=np.zeros(n_keep))
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
        accept_sigma = accept_xi = 0
        keep_idx = 0
        progress_every = self.config.progress_every or max(1, n_iter // 50)
        max_state_tries = int(state_kwargs.get("max_state_tries", 25))
        lasso_var = (
            self.priors.lasso.variance_scale(None) if self.priors.lasso is not None else 1.0
        )

        for it in range(n_iter):
            last_good = (z_path.copy(), dict(params_state), dict(params_obs), dict(tau), float(lambda2))
            iteration_ok = False

            for _ in range(max_state_tries):
                work_state = dict(params_state)
                work_obs = dict(params_obs)
                try:
                    if state_method == "laplace":
                        mu_cur = mu_from_ncp(z_path, work_state, self.layout)
                        z_star, R_t = _laplace_pseudo_mu(
                            y1,
                            mu_cur,
                            self.model,
                            work_obs,
                            curvature_floor=float(state_kwargs.get("curvature_floor", 1e-10)),
                            shift_clip=float(state_kwargs.get("shift_clip", 1e6)),
                        )
                        from .noncentered_utils import baseline_mu_path

                        offset = baseline_mu_path(Tn, work_state, self.layout)
                        cand_z = ffbs_gaussian_1d_tvR(
                            y=z_star - offset,
                            G=G,
                            Q=Q,
                            H=measurement_vector(work_state, self.layout),
                            R_t=R_t,
                            m0=np.zeros(self.layout.ncp_state_dim),
                            C0=float(state_kwargs.get("C0_scale", 1e-6))
                            * np.eye(self.layout.ncp_state_dim),
                            rng=self.rng,
                        )
                    elif state_method == "particle":
                        particle_method = str(state_kwargs.get("particle_method", "bootstrap"))
                        n_particles = int(state_kwargs.get("particle_n_particles", 1000))
                        particle_config = state_kwargs.get("particle_config") or ParticleConfig(
                            n_particles=n_particles, method=particle_method
                        )
                        xs = ncp_particle_state_sample(
                            y=y1,
                            model=self.model,
                            params_state=work_state,
                            params_obs=work_obs,
                            layout=self.layout,
                            rng=self.rng,
                            n_particles=n_particles,
                            method=particle_method,
                            config=particle_config,
                        )
                        cand_z = np.asarray(xs.meta["z_path"], dtype=float)
                        mu_cur = mu_from_ncp(cand_z, work_state, self.layout)
                        z_star, R_t = _laplace_pseudo_mu(y1, mu_cur, self.model, work_obs)
                    else:
                        raise ValueError("state_method must be 'laplace' or 'particle'.")

                    cand_state = dict(work_state)
                    cand_state.update(
                        gev_theta_update(
                            z_pseudo=z_star,
                            R_t=R_t,
                            z_path=cand_z,
                            priors=self.priors,
                            layout=self.layout,
                            rng=self.rng,
                            tau=tau or None,
                            lasso_variance_scale=lasso_var,
                        )
                    )
                    cand_z, cand_state = random_sign_switches(
                        cand_z, cand_state, self.layout, self.rng
                    )
                    mu_exact = mu_from_ncp(cand_z, cand_state, self.layout)
                    if not _gev_support_ok(y1, mu_exact, self.model, work_obs):
                        continue

                    cand_tau, cand_lambda2 = tau, lambda2
                    if self.priors.lasso is not None:
                        cand_tau, cand_lambda2 = update_lasso_scales(
                            cand_state,
                            tau,
                            lambda2,
                            self.priors,
                            self.layout,
                            variance_scale=lasso_var,
                            rng=self.rng,
                        )

                    cand_obs, acc_s = self._mh_update_log_sigma(y1, mu_exact, work_obs)
                    cand_obs, acc_x = self._mh_update_xi(y1, mu_exact, cand_obs)
                    if not _gev_support_ok(y1, mu_exact, self.model, cand_obs):
                        continue

                    z_path, params_state, params_obs = cand_z, cand_state, cand_obs
                    tau, lambda2 = dict(cand_tau), float(cand_lambda2)
                    accept_sigma += int(acc_s)
                    accept_xi += int(acc_x)
                    iteration_ok = True
                    break
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    continue

            if not iteration_ok:
                z_path, params_state, params_obs, tau, lambda2 = last_good

            mu_exact = mu_from_ncp(z_path, params_state, self.layout)
            cur_ll = _exact_gev_loglik(y1, mu_exact, self.model, params_obs)
            x_path = map_ncp_to_centered(z_path, params_state, self.layout)

            if self.config.progress and (((it + 1) % progress_every == 0) or it == n_iter - 1):
                msg = (
                    f"[it {it + 1}/{n_iter}] sigma={params_obs['sigma']:.4f} "
                    f"xi={params_obs['xi']:.4f} Q_level={params_state['q_level']:.3g}"
                )
                if self.layout.has_beta:
                    msg += f" Q_trend={params_state['q_trend']:.3g}"
                if self.layout.season_dim > 0:
                    msg += f" Q_season={params_state['q_season']:.3g}"
                if self.priors.lasso is not None:
                    msg += f" lambda2={lambda2:.3g}"
                if not iteration_ok:
                    msg += " [restored]"
                print(msg)

            if it in save_set:
                draws_states[keep_idx] = x_path
                z_draws[keep_idx] = z_path
                for key in ("alpha0", "s_level", "q_level"):
                    draws_static[key][keep_idx] = float(params_state[key])
                draws_static["sigma"][keep_idx] = float(params_obs["sigma"])
                draws_static["sigma2"][keep_idx] = float(params_obs["sigma"]) ** 2
                draws_static["xi"][keep_idx] = float(params_obs["xi"])
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
                logpost[keep_idx] = cur_ll
                keep_idx += 1

        return PosteriorBundle(
            draws_static=draws_static,
            draws_states=draws_states,
            logpost=logpost,
            acceptance={
                "sigma_mh": accept_sigma / max(n_iter, 1),
                "xi_mh": accept_xi / max(n_iter, 1),
            },
            meta={
                "sampler": "noncentered_gev_laplace_lasso_gibbs",
                "parameterization": "noncentered",
                "n_iter": n_iter,
                "burn": burn,
                "thin": thin,
                "state_method": state_method,
                "state_kwargs": state_kwargs,
                "ncp_state_names": self.layout.ncp_state_names,
                "draws_states_ncp": z_draws,
                "step_log_sigma": self.step_log_sigma,
                "step_xi": self.step_xi,
                "bayesian_lasso": self.priors.lasso is not None,
            },
        )
