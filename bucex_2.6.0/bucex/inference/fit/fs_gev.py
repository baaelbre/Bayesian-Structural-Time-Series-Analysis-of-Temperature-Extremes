from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np

from ._fs_output import FSOutput
from ._progress import (
    mcmc_progress_line,
    progress_interval,
    should_report_progress,
    univariate_progress_parameters,
)
from ...models.base import StateSpaceModel
from ..config import GibbsConfig, MCMC
from .fs_utils import (
    NCPLayout,
    apply_theta_draw,
    asis_centered_scale_update,
    build_ncp_system,
    canonicalize_ncp_params,
    copy_horseshoe_state,
    copy_triple_gamma_state,
    copy_lasso_lambda2,
    design_matrix_ncp,
    elliptical_slice_gaussian_prior,
    fs_theta_prior,
    gev_theta_update,
    infer_ncp_layout,
    initialise_horseshoe,
    initialise_triple_gamma,
    initialise_lasso,
    iterated_laplace_ncp,
    map_centered_to_ncp,
    map_ncp_to_centered,
    mu_from_ncp,
    ncp_pgas,
    random_sign_switches,
    theta_vector_from_params,
    update_horseshoe_scales,
    update_triple_gamma_scales,
    update_lasso_scales,
    update_pc_scales,
)
from .model_space import (
    enforce_structural_state,
    enumerate_structural_models,
    initial_structural_state,
    sample_structural_regression,
    sample_structural_regression_exact,
    structural_model_log_prior,
)
from ...priors.structural import FSGEVPriors, NormalPrior, UniformPrior
from .utils import as_1d_observations

Array = np.ndarray
ParamDict = Dict[str, Any]


def _compact_counter(counter: Counter[str], *, limit: int = 6) -> str:
    """Format failure counts for one-line MCMC progress messages."""
    if not counter:
        return "none"
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    shown = items[: max(1, int(limit))]
    text = ",".join(f"{key}:{value}" for key, value in shown)
    if len(items) > len(shown):
        text += f",other:{sum(value for _, value in items[len(shown):])}"
    return text


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


class FSGEVKernel:
    """DGEV sampler in the Fruehwirth--Schnatter parameterisation.

    ``state_method="pgas"`` combines exact-invariant PGAS state updates with
    an exact GEV elliptical-slice update of the FS regression block.
    ``state_method="laplace"`` retains the fast, explicitly approximate
    iterated-Laplace engine. Observation parameters always use the exact GEV
    likelihood.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        priors: FSGEVPriors,
        config: GibbsConfig | MCMC = GibbsConfig(),
        step_log_sigma: float = 0.05,
        step_xi: float = 0.05,
    ) -> None:
        self.model = model
        self.priors = priors
        self.config = config
        self.step_log_sigma = float(step_log_sigma)
        self.step_xi = float(step_xi)
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
    ) -> FSOutput:
        y1 = as_1d_observations(y, family="GEV")
        Tn, m = y1.size, self.model.state_dim
        if exog is not None:
            raise NotImplementedError("The FS GEV strategy does not support exog.")
        state_kwargs = {} if state_kwargs is None else dict(state_kwargs)
        initial_centered_path = state_kwargs.pop("initial_centered_path", None)
        if state_method == "particle":
            state_method = "pgas"
        if state_method not in {"laplace", "pgas"}:
            raise ValueError("state_method must be 'laplace' or 'pgas'.")
        use_asis = bool(state_kwargs.get("asis", False))
        if use_asis and self.priors.ssvs is not None:
            raise ValueError("ASIS is not combined with structural SSVS.")

        params_state = canonicalize_ncp_params(init_params_state, self.layout)
        params_obs = dict(init_params_obs)
        if "sigma" not in params_obs or "xi" not in params_obs:
            raise KeyError("init_params_obs must contain both 'sigma' and 'xi'.")
        tau, lambda2 = initialise_lasso(self.priors, self.layout)
        horseshoe_state = initialise_horseshoe(self.priors, self.layout)
        triple_gamma_state = initialise_triple_gamma(self.priors, self.layout)
        model_state = initial_structural_state(params_state, self.layout)
        model_candidates = enumerate_structural_models(
            self.layout, None if self.priors.ssvs is None else self.priors.ssvs
        )
        if self.priors.ssvs is not None and not np.isfinite(
            structural_model_log_prior(model_state, self.priors.ssvs)
        ):
            valid_states = [
                state
                for state in model_candidates
                if np.isfinite(structural_model_log_prior(state, self.priors.ssvs))
            ]
            if not valid_states:
                raise ValueError("The SSVS prior assigns zero mass to every model.")
            model_state = max(
                valid_states,
                key=lambda state: structural_model_log_prior(
                    state, self.priors.ssvs
                ),
            )
            params_state = enforce_structural_state(
                params_state, model_state, self.layout
            )
        model_index = (
            model_candidates.index(model_state)
            if self.priors.ssvs is not None
            else -1
        )

        if initial_centered_path is None:
            z_path = np.zeros((Tn + 1, self.layout.ncp_state_dim))
        else:
            centered = np.asarray(initial_centered_path, dtype=float)
            expected = (Tn + 1, self.layout.centered_state_dim)
            if centered.shape != expected:
                raise ValueError(
                    f"initial_centered_path must have shape {expected}; got {centered.shape}."
                )
            z_path = map_centered_to_ncp(centered, params_state, self.layout)
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
            if isinstance(lambda2, dict):
                for block in tau:
                    draws_static[f"lambda2_{block}"] = np.zeros(n_keep)
            else:
                draws_static["lambda2"] = np.zeros(n_keep)
            for block in tau:
                draws_static[f"tau_{block}"] = np.zeros(n_keep)
        if self.priors.horseshoe is not None:
            draws_static["horseshoe_global"] = np.zeros(n_keep)
            draws_static["horseshoe_slab2"] = np.zeros(n_keep)
            for block in horseshoe_state["local"]:
                draws_static[f"horseshoe_local_{block}"] = np.zeros(n_keep)
        if self.priors.triple_gamma is not None:
            draws_static["triple_gamma_global"] = np.zeros(n_keep)
            draws_static["triple_gamma_a"] = np.zeros(n_keep)
            draws_static["triple_gamma_c"] = np.zeros(n_keep)
            if self.priors.triple_gamma.regularized:
                draws_static["triple_gamma_slab2"] = np.zeros(n_keep)
            for block in triple_gamma_state["numerator"]:
                draws_static[f"triple_gamma_numerator_{block}"] = np.zeros(n_keep)
                draws_static[f"triple_gamma_denominator_{block}"] = np.zeros(n_keep)
                draws_static[f"triple_gamma_rho_{block}"] = np.zeros(n_keep)
        if self.priors.pc is not None:
            for block in tau:
                draws_static[f"pc_tau_{block}"] = np.zeros(n_keep)
        if self.priors.ssvs is not None:
            draws_static["state_level"] = np.zeros(n_keep, dtype=np.int8)
            draws_static["state_trend"] = np.zeros(n_keep, dtype=np.int8)
            draws_static["state_season"] = np.zeros(n_keep, dtype=np.int8)
            draws_static["model_index"] = np.zeros(n_keep, dtype=np.int16)

        logpost = np.full(n_keep, np.nan)
        G, Q = build_ncp_system(self.layout)
        accept_sigma = accept_xi = 0
        accept_ssvs_model = accept_ssvs_model_change = propose_ssvs_model_change = 0
        keep_idx = 0
        progress_every = progress_interval(n_iter, self.config.progress_every)
        chain_started = perf_counter()
        progress_chain = int(state_kwargs.get("_progress_chain", 1))
        progress_chains = int(state_kwargs.get("_progress_chains", 1))
        progress_label = str(state_kwargs.get("_progress_label", "univariate"))
        restored_iterations = 0
        attempt_failure_totals: Counter[str] = Counter()
        restore_failure_totals: Counter[str] = Counter()
        window_attempt_failures: Counter[str] = Counter()
        window_restored = 0
        window_iterations = 0
        max_state_tries = int(state_kwargs.get("max_state_tries", 25))
        lasso_var = (
            self.priors.lasso.variance_scale(None) if self.priors.lasso is not None else 1.0
        )
        horseshoe_accepts: Counter[str] = Counter()
        triple_gamma_moves: Counter[str] = Counter()
        asis_accepts: Counter[str] = Counter()
        sign_switch_counts: Counter[str] = Counter()
        engine_diagnostics: dict[str, list[float]] = {
            "laplace_iterations": [],
            "laplace_converged": [],
            "laplace_relative_change": [],
            "laplace_support_rejections": [],
            "particle_min_ess": [],
            "particle_mean_unique_ancestors": [],
            "particle_path_changed": [],
            "particle_path_update_fraction": [],
            "particle_changed_fraction": [],
            "particle_reference_ancestor_change_fraction": [],
            "fs_elliptical_slice_steps": [],
            "ssvs_model_move_accepted": [],
            "ssvs_model_proposed_change": [],
            "ssvs_model_log_acceptance_ratio": [],
            "sign_invariance_error": [],
        }

        for it in range(n_iter):
            window_iterations += 1
            last_good = (
                z_path.copy(), dict(params_state), dict(params_obs), dict(tau),
                copy_lasso_lambda2(lambda2), copy_horseshoe_state(horseshoe_state),
                copy_triple_gamma_state(triple_gamma_state),
                model_state, int(model_index)
            )
            iteration_ok = False
            iteration_failures: Counter[str] = Counter()
            last_failure_detail = ""
            successful_laplace_result = None
            successful_pgas_result = None
            successful_fs_slice_steps = np.nan
            successful_ssvs_move_accepted = np.nan
            successful_ssvs_proposed_change = np.nan
            successful_ssvs_log_acceptance_ratio = np.nan
            successful_horseshoe_outcomes: dict[str, bool] = {}
            successful_triple_gamma_outcomes: dict[str, bool] = {}
            successful_asis_outcomes: dict[str, bool] = {}
            successful_sign_switches: dict[str, bool] = {}
            successful_sign_invariance_error = np.nan

            for attempt in range(max_state_tries):
                work_state = dict(params_state)
                work_obs = dict(params_obs)
                stage = "state_update"
                try:
                    laplace_result = None
                    pgas_result = None
                    if state_method == "laplace":
                        stage = "iterated_laplace_state_update"
                        laplace_result = iterated_laplace_ncp(
                            y1,
                            self.model,
                            work_state,
                            work_obs,
                            self.layout,
                            initial_path=z_path,
                            rng=self.rng,
                            max_iterations=int(state_kwargs.get("laplace_max_iterations", 30)),
                            tolerance=float(state_kwargs.get("laplace_tolerance", 1e-5)),
                            curvature_floor=float(state_kwargs.get("curvature_floor", 1e-6)),
                            maximum_variance=float(state_kwargs.get("maximum_variance", 1e8)),
                            shift_limit=state_kwargs.get("shift_limit"),
                            draw_attempts=int(state_kwargs.get("draw_attempts", 30)),
                        )
                        cand_z = laplace_result.z_path
                        z_star = laplace_result.pseudo_y
                        R_t = laplace_result.pseudo_variance
                    else:
                        stage = "pgas_state_update"
                        pgas_result = ncp_pgas(
                            y1,
                            self.model,
                            work_state,
                            work_obs,
                            self.layout,
                            z_path,
                            n_particles=int(
                                state_kwargs.get(
                                    "particles",
                                    state_kwargs.get("particle_n_particles", 256),
                                )
                            ),
                            proposal=str(state_kwargs.get("particle_proposal", "guided")),
                            rng=self.rng,
                        )
                        cand_z = pgas_result.z_path

                    stage = "structural_parameter_update"
                    cand_state = dict(work_state)
                    cand_model_state = model_state
                    cand_model_index = -1
                    fs_slice_steps = np.nan
                    if self.priors.ssvs is not None and state_method == "laplace":
                        X, theta_names, tbar = design_matrix_ncp(
                            cand_z, self.layout, center_time=True
                        )
                        selection = sample_structural_regression(
                            y=z_star,
                            X=X,
                            theta_names=theta_names,
                            tbar=tbar,
                            noise_variance=R_t,
                            priors=self.priors,
                            layout=self.layout,
                            rng=self.rng,
                            apply_theta_draw=apply_theta_draw,
                        )
                        cand_state.update(selection.params_state)
                        cand_model_state = selection.state
                        cand_model_index = selection.selected_index
                    elif self.priors.ssvs is not None:
                        X, theta_names, tbar = design_matrix_ncp(
                            cand_z, self.layout, center_time=True
                        )
                        # This Gaussian approximation is only an independence
                        # proposal.  The selection kernel below applies the
                        # exact GEV likelihood and both forward/reverse proposal
                        # densities in its trans-dimensional MH ratio.
                        pseudo_y, pseudo_variance = _laplace_pseudo_mu(
                            y1,
                            y1,
                            self.model,
                            work_obs,
                            curvature_floor=float(
                                state_kwargs.get("curvature_floor", 1e-6)
                            ),
                        )

                        def exact_structural_loglik(eta):
                            return _exact_gev_loglik(
                                y1,
                                np.asarray(eta, dtype=float),
                                self.model,
                                work_obs,
                            )

                        selection = sample_structural_regression_exact(
                            y=y1,
                            X=X,
                            theta_names=theta_names,
                            tbar=tbar,
                            pseudo_y=pseudo_y,
                            pseudo_variance=pseudo_variance,
                            current_state=model_state,
                            current_params_state=work_state,
                            priors=self.priors,
                            layout=self.layout,
                            rng=self.rng,
                            apply_theta_draw=apply_theta_draw,
                            theta_vector_from_params=theta_vector_from_params,
                            elliptical_slice=elliptical_slice_gaussian_prior,
                            log_likelihood=exact_structural_loglik,
                            proposal_uniform_weight=float(
                                state_kwargs.get(
                                    "ssvs_proposal_uniform_weight", 0.05
                                )
                            ),
                            elliptical_slice_max_steps=int(
                                state_kwargs.get(
                                    "elliptical_slice_max_steps", 10_000
                                )
                            ),
                        )
                        cand_state.update(selection.params_state)
                        cand_model_state = selection.state
                        cand_model_index = selection.selected_index
                        fs_slice_steps = float(selection.elliptical_slice_steps)
                        cand_ssvs_move_accepted = float(
                            selection.move_accepted
                        )
                        cand_ssvs_proposed_change = float(
                            selection.proposed_index != model_index
                        )
                        cand_ssvs_log_acceptance_ratio = float(
                            selection.log_acceptance_ratio
                        )
                    elif state_method == "laplace":
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
                                horseshoe_state=horseshoe_state or None,
                                triple_gamma_state=triple_gamma_state or None,
                            )
                        )
                    else:
                        X, theta_names, tbar = design_matrix_ncp(
                            cand_z, self.layout, center_time=True
                        )
                        prior_mean, prior_covariance = fs_theta_prior(
                            self.priors,
                            self.layout,
                            theta_names,
                            tbar=tbar,
                            tau=tau or None,
                            lasso_variance_scale=lasso_var,
                            horseshoe_state=horseshoe_state or None,
                            triple_gamma_state=triple_gamma_state or None,
                        )
                        current_theta = theta_vector_from_params(
                            work_state,
                            theta_names,
                            self.layout,
                            tbar=tbar,
                        )

                        def exact_theta_loglik(theta):
                            return _exact_gev_loglik(
                                y1, X @ np.asarray(theta, dtype=float), self.model, work_obs
                            )

                        theta, fs_slice_steps = elliptical_slice_gaussian_prior(
                            current_theta,
                            prior_mean,
                            prior_covariance,
                            exact_theta_loglik,
                            self.rng,
                            max_steps=int(state_kwargs.get("elliptical_slice_max_steps", 10_000)),
                        )
                        cand_state.update(
                            apply_theta_draw(
                                work_state,
                                theta,
                                theta_names,
                                self.layout,
                                tbar=tbar,
                            )
                        )
                    cand_asis_outcomes: dict[str, bool] = {}
                    if use_asis:
                        stage = "asis_centered_scale_update"
                        cand_z, cand_state, cand_asis_outcomes = (
                            asis_centered_scale_update(
                                cand_z,
                                cand_state,
                                self.priors,
                                self.layout,
                                self.rng,
                                tau=tau or None,
                                lasso_variance_scale=lasso_var,
                                horseshoe_state=horseshoe_state or None,
                                triple_gamma_state=triple_gamma_state or None,
                                proposal_step=float(state_kwargs.get("asis_step", 0.20)),
                            )
                        )

                    stage = "sign_switch"
                    predictor_before_sign = mu_from_ncp(
                        cand_z, cand_state, self.layout
                    )
                    cand_z, cand_state, cand_sign_switches = random_sign_switches(
                        cand_z,
                        cand_state,
                        self.layout,
                        self.rng,
                        return_switches=True,
                    )
                    predictor_after_sign = mu_from_ncp(
                        cand_z, cand_state, self.layout
                    )
                    cand_sign_error = float(
                        np.max(
                            np.abs(
                                predictor_before_sign - predictor_after_sign
                            )
                        )
                    )
                    sign_tolerance = 1e-10 * (
                        1.0 + float(np.max(np.abs(predictor_before_sign)))
                    )
                    if cand_sign_error > sign_tolerance:
                        raise RuntimeError(
                            "FS sign switch changed the GEV predictor by "
                            f"{cand_sign_error:.3g}."
                        )
                    stage = "support_after_state_parameters"
                    mu_exact = mu_from_ncp(cand_z, cand_state, self.layout)
                    if not _gev_support_ok(y1, mu_exact, self.model, work_obs):
                        reason = "support_after_state_parameters"
                        iteration_failures[reason] += 1
                        attempt_failure_totals[reason] += 1
                        window_attempt_failures[reason] += 1
                        last_failure_detail = f"attempt={attempt + 1}:{reason}"
                        continue

                    stage = "shrinkage_update"
                    cand_tau, cand_lambda2 = tau, lambda2
                    cand_horseshoe = copy_horseshoe_state(horseshoe_state)
                    cand_horseshoe_outcomes: dict[str, bool] = {}
                    cand_triple_gamma = copy_triple_gamma_state(
                        triple_gamma_state
                    )
                    cand_triple_gamma_outcomes: dict[str, bool] = {}
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
                    if self.priors.horseshoe is not None:
                        cand_horseshoe, cand_horseshoe_outcomes = (
                            update_horseshoe_scales(
                                cand_state,
                                horseshoe_state,
                                self.priors,
                                self.layout,
                                rng=self.rng,
                                step_local=float(
                                    state_kwargs.get("horseshoe_step_local", 0.35)
                                ),
                                step_global=float(
                                    state_kwargs.get("horseshoe_step_global", 0.25)
                                ),
                                step_slab=float(
                                    state_kwargs.get("horseshoe_step_slab", 0.20)
                                ),
                            )
                        )
                    if self.priors.triple_gamma is not None:
                        cand_triple_gamma, cand_triple_gamma_outcomes = (
                            update_triple_gamma_scales(
                                cand_state,
                                triple_gamma_state,
                                self.priors,
                                self.layout,
                                rng=self.rng,
                                width_local=float(
                                    state_kwargs.get(
                                        "triple_gamma_width_local", 1.0
                                    )
                                ),
                                width_global=float(
                                    state_kwargs.get(
                                        "triple_gamma_width_global", 1.0
                                    )
                                ),
                                width_shape=float(
                                    state_kwargs.get(
                                        "triple_gamma_width_shape", 0.8
                                    )
                                ),
                                width_slab=float(
                                    state_kwargs.get(
                                        "triple_gamma_width_slab", 0.8
                                    )
                                ),
                            )
                        )
                    if self.priors.pc is not None:
                        cand_tau = update_pc_scales(
                            cand_state,
                            tau,
                            self.priors,
                            self.layout,
                            rng=self.rng,
                        )

                    stage = "observation_parameter_update"
                    cand_obs, acc_s = self._mh_update_log_sigma(y1, mu_exact, work_obs)
                    cand_obs, acc_x = self._mh_update_xi(y1, mu_exact, cand_obs)
                    stage = "support_after_observation_parameters"
                    if not _gev_support_ok(y1, mu_exact, self.model, cand_obs):
                        reason = "support_after_observation_parameters"
                        iteration_failures[reason] += 1
                        attempt_failure_totals[reason] += 1
                        window_attempt_failures[reason] += 1
                        last_failure_detail = f"attempt={attempt + 1}:{reason}"
                        continue

                    z_path, params_state, params_obs = cand_z, cand_state, cand_obs
                    tau, lambda2 = dict(cand_tau), copy_lasso_lambda2(cand_lambda2)
                    horseshoe_state = copy_horseshoe_state(cand_horseshoe)
                    triple_gamma_state = copy_triple_gamma_state(
                        cand_triple_gamma
                    )
                    model_state, model_index = cand_model_state, int(cand_model_index)
                    accept_sigma += int(acc_s)
                    accept_xi += int(acc_x)
                    successful_laplace_result = laplace_result
                    successful_pgas_result = pgas_result
                    successful_fs_slice_steps = float(fs_slice_steps)
                    if self.priors.ssvs is not None and state_method == "pgas":
                        successful_ssvs_move_accepted = cand_ssvs_move_accepted
                        successful_ssvs_proposed_change = cand_ssvs_proposed_change
                        successful_ssvs_log_acceptance_ratio = (
                            cand_ssvs_log_acceptance_ratio
                        )
                        accept_ssvs_model += int(cand_ssvs_move_accepted)
                        accept_ssvs_model_change += int(
                            cand_ssvs_move_accepted
                            and cand_ssvs_proposed_change
                        )
                        propose_ssvs_model_change += int(
                            cand_ssvs_proposed_change
                        )
                    successful_horseshoe_outcomes = cand_horseshoe_outcomes
                    successful_triple_gamma_outcomes = (
                        cand_triple_gamma_outcomes
                    )
                    successful_asis_outcomes = cand_asis_outcomes
                    successful_sign_switches = cand_sign_switches
                    successful_sign_invariance_error = cand_sign_error
                    for key, accepted in successful_horseshoe_outcomes.items():
                        horseshoe_accepts[key] += int(accepted)
                    for key, moved in successful_triple_gamma_outcomes.items():
                        triple_gamma_moves[key] += int(moved)
                    for key, accepted in successful_asis_outcomes.items():
                        asis_accepts[key] += int(accepted)
                    iteration_ok = True
                    break
                except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                    reason = f"{stage}:{type(exc).__name__}"
                    iteration_failures[reason] += 1
                    attempt_failure_totals[reason] += 1
                    window_attempt_failures[reason] += 1
                    detail = str(exc).replace("\n", " ").strip()
                    if len(detail) > 140:
                        detail = detail[:137] + "..."
                    last_failure_detail = f"attempt={attempt + 1}:{reason}"
                    if detail:
                        last_failure_detail += f":{detail}"
                    continue

            if not iteration_ok:
                (
                    z_path, params_state, params_obs, tau, lambda2,
                    horseshoe_state, triple_gamma_state, model_state, model_index
                ) = last_good
                restored_iterations += 1
                window_restored += 1
                restore_failure_totals.update(iteration_failures)

            if successful_laplace_result is not None:
                engine_diagnostics["laplace_iterations"].append(
                    float(successful_laplace_result.iterations)
                )
                engine_diagnostics["laplace_converged"].append(
                    float(successful_laplace_result.converged)
                )
                engine_diagnostics["laplace_relative_change"].append(
                    float(successful_laplace_result.relative_change)
                )
                engine_diagnostics["laplace_support_rejections"].append(
                    float(successful_laplace_result.support_rejections)
                )
            else:
                for name in (
                    "laplace_iterations",
                    "laplace_converged",
                    "laplace_relative_change",
                    "laplace_support_rejections",
                ):
                    engine_diagnostics[name].append(np.nan)
            if successful_pgas_result is not None:
                engine_diagnostics["particle_min_ess"].append(
                    float(np.min(successful_pgas_result.ess[1:]))
                )
                engine_diagnostics["particle_mean_unique_ancestors"].append(
                    float(np.mean(successful_pgas_result.unique_ancestors[1:]))
                )
                engine_diagnostics["particle_path_changed"].append(
                    float(successful_pgas_result.path_changed)
                )
                update_fraction = float(
                    successful_pgas_result.path_update_fraction
                )
                engine_diagnostics["particle_path_update_fraction"].append(
                    update_fraction
                )
                engine_diagnostics["particle_changed_fraction"].append(
                    update_fraction
                )
                conditioned = int(
                    state_kwargs.get(
                        "particles",
                        state_kwargs.get("particle_n_particles", 256),
                    )
                ) - 1
                engine_diagnostics[
                    "particle_reference_ancestor_change_fraction"
                ].append(
                    float(
                        np.mean(
                            successful_pgas_result.reference_ancestors[1:]
                            != conditioned
                        )
                    )
                )
            else:
                for name in (
                    "particle_min_ess",
                    "particle_mean_unique_ancestors",
                    "particle_path_changed",
                    "particle_path_update_fraction",
                    "particle_changed_fraction",
                    "particle_reference_ancestor_change_fraction",
                ):
                    engine_diagnostics[name].append(np.nan)
            engine_diagnostics["fs_elliptical_slice_steps"].append(
                successful_fs_slice_steps
            )
            engine_diagnostics["ssvs_model_move_accepted"].append(
                successful_ssvs_move_accepted
            )
            engine_diagnostics["ssvs_model_proposed_change"].append(
                successful_ssvs_proposed_change
            )
            engine_diagnostics["ssvs_model_log_acceptance_ratio"].append(
                successful_ssvs_log_acceptance_ratio
            )
            engine_diagnostics["sign_invariance_error"].append(
                successful_sign_invariance_error
            )
            if iteration_ok:
                for name, switched in successful_sign_switches.items():
                    sign_switch_counts[name] += int(switched)

            mu_exact = mu_from_ncp(z_path, params_state, self.layout)
            cur_ll = _exact_gev_loglik(y1, mu_exact, self.model, params_obs)
            x_path = map_ncp_to_centered(z_path, params_state, self.layout)

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
                    if isinstance(lambda2, dict):
                        for block, value in lambda2.items():
                            draws_static[f"lambda2_{block}"][keep_idx] = float(value)
                    else:
                        draws_static["lambda2"][keep_idx] = float(lambda2)
                    for block, value in tau.items():
                        draws_static[f"tau_{block}"][keep_idx] = float(value)
                if self.priors.horseshoe is not None:
                    draws_static["horseshoe_global"][keep_idx] = float(
                        horseshoe_state["global"]
                    )
                    draws_static["horseshoe_slab2"][keep_idx] = float(
                        horseshoe_state["slab2"]
                    )
                    for block, value in horseshoe_state["local"].items():
                        draws_static[f"horseshoe_local_{block}"][keep_idx] = float(value)
                if self.priors.triple_gamma is not None:
                    tg = self.priors.triple_gamma
                    draws_static["triple_gamma_global"][keep_idx] = float(
                        triple_gamma_state["global"]
                    )
                    draws_static["triple_gamma_a"][keep_idx] = float(
                        triple_gamma_state["a"]
                    )
                    draws_static["triple_gamma_c"][keep_idx] = float(
                        triple_gamma_state["c"]
                    )
                    if tg.regularized:
                        draws_static["triple_gamma_slab2"][keep_idx] = float(
                            triple_gamma_state["slab2"]
                        )
                    for block in triple_gamma_state["numerator"]:
                        numerator = float(
                            triple_gamma_state["numerator"][block]
                        )
                        denominator = float(
                            triple_gamma_state["denominator"][block]
                        )
                        global_scale = float(triple_gamma_state["global"])
                        draws_static[
                            f"triple_gamma_numerator_{block}"
                        ][keep_idx] = numerator
                        draws_static[
                            f"triple_gamma_denominator_{block}"
                        ][keep_idx] = denominator
                        draws_static[f"triple_gamma_rho_{block}"][keep_idx] = (
                            tg.shrinkage_factor(
                                numerator=numerator,
                                denominator=denominator,
                                global_scale=global_scale,
                            )
                        )
                if self.priors.pc is not None:
                    for block, value in tau.items():
                        draws_static[f"pc_tau_{block}"][keep_idx] = float(value)
                if self.priors.ssvs is not None:
                    draws_static["state_level"][keep_idx] = int(model_state.level)
                    draws_static["state_trend"][keep_idx] = int(model_state.trend)
                    draws_static["state_season"][keep_idx] = int(model_state.season)
                    draws_static["model_index"][keep_idx] = int(model_index)
                logpost[keep_idx] = cur_ll
                keep_idx += 1

            completed = it + 1
            if self.config.progress and should_report_progress(
                completed,
                total=n_iter,
                warmup=burn,
                every=progress_every,
            ):
                details: list[str] = []
                if self.priors.lasso is not None:
                    if isinstance(lambda2, dict):
                        compact = ",".join(
                            f"{key[0]}:{value:.2g}"
                            for key, value in lambda2.items()
                        )
                        details.append(f"lambda2=({compact})")
                    else:
                        details.append(f"lambda2={lambda2:.3g}")
                if self.priors.pc is not None:
                    compact = ",".join(
                        f"{key[0]}:{value:.2g}" for key, value in tau.items()
                    )
                    details.append(f"pc_tau=({compact})")
                if self.priors.ssvs is not None:
                    details.append(
                        "structure=("
                        f"{model_state.level.label},{model_state.trend.label},"
                        f"{model_state.season.label})"
                    )
                    if state_method == "pgas" and np.isfinite(
                        successful_ssvs_move_accepted
                    ):
                        details.append(
                            "ssvs_mh="
                            f"{int(successful_ssvs_move_accepted)}"
                            f" change={int(successful_ssvs_proposed_change)}"
                            f" slice_steps={int(successful_fs_slice_steps)}"
                        )
                if not iteration_ok:
                    restored = (
                        f"restored_attempts={sum(iteration_failures.values())}"
                        f" reasons=({_compact_counter(iteration_failures)})"
                    )
                    if last_failure_detail:
                        restored += f" last={last_failure_detail}"
                    details.append(restored)
                if window_restored or window_attempt_failures:
                    details.append(
                        f"window_restored={window_restored}/{window_iterations}"
                        f" failures=({_compact_counter(window_attempt_failures)})"
                    )
                current_metrics = {
                    name: values[-1]
                    for name, values in engine_diagnostics.items()
                    if values
                }
                print(
                    mcmc_progress_line(
                        label=progress_label,
                        engine=state_method,
                        chain=progress_chain,
                        chains=progress_chains,
                        completed=completed,
                        total=n_iter,
                        warmup=burn,
                        saved=keep_idx,
                        draws=n_keep,
                        elapsed=perf_counter() - chain_started,
                        parameters=univariate_progress_parameters(
                            {
                                "q_level": params_state["q_level"],
                                **(
                                    {"q_trend": params_state["q_trend"]}
                                    if self.layout.has_beta
                                    else {}
                                ),
                                **(
                                    {"q_season": params_state["q_season"]}
                                    if self.layout.season_dim > 0
                                    else {}
                                ),
                            },
                            params_obs,
                            horseshoe_state=horseshoe_state,
                            triple_gamma_state=triple_gamma_state,
                        ),
                        metrics=current_metrics,
                        particles=(
                            int(state_kwargs.get("particles", 256))
                            if state_method == "pgas"
                            else None
                        ),
                        details=tuple(details),
                    ),
                    flush=True,
                )
                window_restored = 0
                window_iterations = 0
                window_attempt_failures.clear()

        return FSOutput(
            draws_static=draws_static,
            draws_states=draws_states,
            logpost=logpost,
            acceptance={
                "sigma_mh": accept_sigma / max(n_iter, 1),
                "xi_mh": accept_xi / max(n_iter, 1),
                **(
                    {
                        "ssvs_model_mh": accept_ssvs_model / max(n_iter, 1),
                        "ssvs_model_change_mh": accept_ssvs_model_change
                        / max(propose_ssvs_model_change, 1),
                    }
                    if self.priors.ssvs is not None and state_method == "pgas"
                    else {}
                ),
                **{
                    key: value / max(n_iter, 1)
                    for key, value in horseshoe_accepts.items()
                },
                **{
                    key: value / max(n_iter, 1)
                    for key, value in asis_accepts.items()
                },
                **{
                    key: value / max(n_iter, 1)
                    for key, value in triple_gamma_moves.items()
                },
            },
            meta={
                "sampler": (
                    (
                        "fruehwirth_schnatter_gev_pgas_ssvs_exact_rjmh"
                        if state_method == "pgas"
                        else "fruehwirth_schnatter_gev_laplace_ssvs_gibbs"
                    )
                    if self.priors.ssvs is not None
                    else (
                        f"fruehwirth_schnatter_gev_{state_method}_lasso_gibbs"
                        if self.priors.lasso is not None
                        else (
                            f"fruehwirth_schnatter_gev_{state_method}_horseshoe_gibbs"
                            if self.priors.horseshoe is not None
                            else (
                                f"fruehwirth_schnatter_gev_{state_method}_triple_gamma_slice"
                                if self.priors.triple_gamma is not None
                                else (
                                    f"fruehwirth_schnatter_gev_{state_method}_pc_gibbs"
                                    if self.priors.pc is not None
                                    else f"fruehwirth_schnatter_gev_{state_method}_normal_gibbs"
                                )
                            )
                        )
                    )
                ),
                "parameterization": "fruehwirth_schnatter",
                "asis": use_asis,
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
                "regularized_horseshoe": self.priors.horseshoe is not None,
                "triple_gamma": self.priors.triple_gamma is not None,
                "regularized_triple_gamma": bool(
                    self.priors.triple_gamma is not None
                    and self.priors.triple_gamma.regularized
                ),
                "shrinkage_update": (
                    "slice"
                    if self.priors.horseshoe is not None
                    or self.priors.triple_gamma is not None
                    else None
                ),
                "pc_innovation_prior": self.priors.pc is not None,
                "componentwise_lasso": bool(
                    self.priors.lasso is not None
                    and getattr(self.priors.lasso, "componentwise", False)
                ),
                "structural_ssvs": self.priors.ssvs is not None,
                "model_selection_exact": (
                    state_method == "pgas"
                    if self.priors.ssvs is not None
                    else None
                ),
                "model_selection_basis": (
                    (
                        "exact_gev_rjmh_with_laplace_independence_proposals"
                        if state_method == "pgas"
                        else "laplace_pseudo_observations"
                    )
                    if self.priors.ssvs is not None
                    else None
                ),
                "targets_exact_posterior": state_method == "pgas",
                "approximation": (
                    "iterated_laplace" if state_method == "laplace" else None
                ),
                "pgas_exact_invariant": state_method == "pgas",
                "engine_diagnostics": {
                    key: np.asarray(value, dtype=float)
                    for key, value in engine_diagnostics.items()
                },
                "restored_iterations": int(restored_iterations),
                "restored_fraction": float(restored_iterations / max(n_iter, 1)),
                "attempt_failure_counts": dict(attempt_failure_totals),
                "restore_failure_counts": dict(restore_failure_totals),
                "max_state_tries": int(max_state_tries),
                "sign_switching": True,
                "sign_switch_invariance_checked": True,
                "sign_switch_counts": dict(sign_switch_counts),
            },
        )
