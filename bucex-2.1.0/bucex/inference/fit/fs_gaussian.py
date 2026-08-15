from __future__ import annotations

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
    build_ncp_system,
    canonicalize_ncp_params,
    ffbs_gaussian_1d,
    design_matrix_ncp,
    gaussian_theta_update,
    infer_ncp_layout,
    initialise_lasso,
    map_ncp_to_centered,
    measurement_vector,
    mu_from_ncp,
    random_sign_switches,
    update_lasso_scales,
    apply_theta_draw,
    copy_lasso_lambda2,
    lasso_coefficient_scale,
    initialise_horseshoe,
    initialise_triple_gamma,
    update_horseshoe_scales,
    update_triple_gamma_scales,
    update_pc_scales,
    asis_centered_scale_update,
)
from .model_space import (
    ComponentState,
    initial_structural_state,
    sample_structural_regression,
)
from ...priors.structural import InverseGammaPrior, FSGaussianPriors
from .utils import as_1d_observations, sample_inverse_gamma

Array = np.ndarray
ParamDict = Dict[str, Any]


def _sigma2_update_from_mu(
    y: Array,
    mu: Array,
    prior: InverseGammaPrior,
    rng: np.random.Generator,
    *,
    params_state: Optional[ParamDict] = None,
    tau: Optional[dict[str, float]] = None,
    lasso_uses_sigma2: bool = False,
    lasso_prior: Any = None,
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
                coefficient_scale = (
                    lasso_coefficient_scale(lasso_prior, block)
                    if lasso_prior is not None
                    else 1.0
                )
                denominator = coefficient_scale**2 * max(float(tau[block]), 1e-12)
                ss += float(params_state[key]) ** 2 / max(denominator, 1e-16)
                n_extra += 1
    a_post = prior.a + 0.5 * (resid.size + n_extra)
    b_post = prior.b + 0.5 * ss
    return sample_inverse_gamma(a_post, b_post, rng)


class FSGaussianKernel:
    """NCP FFBS sampler with optional hierarchical Bayesian lasso."""

    def __init__(
        self,
        model: StateSpaceModel,
        priors: FSGaussianPriors,
        config: GibbsConfig | MCMC = GibbsConfig(),
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
    ) -> FSOutput:
        y1 = as_1d_observations(y, family="Gaussian")
        Tn = y1.size
        m = self.model.state_dim

        if exog is not None:
            raise NotImplementedError("The FS Gaussian strategy does not support exog.")
        if state_method != "ffbs":
            raise NotImplementedError("The FS Gaussian strategy requires engine='ffbs'.")
        state_kwargs = {} if state_kwargs is None else dict(state_kwargs)

        params_state = canonicalize_ncp_params(init_params_state, self.layout)
        params_obs = dict(init_params_obs)
        if "sigma2" in params_obs and "sigma" not in params_obs:
            params_obs["sigma"] = float(np.sqrt(float(params_obs["sigma2"])))
        if "sigma" not in params_obs:
            raise KeyError("init_params_obs must contain 'sigma' or 'sigma2'.")
        params_obs["sigma2"] = float(params_obs["sigma"]) ** 2

        tau, lambda2 = initialise_lasso(self.priors, self.layout)
        horseshoe_state = initialise_horseshoe(self.priors, self.layout)
        triple_gamma_state = initialise_triple_gamma(self.priors, self.layout)
        model_state = initial_structural_state(params_state, self.layout)
        use_asis = bool(state_kwargs.get("asis", False))

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
        z_path = np.zeros((Tn + 1, self.layout.ncp_state_dim), dtype=float)
        keep_idx = 0
        progress_every = progress_interval(n_iter, self.config.progress_every)
        chain_started = perf_counter()
        progress_chain = int(state_kwargs.get("_progress_chain", 1))
        progress_chains = int(state_kwargs.get("_progress_chains", 1))
        progress_label = str(state_kwargs.get("_progress_label", "univariate"))
        horseshoe_accepts: dict[str, int] = {}
        triple_gamma_moves: dict[str, int] = {}
        asis_accepts: dict[str, int] = {}

        for it in range(n_iter):
            from .fs_utils import baseline_mu_path

            offset = baseline_mu_path(Tn, params_state, self.layout)
            H = measurement_vector(params_state, self.layout)
            z_path = ffbs_gaussian_1d(
                y=y1 - offset,
                G=G,
                Q=Q,
                H=H,
                R=float(params_obs["sigma2"]),
                m0=np.zeros(self.layout.ncp_state_dim),
                C0=float(state_kwargs.get("C0_scale", 0.0)) * np.eye(self.layout.ncp_state_dim),
                rng=self.rng,
            )

            lasso_var = (
                self.priors.lasso.variance_scale(float(params_obs["sigma2"]))
                if self.priors.lasso is not None
                else float(params_obs["sigma2"])
            )
            model_index = -1
            if self.priors.ssvs is not None:
                X, theta_names, tbar = design_matrix_ncp(
                    z_path, self.layout, center_time=True
                )
                selection = sample_structural_regression(
                    y=y1,
                    X=X,
                    theta_names=theta_names,
                    tbar=tbar,
                    noise_variance=float(params_obs["sigma2"]),
                    priors=self.priors,
                    layout=self.layout,
                    rng=self.rng,
                    apply_theta_draw=apply_theta_draw,
                )
                params_state.update(selection.params_state)
                model_state = selection.state
                model_index = selection.selected_index
            else:
                theta_draw = gaussian_theta_update(
                    y=y1,
                    z_path=z_path,
                    sigma2=float(params_obs["sigma2"]),
                    priors=self.priors,
                    layout=self.layout,
                    rng=self.rng,
                    tau=tau or None,
                    lasso_variance_scale=lasso_var,
                    horseshoe_state=horseshoe_state or None,
                    triple_gamma_state=triple_gamma_state or None,
                )
                params_state.update(theta_draw)

            if use_asis:
                z_path, params_state, outcomes = asis_centered_scale_update(
                    z_path,
                    params_state,
                    self.priors,
                    self.layout,
                    self.rng,
                    tau=tau or None,
                    lasso_variance_scale=lasso_var,
                    horseshoe_state=horseshoe_state or None,
                    triple_gamma_state=triple_gamma_state or None,
                    proposal_step=float(state_kwargs.get("asis_step", 0.20)),
                )
                for key, accepted in outcomes.items():
                    asis_accepts[key] = asis_accepts.get(key, 0) + int(accepted)

            z_path, params_state = random_sign_switches(
                z_path, params_state, self.layout, self.rng
            )

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
            if self.priors.horseshoe is not None:
                horseshoe_state, outcomes = update_horseshoe_scales(
                    params_state,
                    horseshoe_state,
                    self.priors,
                    self.layout,
                    rng=self.rng,
                    step_local=float(state_kwargs.get("horseshoe_step_local", 0.35)),
                    step_global=float(state_kwargs.get("horseshoe_step_global", 0.25)),
                    step_slab=float(state_kwargs.get("horseshoe_step_slab", 0.20)),
                )
                for key, accepted in outcomes.items():
                    horseshoe_accepts[key] = horseshoe_accepts.get(key, 0) + int(accepted)
            if self.priors.triple_gamma is not None:
                triple_gamma_state, outcomes = update_triple_gamma_scales(
                    params_state,
                    triple_gamma_state,
                    self.priors,
                    self.layout,
                    rng=self.rng,
                    width_local=float(
                        state_kwargs.get("triple_gamma_width_local", 1.0)
                    ),
                    width_global=float(
                        state_kwargs.get("triple_gamma_width_global", 1.0)
                    ),
                    width_shape=float(
                        state_kwargs.get("triple_gamma_width_shape", 0.8)
                    ),
                    width_slab=float(
                        state_kwargs.get("triple_gamma_width_slab", 0.8)
                    ),
                )
                for key, moved in outcomes.items():
                    triple_gamma_moves[key] = (
                        triple_gamma_moves.get(key, 0) + int(moved)
                    )
            if self.priors.pc is not None:
                tau = update_pc_scales(
                    params_state,
                    tau,
                    self.priors,
                    self.layout,
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
                lasso_prior=self.priors.lasso,
            )
            params_obs["sigma2"] = float(sigma2)
            params_obs["sigma"] = float(np.sqrt(sigma2))
            x_path = map_ncp_to_centered(z_path, params_state, self.layout)

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
                resid = y1 - mu
                logpost[keep_idx] = float(
                    -0.5 * Tn * np.log(2.0 * np.pi * params_obs["sigma2"])
                    - 0.5 * float(resid @ resid) / params_obs["sigma2"]
                )
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
                print(
                    mcmc_progress_line(
                        label=progress_label,
                        engine="ffbs",
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
                        details=tuple(details),
                    ),
                    flush=True,
                )

        return FSOutput(
            draws_static=draws_static,
            draws_states=draws_states,
            logpost=logpost,
            acceptance={
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
                    "fs_gaussian_ssvs_gibbs"
                    if self.priors.ssvs is not None
                    else (
                        "fs_gaussian_lasso_gibbs"
                        if self.priors.lasso is not None
                        else (
                            "fs_gaussian_horseshoe_gibbs"
                            if self.priors.horseshoe is not None
                            else (
                                "fs_gaussian_triple_gamma_slice"
                                if self.priors.triple_gamma is not None
                                else (
                                    "fs_gaussian_pc_gibbs"
                                    if self.priors.pc is not None
                                    else "fs_gaussian_normal_gibbs"
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
                "model_selection_exact": self.priors.ssvs is not None,
            },
        )
