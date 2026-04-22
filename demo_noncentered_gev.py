from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bucex import DummySeasonal, GEVObs, LocalLinearTrend, StructuralModel, fit_bayes
from bucex.inference.dispatch import smooth_states
from bucex.inference.fit.base import GibbsConfig
from bucex.inference.fit.priors import (
    DiagonalNormalPrior,
    NonCenteredGEVPriors,
    NormalPrior,
)
from bucex.simulate.statespace import simulate_statespace


T = 240
RESULTS_DIR = Path("results")

# ---------------------------------------------------------------------
# Truth for simulation: the simulator uses the centered parameterization
# ---------------------------------------------------------------------
TRUE_PARAMS_STATE_SIM = {
    "m0_level": 18.0,
    "v0_level": 0.50,
    "m0_trend": 0.015,
    "v0_trend": 0.0025,
    "m0_season": np.zeros(11),
    "v0_season": 0.20 * np.ones(11),
    "q_level": 0.06**2,
    "q_trend": 0.004**2,
    "q_season": 0.03**2,
}
TRUE_PARAMS_OBS = {"sigma": 1.2, "xi": -0.12}

# ---------------------------------------------------------------------
# The same truth, written in the non-centred parameterization
# ---------------------------------------------------------------------
TRUE_PARAMS_STATE_NCP = {
    "alpha0": 18.0,
    "beta0": 0.015,
    "gamma0_season": np.zeros(11),
    "s_level": 0.06,
    "s_trend": 0.004,
    "s_season": 0.03,
}

# ---------------------------------------------------------------------
# Initial values for the non-centred sampler
# ---------------------------------------------------------------------
INIT_PARAMS_STATE_NCP = {
    "alpha0": 17.0,
    "beta0": 0.00,
    "gamma0_season": np.zeros(11),
    "s_level": 0.12,
    "s_trend": 0.02,
    "s_season": 0.10,
}
INIT_PARAMS_OBS = {"sigma": 3.0, "xi": -0.02}


def build_model() -> StructuralModel:
    return StructuralModel(
        components=[
            LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            DummySeasonal(period=12, mode="dynamic"),
        ],
        obs=GEVObs(),
    )


def build_priors() -> NonCenteredGEVPriors:
    return NonCenteredGEVPriors(
        log_sigma=NormalPrior(mean=np.log(1.2), sd=0.50),
        xi=NormalPrior(mean=-0.10, sd=0.15),
        alpha0=NormalPrior(mean=18.0, sd=5.0),
        beta0=NormalPrior(mean=0.0, sd=0.10),
        gamma0_season=DiagonalNormalPrior(
            mean=np.zeros(11),
            sd=2.0 * np.ones(11),
        ),
        s_level=NormalPrior(mean=0.0, sd=0.20),
        s_trend=NormalPrior(mean=0.0, sd=0.05),
        s_season=NormalPrior(mean=0.0, sd=0.20),
        xi_max_abs=0.45,
    )


def build_config() -> GibbsConfig:
    return GibbsConfig(
        n_iter=3000,
        burn=1000,
        thin=2,
        seed=333,
        progress=True,
        progress_every=100,
    )


def ensure_results_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pointwise_band(draws: np.ndarray, level: float = 0.90):
    alpha = 1.0 - level
    lo = np.quantile(draws, alpha / 2.0, axis=0)
    med = np.quantile(draws, 0.5, axis=0)
    hi = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    return lo, med, hi


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def state_index_map(state_names):
    return {name: i for i, name in enumerate(state_names)}


def reconstruct_eta_path(x_path: np.ndarray, model: StructuralModel, params_state: dict) -> np.ndarray:
    Tn = x_path.shape[0] - 1
    out = np.zeros(Tn, dtype=float)
    for t in range(1, Tn + 1):
        des = model.design(t=t, params_state=params_state)
        eta_t = des.Z @ x_path[t] + des.d
        out[t - 1] = float(np.atleast_1d(eta_t)[0])
    return out


def reconstruct_eta_draws(draws_states: np.ndarray, model: StructuralModel, params_state: dict) -> np.ndarray:
    M, T1, _ = draws_states.shape
    Tn = T1 - 1
    out = np.zeros((M, Tn), dtype=float)
    for m in range(M):
        out[m] = reconstruct_eta_path(draws_states[m], model, params_state)
    return out


def conditional_smoother(y: np.ndarray, model: StructuralModel):
    return smooth_states(
        y=y,
        model=model,
        params_state=TRUE_PARAMS_STATE_SIM,
        params_obs=TRUE_PARAMS_OBS,
        method="laplace",
    )


def save_obs_figure(
    outdir: Path,
    t_obs: np.ndarray,
    y: np.ndarray,
    true_eta: np.ndarray,
    smooth_eta: np.ndarray,
    eta_lo: np.ndarray,
    eta_med: np.ndarray,
    eta_hi: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    ax.plot(t_obs, y, linewidth=1.0, alpha=0.7, label="observed y")
    ax.plot(t_obs, true_eta, linewidth=2.0, label="true eta / mu")
    ax.plot(t_obs, smooth_eta, linewidth=1.5, label="Laplace smoother eta")
    ax.plot(t_obs, eta_med, linewidth=2.0, label="posterior median eta")
    ax.fill_between(t_obs, eta_lo, eta_hi, alpha=0.25, label="90% posterior CI")
    ax.set_title("GEV observation model and latent location (non-centred)")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    outpath = outdir / "noncentered_gev_fit_obs_eta.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {outpath}")
    plt.show()


def save_state_figure(
    outdir: Path,
    filename: str,
    title: str,
    ylabel: str,
    t_state: np.ndarray,
    truth: np.ndarray,
    smooth: np.ndarray,
    lo: np.ndarray,
    med: np.ndarray,
    hi: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    ax.plot(t_state, truth, linewidth=2.0, label=f"true {ylabel}")
    ax.plot(t_state, smooth, linewidth=1.5, label=f"Laplace smoother {ylabel}")
    ax.plot(t_state, med, linewidth=2.0, label=f"posterior median {ylabel}")
    ax.fill_between(t_state, lo, hi, alpha=0.25, label="90% posterior CI")
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    outpath = outdir / filename
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {outpath}")
    plt.show()


def save_trace_figure(
    outdir: Path,
    sigma: np.ndarray,
    xi: np.ndarray,
    s_level: np.ndarray,
    s_trend: np.ndarray,
    s_season: np.ndarray,
    q_level: np.ndarray,
    q_trend: np.ndarray,
    q_season: np.ndarray,
    loglike: np.ndarray,
) -> None:
    fig, axes = plt.subplots(9, 1, figsize=(12, 18), constrained_layout=True)
    axes[0].plot(sigma)
    axes[0].set_title("Trace: sigma")
    axes[1].plot(xi)
    axes[1].set_title("Trace: xi")
    axes[2].plot(s_level)
    axes[2].set_title("Trace: s_level")
    axes[3].plot(s_trend)
    axes[3].set_title("Trace: s_trend")
    axes[4].plot(s_season)
    axes[4].set_title("Trace: s_season")
    axes[5].plot(q_level)
    axes[5].set_title("Trace: q_level = s_level^2")
    axes[6].plot(q_trend)
    axes[6].set_title("Trace: q_trend = s_trend^2")
    axes[7].plot(q_season)
    axes[7].set_title("Trace: q_season = s_season^2")
    axes[8].plot(loglike)
    axes[8].set_title("Trace: exact GEV log-likelihood")
    axes[8].set_xlabel("saved draw index")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    outpath = outdir / "noncentered_gev_parameter_traces.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {outpath}")
    plt.show()


def main() -> None:
    model = build_model()

    sim = simulate_statespace(
        T=T,
        model=model,
        params_state=TRUE_PARAMS_STATE_SIM,
        params_obs=TRUE_PARAMS_OBS,
        rng=np.random.default_rng(321),
        x0=None,
    )

    print("--- GEV simulation complete ---")
    print(f"state names : {sim.state_names}")
    print(f"x shape     : {sim.x.shape}")
    print(f"mu shape    : {sim.mu.shape}")
    print(f"y shape     : {sim.y.shape}")

    sr = conditional_smoother(sim.y, model)

    fit = fit_bayes(
        y=sim.y,
        model=model,
        priors=build_priors(),
        init_params_state=INIT_PARAMS_STATE_NCP,
        init_params_obs=INIT_PARAMS_OBS,
        config=build_config(),
        method="gibbs",
        state_method="laplace",
        parameterization="noncentered",
    )

    print("\n--- Non-centred GEV Gibbs fit complete ---")
    print(f"n_draws      : {fit.n_draws}")
    print(f"states shape : {fit.draws_states.shape if fit.draws_states is not None else None}")
    print(f"static keys  : {list(fit.draws_static.keys())}")
    print(f"acceptance   : {fit.acceptance}")

    post_sigma = fit.draws_static["sigma"]
    post_xi = fit.draws_static["xi"]
    post_s_level = fit.draws_static["s_level"]
    post_s_trend = fit.draws_static["s_trend"]
    post_s_season = fit.draws_static["s_season"]
    post_q_level = fit.draws_static["q_level"]
    post_q_trend = fit.draws_static["q_trend"]
    post_q_season = fit.draws_static["q_season"]

    print("\n--- Posterior means vs truth ---")
    print(f"sigma    : post mean = {np.mean(post_sigma):.4f}, truth = {TRUE_PARAMS_OBS['sigma']:.4f}")
    print(f"xi       : post mean = {np.mean(post_xi):.4f}, truth = {TRUE_PARAMS_OBS['xi']:.4f}")
    print(f"s_level  : post mean = {np.mean(post_s_level):.4f}, truth = {TRUE_PARAMS_STATE_NCP['s_level']:.4f}")
    print(f"s_trend  : post mean = {np.mean(post_s_trend):.6f}, truth = {TRUE_PARAMS_STATE_NCP['s_trend']:.6f}")
    print(f"s_season : post mean = {np.mean(post_s_season):.4f}, truth = {TRUE_PARAMS_STATE_NCP['s_season']:.4f}")
    print(f"q_level  : post mean = {np.mean(post_q_level):.4f}, truth = {TRUE_PARAMS_STATE_SIM['q_level']:.4f}")
    print(f"q_trend  : post mean = {np.mean(post_q_trend):.6f}, truth = {TRUE_PARAMS_STATE_SIM['q_trend']:.6f}")
    print(f"q_season : post mean = {np.mean(post_q_season):.4f}, truth = {TRUE_PARAMS_STATE_SIM['q_season']:.4f}")
    print(f"alpha0   : post mean = {np.mean(fit.draws_static['alpha0']):.4f}, truth = {TRUE_PARAMS_STATE_NCP['alpha0']:.4f}")
    print(f"beta0    : post mean = {np.mean(fit.draws_static['beta0']):.4f}, truth = {TRUE_PARAMS_STATE_NCP['beta0']:.4f}")

    draws_states = fit.draws_states
    eta_draws = reconstruct_eta_draws(draws_states, model, TRUE_PARAMS_STATE_SIM)
    eta_lo, eta_med, eta_hi = pointwise_band(eta_draws)

    idx = state_index_map(sim.state_names)
    alpha_draws = draws_states[:, :, idx["alpha"]]
    beta_draws = draws_states[:, :, idx["beta"]]
    g1_draws = draws_states[:, :, idx["g1"]]

    alpha_lo, alpha_med, alpha_hi = pointwise_band(alpha_draws)
    beta_lo, beta_med, beta_hi = pointwise_band(beta_draws)
    g1_lo, g1_med, g1_hi = pointwise_band(g1_draws)

    eta_smooth = reconstruct_eta_path(sr.m_smooth, model, TRUE_PARAMS_STATE_SIM)
    true_alpha = sim.x[:, idx["alpha"]]
    true_beta = sim.x[:, idx["beta"]]
    true_g1 = sim.x[:, idx["g1"]]
    smooth_alpha = sr.m_smooth[:, idx["alpha"]]
    smooth_beta = sr.m_smooth[:, idx["beta"]]
    smooth_g1 = sr.m_smooth[:, idx["g1"]]

    print("\n--- RMSE diagnostics ---")
    print(f"eta Laplace smoother RMSE     : {rmse(sim.mu, eta_smooth):.4f}")
    print(f"eta posterior median RMSE     : {rmse(sim.mu, eta_med):.4f}")
    print()
    print(f"alpha smoother RMSE           : {rmse(true_alpha, smooth_alpha):.4f}")
    print(f"alpha posterior RMSE          : {rmse(true_alpha, alpha_med):.4f}")
    print()
    print(f"beta smoother RMSE            : {rmse(true_beta, smooth_beta):.4f}")
    print(f"beta posterior RMSE           : {rmse(true_beta, beta_med):.4f}")
    print()
    print(f"g1 smoother RMSE              : {rmse(true_g1, smooth_g1):.4f}")
    print(f"g1 posterior RMSE             : {rmse(true_g1, g1_med):.4f}")

    outdir = ensure_results_dir(RESULTS_DIR)
    t_obs = np.arange(1, T + 1)
    t_state = np.arange(0, T + 1)

    save_obs_figure(
        outdir=outdir,
        t_obs=t_obs,
        y=sim.y,
        true_eta=sim.mu,
        smooth_eta=eta_smooth,
        eta_lo=eta_lo,
        eta_med=eta_med,
        eta_hi=eta_hi,
    )

    save_state_figure(
        outdir=outdir,
        filename="noncentered_gev_level_alpha.png",
        title="Latent level alpha",
        ylabel="alpha",
        t_state=t_state,
        truth=true_alpha,
        smooth=smooth_alpha,
        lo=alpha_lo,
        med=alpha_med,
        hi=alpha_hi,
    )

    save_state_figure(
        outdir=outdir,
        filename="noncentered_gev_slope_beta.png",
        title="Latent slope beta",
        ylabel="beta",
        t_state=t_state,
        truth=true_beta,
        smooth=smooth_beta,
        lo=beta_lo,
        med=beta_med,
        hi=beta_hi,
    )

    save_state_figure(
        outdir=outdir,
        filename="noncentered_gev_seasonality_g1.png",
        title="Latent seasonal state g1",
        ylabel="g1",
        t_state=t_state,
        truth=true_g1,
        smooth=smooth_g1,
        lo=g1_lo,
        med=g1_med,
        hi=g1_hi,
    )

    save_trace_figure(
        outdir=outdir,
        sigma=post_sigma,
        xi=post_xi,
        s_level=post_s_level,
        s_trend=post_s_trend,
        s_season=post_s_season,
        q_level=post_q_level,
        q_trend=post_q_trend,
        q_season=post_q_season,
        loglike=fit.logpost if fit.logpost is not None else np.full_like(post_sigma, np.nan),
    )


if __name__ == "__main__":
    main()
