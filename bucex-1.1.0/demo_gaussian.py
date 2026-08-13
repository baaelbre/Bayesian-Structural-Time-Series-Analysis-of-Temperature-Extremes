from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bucex import DummySeasonal, GaussianObs, LocalLinearTrend, StructuralModel, fit_bayes
from bucex.inference.dispatch import smooth_states
from bucex.inference.fit.base import GibbsConfig
from bucex.inference.fit.priors import (
    CenteredGaussianPriors,
    DiagonalNormalPrior,
    InitialStatePriors,
    InverseGammaPrior,
    NormalPrior,
)
from bucex.simulate.statespace import simulate_statespace

from demo_utils import (
    ensure_results_dir,
    pointwise_band,
    reconstruct_eta_draws,
    reconstruct_eta_path,
    rmse,
    state_index_map,
)


STATE_METHOD = "ffbs"  # "ffbs" or "particle"
STATE_KWARGS = {
    "particle_method": "auxiliary",
    "particle_n_particles": 2500,
}
T = 240
RESULTS_DIR = "results"


TRUE_PARAMS_STATE = {
    "m0_level": 0.0,
    "v0_level": 0.50,
    "m0_trend": 0.03,
    "v0_trend": 0.01,
    "m0_season": np.zeros(11),
    "v0_season": 0.20 * np.ones(11),
    "q_level": 0.03,
    "q_trend": 0.0005,
    "q_season": 0.02,
}

TRUE_PARAMS_OBS = {"sigma": 0.8}

INIT_PARAMS_STATE = {
    "m0_level": 0.5,
    "v0_level": 1.0,
    "m0_trend": 0.00,
    "v0_trend": 0.05,
    "m0_season": np.zeros(11),
    "v0_season": 0.50 * np.ones(11),
    "q_level": 0.10,
    "q_trend": 0.01,
    "q_season": 0.10,
}

INIT_PARAMS_OBS = {"sigma": 1.5}

def build_model() -> StructuralModel:
    return StructuralModel(
        components=[
            LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            DummySeasonal(period=12, mode="dynamic"),
        ],
        obs=GaussianObs(),
    )


def build_priors() -> CenteredGaussianPriors:
    return CenteredGaussianPriors(
        sigma2=InverseGammaPrior(a=2.0, b=1.0),
        q_level=InverseGammaPrior(a=2.0, b=0.10),
        q_trend=InverseGammaPrior(a=2.0, b=0.01),
        q_season=InverseGammaPrior(a=2.0, b=0.10),
        initial=InitialStatePriors(
            m0_level=NormalPrior(mean=0.0, sd=2.0),
            v0_level=InverseGammaPrior(a=3.0, b=0.5),
            m0_trend=NormalPrior(mean=0.0, sd=0.2),
            v0_trend=InverseGammaPrior(a=3.0, b=0.05),
            m0_season=DiagonalNormalPrior(mean=np.zeros(11), sd=2.0 * np.ones(11)),
            v0_season=InverseGammaPrior(a=3.0, b=0.5),
        ),
    )


def build_config() -> GibbsConfig:
    return GibbsConfig(
        n_iter=2500,
        burn=1000,
        thin=2,
        seed=222,
        progress=True,
        progress_every=100,
    )


def conditional_smoother(y: np.ndarray, model: StructuralModel):
    if STATE_METHOD == "ffbs":
        return smooth_states(
            y=y,
            model=model,
            params_state=TRUE_PARAMS_STATE,
            params_obs=TRUE_PARAMS_OBS,
            method="kalman",
        )

    return smooth_states(
        y=y,
        model=model,
        params_state=TRUE_PARAMS_STATE,
        params_obs=TRUE_PARAMS_OBS,
        method="particle",
        particle_method=STATE_KWARGS.get("particle_method", "bootstrap"),
        particle_n_particles=STATE_KWARGS.get("particle_n_particles", 1000),
        particle_n_smoother_draws=200,
        rng=np.random.default_rng(999),
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
    ax.plot(t_obs, smooth_eta, linewidth=1.5, label="conditional smoother eta")
    ax.plot(t_obs, eta_med, linewidth=2.0, label="posterior median eta")
    ax.fill_between(t_obs, eta_lo, eta_hi, alpha=0.25, label="90% posterior CI")
    ax.set_title(f"Gaussian observation model and latent mean ({STATE_METHOD})")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    outpath = outdir / "gaussian_fit_obs_eta.png"
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
    ax.plot(t_state, smooth, linewidth=1.5, label=f"conditional smoother {ylabel}")
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
    q_level: np.ndarray,
    q_trend: np.ndarray,
    q_season: np.ndarray,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(sigma)
    axes[0].set_title("Trace: sigma")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(q_level)
    axes[1].set_title("Trace: q_level")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(q_trend)
    axes[2].set_title("Trace: q_trend")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(q_season)
    axes[3].set_title("Trace: q_season")
    axes[3].set_xlabel("saved draw index")
    axes[3].grid(True, alpha=0.3)
    outpath = outdir / "gaussian_fit_parameter_traces.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {outpath}")
    plt.show()


def main() -> None:
    model = build_model()
    sim = simulate_statespace(
        T=T,
        model=model,
        params_state=TRUE_PARAMS_STATE,
        params_obs=TRUE_PARAMS_OBS,
        rng=np.random.default_rng(123),
        x0=None,
    )

    print("--- Gaussian simulation complete ---")
    print(f"state names : {sim.state_names}")
    print(f"x shape     : {sim.x.shape}")
    print(f"mu shape    : {sim.mu.shape}")
    print(f"y shape     : {sim.y.shape}")

    sr = conditional_smoother(sim.y, model)

    fit = fit_bayes(
        y=sim.y,
        model=model,
        priors=build_priors(),
        init_params_state=INIT_PARAMS_STATE,
        init_params_obs=INIT_PARAMS_OBS,
        config=build_config(),
        method="gibbs",
        state_method=STATE_METHOD,
        parameterization="centered",
        state_kwargs=STATE_KWARGS if STATE_METHOD == "particle" else {},
    )

    print(f"\n--- Gaussian Gibbs fit complete ({STATE_METHOD}) ---")
    print(f"n_draws      : {fit.n_draws}")
    print(f"states shape : {fit.draws_states.shape if fit.draws_states is not None else None}")
    print(f"static keys  : {list(fit.draws_static.keys())}")

    post_sigma = fit.draws_static["sigma"]
    post_q_level = fit.draws_static["q_level"]
    post_q_trend = fit.draws_static["q_trend"]
    post_q_season = fit.draws_static["q_season"]

    print("\n--- Posterior means vs truth ---")
    print(f"sigma    : post mean = {np.mean(post_sigma):.4f}, truth = {TRUE_PARAMS_OBS['sigma']:.4f}")
    print(f"q_level  : post mean = {np.mean(post_q_level):.4f}, truth = {TRUE_PARAMS_STATE['q_level']:.4f}")
    print(f"q_trend  : post mean = {np.mean(post_q_trend):.6f}, truth = {TRUE_PARAMS_STATE['q_trend']:.6f}")
    print(f"q_season : post mean = {np.mean(post_q_season):.4f}, truth = {TRUE_PARAMS_STATE['q_season']:.4f}")

    if "m0_level" in fit.draws_static:
        print(f"m0_level : post mean = {np.mean(fit.draws_static['m0_level']):.4f}, truth = {TRUE_PARAMS_STATE['m0_level']:.4f}")
    if "v0_level" in fit.draws_static:
        print(f"v0_level : post mean = {np.mean(fit.draws_static['v0_level']):.4f}, truth = {TRUE_PARAMS_STATE['v0_level']:.4f}")

    draws_states = fit.draws_states
    eta_draws = reconstruct_eta_draws(draws_states, model, INIT_PARAMS_STATE)
    eta_lo, eta_med, eta_hi = pointwise_band(eta_draws)

    idx = state_index_map(sim.state_names)
    alpha_draws = draws_states[:, :, idx["alpha"]]
    beta_draws = draws_states[:, :, idx["beta"]]
    g1_draws = draws_states[:, :, idx["g1"]]

    alpha_lo, alpha_med, alpha_hi = pointwise_band(alpha_draws)
    beta_lo, beta_med, beta_hi = pointwise_band(beta_draws)
    g1_lo, g1_med, g1_hi = pointwise_band(g1_draws)

    eta_smooth = reconstruct_eta_path(sr.m_smooth, model, TRUE_PARAMS_STATE)
    true_alpha = sim.x[:, idx["alpha"]]
    true_beta = sim.x[:, idx["beta"]]
    true_g1 = sim.x[:, idx["g1"]]
    smooth_alpha = sr.m_smooth[:, idx["alpha"]]
    smooth_beta = sr.m_smooth[:, idx["beta"]]
    smooth_g1 = sr.m_smooth[:, idx["g1"]]

    print("\n--- RMSE diagnostics ---")
    print(f"eta conditional smoother RMSE : {rmse(sim.mu, eta_smooth):.4f}")
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
        filename="gaussian_fit_level_alpha.png",
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
        filename="gaussian_fit_slope_beta.png",
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
        filename="gaussian_fit_seasonality_g1.png",
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
        q_level=post_q_level,
        q_trend=post_q_trend,
        q_season=post_q_season,
    )


if __name__ == "__main__":
    main()
