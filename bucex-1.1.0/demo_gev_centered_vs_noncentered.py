from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bucex import DummySeasonal, GEVObs, LocalLinearTrend, StructuralModel, fit_bayes
from bucex.inference.fit.base import GibbsConfig
from bucex.inference.fit.priors import (
    CenteredGEVPriors,
    DiagonalNormalPrior,
    InitialStatePriors,
    InverseGammaPrior,
    NonCenteredGEVPriors,
    NormalPrior,
)
from bucex.simulate.statespace import simulate_statespace


# ---------------------------------------------------------------------
# User controls
# ---------------------------------------------------------------------
T = 140
SEED = 123
N_ITER = 1500
BURN = 500
THIN = 1
OUTDIR = Path("results")

METHOD_SPECS: list[dict[str, Any]] = [
    {
        "label": "centered-laplace",
        "parameterization": "centered",
        "state_method": "laplace",
        "state_kwargs": {},
    },
    {
        "label": "centered-bootstrap",
        "parameterization": "centered",
        "state_method": "particle",
        "state_kwargs": {"particle_method": "bootstrap", "particle_n_particles": 150},
    },
    {
        "label": "centered-auxiliary",
        "parameterization": "centered",
        "state_method": "particle",
        "state_kwargs": {"particle_method": "auxiliary", "particle_n_particles": 150},
    },
    {
        "label": "noncentered-laplace",
        "parameterization": "noncentered",
        "state_method": "laplace",
        "state_kwargs": {},
    },
    {
        "label": "noncentered-bootstrap",
        "parameterization": "noncentered",
        "state_method": "particle",
        "state_kwargs": {"particle_method": "bootstrap", "particle_n_particles": 150},
    },
    {
        "label": "noncentered-auxiliary",
        "parameterization": "noncentered",
        "state_method": "particle",
        "state_kwargs": {"particle_method": "auxiliary", "particle_n_particles": 150},
    },
]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def get_state_index(state_names: tuple[str, ...], name: str) -> int:
    if name not in state_names:
        raise KeyError(f"State '{name}' not found in {state_names}.")
    return state_names.index(name)


def reconstruct_eta_path(x_path: np.ndarray, model, params_state: dict) -> np.ndarray:
    Tn = x_path.shape[0] - 1
    eta = np.zeros(Tn, dtype=float)
    for t in range(1, Tn + 1):
        des = model.design(t=t, params_state=params_state, exog_t=None)
        eta_t = des.Z @ x_path[t] + des.d
        eta[t - 1] = float(np.atleast_1d(eta_t)[0])
    return eta


def reconstruct_eta_from_draws(draws_states: np.ndarray, model, params_state: dict) -> np.ndarray:
    M = draws_states.shape[0]
    Tn = draws_states.shape[1] - 1
    eta_draws = np.zeros((M, Tn), dtype=float)
    for m in range(M):
        eta_draws[m] = reconstruct_eta_path(draws_states[m], model, params_state)
    return eta_draws


def posterior_median_path(draws: np.ndarray) -> np.ndarray:
    return np.quantile(np.asarray(draws, dtype=float), 0.5, axis=0)


def plot_eta_comparison(
    outdir: Path,
    t_obs: np.ndarray,
    y: np.ndarray,
    true_eta: np.ndarray,
    eta_methods: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(t_obs, y, linewidth=1.0, alpha=0.35, label="observed y")
    ax.plot(t_obs, true_eta, linewidth=2.8, label="true eta")
    for label, eta_hat in eta_methods.items():
        ax.plot(t_obs, eta_hat, linewidth=1.6, label=label)
    ax.set_title("GEV fit comparison: centered vs noncentered, Laplace vs particles")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    outpath = outdir / "gev_centered_vs_noncentered_eta.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {outpath}")
    plt.show()


def plot_state_comparison(
    outdir: Path,
    filename: str,
    title: str,
    ylabel: str,
    t_state: np.ndarray,
    truth: np.ndarray,
    state_methods: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(t_state, truth, linewidth=2.8, label=f"true {ylabel}")
    for label, state_hat in state_methods.items():
        ax.plot(t_state, state_hat, linewidth=1.6, label=label)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    outpath = outdir / filename
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {outpath}")
    plt.show()


def centered_priors() -> CenteredGEVPriors:
    return CenteredGEVPriors(
        log_sigma=NormalPrior(mean=0.0, sd=1.0),
        xi=NormalPrior(mean=0.0, sd=0.20),
        xi_max_abs=0.45,
        q_level=InverseGammaPrior(a=2.0, b=0.10),
        q_trend=InverseGammaPrior(a=2.0, b=0.01),
        q_season=InverseGammaPrior(a=2.0, b=0.10),
        initial=InitialStatePriors(
            m0_level=NormalPrior(mean=10.0, sd=2.0),
            v0_level=InverseGammaPrior(a=3.0, b=0.5),
            m0_trend=NormalPrior(mean=0.0, sd=0.2),
            v0_trend=InverseGammaPrior(a=3.0, b=0.05),
            m0_season=DiagonalNormalPrior(mean=np.zeros(11), sd=2.0 * np.ones(11)),
            v0_season=InverseGammaPrior(a=3.0, b=0.5),
        ),
    )


def noncentered_priors() -> NonCenteredGEVPriors:
    return NonCenteredGEVPriors(
        log_sigma=NormalPrior(mean=0.0, sd=1.0),
        xi=NormalPrior(mean=0.0, sd=0.20),
        xi_max_abs=0.45,
        alpha0=NormalPrior(mean=10.0, sd=2.0),
        beta0=NormalPrior(mean=0.0, sd=0.2),
        gamma0_season=DiagonalNormalPrior(mean=np.zeros(11), sd=2.0 * np.ones(11)),
        s_level=NormalPrior(mean=0.0, sd=0.4),
        s_trend=NormalPrior(mean=0.0, sd=0.1),
        s_season=NormalPrior(mean=0.0, sd=0.4),
    )


def main() -> None:
    rng = np.random.default_rng(SEED)

    trend = LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic")
    seasonal = DummySeasonal(period=12, mode="dynamic")
    model = StructuralModel(components=[trend, seasonal], obs=GEVObs())

    true_params_state = {
        "m0_level": 10.0,
        "v0_level": 0.50,
        "m0_trend": 0.02,
        "v0_trend": 0.01,
        "m0_season": np.zeros(11),
        "v0_season": 0.20 * np.ones(11),
        "q_level": 0.02,
        "q_trend": 0.0004,
        "q_season": 0.015,
    }
    true_params_obs = {"sigma": 0.8, "xi": -0.15}

    sim = simulate_statespace(
        T=T,
        model=model,
        params_state=true_params_state,
        params_obs=true_params_obs,
        rng=rng,
        x0=None,
    )

    centered_init_state = {
        "m0_level": 10.0,
        "v0_level": 1.0,
        "m0_trend": 0.00,
        "v0_trend": 0.05,
        "m0_season": np.zeros(11),
        "v0_season": 0.50 * np.ones(11),
        "q_level": 0.08,
        "q_trend": 0.005,
        "q_season": 0.08,
    }
    noncentered_init_state = {
        "alpha0": 10.0,
        "beta0": 0.00,
        "gamma0_season": np.zeros(11),
        "s_level": np.sqrt(0.08),
        "s_trend": np.sqrt(0.005),
        "s_season": np.sqrt(0.08),
    }
    init_params_obs = {"sigma": 1.2, "xi": 0.0}

    cfg = GibbsConfig(
        n_iter=N_ITER,
        burn=BURN,
        thin=THIN,
        seed=222,
        progress=True,
        progress_every=100,
    )

    ia = get_state_index(sim.state_names, "alpha")
    ib = get_state_index(sim.state_names, "beta")
    ig1 = get_state_index(sim.state_names, "g1")

    summaries: list[dict[str, Any]] = []
    eta_methods: dict[str, np.ndarray] = {}
    alpha_methods: dict[str, np.ndarray] = {}
    beta_methods: dict[str, np.ndarray] = {}
    g1_methods: dict[str, np.ndarray] = {}

    for spec in METHOD_SPECS:
        print("\n" + "=" * 80)
        print(f"Running: {spec['label']}")
        print("=" * 80)

        if spec["parameterization"] == "centered":
            priors = centered_priors()
            init_state = centered_init_state
        else:
            priors = noncentered_priors()
            init_state = noncentered_init_state

        fit = fit_bayes(
            y=sim.y,
            model=model,
            priors=priors,
            init_params_state=init_state,
            init_params_obs=init_params_obs,
            config=cfg,
            parameterization=spec["parameterization"],
            state_method=spec["state_method"],
            state_kwargs=spec["state_kwargs"],
        )

        draws_states = fit.draws_states
        eta_draws = reconstruct_eta_from_draws(draws_states, model, true_params_state)
        eta_med = posterior_median_path(eta_draws)
        alpha_med = posterior_median_path(draws_states[:, :, ia])
        beta_med = posterior_median_path(draws_states[:, :, ib])
        g1_med = posterior_median_path(draws_states[:, :, ig1])

        eta_methods[spec["label"]] = eta_med
        alpha_methods[spec["label"]] = alpha_med
        beta_methods[spec["label"]] = beta_med
        g1_methods[spec["label"]] = g1_med

        res = {
            "method": spec["label"],
            "eta_rmse": rmse(sim.mu, eta_med),
            "alpha_rmse": rmse(sim.x[:, ia], alpha_med),
            "beta_rmse": rmse(sim.x[:, ib], beta_med),
            "g1_rmse": rmse(sim.x[:, ig1], g1_med),
            "sigma_mean": float(np.mean(fit.draws_static["sigma"])),
            "xi_mean": float(np.mean(fit.draws_static["xi"])),
        }
        if fit.acceptance:
            for k, v in fit.acceptance.items():
                res[f"acc_{k}"] = float(v)
        summaries.append(res)

    print("\n=== Posterior median RMSE comparison ===")
    for res in summaries:
        print(f"\nMethod: {res['method']}")
        print(f"  eta   RMSE : {res['eta_rmse']:.4f}")
        print(f"  alpha RMSE : {res['alpha_rmse']:.4f}")
        print(f"  beta  RMSE : {res['beta_rmse']:.4f}")
        print(f"  g1    RMSE : {res['g1_rmse']:.4f}")
        print(f"  sigma mean : {res['sigma_mean']:.4f}")
        print(f"  xi    mean : {res['xi_mean']:.4f}")
        for k, v in res.items():
            if k.startswith("acc_"):
                print(f"  {k} : {v:.3f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    t_obs = np.arange(1, T + 1)
    t_state = np.arange(0, T + 1)

    plot_eta_comparison(
        outdir=OUTDIR,
        t_obs=t_obs,
        y=sim.y,
        true_eta=sim.mu,
        eta_methods=eta_methods,
    )
    plot_state_comparison(
        outdir=OUTDIR,
        filename="gev_centered_vs_noncentered_alpha.png",
        title="GEV fit comparison: latent level alpha",
        ylabel="alpha",
        t_state=t_state,
        truth=sim.x[:, ia],
        state_methods=alpha_methods,
    )
    plot_state_comparison(
        outdir=OUTDIR,
        filename="gev_centered_vs_noncentered_beta.png",
        title="GEV fit comparison: latent slope beta",
        ylabel="beta",
        t_state=t_state,
        truth=sim.x[:, ib],
        state_methods=beta_methods,
    )
    plot_state_comparison(
        outdir=OUTDIR,
        filename="gev_centered_vs_noncentered_g1.png",
        title="GEV fit comparison: latent seasonal state g1",
        ylabel="g1",
        t_state=t_state,
        truth=sim.x[:, ig1],
        state_methods=g1_methods,
    )


if __name__ == "__main__":
    main()
