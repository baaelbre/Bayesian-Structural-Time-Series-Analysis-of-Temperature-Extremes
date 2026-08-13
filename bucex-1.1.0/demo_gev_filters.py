from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bucex import DummySeasonal, GEVObs, LocalLinearTrend, StructuralModel
from bucex.inference.dispatch import filter_states, smooth_states
from bucex.simulate.statespace import simulate_statespace

from demo_utils import ensure_results_dir, reconstruct_eta_path, rmse, state_index_map


T = 240
PARTICLE_N = 100
PARTICLE_SMOOTHER_DRAWS = 200
RESULTS_DIR = "results"


TRUE_PARAMS_STATE = {
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

TRUE_PARAMS_OBS = {
    "sigma": 0.8,
    "xi": -0.15,
}


METHOD_SPECS = {
    "bootstrap": {
        "filter": dict(
            method="particle",
            particle_method="bootstrap",
            particle_n_particles=PARTICLE_N,
            rng=np.random.default_rng(1001),
        ),
        "smooth": dict(
            method="particle",
            particle_method="bootstrap",
            particle_n_particles=PARTICLE_N,
            particle_n_smoother_draws=PARTICLE_SMOOTHER_DRAWS,
            rng=np.random.default_rng(1002),
        ),
    },
    "auxiliary": {
        "filter": dict(
            method="particle",
            particle_method="auxiliary",
            particle_n_particles=PARTICLE_N,
            rng=np.random.default_rng(2001),
        ),
        "smooth": dict(
            method="particle",
            particle_method="auxiliary",
            particle_n_particles=PARTICLE_N,
            particle_n_smoother_draws=PARTICLE_SMOOTHER_DRAWS,
            rng=np.random.default_rng(2002),
        ),
    },
    "laplace": {
        "filter": dict(method="laplace"),
        "smooth": dict(method="laplace"),
    },
}


def build_model() -> StructuralModel:
    return StructuralModel(
        components=[
            LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
            DummySeasonal(period=12, mode="dynamic"),
        ],
        obs=GEVObs(),
    )


def simulate_demo_data(model: StructuralModel):
    x0 = np.concatenate(
        [
            np.array(
                [TRUE_PARAMS_STATE["m0_level"], TRUE_PARAMS_STATE["m0_trend"]],
                dtype=float,
            ),
            np.asarray(TRUE_PARAMS_STATE["m0_season"], dtype=float),
        ]
    )
    return simulate_statespace(
        T=T,
        model=model,
        params_state=TRUE_PARAMS_STATE,
        params_obs=TRUE_PARAMS_OBS,
        rng=np.random.default_rng(123),
        x0=x0,
    )


def run_method(name: str, y: np.ndarray, model: StructuralModel):
    spec = METHOD_SPECS[name]
    fr = filter_states(
        y=y,
        model=model,
        params_state=TRUE_PARAMS_STATE,
        params_obs=TRUE_PARAMS_OBS,
        **spec["filter"],
    )
    sr = smooth_states(
        y=y,
        model=model,
        params_state=TRUE_PARAMS_STATE,
        params_obs=TRUE_PARAMS_OBS,
        **spec["smooth"],
    )
    return fr, sr


def summarize_method(
    name: str,
    fr,
    sr,
    sim,
    model: StructuralModel,
    idx: dict[str, int],
) -> dict[str, float]:
    eta_filt = reconstruct_eta_path(fr.m_filt, model, TRUE_PARAMS_STATE)
    eta_smooth = reconstruct_eta_path(sr.m_smooth, model, TRUE_PARAMS_STATE)

    out = {
        "name": name,
        "eta_filter_rmse": rmse(sim.mu, eta_filt),
        "eta_smooth_rmse": rmse(sim.mu, eta_smooth),
        "alpha_filter_rmse": rmse(sim.x[:, idx["alpha"]], fr.m_filt[:, idx["alpha"]]),
        "alpha_smooth_rmse": rmse(sim.x[:, idx["alpha"]], sr.m_smooth[:, idx["alpha"]]),
        "beta_filter_rmse": rmse(sim.x[:, idx["beta"]], fr.m_filt[:, idx["beta"]]),
        "beta_smooth_rmse": rmse(sim.x[:, idx["beta"]], sr.m_smooth[:, idx["beta"]]),
        "g1_filter_rmse": rmse(sim.x[:, idx["g1"]], fr.m_filt[:, idx["g1"]]),
        "g1_smooth_rmse": rmse(sim.x[:, idx["g1"]], sr.m_smooth[:, idx["g1"]]),
    }

    ess = None if fr.meta is None else fr.meta.get("ess")
    if ess is not None:
        ess = np.asarray(ess, dtype=float)
        out["mean_ess"] = float(np.mean(ess[1:]))
        out["min_ess"] = float(np.min(ess[1:]))

    return out


def print_summary_table(rows: list[dict[str, float]]) -> None:
    print("\n=== GEV filter / smoother comparison ===")
    for row in rows:
        print(f"\nMethod: {row['name']}")
        print(f"  eta   filter RMSE : {row['eta_filter_rmse']:.4f}")
        print(f"  eta   smooth RMSE : {row['eta_smooth_rmse']:.4f}")
        print(f"  alpha filter RMSE : {row['alpha_filter_rmse']:.4f}")
        print(f"  alpha smooth RMSE : {row['alpha_smooth_rmse']:.4f}")
        print(f"  beta  filter RMSE : {row['beta_filter_rmse']:.4f}")
        print(f"  beta  smooth RMSE : {row['beta_smooth_rmse']:.4f}")
        print(f"  g1    filter RMSE : {row['g1_filter_rmse']:.4f}")
        print(f"  g1    smooth RMSE : {row['g1_smooth_rmse']:.4f}")
        if "mean_ess" in row:
            print(f"  mean ESS          : {row['mean_ess']:.1f}")
            print(f"  min ESS           : {row['min_ess']:.1f}")


def plot_eta_comparison(outdir: Path, t_obs: np.ndarray, sim, eta_paths: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.plot(t_obs, sim.y, linewidth=1.0, alpha=0.5, label="observed y")
    ax.plot(t_obs, sim.mu, linewidth=2.5, label="true eta")

    for name, eta in eta_paths.items():
        ax.plot(t_obs, eta, linewidth=1.7, label=f"{name} smoother eta")

    ax.set_title("GEV smoothing comparison: latent mean")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    outpath = outdir / "gev_compare_eta.png"
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
    smooth_paths: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.plot(t_state, truth, linewidth=2.5, label=f"true {ylabel}")

    for name, values in smooth_paths.items():
        ax.plot(t_state, values, linewidth=1.7, label=f"{name} smoother {ylabel}")

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    outpath = outdir / filename
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {outpath}")
    plt.show()


def plot_particle_ess(
    outdir: Path,
    t_state: np.ndarray,
    ess_paths: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)

    for name, ess in ess_paths.items():
        ax.plot(t_state[1:], ess[1:], linewidth=1.7, label=f"{name} ESS")

    ax.set_title("Particle filter ESS comparison")
    ax.set_xlabel("time")
    ax.set_ylabel("ESS")
    ax.legend()
    ax.grid(True, alpha=0.3)
    outpath = outdir / "gev_compare_ess.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {outpath}")
    plt.show()


def main() -> None:
    model = build_model()
    sim = simulate_demo_data(model)
    idx = state_index_map(sim.state_names)

    results = {}
    summaries = []
    for name in ("bootstrap", "auxiliary", "laplace"):
        fr, sr = run_method(name=name, y=sim.y, model=model)
        results[name] = {"filter": fr, "smooth": sr}
        summaries.append(summarize_method(name, fr, sr, sim, model, idx))

        ess = None if fr.meta is None else fr.meta.get("ess")
        if ess is not None:
            print(f"{name.title()} filter ESS (mean over time): {np.mean(np.asarray(ess)[1:]):.1f}")

    print("Finished filtering and smoothing with all methods.")
    print_summary_table(summaries)

    t_obs = np.arange(1, T + 1)
    t_state = np.arange(0, T + 1)
    outdir = ensure_results_dir(RESULTS_DIR)

    eta_paths = {
        name: reconstruct_eta_path(res["smooth"].m_smooth, model, TRUE_PARAMS_STATE)
        for name, res in results.items()
    }
    plot_eta_comparison(outdir=outdir, t_obs=t_obs, sim=sim, eta_paths=eta_paths)

    for state_name, filename in [("alpha", "gev_compare_alpha.png"), ("beta", "gev_compare_beta.png"), ("g1", "gev_compare_g1.png")]:
        plot_state_comparison(
            outdir=outdir,
            filename=filename,
            title=f"GEV smoothing comparison: {state_name}",
            ylabel=state_name,
            t_state=t_state,
            truth=sim.x[:, idx[state_name]],
            smooth_paths={name: res["smooth"].m_smooth[:, idx[state_name]] for name, res in results.items()},
        )

    plot_particle_ess(
        outdir=outdir,
        t_state=t_state,
        ess_paths={
            name: np.asarray(results[name]["filter"].meta["ess"], dtype=float)
            for name in ("bootstrap", "auxiliary")
        },
    )


if __name__ == "__main__":
    main()
