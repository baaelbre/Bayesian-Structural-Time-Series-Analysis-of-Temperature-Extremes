"""Fit all six structural simulations with Laplace state updates.

This is a standalone, sequential example: it declares the simulation models,
declares the larger fitted model, constructs the priors, calls ``bx.fit``, and
writes the posterior summaries without a workflow layer.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (SOURCE_ROOT / "bucex").is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


# Results.
RESULTS_ROOT = Path(os.environ.get("BUCEX_RESULTS_ROOT", "results"))
TIMESTAMP_RESULTS = os.environ.get("BUCEX_TIMESTAMP_RESULTS", "1").lower() not in {"0", "false", "no"}
RUN_ID = os.environ.get("BUCEX_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = RESULTS_ROOT / RUN_ID if TIMESTAMP_RESULTS else RESULTS_ROOT
OVERWRITE = os.environ.get("BUCEX_OVERWRITE", "0").lower() in {"1", "true", "yes"}

# Simulation design. Keep aligned with 02_structural_simulations.py.
N_TIME = int(os.environ.get("BUCEX_N_TIME", "1000"))
PERIOD = int(os.environ.get("BUCEX_PERIOD", "4"))
SIGMA = 1.50
XI = -0.30
INITIAL_LEVEL = 25.0
LINEAR_SLOPE = 0.006
RANDOM_WALK_SD = 0.05
LOCAL_LEVEL_SD = 0.05
LOCAL_SLOPE_SD = 0.00050
LOCAL_INITIAL_SLOPE = 0.003
DYNAMIC_SEASON_AMPLITUDE = 0.25
FIXED_SEASON_AMPLITUDE = 0.25
SEASONAL_SD = 0.05
SIMULATION_SEED = 13_081_997

# Prior hyperparameters.
ALPHA_PRIOR_SD = 3.2
BETA_PRIOR_MEAN = 0.0
BETA_PRIOR_SD = 0.01
INITIAL_SEASON_PRIOR_SD = 0.5
SIGMA2_PRIOR_A = 2.0
SIGMA2_PRIOR_B = 2.25
XI_PRIOR_BOUNDS = (-0.50, 0.50)
XI_MAX_ABS = 0.50
INNOVATION_SLAB_SD = {"level": 0.10, "trend": 0.0008, "season": 0.07}
LEVEL_DYNAMIC_PROBABILITY = 0.50
TREND_PROBABILITIES = (1.0 / 3.0,) * 3  # zero, fixed, dynamic
SEASON_PROBABILITIES = (1.0 / 3.0,) * 3  # zero, fixed, dynamic

# MCMC. Environment variables make the same file convenient on an HPC node.
DRAWS = int(os.environ.get("BUCEX_DRAWS", "400"))
WARMUP = int(os.environ.get("BUCEX_WARMUP", "100"))
CHAINS = int(os.environ.get("BUCEX_CHAINS", "1"))
SEED = int(os.environ.get("BUCEX_SEED", "13081997"))
PROGRESS = os.environ.get("BUCEX_PROGRESS", "1").lower() not in {"0", "false", "no"}

FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
DIAGNOSTIC_FIGURES = False
PHASE_LABELS = tuple(f"phase {index + 1}" for index in range(PERIOD))


phase = np.arange(PERIOD, dtype=float)
dynamic_cycle = -DYNAMIC_SEASON_AMPLITUDE * np.cos(2.0 * np.pi * phase / PERIOD)
dynamic_cycle -= dynamic_cycle.mean()
fixed_cycle = -FIXED_SEASON_AMPLITUDE * np.cos(2.0 * np.pi * phase / PERIOD)
fixed_cycle -= fixed_cycle.mean()

SCENARIOS = (
    {
        "name": "stationary",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="static", trend_mode="off"),), name="stationary"),
        "params": {"sigma": SIGMA, "xi": XI},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SIMULATION_SEED,
        "structural_truth": {"level": 1, "slope": 0, "seasonal": 0},
    },
    {
        "name": "linear_trend",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="static", trend_mode="static"),), name="linear trend"),
        "params": {"sigma": SIGMA, "xi": XI},
        "initial_state": np.array([INITIAL_LEVEL, LINEAR_SLOPE]),
        "seed": SIMULATION_SEED + 1,
        "structural_truth": {"level": 1, "slope": 1, "seasonal": 0},
    },
    {
        "name": "random_walk",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="off"),), name="random walk"),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": RANDOM_WALK_SD},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SIMULATION_SEED + 2,
        "structural_truth": {"level": 2, "slope": 0, "seasonal": 0},
    },
    {
        "name": "local_linear_trend",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),), name="local linear trend"),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": LOCAL_LEVEL_SD, "sd.slope": LOCAL_SLOPE_SD},
        "initial_state": np.array([INITIAL_LEVEL, LOCAL_INITIAL_SLOPE]),
        "seed": SIMULATION_SEED + 3,
        "structural_truth": {"level": 2, "slope": 2, "seasonal": 0},
    },
    {
        "name": "stationary_dynamic_season",
        "model": bx.Model(
            bx.GEV(),
            (bx.LocalLinearTrend(level_mode="static", trend_mode="off"), bx.DummySeasonal(PERIOD, mode="dynamic")),
            name="stationary plus changing seasonality",
        ),
        "params": {"sigma": SIGMA, "xi": XI, "sd.seasonal": SEASONAL_SD},
        "initial_state": np.r_[INITIAL_LEVEL, dynamic_cycle[1:]],
        "seed": SIMULATION_SEED + 4,
        "structural_truth": {"level": 1, "slope": 0, "seasonal": 2},
    },
    {
        "name": "local_linear_trend_fixed_season",
        "model": bx.Model(
            bx.GEV(),
            (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"), bx.DummySeasonal(PERIOD, mode="static")),
            name="local linear trend plus fixed seasonality",
        ),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": LOCAL_LEVEL_SD, "sd.slope": LOCAL_SLOPE_SD},
        "initial_state": np.r_[INITIAL_LEVEL, LOCAL_INITIAL_SLOPE, fixed_cycle[1:]],
        "seed": SIMULATION_SEED + 3,
        "structural_truth": {"level": 2, "slope": 2, "seasonal": 1},
    },
)

# Fit one encompassing model to every scenario. SSVS decides whether each
# process is zero, fixed, or dynamic; no scenario-specific model is supplied.
FIT_MODEL = bx.Model(
    bx.GEV(xi_bounds=XI_PRIOR_BOUNDS),
    (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=PERIOD, mode="dynamic"),
    ),
    name="GEV unobserved-components model",
)

PRIOR_SETTINGS = {
    "alpha_sd": ALPHA_PRIOR_SD,
    "beta_mean": BETA_PRIOR_MEAN,
    "beta_sd": BETA_PRIOR_SD,
    "seasonal_initial_sd": INITIAL_SEASON_PRIOR_SD,
    "sigma2": {"a": SIGMA2_PRIOR_A, "b": SIGMA2_PRIOR_B},
    "xi_bounds": list(XI_PRIOR_BOUNDS),
    "xi_max_abs": XI_MAX_ABS,
    "innovation_slab_sd": INNOVATION_SLAB_SD,
    "level_dynamic_probability": LEVEL_DYNAMIC_PROBABILITY,
    "trend_probabilities": list(TREND_PROBABILITIES),
    "season_probabilities": list(SEASON_PROBABILITIES),
}


def main() -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "legend.frameon": False})
    labels = {0: "zero", 1: "fixed", 2: "dynamic"}

    for number, scenario in enumerate(SCENARIOS):
        data_path = OUTPUT_DIR / "simulations" / "structure" / f"{scenario['name']}.csv"
        truth_path = data_path.with_suffix(".json")
        expected_truth = {
            "name": scenario["name"],
            "n_time": N_TIME,
            "period": PERIOD,
            "model": scenario["model"].to_dict(),
            "params": scenario["params"],
            "parameter_truth": {
                "sigma": SIGMA,
                "xi": XI,
                "sd.level": scenario["params"].get("sd.level", 0.0),
                "sd.slope": scenario["params"].get("sd.slope", 0.0),
                "sd.seasonal": scenario["params"].get("sd.seasonal", 0.0),
            },
            "structural_truth": scenario["structural_truth"],
            "initial_state": scenario["initial_state"].tolist(),
            "seed": scenario["seed"],
        }

        if data_path.is_file() and truth_path.is_file() and not OVERWRITE:
            table = pd.read_csv(data_path)
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            for key in ("name", "n_time", "period", "model", "params", "initial_state", "seed"):
                if truth.get(key) != expected_truth[key]:
                    raise ValueError(f"{truth_path} has different {key}; set BUCEX_OVERWRITE=1.")
        else:
            if not OVERWRITE and (data_path.exists() or truth_path.exists()):
                raise FileExistsError(f"Only one simulation artifact exists for {scenario['name']}; set BUCEX_OVERWRITE=1.")
            simulation = bx.simulate(
                scenario["model"],
                N_TIME,
                scenario["params"],
                initial_state=scenario["initial_state"],
                seed=scenario["seed"],
            )
            state_names = scenario["model"].state_names
            states = simulation.states[1:]
            seasonal_name = next((name for name in state_names if name.startswith("seasonal[")), None)
            table = pd.DataFrame(
                {
                    "time": np.arange(1, N_TIME + 1),
                    "cycle": np.arange(N_TIME) // PERIOD + 1,
                    "phase": np.arange(N_TIME) % PERIOD + 1,
                    "y": simulation.y,
                    "eta": simulation.eta,
                    "level": states[:, state_names.index("level")],
                    "slope": states[:, state_names.index("slope")] if "slope" in state_names else np.zeros(N_TIME),
                    "seasonal": states[:, state_names.index(seasonal_name)] if seasonal_name else np.zeros(N_TIME),
                }
            )
            truth = expected_truth
            data_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(data_path, index=False)
            truth_path.write_text(json.dumps(truth, indent=2, sort_keys=True), encoding="utf-8")
            # Fit the canonical CSV representation. A later standalone PGAS
            # process reads these exact floats, and warm starts deliberately
            # require bit-identical observations.
            table = pd.read_csv(data_path)

        y = table["y"].to_numpy(float)
        priors = bx.ssvs_gev_priors(
            period=PERIOD,
            alpha_mean=float(np.median(y)),
            alpha_sd=ALPHA_PRIOR_SD,
            beta_mean=BETA_PRIOR_MEAN,
            beta_sd=BETA_PRIOR_SD,
            seasonal_initial_sd=INITIAL_SEASON_PRIOR_SD,
            sigma2_prior=bx.InverseGammaPrior(SIGMA2_PRIOR_A, SIGMA2_PRIOR_B),
            xi_prior=bx.UniformPrior(*XI_PRIOR_BOUNDS),
            xi_max_abs=XI_MAX_ABS,
            innovation_slab_sd=INNOVATION_SLAB_SD,
            level_dynamic_probability=LEVEL_DYNAMIC_PROBABILITY,
            trend_probabilities=TREND_PROBABILITIES,
            season_probabilities=SEASON_PROBABILITIES,
        )

        fit_path = OUTPUT_DIR / "fits" / "simulations" / "laplace" / scenario["name"] / "combined.bucex"
        if fit_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(fit_path)
            if (
                laplace_fit.n_time != N_TIME
                or laplace_fit.n_chains != CHAINS
                or laplace_fit.draws_per_chain != DRAWS
                or laplace_fit.family != FIT_MODEL.family
                or laplace_fit.model.period != FIT_MODEL.period
                or laplace_fit.state_names != FIT_MODEL.state_names
                or laplace_fit.compiled.noise_names != FIT_MODEL.noise_names
                or laplace_fit.metadata.get("prior_settings") != PRIOR_SETTINGS
            ):
                raise ValueError(f"{fit_path} does not match the current model, prior, or MCMC settings.")
            print(f"Reusing {fit_path}")
        else:
            laplace_fit = bx.fit(
                y,
                model=FIT_MODEL,
                priors=priors,
                engine="laplace",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(draws=DRAWS, warmup=WARMUP, chains=CHAINS, seed=SEED + 100 * number, progress=PROGRESS),
                name=scenario["name"],
            )
            laplace_fit.metadata.update({"example": "simulation_laplace", "truth": truth, "prior_settings": PRIOR_SETTINGS})
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(fit_path)

        # Tables are written explicitly so the example shows what every
        # FitResult method returns.
        table_dir = OUTPUT_DIR / "tables" / "simulations" / "laplace" / scenario["name"]
        figure_dir = OUTPUT_DIR / "figures" / "simulations" / "laplace" / scenario["name"]
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = laplace_fit.diagnostics()

        parameters = pd.DataFrame.from_dict(laplace_fit.static_summary(), orient="index").rename_axis("parameter")
        parameters.to_csv(table_dir / "parameters.csv")
        diagnostics["parameters"].to_csv(table_dir / "diagnostics.csv")
        pd.DataFrame([{"metric": key, "value": value} for key, value in diagnostics["engine"].items()]).to_csv(
            table_dir / "algorithm_diagnostics.csv", index=False
        )

        eta_draws = laplace_fit.eta_draws(original_scale=True)
        lower, median, upper = np.quantile(eta_draws, [0.05, 0.50, 0.95], axis=0)
        pd.DataFrame(
            {"time": table["time"], "observed": laplace_fit.observed, "lower": lower, "median": median, "upper": upper, "truth": table["eta"]}
        ).to_csv(table_dir / "posterior_trajectory.csv", index=False)

        selection = laplace_fit.component_probabilities().reset_index()
        selection["truth_code"] = selection["process"].map(scenario["structural_truth"])
        selection["truth_state"] = selection["truth_code"].map(labels)
        selection["probability_true_state"] = [row[labels[int(row["truth_code"])]] for _, row in selection.iterrows()]
        selection.to_csv(table_dir / "selection_probabilities.csv", index=False)
        laplace_fit.structural_model_probabilities().to_csv(table_dir / "structural_models.csv", index=False)
        laplace_fit.component_transition_summary().reset_index().to_csv(table_dir / "selection_switching.csv", index=False)
        (table_dir / "fit_summary.json").write_text(
            json.dumps(
                {
                    "fit": str(fit_path),
                    "series": laplace_fit.series_name,
                    "n_time": laplace_fit.n_time,
                    "n_chains": laplace_fit.n_chains,
                    "draws_per_chain": laplace_fit.draws_per_chain,
                    "plan": laplace_fit.plan.to_dict(),
                    "engine_diagnostics": diagnostics["engine"],
                    "prior_settings": PRIOR_SETTINGS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        figure, axis = laplace_fit.plot("predictor", credible_interval=0.90)
        axis.plot(np.arange(N_TIME), table["eta"], color="#123B4A", linestyle="--", linewidth=1.2, label="true predictor")
        axis.set_title(f"{scenario['name']}: posterior latent predictor (Laplace)")
        axis.legend()
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"posterior_trajectory.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("component_probabilities")
        figure.suptitle(f"{scenario['name']}: structural selection (Laplace)")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"selection_probabilities.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("process_sds", truths=truth["parameter_truth"], title=f"{scenario['name']}: prior to posterior (Laplace)")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"prior_to_posterior_process_sd.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("parameter_densities", parameters=("sigma", "xi"), truths=truth["parameter_truth"])
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"gev_parameters.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("season", labels=PHASE_LABELS, show_interval=False)
        figure.suptitle(f"{scenario['name']}: phase-specific posterior trajectories")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"seasonal_trajectories.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        if DIAGNOSTIC_FIGURES:
            for kind, filename in (("traces", "process_sd_traces"), ("acf", "parameter_acfs")):
                figure, _ = laplace_fit.plot(kind)
                for extension in FIGURE_FORMATS:
                    figure.savefig(figure_dir / f"{filename}.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(figure)
        print(f"Laplace fit complete: {scenario['name']}")

    print(f"Laplace simulation outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
