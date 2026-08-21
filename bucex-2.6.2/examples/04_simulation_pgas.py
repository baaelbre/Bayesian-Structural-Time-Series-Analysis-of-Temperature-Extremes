"""Fit all six simulations with Laplace-initialized PGAS.

The file is standalone. It constructs the simulation and fitted models with
the bucex API, creates a missing Laplace initializer, and then calls
``bx.fit(..., engine="pgas", init=laplace_fit)``.
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


# Results. Keep the folder name short: the simulation size, slab scales, and
# sampling effort identify the run; run_config.json contains every setting.
RESULTS_ROOT = Path(os.environ.get("BUCEX_RESULTS_ROOT", "results"))
SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = os.environ.get("BUCEX_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OVERWRITE = os.environ.get("BUCEX_OVERWRITE", "0").lower() in {"1", "true", "yes"}

# Simulation design. Keep aligned with scripts 02 and 03.
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
SIMULATION_SEED = int(os.environ.get("BUCEX_SIMULATION_SEED", "13081997"))

# Prior hyperparameters. Keep aligned with script 03.
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
TREND_PROBABILITIES = (1.0 / 3.0,) * 3
SEASON_PROBABILITIES = (1.0 / 3.0,) * 3

# MCMC. Final runs can set BUCEX_DRAWS=2000, BUCEX_WARMUP=2000,
# BUCEX_CHAINS=4, and BUCEX_PARTICLES=512 without editing this file.
DRAWS = int(os.environ.get("BUCEX_DRAWS", "400"))
WARMUP = int(os.environ.get("BUCEX_WARMUP", "100"))
CHAINS = int(os.environ.get("BUCEX_CHAINS", "1"))
PARTICLES = int(os.environ.get("BUCEX_PARTICLES", "128"))
SEED = int(os.environ.get("BUCEX_SEED", "13081997"))
PROGRESS = os.environ.get("BUCEX_PROGRESS", "1").lower() not in {"0", "false", "no"}
CHAIN_ONLY = os.environ.get("BUCEX_CHAIN_ONLY", "0").lower() in {"1", "true", "yes"}
COMBINE_RUNS = tuple(
    Path(value)
    for value in os.environ.get("BUCEX_COMBINE_RUNS", "").split(os.pathsep)
    if value
)

FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
DIAGNOSTIC_FIGURES = False
PHASE_LABELS = tuple(f"phase {index + 1}" for index in range(PERIOD))

RUN_SIGNATURE = f"n{N_TIME}p{PERIOD}_d{DRAWS}w{WARMUP}c{CHAINS}p{PARTICLES}"
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


phase = np.arange(PERIOD, dtype=float)
dynamic_cycle = -DYNAMIC_SEASON_AMPLITUDE * np.cos(2.0 * np.pi * phase / PERIOD)
dynamic_cycle -= dynamic_cycle.mean()
fixed_cycle = -FIXED_SEASON_AMPLITUDE * np.cos(2.0 * np.pi * phase / PERIOD)
fixed_cycle -= fixed_cycle.mean()

SCENARIOS = (
    {
        "name": "stationary",
        "key": "stationary",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="static", trend_mode="off"),), name="stationary"),
        "params": {"sigma": SIGMA, "xi": XI},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SIMULATION_SEED,
        "structural_truth": {"level": 1, "slope": 0, "seasonal": 0},
    },
    {
        "name": "linear_trend",
        "key": "linear",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="static", trend_mode="static"),), name="linear trend"),
        "params": {"sigma": SIGMA, "xi": XI},
        "initial_state": np.array([INITIAL_LEVEL, LINEAR_SLOPE]),
        "seed": SIMULATION_SEED + 1,
        "structural_truth": {"level": 1, "slope": 1, "seasonal": 0},
    },
    {
        "name": "random_walk",
        "key": "random_walk",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="off"),), name="random walk"),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": RANDOM_WALK_SD},
        "initial_state": np.array([INITIAL_LEVEL]),
        "seed": SIMULATION_SEED + 2,
        "structural_truth": {"level": 2, "slope": 0, "seasonal": 0},
    },
    {
        "name": "local_linear_trend",
        "key": "llt",
        "model": bx.Model(bx.GEV(), (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),), name="local linear trend"),
        "params": {"sigma": SIGMA, "xi": XI, "sd.level": LOCAL_LEVEL_SD, "sd.slope": LOCAL_SLOPE_SD},
        "initial_state": np.array([INITIAL_LEVEL, LOCAL_INITIAL_SLOPE]),
        "seed": SIMULATION_SEED + 3,
        "structural_truth": {"level": 2, "slope": 2, "seasonal": 0},
    },
    {
        "name": "stationary_dynamic_season",
        "key": "dynamic_season",
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
        "key": "llt_season",
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

FIT_MODEL = bx.Model(
    bx.GEV(xi_bounds=XI_PRIOR_BOUNDS),
    (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"), bx.DummySeasonal(PERIOD, mode="dynamic")),
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
    if COMBINE_RUNS and len(COMBINE_RUNS) != CHAINS:
        raise ValueError(
            f"BUCEX_COMBINE_RUNS contains {len(COMBINE_RUNS)} runs, "
            f"but BUCEX_CHAINS={CHAINS}."
        )
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "legend.frameon": False})
    labels = {0: "zero", 1: "fixed", 2: "dynamic"}
    comparison_rows = []

    config_path = OUTPUT_DIR / "run_config.json"
    if config_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Refusing to overwrite {config_path}; set BUCEX_OVERWRITE=1 to rerun."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {
        "script": SCRIPT_NAME,
        "created_at": datetime.now().astimezone().isoformat(),
        "run_timestamp": RUN_TIMESTAMP,
        "run_signature": RUN_SIGNATURE,
        "output_directory": str(OUTPUT_DIR),
        "bucex_version": bx.__version__,
        "engine": "pgas",
        "initializer": "laplace",
        "simulation": {
            "n_time": N_TIME,
            "period": PERIOD,
            "sigma": SIGMA,
            "xi": XI,
            "initial_level": INITIAL_LEVEL,
            "linear_slope": LINEAR_SLOPE,
            "random_walk_sd": RANDOM_WALK_SD,
            "local_level_sd": LOCAL_LEVEL_SD,
            "local_slope_sd": LOCAL_SLOPE_SD,
            "local_initial_slope": LOCAL_INITIAL_SLOPE,
            "dynamic_season_amplitude": DYNAMIC_SEASON_AMPLITUDE,
            "fixed_season_amplitude": FIXED_SEASON_AMPLITUDE,
            "seasonal_sd": SEASONAL_SD,
            "seed": SIMULATION_SEED,
        },
        "fit_model": FIT_MODEL.to_dict(),
        "priors": PRIOR_SETTINGS,
        "mcmc": {
            "draws": DRAWS,
            "warmup": WARMUP,
            "chains": CHAINS,
            "particles": PARTICLES,
            "seed": SEED,
        },
        "scenarios": [scenario["name"] for scenario in SCENARIOS],
        "scenario_directories": {
            scenario["name"]: scenario["key"] for scenario in SCENARIOS
        },
        "figures": {
            "formats": list(FIGURE_FORMATS),
            "dpi": FIGURE_DPI,
            "diagnostics": DIAGNOSTIC_FIGURES,
        },
        "chain_only": CHAIN_ONLY,
        "combined_chain_runs": [str(path) for path in COMBINE_RUNS],
    }
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    for number, scenario in enumerate(SCENARIOS):
        data_path = OUTPUT_DIR / "simulations" / f"{scenario['key']}.csv"
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
            simulation = bx.simulate(scenario["model"], N_TIME, scenario["params"], initial_state=scenario["initial_state"], seed=scenario["seed"])
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

        laplace_path = OUTPUT_DIR / "fits" / scenario["key"] / "laplace" / "combined.bucex"
        if laplace_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(laplace_path)
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
                raise ValueError(f"{laplace_path} does not match the current settings.")
        elif COMBINE_RUNS:
            source_paths = [
                run_dir / "fits" / scenario["key"] / "laplace" / "combined.bucex"
                for run_dir in COMBINE_RUNS
            ]
            missing = [path for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine Laplace initializers; missing fit file(s):\n  "
                    + "\n  ".join(str(path) for path in missing)
                )
            laplace_fit = bx.combine_fits(
                [bx.FitResult.load(path) for path in source_paths]
            )
            laplace_fit.metadata["combined_chain_sources"] = [
                str(path) for path in source_paths
            ]
            laplace_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(laplace_path)
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
            laplace_fit.metadata.update({"example": "simulation_laplace_initializer", "truth": truth, "prior_settings": PRIOR_SETTINGS})
            laplace_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(laplace_path)

        pgas_path = OUTPUT_DIR / "fits" / scenario["key"] / "pgas" / "combined.bucex"
        if pgas_path.is_file() and not OVERWRITE:
            pgas_fit = bx.FitResult.load(pgas_path)
            if (
                pgas_fit.n_time != N_TIME
                or pgas_fit.n_chains != CHAINS
                or pgas_fit.draws_per_chain != DRAWS
                or pgas_fit.family != FIT_MODEL.family
                or pgas_fit.model.period != FIT_MODEL.period
                or pgas_fit.state_names != FIT_MODEL.state_names
                or pgas_fit.compiled.noise_names != FIT_MODEL.noise_names
                or pgas_fit.metadata.get("prior_settings") != PRIOR_SETTINGS
                or int(pgas_fit.metadata.get("particles", PARTICLES)) != PARTICLES
            ):
                raise ValueError(f"{pgas_path} does not match the current settings.")
            print(f"Reusing {pgas_path}")
        elif COMBINE_RUNS:
            source_paths = [
                run_dir / "fits" / scenario["key"] / "pgas" / "combined.bucex"
                for run_dir in COMBINE_RUNS
            ]
            missing = [path for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine PGAS chains; missing fit file(s):\n  "
                    + "\n  ".join(str(path) for path in missing)
                )
            pgas_fit = bx.combine_fits(
                [bx.FitResult.load(path) for path in source_paths]
            )
            pgas_fit.metadata["combined_chain_sources"] = [
                str(path) for path in source_paths
            ]
            pgas_fit.metadata["warm_start_source"] = str(laplace_path)
            pgas_path.parent.mkdir(parents=True, exist_ok=True)
            pgas_fit.save(pgas_path)
        else:
            pgas_fit = bx.fit(
                y,
                model=FIT_MODEL,
                priors=laplace_fit.priors,
                engine="pgas",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(draws=DRAWS, warmup=WARMUP, chains=CHAINS, seed=SEED + 10_000 + 100 * number, progress=PROGRESS),
                particles=bx.Particles(n=PARTICLES, proposal="guided"),
                name=scenario["name"],
                init=laplace_fit,
            )
            pgas_fit.metadata.update(
                {
                    "example": "simulation_pgas",
                    "truth": truth,
                    "warm_start_source": str(laplace_path),
                    "prior_settings": PRIOR_SETTINGS,
                    "particles": PARTICLES,
                }
            )
            pgas_path.parent.mkdir(parents=True, exist_ok=True)
            pgas_fit.save(pgas_path)

        if CHAIN_ONLY:
            print(f"One-chain PGAS fit complete: {scenario['name']}")
            continue

        table_dir = OUTPUT_DIR / "tables" / scenario["key"]
        figure_dir = OUTPUT_DIR / "figures" / scenario["key"]
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = pgas_fit.diagnostics()

        pd.DataFrame.from_dict(pgas_fit.static_summary(), orient="index").rename_axis("parameter").to_csv(table_dir / "parameters.csv")
        diagnostics["parameters"].to_csv(table_dir / "diagnostics.csv")
        pd.DataFrame([{"metric": key, "value": value} for key, value in diagnostics["engine"].items()]).to_csv(table_dir / "algorithm.csv", index=False)
        eta_draws = pgas_fit.eta_draws(original_scale=True)
        lower, median, upper = np.quantile(eta_draws, [0.05, 0.50, 0.95], axis=0)
        pd.DataFrame({"time": table["time"], "observed": pgas_fit.observed, "lower": lower, "median": median, "upper": upper, "truth": table["eta"]}).to_csv(
            table_dir / "trajectory.csv", index=False
        )

        selection = pgas_fit.component_probabilities().reset_index()
        selection["truth_code"] = selection["process"].map(scenario["structural_truth"])
        selection["truth_state"] = selection["truth_code"].map(labels)
        selection["probability_true_state"] = [row[labels[int(row["truth_code"])]] for _, row in selection.iterrows()]
        selection.to_csv(table_dir / "selection.csv", index=False)
        pgas_fit.structural_model_probabilities().to_csv(table_dir / "models.csv", index=False)
        pgas_fit.component_transition_summary().reset_index().to_csv(table_dir / "switching.csv", index=False)
        (table_dir / "summary.json").write_text(
            json.dumps(
                {
                    "fit": str(pgas_path),
                    "warm_start": str(laplace_path),
                    "n_time": pgas_fit.n_time,
                    "n_chains": pgas_fit.n_chains,
                    "draws_per_chain": pgas_fit.draws_per_chain,
                    "particles": PARTICLES,
                    "plan": pgas_fit.plan.to_dict(),
                    "engine_diagnostics": diagnostics["engine"],
                    "prior_settings": PRIOR_SETTINGS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        for engine_name, current_fit in (("laplace", laplace_fit), ("pgas", pgas_fit)):
            current = current_fit.component_probabilities().reset_index()
            for _, row in current.iterrows():
                truth_state = labels[scenario["structural_truth"][row["process"]]]
                comparison_rows.append(
                    {
                        "scenario": scenario["name"],
                        "engine": engine_name,
                        "process": row["process"],
                        "truth_state": truth_state,
                        "probability_true_state": float(row[truth_state]),
                    }
                )

        figure, axis = pgas_fit.plot("predictor", credible_interval=0.90)
        axis.plot(np.arange(N_TIME), table["eta"], color="#123B4A", linestyle="--", linewidth=1.2, label="true predictor")
        axis.set_title(f"{scenario['name']}: posterior latent predictor (PGAS)")
        axis.legend()
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"trajectory.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = pgas_fit.plot("component_probabilities")
        figure.suptitle(f"{scenario['name']}: structural selection (PGAS)")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"selection.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = pgas_fit.plot("process_sds", truths=truth["parameter_truth"], title=f"{scenario['name']}: prior to posterior (PGAS)")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"process_sd.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = pgas_fit.plot("parameter_densities", parameters=("sigma", "xi"), truths=truth["parameter_truth"])
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"gev.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = pgas_fit.plot("season", labels=PHASE_LABELS, show_interval=False)
        figure.suptitle(f"{scenario['name']}: phase-specific posterior trajectories")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"season.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        if DIAGNOSTIC_FIGURES:
            for kind, filename in (("traces", "sd_traces"), ("acf", "acf")):
                figure, _ = pgas_fit.plot(kind)
                for extension in FIGURE_FORMATS:
                    figure.savefig(figure_dir / f"{filename}.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(figure)
        print(f"PGAS fit complete: {scenario['name']}")

    if CHAIN_ONLY:
        print(f"One-chain PGAS simulation outputs: {OUTPUT_DIR}")
        return

    comparison_path = OUTPUT_DIR / "tables" / "engine_selection.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    print(f"PGAS simulation outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
