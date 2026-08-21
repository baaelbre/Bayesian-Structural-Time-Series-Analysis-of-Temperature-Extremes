"""Fit all four Uccle extremes with Laplace-initialized PGAS.

This standalone example repeats the model and prior declarations so it can be
read and submitted without hidden settings or workflow objects.
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


# Results. The folder name keeps only the data range, slab scales, and sampling
# effort. run_config.json contains the full data, prior, and sampler settings.
RESULTS_ROOT = Path(os.environ.get("BUCEX_RESULTS_ROOT", "results"))
SCRIPT_NAME = Path(__file__).stem
RUN_TIMESTAMP = os.environ.get("BUCEX_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
OVERWRITE = os.environ.get("BUCEX_OVERWRITE", "0").lower() in {"1", "true", "yes"}

# Data.
DATA_DIR: Path | None = (
    Path(os.environ["BUCEX_DATA_DIR"]) if "BUCEX_DATA_DIR" in os.environ else None
)
START = os.environ.get("BUCEX_START", "1892-01-01")
END: str | None = os.environ.get("BUCEX_END") or None
SERIES = ("TXx", "TXn", "TNx", "TNn")
PERIOD = 12

# Prior hyperparameters. Keep these identical to 05_uccle_laplace.py.
ALPHA_PRIOR_SD = 3.2
BETA_PRIOR_MEAN = 0.0
BETA_PRIOR_SD = 0.004
INITIAL_SEASON_PRIOR_SD = 2.25
SIGMA2_PRIOR_A = 2.0
SIGMA2_PRIOR_B = 2.0
XI_PRIOR_BOUNDS = (-0.50, 0.50)
XI_MAX_ABS = 0.50
INNOVATION_SLAB_SD = {"level": 0.05, "trend": 0.00010, "season": 0.09}
LEVEL_DYNAMIC_PROBABILITY = 0.50
TREND_PROBABILITIES = (1.0 / 3.0,) * 3
SEASON_PROBABILITIES = (1.0 / 3.0,) * 3

# MCMC. These values can also be supplied through the environment.
DRAWS = int(os.environ.get("BUCEX_DRAWS", "250"))
WARMUP = int(os.environ.get("BUCEX_WARMUP", "250"))
CHAINS = int(os.environ.get("BUCEX_CHAINS", "2"))
PARTICLES = int(os.environ.get("BUCEX_PARTICLES", "128"))
SEED = int(os.environ.get("BUCEX_SEED", "56000"))
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

RUN_SIGNATURE = (
    f"y{START.removesuffix('-01-01')}-{(END or 'latest').removesuffix('-12-31')}"
    f"_d{DRAWS}w{WARMUP}c{CHAINS}p{PARTICLES}"
)
OUTPUT_DIR = RESULTS_ROOT / SCRIPT_NAME / f"{RUN_TIMESTAMP}__{RUN_SIGNATURE}"


MODEL = bx.Model(
    bx.GEV(xi_bounds=XI_PRIOR_BOUNDS),
    (bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"), bx.DummySeasonal(PERIOD, mode="dynamic")),
    name="monthly GEV unobserved-components model",
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
    selection_rows = []

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
        "data": {
            "data_dir": None if DATA_DIR is None else str(DATA_DIR),
            "start": START,
            "end": END,
            "series": list(SERIES),
            "period": PERIOD,
        },
        "model": MODEL.to_dict(),
        "priors": PRIOR_SETTINGS,
        "mcmc": {
            "draws": DRAWS,
            "warmup": WARMUP,
            "chains": CHAINS,
            "particles": PARTICLES,
            "seed": SEED,
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

    for number, name in enumerate(SERIES):
        values = bx.load_uccle_series(name, DATA_DIR, start=START, end=END)
        tail = bx.UCCLE_INFO[name]["tail"]
        sign = -1.0 if tail == "min" else 1.0
        transformed = sign * values.to_numpy(float)
        priors = bx.ssvs_gev_priors(
            period=PERIOD,
            alpha_mean=float(np.median(transformed)),
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

        laplace_path = OUTPUT_DIR / "fits" / name / "laplace" / "combined.bucex"
        if laplace_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(laplace_path)
            if (
                laplace_fit.n_time != len(values)
                or laplace_fit.n_chains != CHAINS
                or laplace_fit.draws_per_chain != DRAWS
                or laplace_fit.family != MODEL.family
                or laplace_fit.model.period != MODEL.period
                or laplace_fit.state_names != MODEL.state_names
                or laplace_fit.compiled.noise_names != MODEL.noise_names
                or laplace_fit.metadata.get("prior_settings") != PRIOR_SETTINGS
            ):
                raise ValueError(f"{laplace_path} does not match the current settings.")
        elif COMBINE_RUNS:
            source_paths = [
                run_dir / "fits" / name / "laplace" / "combined.bucex"
                for run_dir in COMBINE_RUNS
            ]
            missing = [path for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine Uccle Laplace initializers; missing fit file(s):\n  "
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
                values,
                model=MODEL,
                priors=priors,
                engine="laplace",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(draws=DRAWS, warmup=WARMUP, chains=CHAINS, seed=SEED + 100 * number, progress=PROGRESS),
                name=name,
                tail=tail,
            )
            laplace_fit.metadata.update(
                {
                    "example": "uccle_laplace_initializer",
                    "description": bx.UCCLE_INFO[name]["description"],
                    "start": START,
                    "end": END,
                    "prior_settings": PRIOR_SETTINGS,
                }
            )
            laplace_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(laplace_path)

        pgas_path = OUTPUT_DIR / "fits" / name / "pgas" / "combined.bucex"
        if pgas_path.is_file() and not OVERWRITE:
            pgas_fit = bx.FitResult.load(pgas_path)
            if (
                pgas_fit.n_time != len(values)
                or pgas_fit.n_chains != CHAINS
                or pgas_fit.draws_per_chain != DRAWS
                or pgas_fit.family != MODEL.family
                or pgas_fit.model.period != MODEL.period
                or pgas_fit.state_names != MODEL.state_names
                or pgas_fit.compiled.noise_names != MODEL.noise_names
                or pgas_fit.metadata.get("prior_settings") != PRIOR_SETTINGS
                or int(pgas_fit.metadata.get("particles", PARTICLES)) != PARTICLES
            ):
                raise ValueError(f"{pgas_path} does not match the current settings.")
            print(f"Reusing {pgas_path}")
        elif COMBINE_RUNS:
            source_paths = [
                run_dir / "fits" / name / "pgas" / "combined.bucex"
                for run_dir in COMBINE_RUNS
            ]
            missing = [path for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot combine Uccle PGAS chains; missing fit file(s):\n  "
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
                values,
                model=MODEL,
                priors=laplace_fit.priors,
                engine="pgas",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(draws=DRAWS, warmup=WARMUP, chains=CHAINS, seed=SEED + 10_000 + 100 * number, progress=PROGRESS),
                particles=bx.Particles(n=PARTICLES, proposal="guided"),
                name=name,
                tail=tail,
                init=laplace_fit,
            )
            pgas_fit.metadata.update(
                {
                    "example": "uccle_pgas",
                    "description": bx.UCCLE_INFO[name]["description"],
                    "warm_start_source": str(laplace_path),
                    "start": START,
                    "end": END,
                    "prior_settings": PRIOR_SETTINGS,
                    "particles": PARTICLES,
                }
            )
            pgas_path.parent.mkdir(parents=True, exist_ok=True)
            pgas_fit.save(pgas_path)

        if CHAIN_ONLY:
            print(f"One-chain Uccle PGAS fit complete: {name}")
            continue

        table_dir = OUTPUT_DIR / "tables" / name
        figure_dir = OUTPUT_DIR / "figures" / name
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = pgas_fit.diagnostics()

        pd.DataFrame.from_dict(pgas_fit.static_summary(), orient="index").rename_axis("parameter").to_csv(table_dir / "parameters.csv")
        diagnostics["parameters"].to_csv(table_dir / "diagnostics.csv")
        pd.DataFrame([{"metric": key, "value": value} for key, value in diagnostics["engine"].items()]).to_csv(table_dir / "algorithm.csv", index=False)
        eta_draws = pgas_fit.eta_draws(original_scale=True)
        lower, median, upper = np.quantile(eta_draws, [0.05, 0.50, 0.95], axis=0)
        pd.DataFrame({"date": values.index, "observed": values.to_numpy(), "lower": lower, "median": median, "upper": upper}).to_csv(
            table_dir / "trajectory.csv", index=False
        )
        selection = pgas_fit.component_probabilities().reset_index()
        selection.insert(0, "series", name)
        selection.insert(1, "engine", "pgas")
        selection.to_csv(table_dir / "selection.csv", index=False)
        pgas_fit.structural_model_probabilities().to_csv(table_dir / "models.csv", index=False)
        pgas_fit.component_transition_summary().reset_index().to_csv(table_dir / "switching.csv", index=False)
        (table_dir / "summary.json").write_text(
            json.dumps(
                {
                    "fit": str(pgas_path),
                    "warm_start": str(laplace_path),
                    "series": name,
                    "description": bx.UCCLE_INFO[name]["description"],
                    "tail": tail,
                    "start": str(values.index.min().date()),
                    "end": str(values.index.max().date()),
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

        laplace_selection = laplace_fit.component_probabilities().reset_index()
        laplace_selection.insert(0, "series", name)
        laplace_selection.insert(1, "engine", "laplace")
        selection_rows.extend((laplace_selection, selection))

        figure, axis = pgas_fit.plot("predictor", credible_interval=0.90)
        axis.set_title(f"{name}: posterior latent predictor (PGAS)")
        axis.set_ylabel("GEV location / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"trajectory.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = pgas_fit.plot("component_probabilities")
        figure.suptitle(f"{name}: structural selection (PGAS)")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"selection.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = pgas_fit.plot("process_sds", title=f"{name}: prior to posterior (PGAS)")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"process_sd.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = pgas_fit.plot("parameter_densities", parameters=("sigma", "xi"))
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"gev.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = pgas_fit.plot("season", show_interval=False)
        figure.suptitle(f"{name}: monthly level + seasonal trajectories")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"season.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        try:
            figure, _ = pgas_fit.plot("endpoint")
        except ValueError:
            print(f"{name}: no finite endpoint in the retained posterior draws.")
        else:
            for extension in FIGURE_FORMATS:
                figure.savefig(figure_dir / f"endpoint.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(figure)

        if DIAGNOSTIC_FIGURES:
            for kind, filename in (("traces", "sd_traces"), ("acf", "acf")):
                figure, _ = pgas_fit.plot(kind)
                for extension in FIGURE_FORMATS:
                    figure.savefig(figure_dir / f"{filename}.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(figure)
        print(f"PGAS fit complete: {name}")

    if CHAIN_ONLY:
        print(f"One-chain Uccle PGAS outputs: {OUTPUT_DIR}")
        return

    selection_table = pd.concat(selection_rows, ignore_index=True)
    selection_path = OUTPUT_DIR / "tables" / "selection_all.csv"
    selection_table.to_csv(selection_path, index=False)

    # One compact comparison of the two engines. Each row is a series/process
    # pair and each bar is the posterior probability of a dynamic component.
    comparison = selection_table.pivot_table(index=["series", "process"], columns="engine", values="dynamic")
    figure, axis = plt.subplots(figsize=(11, 5.2))
    positions = np.arange(len(comparison))
    width = 0.38
    if "laplace" in comparison:
        axis.bar(positions - width / 2, comparison["laplace"], width, label="Laplace")
    if "pgas" in comparison:
        axis.bar(positions + width / 2, comparison["pgas"], width, label="PGAS")
    axis.set_xticks(positions, [f"{series}\n{process}" for series, process in comparison.index])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("posterior P(dynamic)")
    axis.set_title("Uccle structural selection: Laplace and PGAS")
    axis.legend()
    figure.tight_layout()
    comparison_dir = OUTPUT_DIR / "figures"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for extension in FIGURE_FORMATS:
        figure.savefig(comparison_dir / f"selection_engines.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"Uccle PGAS outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
