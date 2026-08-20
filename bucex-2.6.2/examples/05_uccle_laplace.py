"""Fit TXx, TXn, TNx, and TNn with Laplace state updates.

The model, prior, result tables, and figures are all visible in this file.
Run with ``python examples/05_uccle_laplace.py``.
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

# Data.
DATA_DIR: Path | None = (
    Path(os.environ["BUCEX_DATA_DIR"]) if "BUCEX_DATA_DIR" in os.environ else None
)
START = os.environ.get("BUCEX_START", "1892-01-01")
END: str | None = os.environ.get("BUCEX_END") or None
SERIES = ("TXx", "TXn", "TNx", "TNn")
PERIOD = 12

# Prior hyperparameters. A zero-centred fixed-slope prior is appropriate for
# both upper-tail series and sign-transformed lower-tail series. Its scale is
# still broad relative to observed monthly climate rates.
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

# MCMC. Final runs can set 2000/2000/4 through the environment.
DRAWS = int(os.environ.get("BUCEX_DRAWS", "250"))
WARMUP = int(os.environ.get("BUCEX_WARMUP", "250"))
CHAINS = int(os.environ.get("BUCEX_CHAINS", "2"))
SEED = int(os.environ.get("BUCEX_SEED", "56000"))
PROGRESS = os.environ.get("BUCEX_PROGRESS", "1").lower() not in {"0", "false", "no"}

FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
DIAGNOSTIC_FIGURES = False


MODEL = bx.Model(
    bx.GEV(xi_bounds=XI_PRIOR_BOUNDS),
    (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=PERIOD, mode="dynamic"),
    ),
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
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "legend.frameon": False})
    selection_rows = []

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

        fit_path = OUTPUT_DIR / "fits" / "uccle" / "laplace" / name / "combined.bucex"
        if fit_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(fit_path)
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
                raise ValueError(f"{fit_path} does not match the current model, prior, data, or MCMC settings.")
            print(f"Reusing {fit_path}")
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
                    "example": "uccle_laplace",
                    "description": bx.UCCLE_INFO[name]["description"],
                    "start": START,
                    "end": END,
                    "prior_settings": PRIOR_SETTINGS,
                }
            )
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(fit_path)

        table_dir = OUTPUT_DIR / "tables" / "uccle" / "laplace" / name
        figure_dir = OUTPUT_DIR / "figures" / "uccle" / "laplace" / name
        table_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = laplace_fit.diagnostics()

        pd.DataFrame.from_dict(laplace_fit.static_summary(), orient="index").rename_axis("parameter").to_csv(table_dir / "parameters.csv")
        diagnostics["parameters"].to_csv(table_dir / "diagnostics.csv")
        pd.DataFrame([{"metric": key, "value": value} for key, value in diagnostics["engine"].items()]).to_csv(table_dir / "algorithm_diagnostics.csv", index=False)
        eta_draws = laplace_fit.eta_draws(original_scale=True)
        lower, median, upper = np.quantile(eta_draws, [0.05, 0.50, 0.95], axis=0)
        pd.DataFrame({"date": values.index, "observed": values.to_numpy(), "lower": lower, "median": median, "upper": upper}).to_csv(
            table_dir / "posterior_trajectory.csv", index=False
        )
        selection = laplace_fit.component_probabilities().reset_index()
        selection.insert(0, "series", name)
        selection.insert(1, "engine", "laplace")
        selection.to_csv(table_dir / "selection_probabilities.csv", index=False)
        selection_rows.append(selection)
        laplace_fit.structural_model_probabilities().to_csv(table_dir / "structural_models.csv", index=False)
        laplace_fit.component_transition_summary().reset_index().to_csv(table_dir / "selection_switching.csv", index=False)
        (table_dir / "fit_summary.json").write_text(
            json.dumps(
                {
                    "fit": str(fit_path),
                    "series": name,
                    "description": bx.UCCLE_INFO[name]["description"],
                    "tail": tail,
                    "start": str(values.index.min().date()),
                    "end": str(values.index.max().date()),
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
        axis.set_title(f"{name}: posterior latent predictor (Laplace)")
        axis.set_ylabel("GEV location / °C")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"posterior_trajectory.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("component_probabilities")
        figure.suptitle(f"{name}: structural selection (Laplace)")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"selection_probabilities.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("process_sds", title=f"{name}: prior to posterior (Laplace)")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"prior_to_posterior_process_sd.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("parameter_densities", parameters=("sigma", "xi"))
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"gev_parameters.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        figure, _ = laplace_fit.plot("season", show_interval=False)
        figure.suptitle(f"{name}: monthly level + seasonal trajectories")
        for extension in FIGURE_FORMATS:
            figure.savefig(figure_dir / f"seasonal_trajectories.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)

        try:
            figure, _ = laplace_fit.plot("endpoint")
        except ValueError:
            print(f"{name}: no finite endpoint in the retained posterior draws.")
        else:
            for extension in FIGURE_FORMATS:
                figure.savefig(figure_dir / f"endpoint.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(figure)

        if DIAGNOSTIC_FIGURES:
            for kind, filename in (("traces", "process_sd_traces"), ("acf", "parameter_acfs")):
                figure, _ = laplace_fit.plot(kind)
                for extension in FIGURE_FORMATS:
                    figure.savefig(figure_dir / f"{filename}.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
                plt.close(figure)
        print(f"Laplace fit complete: {name}")

    selection_path = OUTPUT_DIR / "tables" / "uccle" / "uccle_selection_laplace.csv"
    pd.concat(selection_rows, ignore_index=True).to_csv(selection_path, index=False)
    print(f"Uccle Laplace outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
