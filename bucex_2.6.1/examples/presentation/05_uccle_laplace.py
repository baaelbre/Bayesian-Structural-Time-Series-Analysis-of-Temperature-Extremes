"""Load and fit TXx, TXn, TNx, and TNn with Laplace state updates."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if (SOURCE_ROOT / "bucex").is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import bucex as bx


OUTPUT_DIR = Path("results/presentation")
DATA_DIR: Path | None = None
START = "1892-01-01"
END: str | None = None
SERIES = ("TXx", "TXn", "TNx", "TNn")
PERIOD = 12

# Prior hyperparameters for the observed monthly series.
ALPHA_PRIOR_SD = 3.2
# The current Uccle OLS rates are only approximately 0.15–0.21
#C per decade, which is very close to the global average. This is about 0.2/decade, or about 0.2/120 = 0.002 per month. 
BETA_PRIOR_MEAN = 0.002
BETA_PRIOR_SD = 0.01
INITIAL_SEASON_PRIOR_SD = 2.25
SIGMA2_PRIOR_A = 2.0
SIGMA2_PRIOR_B = 2.0
XI_PRIOR_BOUNDS = (-0.50, 0.50)
XI_MAX_ABS = 0.50
# trend slab: 0.00005, 0.00010, 0.00020, 0.00040
#beta SD:    0.0025, 0.0040, 0.0075
INNOVATION_SLAB_SD = {
    "level": 0.05,
    "trend": 0.00010,
    "season": 0.09,
}
LEVEL_DYNAMIC_PROBABILITY = 0.50
TREND_PROBABILITIES = (1.0 / 3.0,) * 3
SEASON_PROBABILITIES = (1.0 / 3.0,) * 3
# or:
#TREND_PROBABILITIES = (0.0, 0.5, 0.5) # or (0.10,0.45,0.45)
#SEASON_PROBABILITIES = (0.0, 0.5, 0.5) 

# Pilot settings. For final results use 2_000 draws, 2_000 warmup, 4 chains.
DRAWS = 250
WARMUP = 250
CHAINS = 2
SEED = 56_000
PROGRESS = True

FIGURE_FORMATS = ("pdf", "png")
FIGURE_DPI = 180
DIAGNOSTIC_FIGURES = False
OVERWRITE = False


def prior_settings() -> dict:
    return {
        "alpha_sd": ALPHA_PRIOR_SD,
        "beta_mean": BETA_PRIOR_MEAN,
        "beta_sd": BETA_PRIOR_SD,
        "seasonal_initial_sd": INITIAL_SEASON_PRIOR_SD,
        "sigma2": {"a": SIGMA2_PRIOR_A, "b": SIGMA2_PRIOR_B},
        "xi_bounds": list(XI_PRIOR_BOUNDS),
        "xi_max_abs": XI_MAX_ABS,
        "innovation_slab_sd": dict(INNOVATION_SLAB_SD),
        "level_dynamic_probability": LEVEL_DYNAMIC_PROBABILITY,
        "trend_probabilities": list(TREND_PROBABILITIES),
        "season_probabilities": list(SEASON_PROBABILITIES),
    }


def make_priors(values, sign: float) -> bx.FSGEVPriors:
    y = sign * values.to_numpy(float)
    return bx.ssvs_gev_priors(
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


def main() -> None:
    fitted: dict[tuple[str, str], bx.FitResult] = {}

    for number, name in enumerate(SERIES):
        values = bx.load_uccle_series(name, DATA_DIR, start=START, end=END)
        tail = bx.UCCLE_INFO[name]["tail"]
        sign = -1.0 if tail == "min" else 1.0

        priors = make_priors(values, sign)

        fit_path = (
            OUTPUT_DIR / "fits" / "uccle" / "laplace" / name / "combined.bucex"
        )
        if fit_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(fit_path)
            if (
                laplace_fit.n_time != len(values)
                or laplace_fit.model.period != PERIOD
                or laplace_fit.n_chains != CHAINS
                or laplace_fit.draws_per_chain != DRAWS
                or laplace_fit.metadata.get("prior_settings") != prior_settings()
            ):
                raise ValueError(
                    f"{fit_path} does not match the current data, prior, or "
                    "MCMC controls; set OVERWRITE=True to refit."
                )
            print(f"Reusing {fit_path}")
        else:
            laplace_fit = bx.fit(
                values,
                family="gev",
                period=PERIOD,
                priors=priors,
                engine="laplace",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(
                    draws=DRAWS,
                    warmup=WARMUP,
                    chains=CHAINS,
                    seed=SEED + 100 * number,
                    progress=PROGRESS,
                ),
                name=name,
                tail=tail,
            )
            laplace_fit.metadata.update(
                {
                    "example": "uccle_laplace",
                    "description": bx.UCCLE_INFO[name]["description"],
                    "prior_settings": prior_settings(),
                }
            )
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(fit_path)
        fitted[("laplace", name)] = laplace_fit

        bx.export_fit_results(
            laplace_fit,
            OUTPUT_DIR / "tables" / "uccle" / "laplace" / name,
            source=fit_path,
        )
        bx.plot_fit_results(
            laplace_fit,
            OUTPUT_DIR / "figures" / "uccle" / "laplace" / name,
            formats=FIGURE_FORMATS,
            dpi=FIGURE_DPI,
            diagnostics=DIAGNOSTIC_FIGURES,
        )
        print(f"Laplace fit complete: {name}")

    selection = bx.collect_selection_probabilities(fitted)
    selection_path = (
        OUTPUT_DIR / "tables" / "40_uccle" / "uccle_selection_laplace.csv"
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(selection_path, index=False)


if __name__ == "__main__":
    main()
