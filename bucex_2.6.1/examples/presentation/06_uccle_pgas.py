"""Fit all four Uccle extremes with standalone Laplace-initialized PGAS."""
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

# Prior hyperparameters: keep aligned with 05_uccle_laplace.py when reusing
# its Laplace initializers.
ALPHA_PRIOR_SD = 3.2
BETA_PRIOR_MEAN = 0.0
BETA_PRIOR_SD = 0.010
INITIAL_SEASON_PRIOR_SD = 2.25
SIGMA2_PRIOR_A = 2.0
SIGMA2_PRIOR_B = 2.0
XI_PRIOR_BOUNDS = (-0.50, 0.50)
XI_MAX_ABS = 0.50
INNOVATION_SLAB_SD = {
    "level": 0.12,
    "trend": 0.0015,
    "season": 0.10,
}
LEVEL_DYNAMIC_PROBABILITY = 0.50
TREND_PROBABILITIES = (1.0 / 3.0,) * 3
SEASON_PROBABILITIES = (1.0 / 3.0,) * 3

# Pilot settings. Final runs should use 2_000 draws, 2_000 warmup iterations,
# 4 chains, and 512 particles.
DRAWS = 250
WARMUP = 250
CHAINS = 2
PARTICLES = 128
SEED = 66_000
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


def validate_fit(fit: bx.FitResult, n_time: int, path: Path) -> None:
    if (
        fit.n_time != n_time
        or fit.model.period != PERIOD
        or fit.n_chains != CHAINS
        or fit.draws_per_chain != DRAWS
        or fit.metadata.get("prior_settings") != prior_settings()
    ):
        raise ValueError(
            f"{path} does not match the current data, prior, or MCMC controls; "
            "set OVERWRITE=True to refit."
        )


def main() -> None:
    fitted: dict[tuple[str, str], bx.FitResult] = {}

    for number, name in enumerate(SERIES):
        values = bx.load_uccle_series(name, DATA_DIR, start=START, end=END)
        tail = bx.UCCLE_INFO[name]["tail"]
        sign = -1.0 if tail == "min" else 1.0

        priors = make_priors(values, sign)

        laplace_path = (
            OUTPUT_DIR / "fits" / "uccle" / "laplace" / name / "combined.bucex"
        )
        if laplace_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(laplace_path)
            validate_fit(laplace_fit, len(values), laplace_path)
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
                    "example": "uccle_laplace_initializer",
                    "description": bx.UCCLE_INFO[name]["description"],
                    "prior_settings": prior_settings(),
                }
            )
            laplace_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(laplace_path)
        fitted[("laplace", name)] = laplace_fit

        pgas_path = (
            OUTPUT_DIR / "fits" / "uccle" / "pgas" / name / "combined.bucex"
        )
        if pgas_path.is_file() and not OVERWRITE:
            pgas_fit = bx.FitResult.load(pgas_path)
            validate_fit(pgas_fit, len(values), pgas_path)
            print(f"Reusing {pgas_path}")
        else:
            # The FitResult initializer contains one coherent Laplace draw:
            # parameters, indicators, and the complete centred state path.
            pgas_fit = bx.fit(
                values,
                family="gev",
                period=PERIOD,
                priors=laplace_fit.priors,
                engine="pgas",
                parameterization="fruehwirth_schnatter",
                asis=False,
                mcmc=bx.MCMC(
                    draws=DRAWS,
                    warmup=WARMUP,
                    chains=CHAINS,
                    seed=SEED + 10_000 + 100 * number,
                    progress=PROGRESS,
                ),
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
                    "prior_settings": prior_settings(),
                }
            )
            pgas_path.parent.mkdir(parents=True, exist_ok=True)
            pgas_fit.save(pgas_path)
        fitted[("pgas", name)] = pgas_fit

        bx.export_fit_results(
            pgas_fit,
            OUTPUT_DIR / "tables" / "uccle" / "pgas" / name,
            source=pgas_path,
        )
        bx.plot_fit_results(
            pgas_fit,
            OUTPUT_DIR / "figures" / "uccle" / "pgas" / name,
            formats=FIGURE_FORMATS,
            dpi=FIGURE_DPI,
            diagnostics=DIAGNOSTIC_FIGURES,
        )
        print(f"PGAS fit complete: {name}")

    selection = bx.collect_selection_probabilities(fitted)
    selection_path = (
        OUTPUT_DIR / "tables" / "40_uccle" / "uccle_selection_probabilities.csv"
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(selection_path, index=False)
    bx.plot_uccle_selection_comparison(
        selection,
        OUTPUT_DIR / "figures" / "40_uccle",
        formats=FIGURE_FORMATS,
        dpi=FIGURE_DPI,
    )


if __name__ == "__main__":
    main()
