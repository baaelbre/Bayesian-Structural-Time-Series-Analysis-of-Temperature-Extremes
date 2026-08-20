"""Load and fit TXx, TXn, TNx, and TNn with Laplace state updates."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import bucex as bx


OUTPUT_DIR = Path("results/presentation")
DATA_DIR: Path | None = None
START = "1892-01-01"
END: str | None = None
SERIES = ("TXx", "TXn", "TNx", "TNn")

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


def main() -> None:
    fitted: dict[tuple[str, str], bx.FitResult] = {}

    for number, name in enumerate(SERIES):
        values = bx.load_uccle_series(name, DATA_DIR, start=START, end=END)
        tail = bx.UCCLE_INFO[name]["tail"]
        sign = -1.0 if tail == "min" else 1.0

        priors = bx.ssvs_gev_priors(
            period=12,
            alpha_mean=float(np.median(sign * values.to_numpy(float))),
            beta_mean=0.0,
            beta_sd=0.01,
            innovation_slab_sd={
                "level": 0.12,
                "trend": 0.0015,
                "season": 0.10,
            },
            level_dynamic_probability=0.5,
            trend_probabilities=(1.0 / 3.0,) * 3,
            season_probabilities=(1.0 / 3.0,) * 3,
        )

        fit_path = (
            OUTPUT_DIR / "fits" / "uccle" / "laplace" / name / "combined.bucex"
        )
        if fit_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(fit_path)
            print(f"Reusing {fit_path}")
        else:
            laplace_fit = bx.fit(
                values,
                family="gev",
                period=12,
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
