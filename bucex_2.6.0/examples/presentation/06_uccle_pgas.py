"""Fit all four Uccle extremes with standalone Laplace-initialized PGAS."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import bucex as bx


OUTPUT_DIR = Path("results/presentation")
DATA_DIR: Path | None = None
START = "1892-01-01"
END: str | None = None
SERIES = ("TXx", "TXn", "TNx", "TNn")

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

        laplace_path = (
            OUTPUT_DIR / "fits" / "uccle" / "laplace" / name / "combined.bucex"
        )
        if laplace_path.is_file() and not OVERWRITE:
            laplace_fit = bx.FitResult.load(laplace_path)
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
            laplace_path.parent.mkdir(parents=True, exist_ok=True)
            laplace_fit.save(laplace_path)
        fitted[("laplace", name)] = laplace_fit

        pgas_path = (
            OUTPUT_DIR / "fits" / "uccle" / "pgas" / name / "combined.bucex"
        )
        if pgas_path.is_file() and not OVERWRITE:
            pgas_fit = bx.FitResult.load(pgas_path)
            print(f"Reusing {pgas_path}")
        else:
            # The FitResult initializer contains one coherent Laplace draw:
            # parameters, indicators, and the complete centred state path.
            pgas_fit = bx.fit(
                values,
                family="gev",
                period=12,
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
