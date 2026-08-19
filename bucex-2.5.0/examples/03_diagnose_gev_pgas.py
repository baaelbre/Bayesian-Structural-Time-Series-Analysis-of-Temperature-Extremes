"""Example 3: determine why a GEV PGAS fit mixes poorly.

This is a controlled sampler experiment, not a scientific analysis.  It fits
the same simulated data with more particles and with a different
parameterization.  Interpret relative changes in restoration failures,
particle diagnostics, R-hat, ESS, and runtime.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 120
DRAWS = 300
WARMUP = 300
CHAINS = 2
SEED = 301
FIGURE_DIR = Path("figures/03_diagnose_gev_pgas")

RUNS = (
    {"label": "FS, 96 particles", "parameterization": "fruehwirth_schnatter", "particles": 96},
    {"label": "FS, 256 particles", "parameterization": "fruehwirth_schnatter", "particles": 256},
    {"label": "disturbance, 256 particles", "parameterization": "disturbance", "particles": 256},
)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    model = bx.Model(
        bx.GEV(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(12)),
        name="GEV sampler diagnostic",
    )
    truth = {
        "sd.level": 0.020,
        "sd.slope": 0.00015,
        "sd.seasonal": 0.015,
        "sigma": 0.50,
        "xi": -0.08,
    }
    compiled = bx.compile_model(model, np.zeros(N_TIME))
    initial_state = np.zeros(compiled.state_dim)
    initial_state[compiled.state_names.index("slope")] = 0.008
    simulation = bx.simulate(
        model,
        N_TIME,
        truth,
        initial_state=initial_state,
        seed=SEED,
    )

    scientific = ["sd.level", "sd.slope", "sd.seasonal", "sigma", "xi"]
    rows = []
    fits = {}

    for index, run in enumerate(RUNS):
        print(f"\n{'=' * 72}\n{run['label']}\n{'=' * 72}")
        started = perf_counter()
        fit = bx.fit(
            simulation.y,
            model,
            parameterization=run["parameterization"],
            engine="pgas",
            priors="pc",
            asis=True,
            mcmc=bx.MCMC(
                draws=DRAWS,
                warmup=WARMUP,
                chains=CHAINS,
                seed=SEED + 10 + index,
                progress=True,
            ),
            particles=bx.Particles(n=run["particles"], proposal="guided"),
        )
        elapsed = perf_counter() - started
        fits[run["label"]] = fit

        diagnostics = fit.diagnostics()
        parameter_table = diagnostics["parameters"].loc[scientific]
        engine = diagnostics["engine"]
        n_particles = run["particles"]

        eta = fit.eta_draws()
        sigma = fit.parameter("sigma")
        xi = fit.parameter("xi")
        support = 1.0 + xi[:, None] * (fit.y[None, :] - eta) / sigma[:, None]
        minimum_support = np.min(support, axis=1)

        rows.append(
            {
                "run": run["label"],
                "seconds": elapsed,
                "max_rhat": float(parameter_table["rhat"].max()),
                "min_ess": float(parameter_table["ess_bulk"].min()),
                "restored_fraction": float(fit.metadata.get("restored_fraction", 0.0)),
                "min_particle_ess/N": engine["median_min_particle_ess"] / n_particles,
                "unique_ancestors/N": engine["mean_unique_ancestors"] / n_particles,
                "path_change_rate": engine["path_change_rate"],
                "path_update_fraction": engine["mean_path_update_fraction"],
                "support_margin_1pct": float(np.quantile(minimum_support, 0.01)),
                "attempt_failures": str(fit.metadata.get("attempt_failure_counts", {})),
            }
        )
        print("\nPARAMETERS\n", parameter_table.round(4))
        print("\nPARTICLES\n", engine)
        print("\nRESTORES\n", fit.metadata.get("restore_failure_counts", {}))

    comparison = pd.DataFrame(rows).set_index("run")
    print("\n\nSAMPLER COMPARISON\n", comparison.round(4).to_string())
    print(
        "\nHOW TO READ THIS:\n"
        "- Improvement from 96 to 256 particles points to particle depletion.\n"
        "- Similar restoration failures at both particle counts point to a "
        "numerical/kernel problem.\n"
        "- Failures restricted to FS point to its regression reconstruction.\n"
        "- No failures but poor R-hat points mainly to insufficient warmup/draws."
    )
    print(
        "\nThe disturbance run is a failure-localization check. Its positive-SD "
        "PC prior is data-calibrated, whereas the FS PC profile uses fixed monthly "
        "reference scales; use Example 5 for a posterior comparison under exactly "
        "matched priors."
    )

    for label in ("FS, 96 particles", "FS, 256 particles"):
        figure, _ = fits[label].plot(
            "traces",
            parameters=scientific,
            truths=truth,
        )
        figure.suptitle(label, y=0.995)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        safe_label = label.lower().replace(", ", "_").replace(" ", "_")
        figure.savefig(FIGURE_DIR / f"{safe_label}_traces.png", bbox_inches="tight")
        fits[label].plot(
            "acf",
            parameters=scientific,
            max_lag=50,
            save=FIGURE_DIR / f"{safe_label}_acf.png",
        )
    plt.show()


if __name__ == "__main__":
    main()
