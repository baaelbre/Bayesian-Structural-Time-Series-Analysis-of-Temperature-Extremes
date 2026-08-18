"""Example 12: exact structural SSVS for a GEV model using PGAS.

Laplace information constructs efficient model proposals only. The retained
chain targets the exact posterior because PGAS updates the state path and a
trans-dimensional MH correction uses the exact GEV likelihood. Compare with
``engine="laplace"`` only as a clearly labelled fast screening analysis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bucex as bx


N_TIME = 120
PERIOD = 12
DRAWS = 500
WARMUP = 750
CHAINS = 4
N_PARTICLES = 256
SEED = 1_201
FIGURE_DIR = Path("figures/12_gev_ssvs_pgas")


def main() -> None:
    model = bx.Model(
        bx.GEV(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
        name="GEV structural selection",
    )
    truth = {
        "sd.level": 0.015,
        "sd.slope": 0.0,
        "sd.seasonal": 0.0,
        "sigma": 0.5,
        "xi": -0.08,
    }
    compiled = bx.compile_model(model, np.zeros(N_TIME))
    initial_state = np.zeros(compiled.state_dim)
    initial_state[compiled.state_names.index("slope")] = 0.006
    simulation = bx.simulate(
        model,
        N_TIME,
        truth,
        initial_state=initial_state,
        seed=SEED,
    )

    fit = bx.fit(
        simulation.y,
        model,
        parameterization="fruehwirth_schnatter",
        engine="pgas",
        priors="ssvs",
        asis=False,
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED + 1,
            progress=True,
        ),
        particles=bx.Particles(n=N_PARTICLES, proposal="guided"),
    )

    scientific = ("sd.level", "sd.slope", "sd.seasonal", "sigma", "xi")
    print("\nINFERENCE PLAN\n", fit.plan)
    print(
        "\nEXACT SSVS CONTRACT\n",
        {
            "model_selection_exact": fit.metadata["model_selection_exact"],
            "basis": fit.metadata["model_selection_basis"],
            "model_MH_acceptance": fit.acceptance.get("ssvs_model_mh"),
            "changed-model_acceptance": fit.acceptance.get(
                "ssvs_model_change_mh"
            ),
        },
    )
    print(
        "\nCONVERGENCE\n",
        fit.diagnostics()["parameters"].loc[list(scientific)].round(4),
    )
    print("\nPARTICLES\n", fit.diagnostics()["engine"])
    print("\nCOMPONENT PROBABILITIES\n", fit.component_probabilities().round(3))
    print(
        "\nSWITCHING\n",
        fit.component_transition_summary().round(3),
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fit.plot("predictor", save=FIGURE_DIR / "predictor.png")
    fit.plot(
        "component_probabilities",
        save=FIGURE_DIR / "component_probabilities.png",
    )
    fit.plot(
        "traces",
        parameters=scientific,
        truths=truth,
        save=FIGURE_DIR / "traces.png",
    )
    fit.plot(
        "acf",
        parameters=scientific,
        max_lag=50,
        save=FIGURE_DIR / "acf.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
