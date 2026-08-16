"""Example 15: joint hierarchical SSVS for all six Uccle summaries.

This model partially pools structural *selection* across the six summaries but
does not force their temperature paths to be synchronized.  Gaussian channels
use exact FFBS; because four channels are GEV, the joint run uses exact-
invariant PGAS and exact GEV-corrected structural model moves.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bucex as bx


START = "1980-01-01"       # use None only after validating the shorter run
END = None
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
N_PARTICLES = 512
SEED = 1_501
SAVE_FIT = False
SHOW_PLOTS = False
FIGURE_DIR = Path("figures/15_uccle_hierarchical_ssvs")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    model = bx.make_uccle_multiseries_model()
    data = bx.load_uccle_factor_data(start=START, end=END)
    print("\nDATA\n", data.describe().round(2))
    print("\nMODEL\n", model)
    print("\nINFERENCE PLAN\n", bx.plan(model, data))

    fit = bx.fit_uccle_multiseries(
        model=model,
        start=START,
        end=END,
        priors="hierarchical_ssvs",
        parameterization="fruehwirth_schnatter",
        engine="pgas",
        asis=False,
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED,
            progress=True,
        ),
        particles=bx.Particles(n=N_PARTICLES, proposal="guided"),
    )

    scientific = [
        name
        for name, values in fit.parameter_draws.items()
        if np.asarray(values).ndim == 2
        and name.startswith(("sd.channel.", "sigma.", "xi."))
    ]
    print("\nRESOLVED INFERENCE PLAN\n", fit.plan)
    print(
        "\nSCIENTIFIC-PARAMETER DIAGNOSTICS\n",
        fit.diagnostics()["parameters"].loc[scientific].round(4),
    )
    print("\nPARTICLE AND MODEL-MOVE DIAGNOSTICS\n", fit.diagnostics()["engine"])
    print("\nSERIES-SPECIFIC STRUCTURAL PROBABILITIES\n")
    print(fit.component_probabilities().round(3).to_string())
    print("\nPOPULATION ALLOCATION PROBABILITIES\n")
    print(fit.hierarchical_probabilities().round(3).to_string())
    print("\nSHARED DYNAMIC-SLAB MULTIPLIERS\n")
    print(fit.hierarchical_slab_summary().round(3).to_string())
    print("\nMODEL-SWITCHING DIAGNOSTICS\n")
    print(fit.component_transition_summary().round(3).to_string())
    print(
        "\nEXACTNESS AND SIGN-SYMMETRY CONTRACT\n",
        {
            "targets_exact_posterior": fit.plan.targets_exact_posterior,
            "model_selection_exact": fit.meta["model_selection_exact"],
            "pgas_exact_invariant": fit.meta["pgas_exact_invariant"],
            "sign_switching": fit.meta["sign_switching"],
            "sign_invariance_checked": fit.meta["sign_switch_invariance_checked"],
            "restored_iterations": fit.meta["restored_iterations"],
        },
    )

    start_year = int(data.index[0].year)
    end_year = int(data.index[-1].year)
    print("\nCOMPLETE-PREDICTOR RATES PER DECADE")
    for name in model.channel_names:
        print(f"  {name}: {fit.channel_rate_summary(name, start_year, end_year)}")

    forecast = fit.forecast(12, draws=min(fit.n_draws, 2_000), seed=SEED + 1)
    print("\n12-MONTH FORECAST (FIRST 12 ROWS)\n")
    print(forecast.summary().head(12).round(3).to_string(index=False))

    for name in model.channel_names:
        fit.plot(
            "channel",
            channel=name,
            save=FIGURE_DIR / f"{name}_predictor.png",
        )
    fit.plot(
        "component_probabilities",
        save=FIGURE_DIR / "component_probabilities.png",
    )
    fit.plot("hierarchy", save=FIGURE_DIR / "hierarchy.png")
    fit.plot("process_sd", save=FIGURE_DIR / "process_sds.png")
    fit.plot("traces", parameters=scientific, save=FIGURE_DIR / "traces.png")
    fit.plot(
        "acf", parameters=scientific, max_lag=50,
        save=FIGURE_DIR / "acf.png",
    )

    if SAVE_FIT:
        Path("results").mkdir(exist_ok=True)
        fit.save("results/uccle_hierarchical_ssvs.bucex")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
