"""Example 10: hierarchically pool all six Uccle summaries.

This is the proposed joint analysis. Each summary keeps its own temperature
path and likelihood. The population hierarchy asks whether the same kinds of
dynamics recur across summaries; it can pool SSVS probabilities, normal-slab
magnitudes, or both. It does not impose a shared trajectory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bucex as bx


START = "1980-01-01"      # validate this run before extending to 1892
END = None
POOL = "selection"        # try "slab" or "both" as sensitivity analyses
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
N_PARTICLES = 512
SEED = 1_001
SAVE_FIT = False
SHOW_PLOTS = False
FIGURE_DIR = Path("figures/10_uccle_hierarchical")


def main() -> None:
    model = bx.make_uccle_hierarchical_model()
    data = bx.load_uccle_multiseries(start=START, end=END)
    prior = bx.HierarchicalPrior(
        pool=POOL,
        season_states=("fixed", "dynamic"),
        slab_df=4.0,
    )
    fit = bx.fit_uccle_hierarchical(
        model=model,
        start=START,
        end=END,
        priors=prior,
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
        for name in fit.parameter_draws
        if name.startswith(("sd.channel.", "sigma.", "xi.", "hierarchy."))
    ]
    diagnostics = fit.diagnostics()
    print("\nDATA\n", data.describe().round(2))
    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nSCIENTIFIC DIAGNOSTICS\n", diagnostics["parameters"].loc[scientific].round(4))
    print("\nPARTICLE DIAGNOSTICS\n", diagnostics["engine"])
    print("\nCHANNEL ALLOCATIONS\n", fit.component_probabilities().round(3))
    print("\nPOPULATION PROBABILITIES\n", fit.hierarchical_probabilities().round(3))
    print("\nPOOLED SLAB MULTIPLIERS\n", fit.hierarchical_slab_summary().round(3))
    print("\nALLOCATION SWITCHING\n", fit.component_transition_summary().round(3))
    print(
        "\nSAMPLER CONTRACT\n",
        {
            "exact_target": fit.plan.targets_exact_posterior,
            "sign_switching": fit.metadata.get("sign_switching"),
            "sign_invariance_error": float(np.nanmax(
                fit.sampler_diagnostics["draw_metrics"]["sign_invariance_error"]
            )),
            "restored_iterations": fit.metadata.get("restored_iterations"),
        },
    )

    print("\nCOMPLETE-PREDICTOR RATES PER DECADE")
    for channel in model.channel_names:
        print(channel, fit.channel_rate_summary(channel))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for channel in model.channel_names:
        fit.plot("channel", channel=channel, save=FIGURE_DIR / f"{channel}.png")
    fit.plot("component_probabilities", save=FIGURE_DIR / "allocations.png")
    fit.plot("hierarchy", save=FIGURE_DIR / "hierarchy.png")
    fit.plot("process_sd", save=FIGURE_DIR / "process_sds.png")
    fit.plot("traces", parameters=scientific, save=FIGURE_DIR / "traces.png")
    fit.plot("acf", parameters=scientific, max_lag=50, save=FIGURE_DIR / "acf.png")

    if SAVE_FIT:
        Path("results").mkdir(exist_ok=True)
        fit.save("results/uccle_hierarchical.bucex")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
