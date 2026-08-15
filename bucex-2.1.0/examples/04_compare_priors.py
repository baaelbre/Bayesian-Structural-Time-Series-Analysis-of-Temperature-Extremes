"""Example 4: prior sensitivity for structural innovation SDs.

All fits use exactly the same Gaussian data and FS parameterization.  This
script compares continuous shrinkage, exact structural selection, the
paper-exact triple gamma, and its optional regularizing slab.  SSVS is
reported separately because it assigns exact posterior probability to
zero/fixed/dynamic model states.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 180
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
SEED = 401
PROFILES = (
    "normal",
    "pc",
    "regularized_horseshoe",
    "triple_gamma",
    "regularized_triple_gamma",
    "ssvs",
)
FIGURE_DIR = Path("figures/04_compare_priors")


def describe_prior(priors) -> str:
    """Return the calibration that matters scientifically."""

    if priors.pc is not None:
        return f"PC: P(SD > upper)=alpha; upper={dict(priors.pc.upper)}, alpha={priors.pc.alpha}"
    if priors.horseshoe is not None:
        return (
            "regularized horseshoe: coefficient_scale="
            f"{dict(priors.horseshoe.coefficient_scale)}, "
            f"global_scale={priors.horseshoe.global_scale}, "
            f"slab_scale={priors.horseshoe.slab_scale}"
        )
    if priors.triple_gamma is not None:
        prior = priors.triple_gamma
        return (
            f"{'regularized ' if prior.regularized else ''}triple gamma: "
            f"a={prior.spike_shape}, c={prior.tail_shape}, "
            f"learn_global={prior.learn_global}, "
            f"learn_shapes={prior.learn_shapes}, "
            f"coefficient_scale={dict(prior.coefficient_scale)}, "
            f"slab_scale={prior.slab_scale if prior.regularized else 'none'}"
        )
    if priors.ssvs is not None:
        return (
            "SSVS: exact zero/fixed/dynamic states; innovation slab SD="
            f"{dict(priors.ssvs.innovation_slab_sd)}"
        )
    return (
        "normal signed scales: "
        f"level={priors.s_level.sd}, trend={priors.s_trend.sd}, "
        f"season={priors.s_season.sd}"
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(12)),
        name="prior sensitivity",
    )
    truth = {
        "sd.level": 0.012,
        "sd.slope": 0.00012,
        "sd.seasonal": 0.0,
        "sigma": 0.25,
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

    parameters = ["sd.level", "sd.slope", "sd.seasonal"]
    rows = []
    fits = {}
    for index, profile in enumerate(PROFILES):
        print(f"\n{'=' * 72}\nPRIOR: {profile}\n{'=' * 72}")
        fit = bx.fit(
            simulation.y,
            model,
            parameterization="fruehwirth_schnatter",
            engine="ffbs",
            priors=profile,
            asis=profile != "ssvs",
            mcmc=bx.MCMC(
                draws=DRAWS,
                warmup=WARMUP,
                chains=CHAINS,
                seed=SEED + 10 + index,
                progress=True,
            ),
        )
        fits[profile] = fit
        print(describe_prior(fit.priors))
        diagnostics = fit.diagnostics()["parameters"]
        summaries = fit.static_summary()
        for parameter in parameters:
            posterior = summaries[parameter]
            rows.append(
                {
                    "prior": profile,
                    "parameter": parameter,
                    "truth": truth[parameter],
                    "median": posterior["median"],
                    "lower90": posterior["lower"],
                    "upper90": posterior["upper"],
                    "rhat": diagnostics.loc[parameter, "rhat"],
                    "ess_bulk": diagnostics.loc[parameter, "ess_bulk"],
                }
            )

        fit.plot(
            "process_sd",
            truths=truth,
            title=profile.replace("_", " "),
            save=FIGURE_DIR / f"{profile}_prior_posterior.png",
        )
        fit.plot(
            "traces", parameters=parameters, truths=truth,
            save=FIGURE_DIR / f"{profile}_traces.png",
        )
        fit.plot(
            "acf", parameters=parameters, max_lag=50,
            save=FIGURE_DIR / f"{profile}_acf.png",
        )

        if fit.priors.horseshoe is not None:
            hierarchy = [
                name for name in diagnostics.index
                if name.startswith("horseshoe_")
            ]
            print("\nHORSESHOE HIERARCHY (slice-updated)")
            print(diagnostics.loc[hierarchy, ["mean", "sd", "rhat", "ess_bulk"]].round(4))
            print(
                "These are stepping-out slice updates: there is no Metropolis "
                "acceptance probability to tune."
            )
        if fit.priors.triple_gamma is not None:
            rho_names = [
                name for name in fit.parameter_draws
                if name.startswith("triple_gamma_rho_")
            ]
            rho_rows = []
            for name in rho_names:
                values = fit.parameter(name)
                rho_rows.append(
                    {
                        "process": name.removeprefix("triple_gamma_rho_"),
                        "median_rho": np.median(values),
                        "P(rho>0.5)": np.mean(values > 0.5),
                        "P(rho>0.9)": np.mean(values > 0.9),
                    }
                )
            print("\nTRIPLE-GAMMA SHRINKAGE FACTORS")
            print(pd.DataFrame(rho_rows).set_index("process").round(3))

    comparison = pd.DataFrame(rows).set_index(["prior", "parameter"])
    print("\nPRIOR-SENSITIVITY TABLE\n", comparison.round(6).to_string())

    print("\nSSVS COMPONENT PROBABILITIES")
    print(fits["ssvs"].component_probabilities().round(3))
    print("\nSSVS SWITCHING DIAGNOSTICS")
    print(fits["ssvs"].component_transition_summary().round(3))
    fits["ssvs"].plot(
        "component_probabilities",
        save=FIGURE_DIR / "ssvs_component_probabilities.png",
    )

    # One compact comparison on common axes.
    figure, axes = plt.subplots(1, len(parameters), figsize=(13, 4))
    for axis, parameter in zip(axes, parameters):
        subset = comparison.xs(parameter, level="parameter")
        positions = np.arange(len(PROFILES))
        median = subset.loc[list(PROFILES), "median"].to_numpy()
        lower = subset.loc[list(PROFILES), "lower90"].to_numpy()
        upper = subset.loc[list(PROFILES), "upper90"].to_numpy()
        axis.errorbar(
            positions,
            median,
            yerr=np.vstack((median - lower, upper - median)),
            fmt="o",
            capsize=3,
        )
        axis.axhline(truth[parameter], color="black", linestyle="--", label="truth")
        axis.set_xticks(positions, [name.replace("_", "\n") for name in PROFILES])
        axis.set_title(parameter)
        axis.legend()
    figure.suptitle("Same likelihood and data; different innovation priors")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "prior_comparison.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
