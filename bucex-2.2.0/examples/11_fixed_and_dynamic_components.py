"""Example 11: verify zero, fixed, and dynamic FS component semantics.

In the FS parameterization a fixed level is an intercept ``alpha0`` with
``sd.level == 0``; a fixed trend is a slope ``beta0`` with
``sd.slope == 0``; and fixed seasonality is a vector of static initial-season
coefficients with ``sd.seasonal == 0``. SSVS assigns exact probability to
those alternatives rather than approximating zero with a tiny variance.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bucex as bx


N_TIME = 180
PERIOD = 12
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
SEED = 1_101
FIGURE_DIR = Path("figures/11_fixed_and_dynamic_components")


def main() -> None:
    # The dynamic grammar supplies all candidate columns. Zero innovation SDs
    # make the data-generating trend and seasonality deterministic.
    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
        name="deterministic trend and seasonality",
    )
    truth = {
        "sd.level": 0.0,
        "sd.slope": 0.0,
        "sd.seasonal": 0.0,
        "sigma": 0.25,
    }
    compiled = bx.compile_model(model, np.zeros(N_TIME))
    initial_state = np.zeros(compiled.state_dim)
    initial_state[compiled.state_names.index("level")] = 0.5
    initial_state[compiled.state_names.index("slope")] = 0.004
    seasonal_slice = compiled.component_slices["seasonal"]
    initial_state[seasonal_slice] = np.linspace(0.12, -0.08, PERIOD - 1)
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
        engine="ffbs",
        priors="ssvs",
        asis=False,  # SSVS itself switches between fixed and dynamic states
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED + 1,
            progress=True,
        ),
    )

    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nCOMPONENT PROBABILITIES\n", fit.component_probabilities().round(3))
    print(
        "\nSWITCHING DIAGNOSTICS\n",
        fit.component_transition_summary().round(3),
    )
    print(
        "\nSTATIC COEFFICIENTS\n",
        {
            "alpha0 median": float(np.median(fit.parameter("alpha0"))),
            "beta0 median": float(np.median(fit.parameter("beta0"))),
            "gamma0 seasonal medians": np.median(
                fit.parameter("gamma0_season"), axis=0
            ).round(4),
        },
    )

    # Algebraic contract: whenever a component is stored as fixed, its
    # innovation SD is exactly zero—not merely small.
    for state_name, sd_name in (
        ("state_level", "sd.level"),
        ("state_trend", "sd.slope"),
        ("state_season", "sd.seasonal"),
    ):
        states = fit.parameter(state_name)
        scales = fit.parameter(sd_name)
        fixed = states == int(bx.ComponentState.FIXED)
        print(
            f"{state_name}: fixed draws={int(np.sum(fixed))}, "
            f"largest fixed SD={float(np.max(scales[fixed])) if np.any(fixed) else np.nan}"
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fit.plot(
        "predictor",
        save=FIGURE_DIR / "predictor.png",
    )
    fit.plot(
        "component_probabilities",
        save=FIGURE_DIR / "component_probabilities.png",
    )
    fit.plot(
        "traces",
        parameters=("sd.level", "sd.slope", "sd.seasonal", "sigma"),
        truths=truth,
        save=FIGURE_DIR / "traces.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
