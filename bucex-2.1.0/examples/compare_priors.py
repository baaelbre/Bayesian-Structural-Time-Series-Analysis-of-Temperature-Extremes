"""Play 03: compare innovation-SD prior profiles on one Gaussian series."""
from __future__ import annotations

import pandas as pd

import bucex as bx
from play_config import describe, mcmc, show_figures


PROFILES = ("normal", "pc", "regularized_horseshoe", "ssvs")
PARAMETERIZATION = "fruehwirth_schnatter"
N_TIME = 120


def main() -> None:
    print(describe())
    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(12)),
        name="prior sensitivity",
    )
    truth = {
        "sd.level": 0.004,
        "sd.slope": 0.001,
        "sd.seasonal": 0.0,
        "sigma": 0.25,
    }
    data = bx.simulate(model, N_TIME, truth, seed=30).y

    rows = []
    fits = {}
    for index, profile in enumerate(PROFILES):
        print(f"\nPRIOR PROFILE: {profile}")
        fit = bx.fit(
            data,
            model,
            parameterization=PARAMETERIZATION,
            engine="ffbs",
            priors=profile,
            # Exact structural SSVS changes component states and therefore
            # is not interwoven with the continuous ASIS scale update.
            asis=profile != "ssvs",
            mcmc=mcmc(31 + index),
        )
        fits[profile] = fit
        print(fit.plan)
        for parameter in ("sd.level", "sd.slope", "sd.seasonal"):
            summary = fit.static_summary()[parameter]
            rows.append(
                {
                    "prior": profile,
                    "parameter": parameter,
                    "truth": truth[parameter],
                    "median": summary["median"],
                    "lower90": summary["lower"],
                    "upper90": summary["upper"],
                }
            )

    table = pd.DataFrame(rows).set_index(["prior", "parameter"])
    print("\nPRIOR SENSITIVITY\n", table.round(5))
    print(
        "\nPC and horseshoe profiles regularize small innovations continuously; "
        "SSVS additionally reports component-allocation probabilities."
    )

    for profile, fit in fits.items():
        figure, _ = fit.plot("process_sd")
        figure.suptitle(profile)
    fits["ssvs"].plot("component_probabilities")
    show_figures()


if __name__ == "__main__":
    main()
