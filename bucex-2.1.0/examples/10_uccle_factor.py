"""Example 10: the proposed six-summary Uccle dynamic-factor model.

TXm anchors the common factor.  Every summary has its own seasonal component;
non-reference channels may also have a persistent idiosyncratic local level.
Set START=None for the complete record after validating the shorter run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import bucex as bx


START = "1980-01-01"
END = None
REFERENCE = "TXm"
PRIOR = "regularized_horseshoe"
DRAWS = 500
WARMUP = 750
CHAINS = 4
N_PARTICLES = 512
SEED = 1_001
SAVE_FIT = False


def main() -> None:
    model = bx.make_uccle_factor_model(
        structure="estimated",
        individual="local_level",
        seasonal="series_specific",
        seasonal_mode="dynamic",
    )
    data = bx.load_uccle_factor_data(start=START, end=END)
    print("\nDATA\n", data.describe().round(2))

    compiled = bx.compile_model(model, data)
    priors = bx.identified_factor_priors(
        compiled,
        profile=PRIOR,
        smooth_factor=True,
        reference_channel=REFERENCE,
    )
    fit = bx.fit(
        data,
        model,
        parameterization="fruehwirth_schnatter",
        engine="pgas",
        priors=priors,
        asis=True,
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED,
            progress=True,
        ),
        particles=bx.Particles(n=N_PARTICLES, proposal="guided"),
    )

    baseline = slice(0, min(120, fit.n_time))
    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nPARAMETER DIAGNOSTICS\n", fit.diagnostics()["parameters"].round(4))
    print("\nPARTICLE DIAGNOSTICS\n", fit.diagnostics()["engine"])
    print(
        "\nIDENTIFICATION DIAGNOSTICS\n",
        fit.factor_identification_diagnostics(baseline=baseline).round(3),
    )

    start_year = int(data.index[0].year)
    end_year = int(data.index[-1].year)
    print(
        "\nCOMMON-FACTOR RATE\n",
        fit.factor_rate_summary("common", start_year, end_year),
    )

    if SAVE_FIT:
        Path("results").mkdir(exist_ok=True)
        output = Path("results/uccle_factor.bucex")
        fit.save(output)
        print(f"\nSaved {output}")

    fit.plot("factor", factor="common")
    fit.plot(
        "factor_decomposition",
        channels=("TXm", "TNm", "TXx", "TNn"),
        baseline=baseline,
    )
    fit.plot(
        "parameter_density",
        parameters=(
            "loading.common.TNm",
            "loading.common.TXx",
            "loading.common.TNn",
            "sd.factor.common.slope",
            "sd.channel.TXx.level",
            "xi.TXx",
            "xi.TNn",
        ),
    )
    fit.plot("traces")
    fit.plot("identification", baseline=baseline)
    for channel in ("TXx", "TNn"):
        fit.plot("loading_deviation", channel=channel, baseline=baseline)
        fit.plot("idiosyncratic_innovations", channel=channel)
    plt.show()


if __name__ == "__main__":
    main()
