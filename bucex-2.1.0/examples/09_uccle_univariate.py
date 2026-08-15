"""Example 9: analyse the six Uccle summaries separately.

This is the simple alternative to the joint factor model in Example 10. The
default SSVS run asks, for each summary separately, whether level, slope, and
seasonality are absent, fixed, or dynamic. Change ``PRIOR`` to ``pc``,
``triple_gamma``, or ``regularized_triple_gamma`` for continuous shrinkage.

For SSVS, Gaussian summaries use exact FFBS model-space updates. GEV SSVS is
currently paired with the faster Laplace pseudo-observation update and is
therefore labelled approximate. Continuous-prior GEV fits use PGAS.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import bucex as bx


# Start with one name, for example ("TXm",), while learning the workflow.
# The default below performs the six scientifically separate analyses.
SERIES_TO_FIT = ("TXm", "TNm", "TXx", "TXn", "TNx", "TNn")
START = "1980-01-01"       # use None for the complete 1892--2022 record
END = None
PRIOR = "ssvs"             # pc, triple_gamma, regularized_triple_gamma, ...
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
N_PARTICLES = 512
SEED = 901
SAVE_FITS = False
SHOW_PLOTS = False             # set True after selecting one or two series
FIGURE_DIR = Path("figures/09_uccle_univariate")
RESULT_DIR = Path("results/09_uccle_univariate")


def engine_for(family: str) -> str:
    if family == "gaussian":
        return "ffbs"
    return "laplace" if PRIOR == "ssvs" else "pgas"


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_FITS:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    probability_tables = []

    for index, series in enumerate(SERIES_TO_FIT):
        info = bx.UCCLE_INFO[series]
        engine = engine_for(info["family"])
        print(f"\n{'=' * 78}\n{series}: {info['description']}\n{'=' * 78}")
        print(
            f"family={info['family']}, tail={info['tail']}, "
            f"prior={PRIOR}, engine={engine}"
        )
        if info["family"] == "gev" and PRIOR == "ssvs":
            print(
                "NOTE: this GEV SSVS run uses the documented Laplace "
                "pseudo-observation approximation."
            )

        fit = bx.fit_uccle_series(
            series,
            start=START,
            end=END,
            priors=PRIOR,
            parameterization="fruehwirth_schnatter",
            engine=engine,
            asis=PRIOR != "ssvs",
            mcmc=bx.MCMC(
                draws=DRAWS,
                warmup=WARMUP,
                chains=CHAINS,
                seed=SEED + index,
                progress=True,
            ),
            particles=bx.Particles(n=N_PARTICLES, proposal="guided"),
        )

        diagnostics = fit.diagnostics()
        scientific = [
            *(f"sd.{name}" for name in fit.compiled.noise_names),
            "sigma",
            *(["xi"] if fit.family == "gev" else []),
        ]
        print("\nINFERENCE PLAN\n", fit.plan)
        print(
            "\nSCIENTIFIC-PARAMETER DIAGNOSTICS\n",
            diagnostics["parameters"].loc[scientific].round(4),
        )
        if diagnostics["engine"]:
            print("\nENGINE DIAGNOSTICS\n", diagnostics["engine"])
        if fit.family == "gev":
            print(
                "\nRESTORED FRACTION\n",
                fit.metadata.get("restored_fraction", 0.0),
            )

        summaries = fit.static_summary()
        for parameter in scientific:
            row = summaries[parameter]
            summary_rows.append(
                {
                    "series": series,
                    "family": fit.family,
                    "parameter": parameter,
                    "median": row["median"],
                    "lower90": row["lower"],
                    "upper90": row["upper"],
                    "rhat": diagnostics["parameters"].loc[parameter, "rhat"],
                    "ess_bulk": diagnostics["parameters"].loc[
                        parameter, "ess_bulk"
                    ],
                    "diagnostic": diagnostics["parameters"].loc[
                        parameter, "diagnostic"
                    ],
                }
            )

        if PRIOR == "ssvs":
            probabilities = fit.component_probabilities().copy()
            probabilities.insert(0, "series", series)
            probability_tables.append(probabilities.reset_index())
            print("\nSSVS COMPONENT PROBABILITIES\n", probabilities.round(3))
            print(
                "\nSSVS SWITCHING DIAGNOSTICS\n",
                fit.component_transition_summary().round(3),
            )

        start_year = int(pd.Timestamp(fit.dates[0]).year)
        end_year = int(pd.Timestamp(fit.dates[-1]).year)
        print(
            "\nAVERAGE RATE OVER THE FITTED PERIOD\n",
            fit.period_rate_summary(
                {"fitted_period": (start_year, end_year)}
            ),
        )

        if SAVE_FITS:
            output = RESULT_DIR / f"{series}_{PRIOR}.bucex"
            fit.save(output)
            print(f"\nSaved {output}")

        prefix = FIGURE_DIR / f"{series}_{PRIOR}"
        fit.plot("predictor", save=f"{prefix}_predictor.png")
        fit.plot("level_slope", save=f"{prefix}_level_slope.png")
        fit.plot(
            "process_sd",
            title=f"{series}: {PRIOR} prior",
            save=f"{prefix}_prior_posterior_sd.png",
        )
        fit.plot(
            "traces", parameters=scientific,
            save=f"{prefix}_traces.png",
        )
        fit.plot(
            "acf", parameters=scientific, max_lag=50,
            save=f"{prefix}_acf.png",
        )
        if PRIOR == "ssvs":
            fit.plot(
                "component_probabilities",
                save=f"{prefix}_component_probabilities.png",
            )
        if fit.family == "gev":
            fit.plot("endpoint", save=f"{prefix}_endpoint.png")
        if not SHOW_PLOTS:
            # All figures are already saved. Closing per series keeps the
            # six-series workflow from holding forty Matplotlib windows.
            plt.close("all")

    comparison = pd.DataFrame(summary_rows).set_index(
        ["series", "parameter"]
    )
    print("\n\nRESULTS ACROSS THE SIX SEPARATE ANALYSES\n")
    print(comparison.round(5).to_string())
    if probability_tables:
        print("\n\nSSVS PROBABILITIES ACROSS ALL SIX SUMMARIES\n")
        print(
            pd.concat(probability_tables, ignore_index=True)
            .set_index(["series", "process"])
            .round(3)
            .to_string()
        )
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
