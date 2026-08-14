"""Example 9: fit one bundled Uccle temperature summary.

Choose one of TXm, TNm, TXx, TXn, TNx, or TNn below.  Gaussian summaries use
FFBS.  GEV summaries use PGAS here; set GEV_ENGINE='laplace' for a faster but
approximate exploratory fit.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import bucex as bx


SERIES = "TXx"
START = "1980-01-01"       # use None for the complete 1892--2022 record
END = None
GEV_ENGINE = "pgas"        # alternatively: "laplace"
PRIOR = "pc"
DRAWS = 1_000
WARMUP = 1_000
CHAINS = 4
N_PARTICLES = 512
SEED = 901
SAVE_FIT = False


def main() -> None:
    info = bx.UCCLE_INFO[SERIES]
    engine = "ffbs" if info["family"] == "gaussian" else GEV_ENGINE
    print(f"Fitting {SERIES}: {info['description']}")
    print(f"family={info['family']}, tail={info['tail']}, engine={engine}")

    fit = bx.fit_uccle_series(
        SERIES,
        start=START,
        end=END,
        priors=PRIOR,
        parameterization="fruehwirth_schnatter",
        engine=engine,
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

    print("\nINFERENCE PLAN\n", fit.plan)
    print("\nPARAMETER DIAGNOSTICS\n", fit.diagnostics()["parameters"].round(4))
    print("\nENGINE DIAGNOSTICS\n", fit.diagnostics()["engine"])
    if fit.family == "gev":
        print("\nRESTORED FRACTION\n", fit.metadata.get("restored_fraction", 0.0))

    start_year = fit.dates[0].astype("datetime64[Y]").astype(int) + 1970
    end_year = fit.dates[-1].astype("datetime64[Y]").astype(int) + 1970
    print(
        "\nAVERAGE RATE OVER THE FITTED PERIOD\n",
        fit.period_rate_summary({"fitted_period": (int(start_year), int(end_year))}),
    )

    if SAVE_FIT:
        Path("results").mkdir(exist_ok=True)
        output = Path("results") / f"{SERIES}.bucex"
        fit.save(output)
        print(f"\nSaved {output}")

    fit.plot("predictor")
    fit.plot("level_slope")
    fit.plot("process_sd", title=f"{SERIES}: {PRIOR} prior")
    fit.plot("traces")
    if fit.family == "gev":
        fit.plot("endpoint")
    plt.show()


if __name__ == "__main__":
    main()
