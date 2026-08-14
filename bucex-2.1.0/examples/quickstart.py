"""Play 01: the same structural model with Gaussian and GEV observations."""
from __future__ import annotations

import bucex as bx
from play_config import describe, mcmc, particles, show_figures


PERIOD = 12
N_TIME = 120
PARAMETERIZATION = "fruehwirth_schnatter"
GAUSSIAN_PRIOR = "pc"
GEV_PRIOR = "regularized_horseshoe"


def main() -> None:
    print(describe())
    gaussian_model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
        name="monthly Gaussian",
    )
    gev_model = bx.Model(
        bx.GEV(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
        name="monthly GEV",
    )

    gaussian_truth = {
        "sd.level": 0.025,
        "sd.slope": 0.002,
        "sd.seasonal": 0.018,
        "sigma": 0.30,
    }
    gev_truth = {
        "sd.level": 0.025,
        "sd.slope": 0.002,
        "sd.seasonal": 0.018,
        "sigma": 0.50,
        "xi": -0.08,
    }
    gaussian_data = bx.simulate(
        gaussian_model, N_TIME, gaussian_truth, seed=10
    ).y
    gev_data = bx.simulate(gev_model, N_TIME, gev_truth, seed=11).y

    print("\nGAUSSIAN FIT")
    gaussian = bx.fit(
        gaussian_data,
        gaussian_model,
        parameterization=PARAMETERIZATION,
        engine="ffbs",
        priors=GAUSSIAN_PRIOR,
        asis=True,
        mcmc=mcmc(12),
    )
    print(gaussian.plan)
    print(gaussian.diagnostics()["parameters"].round(3))

    print("\nGEV FIT")
    gev = bx.fit(
        gev_data,
        gev_model,
        parameterization=PARAMETERIZATION,
        engine="pgas",
        priors=GEV_PRIOR,
        asis=True,
        mcmc=mcmc(13),
        particles=particles(),
    )
    print(gev.plan)
    print(gev.diagnostics()["parameters"].round(3))
    print(
        "\n12-step GEV forecast\n",
        gev.forecast(12, draws=500, seed=14).summary(),
    )

    gaussian.plot("level_slope")
    gev.plot("level_slope")
    gev.plot("endpoint")
    show_figures()


if __name__ == "__main__":
    main()
