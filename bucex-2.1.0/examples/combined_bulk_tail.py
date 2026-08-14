"""Play 06: parallel Gaussian bulk and GEV tail analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd

import bucex as bx
from play_config import describe, mcmc, particles, show_figures


N_TIME = 120


def main() -> None:
    print(describe())
    rng = np.random.default_rng(60)
    time = np.arange(N_TIME)
    common = 8.0 + 0.012 * time + 0.7 * np.sin(2.0 * np.pi * time / 12.0)
    dates = pd.date_range("2015-01-01", periods=N_TIME, freq="MS")
    bulk = pd.Series(common + rng.normal(scale=0.30, size=N_TIME), index=dates)
    tail = pd.Series(common + rng.gumbel(scale=0.50, size=N_TIME), index=dates)

    pair = bx.fit_bulk_tail(
        bulk,
        tail,
        period=12,
        bulk_priors="pc",
        tail_priors="regularized_horseshoe",
        bulk_mcmc=mcmc(61),
        tail_mcmc=mcmc(62),
        tail_engine="pgas",
        parameterization="fruehwirth_schnatter",
        asis=True,
        particles=particles(),
    )
    print("\nGAUSSIAN PLAN\n", pair.bulk.plan)
    print("\nGEV PLAN\n", pair.tail.plan)
    print("\nGAUSSIAN PARAMETERS\n", pair.bulk.static_summary())
    print("\nGEV PARAMETERS\n", pair.tail.static_summary())
    print(
        "\nfit_bulk_tail aligns the APIs and plots, but the two posterior fits "
        "remain independent. Use a FactorModel when their latent evolution "
        "should be estimated jointly."
    )
    pair.plot()
    pair.tail.plot("endpoint")
    show_figures()


if __name__ == "__main__":
    main()
