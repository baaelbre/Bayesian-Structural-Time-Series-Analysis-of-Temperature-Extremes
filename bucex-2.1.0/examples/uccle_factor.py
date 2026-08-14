"""Play 07: the proposed six-summary Uccle dynamic factor model."""
from __future__ import annotations

import os

import bucex as bx
from play_config import QUICK, describe, mcmc, particles, show_figures


START = os.getenv("BUCEX_UCCLE_START", "2010-01-01" if QUICK else "") or None
END = os.getenv("BUCEX_UCCLE_END") or None
REFERENCE = "TXm"
PRIOR = os.getenv("BUCEX_FACTOR_PRIOR", "regularized_horseshoe")


def main() -> None:
    print(describe())
    print(f"Uccle window: {START or 'first observation'} to {END or 'last observation'}")
    model = bx.make_uccle_factor_model(
        structure="estimated",
        individual="local_level",
        seasonal="series_specific",
        seasonal_mode="dynamic",
    )
    data = bx.load_uccle_factor_data(start=START, end=END)
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
        mcmc=mcmc(70, quick_draws=80, quick_warmup=120),
        particles=particles(),
    )
    print("\nINFERENCE PLAN\n", fit.plan)
    print(
        "\nIDENTIFICATION DIAGNOSTICS\n",
        fit.factor_identification_diagnostics(
            baseline=slice(0, min(60, fit.n_time))
        ).round(3),
    )
    start_year = int(data.index[0].year)
    end_year = int(data.index[-1].year)
    print(
        "\nCOMMON-FACTOR CHANGE\n",
        fit.factor_rate_summary("common", start_year, end_year),
    )

    baseline = slice(0, min(60, fit.n_time))
    fit.plot("factor", factor="common")
    fit.plot(
        "factor_decomposition",
        channels=("TXm", "TXx", "TNn"),
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
    show_figures()


if __name__ == "__main__":
    main()
