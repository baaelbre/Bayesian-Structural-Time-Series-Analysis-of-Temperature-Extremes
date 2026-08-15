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
PRIOR = "regularized_triple_gamma"  # try triple_gamma or regularized_horseshoe
# Used only when PRIOR contains "triple_gamma". Fixed shapes are a stable
# starting point; learn_shapes=True is best treated as a sensitivity run.
TRIPLE_GAMMA_OPTIONS = {
    "spike_shape": 0.10,
    "tail_shape": 0.10,
    "learn_global": True,
    "learn_shapes": False,
    "slab_scale": 2.0,
}
DRAWS = 500
WARMUP = 750
CHAINS = 4
N_PARTICLES = 512
SEED = 1_001
SAVE_FIT = False
FIGURE_DIR = Path("figures/10_uccle_factor")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
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
        triple_gamma_options=(
            TRIPLE_GAMMA_OPTIONS if "triple_gamma" in PRIOR else None
        ),
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
    if fit.priors.triple_gamma is not None:
        rho_names = [
            name for name in fit.parameter_draws
            if name.startswith("triple_gamma.rho.")
        ]
        print(
            "\nTRIPLE-GAMMA SHRINKAGE FACTORS (rho near 1 = strong shrinkage)\n",
            fit.diagnostics()["parameters"].loc[rho_names].round(3),
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

    fit.plot(
        "factor", factor="common",
        save=FIGURE_DIR / "common_factor.png",
    )
    fit.plot(
        "factor_decomposition",
        channels=("TXm", "TNm", "TXx", "TNn"),
        baseline=baseline,
        save=FIGURE_DIR / "decomposition.png",
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
        save=FIGURE_DIR / "densities.png",
    )
    fit.plot("traces", save=FIGURE_DIR / "traces.png")
    fit.plot("acf", max_lag=50, save=FIGURE_DIR / "acf.png")
    fit.plot(
        "identification", baseline=baseline,
        save=FIGURE_DIR / "identification.png",
    )
    for channel in ("TXx", "TNn"):
        fit.plot(
            "loading_deviation", channel=channel, baseline=baseline,
            save=FIGURE_DIR / f"{channel}_loading_deviation.png",
        )
        fit.plot(
            "idiosyncratic_innovations", channel=channel,
            save=FIGURE_DIR / f"{channel}_innovations.png",
        )
    plt.show()


if __name__ == "__main__":
    main()
