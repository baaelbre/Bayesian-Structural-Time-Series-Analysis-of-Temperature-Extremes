"""Example 2: fit a GEV local-linear-trend model with exact PGAS.

Compared with Example 1, the Gaussian likelihood is replaced by a GEV
likelihood and FFBS by PGAS.  The script explicitly checks particle mixing,
iteration restoration, and the GEV support margin.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 120
PERIOD = 12
DRAWS = 500
WARMUP = 750
CHAINS = 4
N_PARTICLES = 256
SEED = 201
FIGURE_DIR = Path("figures/02_gev_local_trend")


def main() -> None:
    model = bx.Model(
        bx.GEV(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(PERIOD)),
        name="simulated monthly GEV series",
    )
    truth = {
        "sd.level": 0.020,
        "sd.slope": 0.00015,
        "sd.seasonal": 0.015,
        "sigma": 0.50,
        "xi": -0.08,
    }

    compiled = bx.compile_model(model, np.zeros(N_TIME))
    initial_state = np.zeros(compiled.state_dim)
    initial_state[compiled.state_names.index("slope")] = 0.008
    simulation = bx.simulate(
        model,
        N_TIME,
        truth,
        initial_state=initial_state,
        seed=SEED,
    )
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    data = pd.Series(simulation.y, index=dates, name="block maximum")

    fit = bx.fit(
        data,
        model,
        parameterization="fruehwirth_schnatter",
        engine="pgas",                    # exact-invariant particle update
        priors="pc",
        asis=True,
        mcmc=bx.MCMC(
            draws=DRAWS,
            warmup=WARMUP,
            chains=CHAINS,
            seed=SEED + 1,
            progress=True,
        ),
        particles=bx.Particles(n=N_PARTICLES, proposal="guided"),
    )

    parameters = (
        "sd.level",
        "sd.slope",
        "sd.seasonal",
        "sigma",
        "xi",
    )
    diagnostics = fit.diagnostics()
    print("\nINFERENCE PLAN\n", fit.plan)
    print(
        "\nCONVERGENCE DIAGNOSTICS\n",
        diagnostics["parameters"].loc[list(parameters)].round(4),
    )
    print("\nPARTICLE DIAGNOSTICS\n", diagnostics["engine"])

    print("\nRESTORATION DIAGNOSTICS")
    for name in (
        "restored_iterations_by_chain",
        "restored_fraction",
        "attempt_failure_counts",
        "restore_failure_counts",
    ):
        print(f"  {name}: {fit.metadata.get(name, 0)}")

    # Every retained draw must satisfy the GEV support constraint.
    eta = fit.eta_draws()
    sigma = fit.parameter("sigma")
    xi = fit.parameter("xi")
    margin = 1.0 + xi[:, None] * (fit.y[None, :] - eta) / sigma[:, None]
    minimum_margin = np.min(margin, axis=1)
    print("\nGEV SUPPORT CHECK")
    print("  invalid retained draws:", int(np.sum(minimum_margin <= 0.0)))
    print(
        "  minimum-margin quantiles:",
        np.quantile(minimum_margin, (0.00, 0.01, 0.05, 0.50)).round(8),
    )

    if fit.metadata.get("restored_iterations", 0):
        print(
            "\nWARNING: restored iterations occurred. Do not interpret this fit "
            "until examples/03_diagnose_gev_pgas.py has localized the cause."
        )

    rows = []
    summaries = fit.static_summary()
    for parameter in parameters:
        row = summaries[parameter]
        rows.append(
            {
                "parameter": parameter,
                "truth": truth[parameter],
                "median": row["median"],
                "lower90": row["lower"],
                "upper90": row["upper"],
            }
        )
    print(
        "\nPARAMETER RECOVERY\n",
        pd.DataFrame(rows).set_index("parameter").round(5),
    )

    print(
        "\n12-STEP FORECAST\n",
        fit.forecast(12, draws=1_000, seed=SEED + 2).summary().round(3),
    )

    predictor_figure, predictor_axis = fit.plot("predictor")
    predictor_axis.plot(dates, simulation.eta, color="black", label="true predictor")
    predictor_axis.legend()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    predictor_figure.savefig(FIGURE_DIR / "predictor.png", bbox_inches="tight")
    fit.plot("level_slope", save=FIGURE_DIR / "level_slope.png")
    fit.plot(
        "process_sd", truths=truth, title="PC prior: GEV model",
        save=FIGURE_DIR / "prior_posterior_sd.png",
    )
    fit.plot(
        "traces", parameters=parameters, truths=truth,
        save=FIGURE_DIR / "traces.png",
    )
    fit.plot(
        "acf", parameters=parameters, max_lag=50,
        save=FIGURE_DIR / "acf.png",
    )
    fit.plot(
        "parameter_density", parameters=parameters, truths=truth,
        save=FIGURE_DIR / "densities.png",
    )
    fit.plot("endpoint", save=FIGURE_DIR / "endpoint.png")
    plt.show()


if __name__ == "__main__":
    main()
