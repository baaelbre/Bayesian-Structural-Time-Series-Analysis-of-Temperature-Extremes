"""Example 13: leave-future-out prediction and proper forecast scores.

The model is repeatedly refitted to an expanding history. CRPS, analytic log
predictive score, tail-weighted CRPS, exceedance scores, quantile scores, and
PIT values are evaluated only on observations that were genuinely held out.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bucex as bx


N_TIME = 144
INITIAL = 96
HORIZON = 12
STEP = 12
SEED = 1_301
FIGURE_DIR = Path("figures/13_leave_future_out")


def main() -> None:
    model = bx.Model(
        bx.Gaussian(),
        (bx.LocalLinearTrend(), bx.DummySeasonal(12)),
        name="forecast validation example",
    )
    truth = {
        "sd.level": 0.015,
        "sd.slope": 0.0001,
        "sd.seasonal": 0.01,
        "sigma": 0.30,
    }
    compiled = bx.compile_model(model, np.zeros(N_TIME))
    initial_state = np.zeros(compiled.state_dim)
    initial_state[compiled.state_names.index("slope")] = 0.004
    simulation = bx.simulate(
        model,
        N_TIME,
        truth,
        initial_state=initial_state,
        seed=SEED,
    )
    dates = pd.date_range("2000-01-01", periods=N_TIME, freq="MS")
    data = pd.Series(simulation.y, index=dates, name="temperature")
    high_threshold = float(np.quantile(data.iloc[:INITIAL], 0.90))

    result = bx.leave_future_out(
        data,
        model,
        initial=INITIAL,
        horizon=HORIZON,
        step=STEP,
        fit_options={
            "priors": "pc",
            "parameterization": "fruehwirth_schnatter",
            "engine": "ffbs",
            "asis": True,
            "mcmc": bx.MCMC(
                draws=300,
                warmup=300,
                chains=2,
                seed=SEED + 1,
                progress=True,
            ),
        },
        forecast_options={"draws": 500, "seed": SEED + 2},
        thresholds=(high_threshold,),
        quantiles=(0.90, 0.95, 0.99),
        progress=True,
    )

    print("\nNUMBER OF ORIGINS\n", result.n_origins)
    print("\nSCORE SUMMARY (LOWER IS BETTER)\n", result.score_summary().round(4))
    print("\nPIT SUMMARY\n", result.pit_diagnostics().summary)
    print("\nFIRST HELD-OUT PREDICTIONS\n", result.predictions.head(12).round(3))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    result.plot_pit(save=FIGURE_DIR / "pit.png")
    result.scores.to_csv(FIGURE_DIR / "scores.csv", index=False)
    result.predictions.to_csv(FIGURE_DIR / "predictions.csv", index=False)
    plt.show()


if __name__ == "__main__":
    main()
