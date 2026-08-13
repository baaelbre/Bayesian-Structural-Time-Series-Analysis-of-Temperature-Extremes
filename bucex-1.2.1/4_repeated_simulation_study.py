records = []

for replication in range(100):
    sim_r = bx.simulate(
        model,
        n_time=300,
        params=truth,
        initial_state=np.array([10.0, 0.02]),
        seed=10_000 + replication,
    )

    fit_r = bx.fit(
        sim_r.y,
        model,
        engine="ffbs",
        parameterization="fruehwirth_schnatter",
        priors="normal",
        asis=True,
        mcmc=bx.MCMC(
            draws=500,
            warmup=500,
            chains=1,
            seed=20_000 + replication,
        ),
    )

    for parameter, true_value in truth.items():
        draws = fit_r.parameter(parameter)
        lower, median, upper = np.quantile(
            draws, [0.025, 0.50, 0.975]
        )

        records.append({
            "replication": replication,
            "parameter": parameter,
            "truth": true_value,
            "estimate": median,
            "lower": lower,
            "upper": upper,
            "covered": lower <= true_value <= upper,
        })
        
        
import pandas as pd

results = pd.DataFrame(records)

summary = (
    results
    .groupby("parameter")
    .agg(
        mean_estimate=("estimate", "mean"),
        bias=("estimate", lambda x: x.mean()),
        coverage=("covered", "mean"),
        mean_width=("upper", lambda x: np.nan),
    )
)

summary["bias"] -= (
    results.groupby("parameter")["truth"].first()
)

print(summary)