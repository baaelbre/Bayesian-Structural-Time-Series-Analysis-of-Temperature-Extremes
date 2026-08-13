import numpy as np
import bucex as bx

model = bx.Model(
    bx.Gaussian(),
    [bx.LocalLinearTrend()],
)

truth = {
    "sd.level": 0.10,
    "sd.slope": 0.008,
    "sigma": 0.40,
}

sim = bx.simulate(
    model,
    n_time=300,
    params=truth,
    initial_state=np.array([10.0, 0.02]),
    seed=123,
)

print(sim.y.shape)       # (300,)
print(sim.states.shape)  # (301, 2)
print(sim.eta.shape)     # (300,)

fit = bx.fit(
    sim.y,
    model,
    engine="ffbs",
    parameterization="fruehwirth_schnatter",
    priors="normal",
    asis=True,
    mcmc=bx.MCMC(
        draws=1000,
        warmup=1000,
        chains=4,
        seed=456,
        progress=True,
    ),
)

print(fit.static_summary())

# compare posterior intervals with the truth
for name, true_value in truth.items():
    draws = fit.parameter(name)

    lower, median, upper = np.quantile(
        draws, [0.025, 0.50, 0.975]
    )

    print(
        f"{name:12s}",
        f"truth={true_value:.4f}",
        f"posterior={median:.4f}",
        f"95% CI=({lower:.4f}, {upper:.4f})",
    )
    
    # compare estimated and true level
    level_draws = fit.state("level")
level_mean = level_draws.mean(axis=0)
true_level = sim.states[1:, 0]

rmse = np.sqrt(np.mean((level_mean - true_level) ** 2))
coverage = np.mean(
    (true_level >= np.quantile(level_draws, 0.025, axis=0))
    &
    (true_level <= np.quantile(level_draws, 0.975, axis=0))
)

print("Level RMSE:", rmse)
print("Pointwise 95% coverage:", coverage)






# compare parametrizations
fits = {}

for i, parameterization in enumerate(
    [
        "centered",
        "disturbance",
        "fruehwirth_schnatter",
    ]
):
    fits[parameterization] = bx.fit(
        sim.y,
        model,
        engine="ffbs",
        parameterization=parameterization,
        priors="normal",
        asis=False,
        mcmc=bx.MCMC(
            draws=1000,
            warmup=1000,
            chains=4,
            seed=1000 + i,
        ),
    )
    
    for name, result in fits.items():
        print("\n", name)
        print(result.static_summary())
        print(result.diagnostics()["parameters"])