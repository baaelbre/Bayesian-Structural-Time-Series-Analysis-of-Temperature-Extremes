import bucex as bx
import numpy as np

model_gev = bx.Model(
    bx.GEV(),
    [
        bx.LocalLinearTrend(),
        bx.DummySeasonal(period=12),
    ],
)

truth_gev = {
    "sd.level": 0.06,
    "sd.slope": 0.003,
    "sd.seasonal": 0.04,
    "sigma": 0.80,
    "xi": -0.15,
}

sim_gev = bx.simulate(
    model_gev,
    n_time=360,
    params=truth_gev,
    initial_state=np.r_[20.0, 0.01, np.zeros(11)],
    seed=200,
)
print('Laplace fit')
fit_laplace = bx.fit(
    sim_gev.y,
    model_gev,
    engine="laplace",
    parameterization="fruehwirth_schnatter",
    priors="regularized_horseshoe",
    asis=True,
    mcmc=bx.MCMC(
        draws=1000,
        warmup=1000,
        chains=1,
        seed=201,
        progress=True
    ),
)

# PGAS
print("\n\nPGAS fit")

fit_pgas = bx.fit(
    sim_gev.y,
    model_gev,
    engine="pgas",
    parameterization="fruehwirth_schnatter",
    priors="regularized_horseshoe",
    asis=True,
    particles=bx.Particles(
        n=256,
        proposal="guided",
    ),
    mcmc=bx.MCMC(
        draws=500,
        warmup=500,
        chains=4,
        seed=202,
        progress=True
    ),
)

# compare gev parameters
for parameter in ["sigma", "xi"]:
    print("\n", parameter, "truth:", truth_gev[parameter])

    for name, result in {
        "Laplace": fit_laplace,
        "PGAS": fit_pgas,
    }.items():
        draws = result.parameter(parameter)
        print(name, np.quantile(draws, [0.025, 0.5, 0.975]))