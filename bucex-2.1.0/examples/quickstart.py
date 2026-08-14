"""Small synthetic Gaussian and GEV fits through the same framework."""
import numpy as np

import bucex as bx


rng = np.random.default_rng(7)
time = np.arange(120)
level = 8.0 + 0.01 * time + np.sin(2.0 * np.pi * time / 12.0)

gaussian = bx.fit(
    level + rng.normal(scale=0.3, size=time.size),
    family="gaussian",
    period=12,
    parameterization="disturbance",
    priors="pc",
    asis=True,
    mcmc=bx.MCMC(draws=250, warmup=250, chains=2, seed=8),
)

gev = bx.fit(
    level + rng.gumbel(scale=0.5, size=time.size),
    family="gev",
    period=12,
    parameterization="fruehwirth_schnatter",
    engine="pgas",
    priors="regularized_horseshoe",
    asis=True,
    mcmc=bx.MCMC(draws=250, warmup=250, chains=2, seed=9),
    particles=bx.Particles(n=128),
)

print(gaussian.summary_dict())
print(gev.diagnostics()["engine"])
print(gev.forecast(12, draws=500, seed=10).summary())
