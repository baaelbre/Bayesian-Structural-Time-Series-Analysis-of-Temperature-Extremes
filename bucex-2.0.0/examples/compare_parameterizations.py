"""Compare parameterizations without changing model or result APIs."""
import numpy as np

import bucex as bx


rng = np.random.default_rng(20)
y = np.cumsum(rng.normal(scale=0.03, size=96)) + rng.normal(scale=0.25, size=96)
model = bx.structural_model("gaussian", period=12)

for index, parameterization in enumerate(
    ("centered", "disturbance", "fruehwirth_schnatter")
):
    fit = bx.fit(
        y,
        model,
        parameterization=parameterization,
        priors="normal",
        asis=True,
        mcmc=bx.MCMC(draws=500, warmup=500, chains=4, seed=21 + index),
    )
    print(parameterization, fit.plan)
    print(fit.diagnostics()["parameters"].loc[["sd.level", "sd.slope"]])
