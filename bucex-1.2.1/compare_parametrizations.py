import bucex as bx
"""
This script compares the three parameterizations of the Gaussian structural model.
We look at posterior levels, innovation SDs, R-hat, ESS, runtime and autocorrelation
"""

y = bx.load_uccle_series("TXm").to_numpy()
model = bx.structural_model("gaussian", period=12)

fits = {}

for i, parameterization in enumerate(
    ["centered", "disturbance", "fruehwirth_schnatter"]
):
    fits[parameterization] = bx.fit(
        y,
        model,
        parameterization=parameterization,
        priors="normal",
        asis=True,
        mcmc=bx.MCMC(
            draws=500,
            warmup=500,
            chains=4,
            seed=100 + i,
        ),
    )

for name, result in fits.items():
    print(name)
    print(result.diagnostics()["parameters"])