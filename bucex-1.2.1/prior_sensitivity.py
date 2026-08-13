import bucex as bx
"""
This script evaluates the sensitivity of the model to different prior specifications.
"""

prior_fits = {}

for i, prior in enumerate(
    [
        "manuscript_lasso",
        "regularized_lasso",
        "regularized_horseshoe",
        "pc",
        "normal",
    ]
):
    prior_fits[prior] = bx.fit_uccle_series(
        "TXx",
        engine="laplace",
        parameterization="fruehwirth_schnatter",
        priors=prior,
        asis=True,
        mcmc=bx.MCMC(
            draws=1000,
            warmup=1000,
            chains=4,
            seed=200 + i,
        ),
    )
    
for prior, result in prior_fits.items():
    print("\n", prior)
    print(result.static_summary())
    print(
        result.period_rate_summary(
            {"recent": (1980, 2022)},
            scale="decade",
        )
    )