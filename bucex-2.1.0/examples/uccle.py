"""Fit one bundled Uccle series and write a safe fit archive."""
from pathlib import Path

import bucex as bx


fit = bx.fit_uccle_series(
    "TXx",
    priors="manuscript_lasso",
    parameterization="fruehwirth_schnatter",
    engine="laplace",
    asis=True,
    mcmc=bx.MCMC(
        draws=1_000,
        warmup=1_000,
        chains=4,
        seed=40,
        progress=True,
    ),
)

Path("results").mkdir(exist_ok=True)
fit.save("results/TXx.bucex")
print(fit.static_summary())
print(fit.period_rate_summary({"recent": (1980, 2022)}))
