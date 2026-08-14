"""Fit the bucex 2.1 one-factor Uccle model."""
from pathlib import Path

import bucex as bx


data = bx.load_uccle_factor_data()
model = bx.make_uccle_factor_model()
fit = bx.fit(
    data,
    model,
    parameterization="fruehwirth_schnatter",
    engine="pgas",
    priors="regularized_horseshoe",
    asis=True,
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=50),
    particles=bx.Particles(n=1_024, proposal="guided"),
)

Path("results").mkdir(exist_ok=True)
fit.save("results/uccle-factor-2.1.bucex")
print(fit.factor_rate_summary("common", 1950, 2022))
print(fit.factor_probabilities("common", start_year=1950, end_year=2022))
print(fit.loading_probability("common", "TXx", threshold=1.0))
