"""Complete simulated parallel bulk/tail workflow."""
import numpy as np

import bucex as bx


model = bx.Model(
    observation=bx.Gaussian(),
    components=[bx.LocalLinearTrend(), bx.DummySeasonal(period=4)],
)

parameters = {
    "sd.level": 0.04,
    "sd.slope": 0.003,
    "sd.seasonal": 0.03,
    "sigma": 0.25,
}
bulk = bx.simulate(model, 80, parameters, initial_state=np.zeros(5), seed=10)

tail_model = model.with_observation(bx.GEV())
tail_parameters = {**parameters, "sigma": 0.8, "xi": -0.15}
tail = bx.simulate(tail_model, 80, tail_parameters, initial_state=np.zeros(5), seed=11)

quick = bx.MCMC(draws=100, warmup=100, chains=2, seed=20)
pair = bx.fit_bulk_tail(
    bulk.y,
    tail.y,
    components=model.components,
    bulk_mcmc=quick,
    tail_mcmc=bx.MCMC(draws=100, warmup=100, chains=2, seed=21),
    tail_engine="laplace",
    asis=True,
)

forecast = pair.forecast(12, draws=100, seed=30)
print(pair.process_sd_summary())
print(forecast["tail"].summary(level=0.95))

