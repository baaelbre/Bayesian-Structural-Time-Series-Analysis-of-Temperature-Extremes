"""Small mixed Gaussian/GEV dynamic-factor example."""
from __future__ import annotations

import bucex as bx


model = bx.FactorModel(
    channels=(
        bx.Channel(
            "mean",
            bx.Gaussian(),
            components=(bx.LocalLevel(), bx.DummySeasonal(period=12)),
        ),
        bx.Channel(
            "maximum",
            bx.GEV(),
            components=(bx.LocalLevel(), bx.DummySeasonal(period=12)),
        ),
    ),
    factors=(
        bx.Factor(
            "climate",
            components=(
                bx.LocalLinearTrend(
                    initial_level=0.0,
                    initial_slope=0.0,
                    initial_level_sd=0.0,
                ),
            ),
            loadings={
                "mean": 1.0,
                "maximum": bx.Loading.estimated(0.8, sd=1.0),
            },
        ),
    ),
    name="synthetic shared climate trend",
)

truth = {
    "sd.factor.climate.level": 0.05,
    "sd.factor.climate.slope": 0.005,
    "sd.channel.mean.level": 0.01,
    "sd.channel.mean.seasonal": 0.02,
    "sd.channel.maximum.level": 0.015,
    "sd.channel.maximum.seasonal": 0.025,
    "sigma.mean": 0.30,
    "sigma.maximum": 0.45,
    "xi.maximum": -0.05,
    "loading.climate.maximum": 0.8,
}

simulation = bx.simulate(model, 60, truth, seed=10)
fit = bx.fit(
    simulation.y,
    model,
    engine="pgas",
    parameterization="fruehwirth_schnatter",
    priors="regularized_horseshoe",
    asis=True,
    mcmc=bx.MCMC(draws=100, warmup=100, chains=2, seed=11),
    particles=bx.Particles(n=128, proposal="guided"),
)

print(fit.plan)
print(fit.static_summary())
print(fit.factor("climate"))
print(fit.reconstructed_state("maximum"))
print(fit.diagnostics()["engine"])
print(fit.forecast(12, draws=100, seed=12).summary())
