import bucex as bx

txx = bx.load_uccle_series("TXx")

print(txx.head())
print(txx.tail())
print(len(txx))

fit = bx.fit_uccle_series(
    "TXx",
    engine="laplace",
    parameterization="disturbance",
    priors="pc",
    asis=True,
    mcmc=bx.MCMC(
        draws=500,
        warmup=500,
        chains=1,
        seed=40,
        progress=True,
    ),
)
print(fit.plan)
print(fit.meta)
print(fit.state_names)
print(fit.parameter_draws.keys())
print(fit.static_summary())

# (chains, draws, T+1, state_dim)
print(fit.state_draws.shape)

# access particular quantities
level = fit.state("level")
slope = fit.state("slope")

sd_level = fit.parameter("sd.level")
sd_slope = fit.parameter("sd.slope")
sd_seasonal = fit.parameter("sd.seasonal")

sigma = fit.parameter("sigma")
xi = fit.parameter("xi")

# for multiple chains, if you don't want to combine them
xi_by_chain = fit.parameter("xi", combine_chains=False)
level_by_chain = fit.state("level", combine_chains=False)

fit.plot("level")
fit.plot("level_slope")
# prior sensitivity plots; overlays marginal prior and posterior distributions
fit.plot("process_sd")
fit.plot("endpoint")

# risk plots
fit.plot(
    "exceedance",
    threshold=35,
    annual=True,
)

fit.plot(
    "return_period",
    threshold=35,
    annual=True,
    max_return_period=10_000,
)


###############
# FORECASTING
################
forecast = fit.forecast(
    horizon=12,
    draws=1000,
    seed=500,
)

print(forecast.summary())
forecast.plot()

# forecast evaluation
# This produces ensemble-based CRPS, threshold-weighted CRPS, 
# quantile scores, and threshold-event Brier/log scores.

fit.save("results/TXx_laplace_lasso.bucex")

restored = bx.FitResult.load(
    "results/TXx_laplace_lasso.bucex"
)

print(restored.plan)
print(restored.static_summary())