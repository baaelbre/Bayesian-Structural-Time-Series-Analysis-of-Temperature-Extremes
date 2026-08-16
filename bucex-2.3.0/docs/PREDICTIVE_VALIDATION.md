# Predictive validation

## Forecast-level API

Every Gaussian, GEV, or factor `FitResult` produces the same `Forecast` object:

```python
forecast = fit.forecast(12, draws=2_000, seed=42)
forecast.summary()
forecast.score(observed, thresholds=[threshold])
forecast.score(observed, aggregate=False)
forecast.pit(observed)
forecast.pit_diagnostics(observed)
forecast.plot(save="figures/forecast.png")
```

For factor forecasts, pass `channel="TXx"` or provide a held-out matrix with
shape `(horizon, n_channels)` to `score()`.

## Implemented proper scores

- `log`: negative log posterior-predictive density. BUCEX evaluates the
  Gaussian or GEV log density for every posterior draw and uses a stable
  log-sum-exp mixture. It is not a KDE of simulated observations.
- `crps`: ensemble continuous ranked probability score.
- `twcrps`: threshold-weighted CRPS through the corresponding upper- or
  lower-tail censoring transform.
- `exceedance_brier`: Brier score for a threshold event.
- `exceedance_log`: binary log score for a threshold event.
- `quantile`: pinball/quantile score at each requested probability.

All are negatively oriented: lower is better. A continuous negative log score
may itself be negative when the predictive density exceeds one; comparisons,
not its sign, matter.

## Leave-future-out workflow

`leave_future_out()` performs expanding-window refits:

```python
result = bx.leave_future_out(
    data,
    model,
    initial=360,
    horizon=12,
    step=12,
    fit_options={
        "priors": "pc",
        "engine": "ffbs",
        "parameterization": "fruehwirth_schnatter",
        "mcmc": bx.MCMC(draws=2_000, warmup=2_000, chains=4),
    },
    forecast_options={"draws": 2_000, "seed": 43},
    thresholds=[30.0],
)
```

The returned object contains:

- `predictions`: interval summaries and observations by origin/horizon;
- `scores`: one row per origin, horizon, score, setting, and channel;
- `pits`: held-out PIT values by origin/horizon/channel;
- `score_summary()`: pooled mean, sum, and number of evaluations;
- `pit_diagnostics()`: histogram counts, mean, variance, KS description, and
  lag-one correlation;
- `plot_pit(save=...)`: a directly saveable calibration plot.

When `horizon > step`, held-out targets overlap. BUCEX retains those rows and
their forecast origin rather than pretending they are independent. Decide in
the analysis whether to report each horizon separately, select one forecast
per date, or use dependence-aware uncertainty for score differences.

## PIT interpretation

For a continuous calibrated predictive distribution, held-out PIT values are
uniform. Inspect the histogram for U-shapes (underdispersion), central humps
(overdispersion), and skew (location bias), then inspect temporal correlation.
The included KS p-value is descriptive: overlapping horizons, parameter
learning, and serial dependence invalidate a naive iid test interpretation.

`posterior_pit(fit)` remains available as an in-sample posterior-predictive
check. Because the same observations helped estimate their latent states, it
is generally too optimistic for calibration claims. Use leave-future-out PITs
for publication validation.

## Climate-extreme reporting

For the six Uccle summaries, report scores by channel and forecast horizon.
CRPS measures the whole distribution; tail-weighted CRPS and exceedance scores
focus on scientifically chosen temperature thresholds; the continuous log
score is sensitive to the full density and harshly penalizes support or tail
misspecification. No single score establishes adequacy, so pair them with PIT,
coverage, return-level checks, and sampler diagnostics.
