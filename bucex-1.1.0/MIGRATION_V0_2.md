# Migrating from v0.1 to v0.2

## Recommended public imports

```python
from bucex import (
    fit_bayes,
    fit_uccle_series,
    fit_uccle_all,
    plot,
)
```

## Old explicit workflow

```python
model = StructuralModel(...)
priors = NonCenteredGEVPriors(...)
fit = fit_bayes(
    y,
    model,
    priors,
    init_params_state,
    init_params_obs,
)
```

This still works with named arguments.

## New workflow

```python
fit = fit_bayes(
    y,
    family="gev",
    dates=dates,
    name="TXx",
    priors="manuscript",
    n_iter=20_000,
    burn=5_000,
)
```

For minima:

```python
fit = fit_bayes(y, family="gev", tail="min")
```

## Result access

```python
fit.state_draws("alpha")
fit.state_draws("beta")
fit.mu_draws()
fit.exceedance_probability_draws(20, annual=True)
fit.return_period_draws(39.7, annual=True)
fit.endpoint_draws()
fit.plot(type="level_slope")
fit.save("fit.pkl")
```

## Parameter-name changes

The process variance names remain `q_level`, `q_trend`, and `q_season`. The
Bayesian-lasso hierarchy adds:

- `s_level`, `s_trend`, `s_season` (signed process standard deviations);
- `tau_level`, `tau_trend`, `tau_season` (local variances);
- `lambda2` (global shrinkage parameter).
