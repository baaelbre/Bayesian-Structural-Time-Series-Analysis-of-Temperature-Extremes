# bucex v0.2.0

`bucex` provides Bayesian structural state-space models for ordinary and extreme
series. Version 0.2 is the first manuscript-oriented release: it contains the
non-centred local-level/local-slope/dummy-seasonal model, the hierarchical
Bayesian lasso used in the paper, the paper's observation priors, Uccle wrappers,
and fit objects that know how to plot and calculate risk.

## Main additions in v0.2

- hierarchical Bayesian lasso on the signed innovation standard deviations
  `s_level`, `s_trend`, and `s_season`;
- local scales `tau_level`, `tau_trend`, `tau_season` and global scale `lambda2`
  are sampled and stored;
- manuscript priors:
  - Gaussian: `sigma2 ~ IG(2, 1)`;
  - DGEV: `sigma2 ~ IG(2, 2)`, `xi ~ Uniform(-0.5, 0.5)`;
- centred-time Fruehwirth-Schnatter regression update with the correct correlated
  prior for the centred intercept and slope;
- automatic sign transformation for TXn and TNn;
- a complete `PosteriorBundle` with data, dates, model, state names, transformations,
  posterior risk methods, save/load, and `fit.plot(type=...)`;
- one-call wrappers for TXm, TNm, TXx, TXn, TNx, and TNn.

## Install locally

From the repository root:

```bash
python -m pip install -e .
```

Or install the built wheel:

```bash
python -m pip install dist/bucex-0.2.0-py3-none-any.whl
```

## Fit one series

```python
from bucex import fit_uccle_series

fit = fit_uccle_series(
    "TXx",
    data_dir="data",
    n_iter=20_000,
    burn=5_000,
    thin=1,
    seed=40,
)

print(fit.static_summary())
```

The wrapper selects the Gaussian model for TXm/TNm and the DGEV model for
TXx/TXn/TNx/TNn. Minima are negated internally and transformed back in all
high-level output.

## Fit all six Uccle series

```python
from bucex import fit_uccle_all

fits = fit_uccle_all(
    data_dir="data",
    n_iter=20_000,
    burn=5_000,
    thin=1,
    seed=40,
)

fits.save("results/bucex_v0.2")
print(fits.summary())
```

## High-level plotting

```python
# One fit
fit.plot(type="level")
fit.plot(type="slope")
fit.plot(type="level_slope")

# All series in a manuscript-style panel
fits.plot(type="level_slope", credible_interval=0.90)

# Tail-risk plots
fits["TXx"].plot(
    type="return_period",
    threshold=[36.8, 39.7],
    annual=True,
    max_return_period=10_000,
)

fits["TNx"].plot(
    type="exceedance",
    threshold=20.0,
    annual=True,
)

fits["TXx"].plot(type="endpoint", threshold=39.7)
```

The slope plot uses degrees Celsius per decade by default (`beta * 120`) for a
monthly model. Pass `slope_scale="raw"` for the model-scale monthly slope.

## Generic high-level API

```python
from bucex import fit_bayes

fit = fit_bayes(
    y,
    family="gev",
    period=12,
    dates=dates,
    name="my_extreme_series",
    tail="max",                 # use "min" for block minima
    priors="manuscript",
    n_iter=10_000,
    burn=2_500,
)
```

A custom model and custom prior object may still be supplied explicitly.

## Important inference note

The DGEV latent-state and regression updates use the local Laplace
pseudo-observation approximation described in the manuscript. The observation
parameter Metropolis-Hastings steps use the exact GEV likelihood. This is the
same approximation structure as the manuscript code, not an exact particle-MCMC
fit.
