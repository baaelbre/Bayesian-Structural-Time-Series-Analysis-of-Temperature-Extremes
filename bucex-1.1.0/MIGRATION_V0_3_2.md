# Migration from v0.3.1 to v0.3.2

## Existing analyses

No changes are required for the manuscript profile:

```python
fit_uccle_series("TXx", priors="manuscript")
```

The original shared Bayesian-lasso prior and its saved `lambda2` draw remain
unchanged.

## New recommended sensitivity profile

```python
fit_uccle_series("TXm", priors="regularized")
```

This profile saves three shrinkage parameters:

```text
lambda2_level
lambda2_trend
lambda2_season
```

instead of one shared `lambda2`.

## Normal and SSVS profiles

The built-in `normal` and `ssvs` profiles now use a monthly initial-slope prior
standard deviation of `0.005`, and smaller innovation scales. Code using custom
prior objects is unchanged.

## Poster summaries

The v0.3.2 poster script interprets acceleration through finite changes in the
posterior level. Output names changed from `poster_period_slopes.*` to
`poster_period_rates.*`.
