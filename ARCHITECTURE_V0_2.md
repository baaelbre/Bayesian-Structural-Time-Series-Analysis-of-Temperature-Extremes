# bucex v0.2 architecture

## Public layer

- `bucex.fit_bayes`: generic high-level fitting entry point.
- `bucex.fit_uccle_series`: manuscript wrapper for one Uccle index.
- `bucex.fit_uccle_all`: wrapper for all six indices.
- `bucex.plot` and `PosteriorBundle.plot`: analysis plotting API.

## Model layer

A `StructuralModel` composes:

- `LocalLinearTrend(alpha, beta)`;
- `DummySeasonal(g1, ..., g11)` with innovation on `g1` only;
- `GaussianObs` or `GEVObs`.

## Inference layer

The manuscript backend is the non-centred parametrisation:

- Gaussian: exact FFBS state update;
- DGEV: local Laplace pseudo-observations followed by FFBS;
- joint centred-time Gaussian update of baseline parameters and signed process
  scales;
- Park-Casella hierarchical Bayesian-lasso updates;
- exact-likelihood MH updates for DGEV `sigma` and `xi`.

## Result layer

`PosteriorBundle` stores draws, data, dates, model, state names, series name and
minimum/maximum transformation. It provides state extraction, risk calculations,
GEV endpoints, summaries, plots, and pickle persistence.
