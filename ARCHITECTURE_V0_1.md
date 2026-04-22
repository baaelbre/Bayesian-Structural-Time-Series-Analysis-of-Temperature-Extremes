# bucex architecture notes: v0.1

## Positioning

`bucex` is a modular Bayesian state-space framework with pluggable observation
models and interchangeable inference backends. Gaussian structural time series
is treated as one special case among several rather than the whole package.

## Design choices implemented in v0.1

- the latent state evolution remains linear-Gaussian and component based
- observation models sit on top of that layer
- inference is selected separately from model specification
- a public high-level API hides backend-specific class names
- centered vs noncentered is exposed as a fitter option, even where the
  noncentered backend is still a placeholder
- a dedicated `risk/` namespace makes dynamic risk summaries first-class

## Current boundaries

v0.1 is intentionally conservative. Internally, the package still uses a scalar
predictor `eta_t` and stores the implementation of observation models in
`bucex.obs`. To reduce churn, `bucex.observation` is added as a forward-looking
compatibility namespace while the deeper multi-predictor refactor is deferred.

## Immediate next architectural step

The next major refactor should replace scalar `eta_t` design objects with named
predictor maps such as `location`, `log_scale`, and `shape`. That refactor is
more intrusive than the rest of the v0.1 packaging cleanup and is therefore
kept out of this first research release.
