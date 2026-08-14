# bucex 2.1.4 release notes

`bucex` 2.1.4 is a modular Bayesian structural time-series package for both
single-series models and shared-factor models with Gaussian and/or GEV
observation channels.

## What changed in 2.1.4

- Gaussian, GEV, centered/disturbance, FS, and factor samplers now use one
  progress vocabulary. Lines show chain, `it`, phase, saved draws, current
  scientific parameters, elapsed time, and ETA. PGAS adds particle ESS,
  ancestor diversity, and path change. `MCMC.progress_every` controls cadence.
- Factor-prior profile names now tolerate whitespace, hyphens, UK spelling,
  and documented aliases such as `horseshoe`, `pc`, and `normal`. Invalid
  values report the received name and distinguish factor from univariate-only
  profiles.
- `identified_factor_priors()` declares smooth-factor, pure-reference, or
  fixed-idiosyncratic sensitivity constraints without manual prior surgery.
- `factor_identification_diagnostics()`, loading/deviation correlations, and
  idiosyncratic-innovation draws expose the remaining dynamic identification
  ridge. The release no longer implies that a sampler move alone identifies
  `lambda[i] * f[t]` and `alpha[i,t]` separately.
- The factor plotting API now covers decompositions with credible bands,
  posterior densities with truth markers, chain-specific innovation-SD
  traces, loading/deviation joint plots and correlations, and
  idiosyncratic-innovation paths.
- Numbered, configurable play scripts compare Gaussian versus GEV fits,
  parameterizations, priors, Gaussian and mixed factors, independent
  bulk/tail fits, and the six-channel Uccle model.

## Which API to use

- Build a univariate model from `Model`, one observation family, and reusable
  components such as `LocalLevel`, `LocalLinearTrend`, and seasonality.
- Build a multivariate shared-trend model from `FactorModel`, named `Channel`
  objects, one or more `Factor` objects, and explicit `Loading` anchors.
- Call the same top-level `bucex.fit(...)` entry point for either graph. The
  inference planner selects an exact Gaussian route when the graph permits it
  and PGAS for mixed Gaussian/GEV graphs; an explicit Laplace route remains
  available as an approximation.

The public model, fit, diagnostic, forecast, plotting, and archive layers are
kept separate from inference internals. New components and observation
families can therefore be added without turning the user-facing API into a
sampler-specific interface.

## Identification and factor-loading behavior

- `LocalLinearTrend(initial_slope_sd=0.0)` now fixes a factor's initial slope
  exactly instead of attempting a degenerate MCMC update.
- Eligible Gaussian channels use a collapsed Kalman likelihood update for the
  idiosyncratic innovation scale and an exact joint FFBS draw for intercept,
  loading, and deviation path.
- GEV channels use a predictor-preserving loading/deviation interweaving move,
  leaving the nonlinear likelihood and support unchanged.
- `FitResult.normalized_factor(...)` provides an explicit baseline convention.
- `FitResult.channel_decomposition(...)` separates baseline, shared factor,
  idiosyncratic deviation, seasonality, and reconstructed predictor draws.
- `FitResult.factor_identification_diagnostics(...)` reports posterior
  loading correlations with factor-like and endpoint summaries of each
  persistent deviation. A ridge flag is an interpretation warning, not a
  convergence diagnosis.

Resolved loading kernels and fixed factor slopes are recorded in sampler
diagnostics, making inference choices inspectable after a fit.

## Compatibility

The established univariate workflows and `FitResult` contract remain intact.
Safe archive loading remains compatible with schemas 1.2, 2.0, and 2.1.
Python 3.10 or newer is required.

## Included checks

The release contains 63 source tests plus fixed-seed univariate and mixed
factor validators. The mixed validator covers two Gaussian and four GEV
channels, exact FS/centered predictor agreement, loading-kernel routing,
archive round trips, and exact channel-decomposition reconstruction.

These short deterministic runs are release checks, not evidence of scientific
convergence. Applied analyses should still use multiple long chains,
sufficient PGAS particles, simulation recovery, and posterior predictive
checks.
