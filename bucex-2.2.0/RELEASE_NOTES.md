# bucex 2.2.0 release notes

`bucex` 2.2.0 is a modular Bayesian structural time-series package for both
single-series models and shared-factor models with Gaussian and/or GEV
observation channels.

## What changed in 2.2.0

- GEV models now support structural SSVS with the exact-invariant PGAS
  backend. Laplace approximations construct efficient full-support proposals,
  while a trans-dimensional Metropolis--Hastings correction evaluates the
  exact GEV likelihood and normalized priors. Active coefficients receive an
  exact elliptical-slice refresh. The result records model-move acceptance,
  proposed changes, slice steps, component probabilities, and switching
  diagnostics.
- The FS backward sampler uses the Joseph-form conditional covariance
  `(I - JG) C (I - JG)' + J Q J'`. This avoids the cancellation in
  `C - J R J'` that could create a small negative eigenvalue for long series,
  singular transition systems, or nearly deterministic innovations. Exact
  deterministic directions remain deterministic; no artificial process
  variance is added.
- The static dummy-seasonal design in the FS parameterization is corrected.
  Static coefficients are now propagated by the same sum-to-zero seasonal
  rotation used by state reconstruction. Regression and reconstructed
  predictors agree to machine precision. **All 2.1.x FS analyses containing
  `DummySeasonal` should be rerun.**
- Fixed and zero structural components now have explicit, tested semantics.
  Fixed level, trend, and seasonality are represented by their separate static
  FS coefficients with exactly zero innovation scales; zero removes the
  corresponding static trend or seasonal coefficient as well. Dynamic
  components retain the static baseline and add a signed innovation scale.
- `Forecast` now provides analytic conditional log densities, log predictive
  scores, held-out PIT values, and PIT summaries. `Forecast.score()` combines
  log score, CRPS, tail-weighted CRPS, quantile scores, and threshold-event
  scores for univariate and factor forecasts.
- `bucex.leave_future_out(...)` performs genuine expanding-window refits and
  returns predictions, per-origin/per-horizon scores, held-out PIT values,
  fitted origins, and optional retained fits. This is deliberately distinct
  from the in-sample posterior PIT convenience method.
- Examples 11--13 demonstrate deterministic versus dynamic components, exact
  GEV PGAS--SSVS, and leave-future-out validation as sequential standalone
  scripts. The existing Gaussian, GEV, prior, parameterization, Uccle, and
  factor examples use the same fit, diagnostics, plotting, and save APIs.

The fixed-seed v2.2.0 source pass contains 80 successful regression tests.
The numerical validator additionally exercises a 1,000-observation singular
Joseph-form FFBS path, exact PGAS--SSVS metadata and moves, corrected seasonal
algebra, log scores, and PITs. These are software checks; manuscript inference
still requires long independent chains, a particle-sensitivity study, and
held-out predictive validation.

## Changes retained from 2.1.5

- `triple_gamma` implements the normal--gamma--gamma variance-selection prior
  of Cadonna, Frühwirth-Schnatter, and Knaus (2020), including optional global
  beta-prime learning, optional beta shape learning, and stored shrinkage
  factors `rho`. It works for univariate Gaussian/GEV FS fits and for the full
  centered, disturbance, or FS factor model.
- `regularized_triple_gamma` adds an inverse-gamma slab cap while retaining the
  triple-gamma spike and tail shapes. `triple_gamma_options={...}` exposes the
  shape/global/slab choices directly in `identified_factor_priors()`.
- The regularized horseshoe now uses exact stepping-out slice updates for its
  local, global, and slab hierarchy. This replaces the slow random-walk block
  that caused low ESS in the old prior-comparison example.
- `fit.plot("acf")` plots chain-specific ACFs. Every plot dispatcher accepts
  `save=`, either as a path or a mapping such as
  `{"path": "figure.png", "dpi": 300}`.
- Prior/posterior process-SD plots use analytic PC, folded-normal, SSVS-slab,
  ordinary process, and fixed-global triple-gamma densities when available.
  Integrated hierarchies use a smooth labelled KDE; heavy tails are displayed
  on a bounded scientifically useful range without altering the density.
- Constant SSVS allocations now report undefined (`NaN`) R-hat and ESS plus a
  clear status. The old mechanical `R-hat=1, ESS=all draws` was not a valid
  convergence conclusion.
- Examples 1--10 are standalone sequential scripts. Example 4 performs the
  full prior comparison, Example 9 reports separate SSVS analyses for all six
  Uccle summaries, and Example 10 runs regularized triple gamma in the joint
  six-summary factor model.

The progress, identification, and decomposition improvements introduced in
2.1.4 remain part of the release:

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

The release contains 80 source-test cases plus fixed-seed univariate and mixed
factor validators. The mixed validator covers two Gaussian and four GEV
channels, exact FS/centered predictor agreement, loading-kernel routing,
archive round trips, and exact channel-decomposition reconstruction. New tests
cover both triple-gamma variants, all factor parameterizations, analytic prior
plots, ACF/save behavior, constant SSVS diagnostics, exact GEV PGAS--SSVS,
Joseph-form covariance handling, static seasonal consistency, and predictive
validation.

These short deterministic runs are release checks, not evidence of scientific
convergence. Applied analyses should still use multiple long chains,
sufficient PGAS particles, simulation recovery, and posterior predictive
checks.
