# Changelog

## 2.1.4

- Unified MCMC progress output across Gaussian, GEV, generic state-space, FS,
  and factor samplers. Progress now consistently reports chain, `it`, phase,
  saved draws, current model parameters, elapsed time, and ETA, with PGAS and
  Laplace diagnostics where relevant. Added `MCMC.progress_every`.
- Normalized factor-prior aliases and improved invalid-profile errors. Added
  `identified_factor_priors()` for smooth-factor, pure-reference, and
  fixed-idiosyncratic sensitivity specifications.
- Added posterior loading/deviation diagnostics and idiosyncratic-innovation
  extraction to `FitResult`. Inference plans now warn when estimated loadings
  coexist with persistent channel deviations.
- Added factor decomposition bands, truth-marked parameter densities,
  chain-specific innovation-SD traces, loading/deviation joint and correlation
  plots, and idiosyncratic-innovation plots through `fit.plot(...)`.
- Replaced the large monolithic demonstrations with configurable play scripts
  for Gaussian, GEV, parameterization, prior, factor, mixed, combined
  bulk/tail, and Uccle analyses. Every fit example enables progress by default.
- Added regression and end-to-end plotting tests for the v2.1.4 APIs.

## 2.1.2

- Made `LocalLinearTrend(initial_slope_sd=0.0)` a true fixed initial factor
  slope in factor FS inference. The coefficient remains stored as a constant
  draw but is no longer proposed or reported as a failed MCMC update.
- Added an exact all-Gaussian loading block. Conditional on the shared factor
  and seasonal path, each eligible channel jointly draws its intercept,
  loading, and complete idiosyncratic random walk by FFBS. The loading is
  therefore sampled marginally rather than conditional on a compensating
  deviation path.
- Added a collapsed Kalman-likelihood update for the corresponding Gaussian
  idiosyncratic innovation SD before drawing the loading/deviation block back.
  This directly targets the loading--variance ridge.
- Added a predictor-preserving loading/deviation interweaving move for GEV
  channels. The move leaves the likelihood and GEV support unchanged and is
  followed by a path-conditional fallback when the deviation SD is fixed at
  zero.
- Added `FitResult.normalized_factor()` and
  `FitResult.channel_decomposition()` so baseline, shared, idiosyncratic, and
  complete predictor draws are separated without relabeling an intercept as
  dynamic deviation.
- Updated the Uccle one-factor helper and factor examples to use fixed initial
  slopes, smooth/pure-reference identification examples, and the new
  decomposition API.
- Added v2.1.2 regression tests for fixed coefficients, collapsed Gaussian
  coefficient recovery, loading-kernel routing, exact decomposition, and
  GEV predictor preservation.

## 2.1.1

- Added an all-Gaussian one-factor channel tutorial using exact FFBS, with a
  pure reference channel, estimated amplified/damped loadings, horseshoe-
  regularized idiosyncratic levels, full scalar output, recovery summaries,
  and shared-versus-individual decomposition plots.
- Made positive-semidefinite covariance handling robust to scale-aware
  floating-point Schur-complement remnants around `1e-8` in singular FS
  smoothers, while continuing to reject materially indefinite matrices.

## 2.1.0

- Added the exact one-factor Uccle graph
  `mu[i,t] = c[i] + lambda[i] * f[t] + S[i,t] + alpha[i,t]`, with one
  shared local-linear trend, a fixed `TXm=1` loading anchor, estimated remaining
  loadings, six channel local levels, and six series-specific dummy-seasonal
  blocks.
- Added a full factor Frühwirth--Schnatter non-centred compiler. Shared,
  idiosyncratic, and seasonal latent blocks have unit innovation variance;
  signed process scales enter the observation design and semantic centered
  states are reconstructed for the common `FitResult` contract.
- Added factor FS FFBS, mixed Gaussian/GEV PGAS, optional Laplace screening,
  sign switching, and centered-scale ASIS. The six-dimensional PGAS weight is
  the product of two Gaussian and four GEV channel likelihoods.
- Preserved centered and standardized-disturbance parameterizations for the
  factor graph and kept the univariate model, inference, forecast, diagnostic,
  and archive paths available.
- Added a factor regularized-horseshoe hierarchy restricted to independent
  channel local-level innovation scales. Shared factor and seasonal scales
  keep calibrated process priors.
- Made the Uccle factor helper default to the v2.1 model and made eligible
  one-factor graphs resolve `parameterization="auto"` to FS.
- Added factor result conveniences: `factor()`, `factor_rate_summary()`,
  `factor_probabilities()`, `loading_probability()`, and
  `reconstructed_state()`.
- Added schema 2.1 archives while retaining safe read compatibility with 1.2
  and 2.0 archives.
- Expanded the factor-model sandbox with complete posterior/diagnostic and
  simulation-recovery tables. Factor MCMC progress now separates warmup from
  retained sampling and reports PGAS particle ESS, ancestor diversity, elapsed
  time, and ETA.

## 2.0.0

- Added `Channel`, `Factor`, `Loading`, and `FactorModel` for modular shared and
  individual structural state blocks across named observation channels.
- Added explicit factor scale/sign identification: estimated loadings require a
  fixed non-zero anchor, multi-factor fixed anchors must have full column rank,
  and loading normal priors remain in the model graph.
- Added `CompiledFactorModel`, which namespaces factor/channel states and
  disturbances and compiles one channel-by-state design over a global linear
  transition.
- Generalized Kalman filtering and FFBS to multivariate observations with
  channelwise missingness and diagonal conditional observation covariance.
- Generalized iterated Laplace and exact-invariant PGAS to mixed Gaussian/GEV
  channel likelihoods on one global state path.
- Added factor-specific process/observation/shape priors, centered/disturbance
  ASIS updates, estimated-loading updates, and exactness-aware planning.
- Made mixed-factor `engine="auto"` choose PGAS; Laplace remains explicitly
  labeled approximate. The FS augmentation remains univariate and fails early
  for factor graphs.
- Extended the existing `FitResult`, forecast, diagnostics, plotting, risk,
  simulation, combination, and checksummed archive machinery to factor models
  without introducing a second result class.
- Added v2 safe archives while retaining read compatibility with v1.2 archives.
- Preserved all 39 v1.2.1 compatibility tests and added factor identification,
  exact-likelihood, mixed-PGAS, recovery, forecast and archive validation.
- Corrected component deserialization for a scalar `LocalLevel.initial_mean`.

## 1.2.1

- Replaced the parallel general and FS public APIs with one `fit()` framework
  and one chain-preserving `FitResult`.
- Made centered, scaled-disturbance and Frühwirth–Schnatter parameterizations
  explicit strategies selected by `parameterization=`.
- Separated parameterization from exact Gaussian FFBS, approximate iterated
  Laplace and exact-invariant PGAS state engines.
- Integrated ASIS through the inference plan with an explicit interweaving
  partner.
- Integrated manuscript/componentwise lasso, regularized horseshoe, PC, Normal
  and SSVS prior profiles with pre-sampling compatibility validation.
- Unified semantic states, observations, forecasting, diagnostics, plotting,
  Uccle workflows and safe serialization across every strategy.
- Preserved FS signed scales and latent paths as parameter/auxiliary draws
  without exposing a second result type.
- Removed the duplicate `_general` tree, centered fitter copies, root shim
  modules, old result implementation, prototype risk/simulation paths,
  version-stamped documentation, stale demos and generated package metadata.
- Added a compact current documentation set and a new release test/validation
  suite.
- Added the `bucex-uccle` command-line workflow for data validation, fitting,
  archive inspection and combining independently fitted chains.
- Corrected prior auto-resolution, initialization compatibility, ASIS parameter
  labeling, compact Uccle MCMC overrides and default risk-summary return values.

## 1.1.0

- Combined the flexible compiled-model work with the v0.3 FS research code,
  including horseshoe/PC priors, ASIS, PGAS and safe archives. The two paths
  still had separate public fit and result contracts.

## 0.3 series

- Developed the FS augmented structural sampler, hierarchical Bayesian lasso,
  structural SSVS, Uccle workflows and restoration diagnostics.
