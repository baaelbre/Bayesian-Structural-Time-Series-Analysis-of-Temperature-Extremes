# Changelog

## 2.3.0

- Added `MultiSeriesModel`/`PanelModel`, a factor-free grammar for related
  named channels using the same compiler, `fit`, `FitResult`, forecasting,
  plotting, and safe persistence APIs as univariate and factor models.
- Added genuine joint hierarchical structural SSVS. Series allocations share
  conjugately learned Dirichlet probabilities and dynamically learned half-t
  slab multipliers; Gaussian panels use exact FFBS and mixed/GEV panels use
  exact-invariant PGAS with exact-likelihood-corrected model moves.
- Added `HierarchicalSSVSPrior` and `HierarchicalSSVSPriors` for explicit
  prevalence, component-scale, slab, and channel nuisance-prior control.
- Added panel-aware component probabilities, model-switch summaries,
  population probability/slab summaries, complete-predictor channel rates,
  forecasts, diagnostics, process-SD plots, traces, ACFs, direct plot saving,
  and schema-2.3 archive round-trips.
- Audited random FS sign-symmetry moves in univariate, hierarchical, and factor
  samplers. Each accepted move verifies predictor/centered-path invariance and
  stores switch counts plus numerical errors.
- Added Uccle multiseries constructors/fit helpers and standalone Examples
  14--16 for hierarchical simulation, the six-summary PGAS analysis, and the
  hierarchy-versus-factor decision.
- Preserved the Joseph-form FS covariance route and the complete 2.2 test and
  predictive-validation contracts.

## 2.2.0

- Added exact-invariant PGAS inference for univariate GEV SSVS. Laplace
  pseudo-information is used only for full-support independence proposals;
  model moves are corrected with the exact GEV likelihood, normalized model
  and slab priors, and forward/reverse proposal densities, then refreshed by
  elliptical-slice sampling. The existing Laplace SSVS route remains an
  explicitly approximate screening option.
- Replaced cancellation-prone FS backward covariance subtraction with an
  equivalent Joseph-form conditional covariance and direct symmetric solves.
  Singular structural transitions and innovation scales at or near zero stay
  on their declared affine support; no artificial process jitter is added.
- Corrected the FS static dummy-seasonal design. The coefficient update and
  state reconstruction now use the same rotated initial-season state vector.
  Earlier FS seasonal fits could estimate a valid regression under one phase
  ordering and reconstruct it under another; those analyses should be rerun.
- Added expanding-window `leave_future_out()` for univariate and factor
  models. Forecasts now expose analytic posterior-mixture log scores,
  per-time/aggregate CRPS, threshold-weighted CRPS, exceedance and quantile
  scores, held-out PIT values, PIT summaries, and saveable PIT histograms.
- Added exact fixed-component contract tests: under FS SSVS, a fixed level is
  `alpha0` with zero level innovation, a fixed trend is `beta0` with zero slope
  innovation, and fixed seasonality is a static coefficient vector with zero
  seasonal innovation.
- Added Examples 11--13 for deterministic component semantics, exact GEV
  PGAS--SSVS, and end-to-end leave-future-out validation. Example 9 now uses
  exact PGAS for GEV SSVS by default, and Example 1 no longer hard-codes a PC
  label when another prior is selected.

## 2.1.5

- Added the Cadonna--Frühwirth-Schnatter--Knaus triple-gamma prior for
  signed structural innovation scales. The paper's normal--gamma--gamma
  hierarchy, optional global beta-prime update, optional shape learning, and
  interpretable shrinkage factors are retained in posterior output.
- Added `regularized_triple_gamma`, which applies an optional inverse-gamma
  slab cap to the triple-gamma local variance. Both triple-gamma profiles are
  supported by univariate Gaussian/GEV FS models and by centered,
  disturbance, and FS factor models.
- Replaced random-walk updates for the regularized-horseshoe hierarchy with
  exact stepping-out slice updates. This addresses the avoidable
  local/global/slab mixing bottleneck exposed by `04_compare_priors.py`.
- Added chain-specific ACF plots through `fit.plot("acf")`. Every fit,
  forecast, collection, and bulk/tail plot now accepts `save=`, including a
  path or a `{"path": ..., "dpi": ...}` mapping.
- Made prior/posterior process-SD plots analytic where possible: PC,
  folded-normal, SSVS slab, ordinary process priors, and fixed-global
  unregularized triple gamma. Integrated hierarchical priors use a smooth,
  explicitly labelled Monte Carlo KDE instead of a jagged histogram.
- Corrected constant-chain diagnostics. R-hat and ESS are now `NaN` for a
  constant SSVS allocation and carry an explicit diagnostic/status message;
  the former mechanical `R-hat=1, ESS=all draws` is no longer presented as
  evidence of mixing.
- Reworked all ten examples as standalone, sequential scripts. Example 4
  compares normal, PC, regularized horseshoe, triple gamma, regularized triple
  gamma, and SSVS; Example 9 reports six separate Uccle SSVS analyses; Example
  10 uses regularized triple gamma in the full six-summary factor model.

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
