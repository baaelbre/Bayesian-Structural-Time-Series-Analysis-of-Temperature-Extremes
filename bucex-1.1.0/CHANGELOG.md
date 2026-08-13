# Changelog

## 1.1.0

- Restored the 0.3 Fruehwirth--Schnatter augmented non-centred sampler as the
  first-class `fit_fs` / `fit_bayes` interface while retaining the 1.0
  declarative `fit` engine for regression and broader model composition.
- Preserved the manuscript Bayesian lasso, componentwise regularized lasso,
  direct normal priors, exact zero/fixed/dynamic structural SSVS, random sign
  switching, transactional restoration diagnostics, and legacy result methods.
- Added a genuine regularized horseshoe hierarchy and a calibrated PC
  innovation prior to the FS engine.
- Added exact-invariant FS PGAS with ancestor sampling, guided-proposal
  corrections, an exact GEV FS elliptical-slice coefficient block, and ASIS.
- Retained the explicitly approximate, damped iterated-Laplace GEV path with
  convergence, support, and restoration diagnostics.
- Fixed particle-history underflow in both engines by retaining normalized log
  weights separately from floating-point probabilities.
- Unified independent multi-chain fitting, rank-normalized diagnostics,
  posterior forecasting, prior-versus-posterior process-SD plots, and safe
  checksummed `.bucex` archives without pickle execution.
- Bundled the six monthly Uccle series, retained daily-to-monthly validation,
  refreshed local/HPC workflows, and added a reproducible release-validation
  record. The data citation and redistribution license remain an explicit
  publication gate.
- Consolidated the package around the readable 0.3 module categories; the 1.0
  compiled implementation is retained internally and re-exported through one
  versioned public API.

## 0.3.3

- Kept the v0.3.2 Laplace and FFBS updates unchanged.
- Added stage-specific diagnostics for every failed DGEV state-update attempt.
- Restored progress lines now report retry counts, failure reasons, the last
  exception detail, and the number of restored iterations in the latest window.
- Saved aggregate restoration diagnostics in `PosteriorBundle.meta` and the HPC
  JSON metadata files.
- Reworked the Torque PBS array into three independent, shorter chains for each
  of the six Uccle series: 18 one-core jobs in total.
- Added a targeted single-series PBS script for diagnostics and reruns.
- Added `examples/pool_uccle_chains.py` to pool completed post-burn chains and
  write classical split-Rhat and restoration summaries.
- Changed the HPC default to 8,000 iterations, 1,500 burn-in and the
  `regularized` prior profile; all values remain overridable through environment
  variables.

## 0.3.2

- Replaced warning-based Gaussian covariance sampling with deterministic,
  scale-aware positive-semidefinite repair.
- Added Joseph-form covariance updates to the non-centred Gaussian FFBS and
  time-varying-variance FFBS routines.
- Added `ComponentwiseBayesianLassoPrior` with separate local/global shrinkage
  for level, trend and seasonality.
- Added the built-in `regularized` profile with monthly scale-aware defaults.
- Recalibrated the `normal` and `ssvs` monthly slope priors and SSVS innovation
  slab scales.
- Preserved the original `manuscript` profile unchanged.
- Added finite-level-change rate methods: `level_rate_draws`,
  `period_rate_summary`, `rate_contrast_draws`, and `rate_contrast_summary`.
- Added SSVS transition diagnostics through `component_transition_summary`.
- Updated the poster script to use finite-period level rates rather than latent
  slope averages.
- Added local and HPC outputs for period rates and acceleration summaries.
- Added five v0.3.2 tests; the complete suite now contains 13 tests.

## 0.3.1

- Added ordinary Normal prior profiles for signed innovation scales.
- Added structural SSVS with exact zero/fixed/dynamic component states.
- The level is always present; no zero-level option is exposed.
- Added full enumeration of the 18 default structural models.
- Added grouped selection for the complete dummy-seasonal baseline.
- Added exact Gaussian structural probabilities conditional on the latent
  non-centred states and observation variance.
- Added Laplace-pseudo-observation structural selection for DGEV models, with
  explicit approximation metadata.
- Added `component_probabilities()`, `structural_model_probabilities()`, and
  `most_probable_structure()` to `PosteriorBundle`.
- Added `plot(type="component_probabilities")`.
- Added built-in prior profiles `"normal"` and `"ssvs"` while preserving the
  `"manuscript"` Bayesian-lasso profile.
- Updated Uccle scripts, poster generation, and Torque PBS array resources.

## 0.2.0

- Added hierarchical Bayesian-lasso priors and Gibbs updates.
- Added inverse-gamma scale prior and bounded-uniform GEV shape prior.
- Corrected the long-series intercept/slope update using centred time and the
  induced correlated prior.
- Added complete posterior metadata and analysis methods.
- Added Uccle data loaders and one/all-series fitting wrappers.
- Added high-level level, slope, exceedance, return-period and endpoint plots.
- Added manuscript analysis example and smoke tests.
