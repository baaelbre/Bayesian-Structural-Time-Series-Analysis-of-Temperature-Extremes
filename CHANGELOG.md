# Changelog

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
