# Changelog

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
