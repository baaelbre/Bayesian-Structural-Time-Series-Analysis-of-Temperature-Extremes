# Changelog

## 2.6.0

### Presentation workflow

- Replaced the broad 2.5 presentation workflow with the requested Uccle TXx
  opening, GEV-tail illustrations, controlled structural simulations,
  Laplace-versus-PGAS comparison, and TXx/TXn/TNx/TNn analyses.
- Added seven fixed-truth SSVS recovery scenarios with common negative shape
  and scale.
- Strengthened the simulated structural innovations, widened the GEV shape
  contrast to `-0.30/0/+0.30`, and added a matched `sigma=0.75/1.50/3.00`
  experiment.
- Replaced simulation grids with one figure per time series and one separate
  three-panel decomposition per structural scenario.
- Standardized result tables, figures, paths, manifests, runtime profiles, and
  CLI behavior across simulated and observed fits.
- Replaced the presentation examples with exactly seven standalone direct-API
  scripts; retained the complete PBS dependency graph for batch execution.

### Inference and diagnostics

- Added true univariate `FitResult` warm starts, including the complete centred
  latent path and signed FS coefficients.
- Added Laplace-to-PGAS warm-start provenance to saved results.
- Added reference-ancestor change diagnostics for standard PGAS with singular
  or deterministic transition directions.

### Release

- Restored the complete top-level package API.
- Advanced safe archives to schema 2.6.0 with backward readers.
- Added v2.6 workflow tests and validation scripts.

## 2.5.0

### Scientific workflow

- Made componentwise fixed/dynamic level, zero/fixed/dynamic slope, and
  fixed/dynamic seasonality the primary univariate and hierarchical SSVS story.
- Changed hierarchical defaults to pooled selection; retained joint trend
  classes and shared slab magnitudes as explicit sensitivity models.
- Added the staged TXx-to-six-series presentation workflow with runtime
  profiles, deterministic paths, manifests, tidy exports, and report-only mode.
- Added controlled TXx structural analogues and leave-future-out model
  comparison using proper scores and PIT diagnostics.

### Execution and release

- Added `bucex-presentation` and the reusable `bucex.workflows` API.
- Added full PBS/Torque arrays for benchmark, validation, univariate,
  hierarchical, sensitivity, combination, and reporting stages.
- Fixed chain combination for deserialized dataclass priors containing NumPy
  arrays, which previously made some SSVS HPC archives incomparable in Python.
- Advanced checksummed archives to schema 2.5.0 with backward readers.
- Added end-to-end workflow validation and package/release build checks.

## 2.4.1

### Hierarchical model space

- Added the default four-class joint level/slope innovation space: linear
  trend, RW1 with drift, RW2 smooth trend, and full local linear trend.
- Kept initial slope estimated in every class and retained componentwise
  no-slope SSVS only as an explicit legacy sensitivity.
- Added posterior summaries and plotting for shared trend-class probabilities.
- Added expert-scale calibration and implication helpers for normal/half-t
  hierarchical slabs.

### Mixed/GEV inference

- Added exploratory hierarchical Laplace SSVS for mixed and all-GEV models.
- Added validated `FitResult.warm_start()` and `init=<laplace fit>` support for
  exact PGAS.
- Added `HierarchicalSampler` controls for Laplace initialization and optional
  concurrent channel updates.
- Vectorized GEV particle weights, complete-path likelihoods, and Laplace
  pseudo-data calculations.
- Grouped channel-specific GEV shape values in progress output.
- Renamed changed fraction to the scientifically explicit path-update fraction,
  retaining a deprecated compatibility alias.

### Workflows and persistence

- Added a Laplace-to-PGAS example, four-process HPC chain example, Slurm array,
  and chain-combination example.
- Updated the checksummed archive schema to 2.4.1 while retaining readers for
  prior supported schemas.

## 2.4.0

### Uccle workflow hotfix

- Aligned `ssvs_gaussian_priors()` and `ssvs_gev_priors()` with Example 09 by
  accepting direct level, trend, season, and slab SSVS settings as well as an
  explicit `SSVSPrior` object.
- Made one-series Uccle selections robust to a bare string, so `"TXm"` cannot
  be accidentally interpreted as the three names `"T"`, `"X"`, and `"m"`.
- Made a requested single-series CSV authoritative without requiring the
  explicit directory to contain all six Uccle summaries.
- Restored the zero-restoration metadata contract for successful
  centered/disturbance PGAS fits.

### Model surface

- Unified univariate `Model` and hierarchical `MultiSeriesModel` under
  `bx.fit()` and `FitResult`.
- Removed obsolete multichannel implementations, compatibility aliases,
  specialised samplers, result methods, plots, examples, tests, and documents.
- Added `HierarchicalPrior(pool="selection"|"slab"|"both")` with normal
  dynamic slabs, Dirichlet population allocation probabilities, and optional
  half-t pooled slab multipliers.
- Made monthly seasonality fixed/dynamic by default; absence is available only
  when explicitly requested.

### Inference

- Added joint Gaussian FFBS and mixed/GEV PGAS hierarchical inference.
- Added audited random sign switches for signed FS innovation coefficients.
- Estimated and stored initial level, slope, and seasonal coefficients for
  univariate and hierarchical models.
- Hardened guided disturbance PGAS on singular affine support by using the
  scaled transition loading, projecting the reference trajectory, and safely
  rejecting invalid optional ancestor moves.
- Applied the same conditioned-predecessor fallback to FS PGAS.
- Retained Joseph-form Gaussian covariance updates for long or nearly
  deterministic series.

### Diagnostics and plots

- Added population allocation, pooled slab, structural transition, and scoped
  channel summaries.
- Made the level--slope plot compare latent level with seasonally adjusted
  observations.
- Standardised progress output across engines with chain, `it`, phase, saved
  draws, scientific parameters, elapsed time, and ETA.
- Retained trace, ACF, analytic-prior, forecast, score, PIT, and direct-save
  plotting APIs.

### Workflows

- Rewrote examples as sequential, editable user scripts.
- Rebuilt the Uccle workflow around independent baselines and hierarchical
  pooled-selection/slab analyses.
- Added release validation for the PGAS regression, initial states, hierarchy,
  predictive checks, plotting, persistence, and package builds.
