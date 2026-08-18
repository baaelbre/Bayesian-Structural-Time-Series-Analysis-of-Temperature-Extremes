# Changelog

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
