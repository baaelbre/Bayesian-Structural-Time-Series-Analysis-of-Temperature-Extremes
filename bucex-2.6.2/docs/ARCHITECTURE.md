# Architecture

Version 2.6.2 keeps one model compiler, one fitting entry point, and one result
type. Analysis scripts compose these public APIs directly.

```text
bucex/
  api/                 fit, forecast, prediction
  components/          trend, seasonal, regression components
  models/              Model, MultiSeriesModel, Channel, compilers
  observation/         Gaussian and GEV families
  priors/              univariate and hierarchical priors
  inference/           plans, FFBS, Laplace, PGAS, samplers
  core/                FitResult and numerical primitives
  diagnostics/         MCMC, scores, PIT, LFO
  datasets/            Uccle loaders and direct fit helpers
  io/                  checksummed non-pickle archives
  plotting/            FitResult-driven figures
```

## Fit lifecycle

1. `fit()` receives a declarative model and data.
2. The compiler creates semantic states, transitions, and innovation loadings.
3. `inference_plan()` rejects incompatible engine/parameterization choices and
   declares exactness.
4. Prior resolution constructs univariate or hierarchical graphs.
5. The selected engine produces states, parameters, and diagnostics.
6. Results are normalized immediately into `FitResult`.

FS auxiliary paths and signed coefficients never replace the centered
scientific state stored in `FitResult`.

## Hierarchical sampler

`MultiSeriesModel` compiles named channel blocks. One Gibbs sweep updates:

1. each channel path by Gaussian FFBS, approximate Laplace, or PGAS;
2. channel component allocations;
3. observation parameters;
4. population Dirichlet probabilities;
5. optional shared half-t slab multipliers.

Channels may update concurrently because they are conditionally independent
given hierarchy values. They still retain separate latent paths. The default
componentwise model pools level, slope, and seasonal decisions separately.

## Numerical stability and singular support

Gaussian filtering uses symmetric/Joseph-form covariance operations and
repairs only small round-off eigenvalues. Materially indefinite matrices raise
instead of being silently projected.

Exact SSVS creates deterministic transition directions. PGAS evaluates
ancestor moves on the affine support of active innovations. The conditioned
predecessor is retained when optional candidates are off-support. This is an
algorithmic requirement, not a numerical convenience.

## Analysis-script lifecycle

The seven files under `examples/` declare models, priors, simulation truths,
fit calls, summaries, and figures in one readable sequence. The seven PBS jobs
invoke those exact files and share a `BUCEX_RUN_ID`; scheduler code does not
define the statistics. Reusable scientific behavior belongs in the core API,
while analysis-specific choices remain visible in the scripts.

## Persistence

Schema 2.6.2 archives contain allowlisted JSON metadata plus compressed NumPy
arrays, verify SHA-256 checksums, and load with `allow_pickle=False`. The model
is recompiled from its stored declaration and observations. `warm_start()`
exports one compatible univariate or multiseries draw without changing the
next fit's target. Univariate FS exports include the complete centred path.

## Extension rule

A new component compiles into the common state contract; a new engine consumes
that contract and returns standard result blocks. Cross-channel dependence must
be an explicit observation/transition feature. It must not be hidden inside a
workflow or a prior-pooling label.
