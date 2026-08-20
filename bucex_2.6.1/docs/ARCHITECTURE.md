# Architecture

Version 2.6.1 keeps one model compiler, one fitting entry point, and one result
type. The new workflow layer orchestrates these public APIs; it is not a second
inference implementation.

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
  workflows/           staged config, execution, exports, CLI
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

## Workflow lifecycle

`PresentationConfig` resolves a runtime profile, Uccle window, simulation
length and period, and figure contract. `WorkflowPaths` maps each scenario, series,
engine, and HPC chain to a deterministic filename. `PresentationWorkflow`
calls ordinary fit/data/diagnostic APIs, saves archives, exports tidy tables,
and updates a locked manifest. `report()` consumes only saved combined fits.

This separation ensures that local scripts and PBS tasks use identical model
definitions. Scheduler code chooses execution units; it does not define the
statistics.

## Persistence

Schema 2.6.1 archives contain allowlisted JSON metadata plus compressed NumPy
arrays, verify SHA-256 checksums, and load with `allow_pickle=False`. The model
is recompiled from its stored declaration and observations. `warm_start()`
exports one compatible univariate or multiseries draw without changing the
next fit's target. Univariate FS exports include the complete centred path.

## Extension rule

A new component compiles into the common state contract; a new engine consumes
that contract and returns standard result blocks. Cross-channel dependence must
be an explicit observation/transition feature. It must not be hidden inside a
workflow or a prior-pooling label.
