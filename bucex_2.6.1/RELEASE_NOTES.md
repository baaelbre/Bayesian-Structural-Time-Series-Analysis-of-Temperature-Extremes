# bucex 2.6.1 release notes

Version 2.6.1 makes the presentation simulations explicit, tunable examples of
the public `bucex` API.

## Tunable period-4 simulations

- `GEVScenario` now carries `period` and builds `DummySeasonal(period)` rather
  than fixing a twelve-block cycle.
- Added public `make_tail_scenarios()`, `make_scale_scenarios()`, and
  `make_structural_scenarios()` constructors. Their arguments expose series
  length, period, `sigma`, `xi`, initial states, process standard deviations,
  seasonal amplitudes, and seeds.
- The default catalogue is exactly the requested six designs: stationary,
  linear trend, random walk, local linear trend, stationary level with changing
  seasonality, and local linear trend with fixed seasonality.
- Structural series now contain 800 period-4 blocks. Smaller process noises and
  seasonal amplitudes keep the signals plausible while the longer record makes
  the component evidence clearer.

## Examples, priors, and HPC

- Scripts 01--04 use the public scenario factories and
  `bx.simulate_scenario()`; all scientific controls are editable at the top.
- Scripts 03--06 expose the inverse-gamma prior for `sigma^2`, bounds for `xi`,
  initial-state scales, innovation slabs, and component model probabilities.
- SSVS prior builders accept observation-prior and initial-state
  hyperparameters.
- Existing simulations and fits are checked against current controls, avoiding
  silent reuse after period, prior, or MCMC changes.
- The CLI and PBS arrays now use the six scenarios, accept a simulation period,
  and default to 800 publication blocks.
- Fit archives use schema 2.6.1 while retaining all earlier supported readers.

## 2.6.0 foundation

Version 2.6.0 reorganizes the package around the COMPSTAT scientific story and
removes the unrelated legacy presentation scripts.

## Focused experimental design

- Added three 30-year local-level GEV tail illustrations with common latent
  evolution and `xi=-0.30`, `0`, and `+0.30`.
- Added a matched observation-scale experiment with `sigma=0.75`, `1.50`, and
  `3.00` and fixed `xi=-0.30`.
- Added seven structural-selection scenarios with common `sigma=1.5` and
  `xi=-0.30`, spanning absent, fixed, and dynamic level, slope, and seasonal
  components with more visible process innovations.
- Simulation time series are separate figures; only each scenario's true
  level/slope/seasonal decomposition uses three panels.
- Added one results contract for simulations and TXx, TXn, TNx, and TNn:
  selection probabilities, structural-model switching, posterior trajectories,
  prior-to-posterior process-scale plots, GEV parameters, and diagnostics.

## Laplace-to-PGAS handoff

- Extended univariate `FitResult.warm_start()` and `fit(init=<FitResult>)`.
- The selected Laplace posterior draw now transfers observation parameters,
  signed FS coefficients, initial components, and the entire centred state
  path to PGAS.
- Warm-start provenance is retained in fit metadata and exported summaries.
- Added the reference-ancestor change rate to make conditioned-lineage
  stickiness visible in models with fixed or excluded transition directions.

PGAS continues to evaluate singular transitions on their affine support. The
implementation is standard one-step ancestor sampling, not a bridge sampler;
the new diagnostic makes this limitation explicit rather than hiding it.

## Uniform workflow API

The complete workflow is now expressed by six stages:

```text
data
tail-simulations
structural-simulations
simulation-fit
uccle-fit
report
```

The seven numbered Python examples now call the public simulation, prior,
fitting, persistence, and plotting API directly. Each is self-contained and
keeps its editable constants in the file; the PGAS examples create a missing
Laplace initializer themselves. `PresentationWorkflow` and
`bucex-presentation` remain available for PBS orchestration. The PBS workflow
uses scenario/series-by-chain arrays and dependency-gated combine jobs, so PGAS
cannot start before the corresponding Laplace fit exists.

## Packaging

- Restored a complete top-level `import bucex as bx` namespace.
- Advanced checksummed fit archives to schema 2.6.0 while retaining readers
  for all previously supported schemas.
- Added release and workflow validation for the univariate full-path warm
  start and particle-degeneracy diagnostic.
