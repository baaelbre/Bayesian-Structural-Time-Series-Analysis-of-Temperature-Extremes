# bucex 2.6.0 release notes

Version 2.6.0 reorganizes the package around the COMPSTAT scientific story and
removes the unrelated legacy presentation scripts.

## Focused experimental design

- Added three local-level GEV tail illustrations with common latent evolution
  and `xi=0.20`, `0`, and `-0.20`.
- Added seven structural-selection scenarios with common `sigma=1.5` and
  `xi=-0.20`, spanning absent, fixed, and dynamic level, slope, and seasonal
  components.
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

`PresentationWorkflow`, `bucex-presentation`, numbered Python examples, and PBS
jobs all call these same stages and write the same deterministic paths. The PBS
workflow uses scenario/series-by-chain arrays and dependency-gated combine
jobs, so PGAS cannot start before the corresponding Laplace fit exists.

## Packaging

- Restored a complete top-level `import bucex as bx` namespace.
- Advanced checksummed fit archives to schema 2.6.0 while retaining readers
  for all previously supported schemas.
- Added release and workflow validation for the univariate full-path warm
  start and particle-degeneracy diagnostic.
