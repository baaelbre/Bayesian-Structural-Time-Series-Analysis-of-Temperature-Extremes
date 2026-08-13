# Migrating to bucex 1.2.1

Version 1.2.1 removes the parallel public implementations that existed in 1.1.
All inference now enters through `bucex.fit` and returns `bucex.FitResult`.

## Calls

| Before | 1.2.1 |
| --- | --- |
| `fit_fs(y, ...)` | `fit(y, parameterization="fruehwirth_schnatter", ...)` |
| `fit_bayes(y, ...)` | Same replacement; `fit_bayes` remains a deprecated delegating wrapper |
| General `fit(y, parameterization="noncentered")` | Use `parameterization="disturbance"` for scaled disturbances |
| `state_method="particle"` | `engine="pgas"`, with `Particles(...)` |
| Laplace settings in `state_kwargs` | `laplace=Laplace(...)` |
| Particle count in `state_kwargs` | `particles=Particles(n=...)` |
| `PosteriorBundle` | `FitResult`; the old name aliases the same class |
| `combine_fs_fits(...)` | `combine_fits(...)` |

The aliases `"fs"`, `"noncentered"`, and `"ncp"` intentionally resolve to the
historical FS augmentation. The unambiguous name for the general standardized
disturbance representation is `"disturbance"`.

## Models and states

The canonical constructors are:

```python
bx.Model(
    observation=bx.GEV(),
    components=[bx.LocalLinearTrend(), bx.DummySeasonal(12)],
)
```

`StructuralModel` and `StructuralSSM` alias `Model`; `GaussianObs` and `GEVObs`
alias `Gaussian` and `GEV`. State names no longer depend on the sampler:

| Old FS name | Canonical name |
| --- | --- |
| `alpha` | `level` |
| `beta` | `slope` |
| `g1`, ... | `seasonal[1]`, ... |
| signed `s_level` | `signed_sd.level` |
| process magnitude | `sd.level` |

Use `fit.state("level")`, `fit.parameter("sd.level")`, and
`fit.auxiliary_draws` rather than indexing implementation-specific arrays.

## Priors

FS fits accept the six named profiles documented in the inference matrix.
Centered and disturbance fits use the general `Priors` object and the `pc`,
`normal`, or `ssvs` shorthands. Signed hierarchical lasso and horseshoe priors
are rejected outside FS rather than silently approximated.

The obsolete centered-prior dataclasses and centered sampler classes were
removed. Centered is now a parameterization in the same state-space sampler.

## Imports

Low-level kernels live in explicit packages:

```python
from bucex.inference.state import ffbs, iterated_laplace, pgas
from bucex.core.numerics import gaussian_support
from bucex.models.compiler import compile_model
```

The old root shim modules such as `bucex.kalman`, `bucex.particle`, and
`bucex.results` were removed.

## Archives

Version 1.2 uses one safe `bucex-fit` schema for all parameterizations. Old
1.1 `bucex-fs-fit` and 1.0 schemas are not loaded implicitly because their
object layouts differ. Refit or convert trusted old results in the old
environment, then save analysis summaries in a neutral format before upgrading.
