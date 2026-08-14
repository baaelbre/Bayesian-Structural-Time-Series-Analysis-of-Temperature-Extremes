# Migrating to bucex 2.1

## From 2.1.2 to 2.1.4

Existing model and fit calls remain source compatible. Progress output is now
uniform and uses `it`; set `MCMC(progress_every=...)` if a script depends on a
specific reporting cadence.

Factor prior strings are normalized more generously. The new helper below
replaces manual `FactorPriors` edits used by earlier examples:

```python
compiled = bx.compile_model(model, data)
priors = bx.identified_factor_priors(
    compiled,
    profile="regularized_horseshoe",
    smooth_factor=True,
    reference_channel="TXm",
)
```

Loading anchors still identify factor scale/sign, not the allocation of a
persistent signal between `lambda[i] * f[t]` and `alpha[i,t]`. Existing fits
are valid, but decomposition claims should now be accompanied by
`factor_identification_diagnostics()` and sensitivity fits. New result and
plot methods are additive; no archive fields were removed.

## From 2.1.1 to 2.1.2

The public construction and fit APIs remain source compatible. Three
behaviours are intentionally stronger:

- `LocalLinearTrend(initial_slope_sd=0.0)` is now honoured as an exact fixed
  factor initial condition under FS inference;
- eligible Gaussian factor loadings and their channel deviations use a joint
  collapsed/FFBS block instead of scalar path-conditional loading proposals;
- eligible GEV loadings receive a predictor-preserving interweaving move.

Existing result code continues to work. New analyses can replace manual
factor centering and channel bookkeeping with `normalized_factor()` and
`channel_decomposition()`.

## From 2.0 to 2.1

The univariate API is unchanged. Existing multi-factor models continue to use
centered or disturbance parameterizations.

The Uccle factor-helper defaults changed intentionally. In 2.1,
`make_uccle_factor_model()` creates one estimated common factor, dynamic
channel local levels, and dynamic series-specific dummy seasonality. It is
eligible for `parameterization="fruehwirth_schnatter"`; `auto` now selects FS
for that exact layout. To reproduce a v2.0 contrast graph explicitly, use:

```python
bx.make_uccle_factor_model(
    structure="contrasts",
    individual="static",
    seasonal=None,
)
```

The default factor prior profile is now `regularized_horseshoe`. It applies
only to dynamic channel local-level innovations. Select `priors="regularized"`
to retain the v2.0 PC process-scale profile.

Factor fits written by 2.1 use safe archive schema 2.1. The loader continues to
read schema 1.2 and 2.0 archives.

## From 1.2 to 2.0

The v1.2 univariate API remains source compatible. `Model`, `fit`, `plan`,
`FitResult`, parameterization names, prior profiles, Uccle helpers, forecasting
and risk methods retain their meanings. Version 2 adds `Channel`, `Factor`,
`Loading`, and `FactorModel` alongside that API.

## From separate fits to a shared factor

A pair of independent fits such as

```python
bulk = bx.fit(bulk_y, family="gaussian", ...)
tail = bx.fit(tail_y, family="gev", ...)
```

is not automatically converted, because factor loadings and individual state
blocks are substantive modeling choices. Build the joint graph explicitly:

```python
model = bx.FactorModel(
    channels=(
        bx.Channel("bulk", bx.Gaussian()),
        bx.Channel("tail", bx.GEV()),
    ),
    factors=(
        bx.Factor(
            "shared",
            (bx.LocalLinearTrend(initial_level=0.0, initial_level_sd=0.0),),
            {
                "bulk": 1.0,
                "tail": bx.Loading.estimated(0.8),
            },
        ),
    ),
)
joint = bx.fit(data, model, engine="pgas", parameterization="disturbance")
```

`fit_bulk_tail()` deliberately remains an independent-fit compatibility
helper and continues to report `joint_likelihood=False`.

Factor parameter names are namespaced. For example, use
`sd.factor.shared.level`, `sigma.bulk`, `xi.tail`, and
`loading.shared.tail`. Factor-model `eta_draws()` and forecasts add a final
channel dimension. Risk methods require `channel=` when a result contains more
than one observation family.

The specialized FS parameterization remains available for supported
univariate models. Factor models support centered and disturbance strategies;
they fail early if FS is requested.

## Earlier 1.1 to 1.2 call changes

Version 1.2.1 removed the parallel public implementations that existed in
1.1. All inference enters through `bucex.fit` and returns `bucex.FitResult`.

### Calls

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

### Models and states

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

### Priors

FS fits accept the six named profiles documented in the inference matrix.
Centered and disturbance fits use the general `Priors` object and the `pc`,
`normal`, or `ssvs` shorthands. Signed hierarchical lasso and horseshoe priors
are rejected outside FS rather than silently approximated.

The obsolete centered-prior dataclasses and centered sampler classes were
removed. Centered is now a parameterization in the same state-space sampler.

### Imports

Low-level kernels live in explicit packages:

```python
from bucex.inference.state import ffbs, iterated_laplace, pgas
from bucex.core.numerics import gaussian_support
from bucex.models.compiler import compile_model
```

The old root shim modules such as `bucex.kalman`, `bucex.particle`, and
`bucex.results` were removed.

## Archives

Version 2 writes safe `bucex-fit` schema 2.0 for both model grammars and reads
safe v1.2 archives. Old 1.1 `bucex-fs-fit` and 1.0 schemas are still not loaded
implicitly because their object layouts differ. Refit or convert trusted old
results in the old environment, then save analysis summaries in a neutral
format before upgrading.
