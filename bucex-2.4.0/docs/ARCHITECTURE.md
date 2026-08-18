# Architecture

Version 2.4 uses one compiled state-space contract, one fitting entry point,
and one result type for univariate and hierarchical analyses.

```text
bucex/
  api/                 fit, forecast, prediction helpers
  components/          local level/trend, dummy seasonal, regression
  models/              Model, MultiSeriesModel, Channel, compilers
  observation/         Gaussian and GEV families
  priors/              univariate and hierarchical prior specifications
  inference/
    config.py          MCMC, Laplace, Particles
    plan.py            compatibility and exactness resolution
    state/             FFBS, Laplace, and PGAS primitives
    fit/               univariate FS/general and hierarchical samplers
  core/                FitResult and stable numerical primitives
  diagnostics/         chain diagnostics, scores, PIT, leave-future-out
  datasets/            reproducible Uccle workflows
  io/                  checksummed non-pickle archives
  plotting/            result-driven plotting and save dispatch
  simulate/            univariate simulation
```

## Fit lifecycle

1. `fit()` accepts a `Model` or `MultiSeriesModel` and preserves pandas dates
   and names.
2. `compile_model()` creates semantic state names, transition matrices,
   innovation loadings, and initial priors.
3. `inference_plan()` validates the observation families, engine,
   parameterization, and ASIS choice before sampling.
4. Prior resolution selects a univariate prior graph or `HierarchicalPrior`.
5. An inference strategy produces states, parameters, diagnostics, and
   metadata.
6. The output is normalized immediately into `FitResult`.

The private `FSOutput` is a transport record between an FS kernel and the
normalizer. It is not exported or serialized.

## Stable semantic boundary

Every strategy stores centered scientific states in the same result contract.
FS unit-innovation states and signed coefficients are auxiliary draws.
Forecasting, plotting, scoring, rate summaries, and persistence consume the
semantic `FitResult` representation rather than branching on sampler type.

The `InferencePlan` records:

- family, engine, and parameterization;
- state-update description;
- exact versus approximate posterior contract;
- ASIS partner, if any;
- warnings such as singular-transition affine support.

## Multiseries compilation

`MultiSeriesModel` contains named `Channel` objects. Each channel is compiled
once, then its transition, innovation loading, initial prior, and state names
are inserted as a block. Parameters use explicit namespaces such as
`sd.channel.TXm.level`, `sigma.TXm`, and `initial.channel.TXm.slope`.

The channels retain separate paths. Dependence is introduced only by the
prior hierarchy. The joint sampler cycles through:

1. a Gaussian FFBS or GEV PGAS state update for every channel;
2. exact structural allocation updates;
3. channel observation-parameter updates;
4. conjugate Dirichlet updates of population allocation probabilities;
5. exact log-scale slice updates of pooled half-t slab multipliers.

ASIS is disabled for this sampler because structural state changes already
alter the active parameter dimension. Random FS sign switches remain enabled
and are checked for predictor invariance.

## Parameterizations

- `centered`: process standard deviations multiply innovations in the state
  transition directly.
- `disturbance`: standardized disturbances are reconstructed through the
  compiled transition loading.
- `fruehwirth_schnatter` / `fs`: signed coefficients multiply unit-innovation
  state paths and support exact structural selection.

`MultiSeriesModel` currently requires FS. Univariate models retain all valid
parameterizations.

The shared Gaussian FFBS primitive uses symmetric covariance updates,
Joseph-form filtering, and a Joseph-style backward conditional covariance.
Small round-off eigenvalues are repaired; materially indefinite matrices still
raise. This is critical for long series with singular transitions and process
scales close to zero.

## PGAS affine support

Singular transitions are handled on their affine support. Guided disturbance
PGAS performs three safeguards:

1. project the conditioned trajectory onto the compiled affine support;
2. recover disturbances using the scaled transition loading rather than an
   unscaled template;
3. when all optional ancestor candidates violate support, keep the validated
   conditioned predecessor.

The last step rejects an invalid optional move; it does not replace the
conditioned trajectory or alter the invariant posterior target.

## Persistence

`.bucex` schema 2.4 archives contain JSON metadata and compressed NumPy arrays.
Loading uses `allow_pickle=False`, verifies SHA-256 checksums, accepts only
allowlisted archive members, and regenerates the compiled model from the
stored declarative specification and data.

## Extension rule

A new component implements the component contract and compiles into the common
state layout. A new engine consumes that compiled contract and returns standard
state and diagnostic data. Neither should add a second public fitter or result
class. New cross-channel dependence must be an explicit likelihood or
transition feature; it must not be hidden inside prior pooling.
