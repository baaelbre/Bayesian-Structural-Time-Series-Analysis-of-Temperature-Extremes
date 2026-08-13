# Architecture

The package is organized around one statistical model and one inference
contract. Parameterization-specific algebra is internal; it does not define a
second user-facing model, observation family, fitter, or result.

```text
bucex/
  api/              fit, planning helpers, posterior prediction
  components/       trend, seasonal and regression components
  models/           Model and compiler
  observation/      Gaussian and GEV families
  priors/           general SD priors and FS hierarchical profiles
  inference/
    config.py       MCMC, Laplace and Particles
    plan.py         compatibility and exactness resolution
    state/           FFBS, iterated Laplace and PGAS kernels
    fit/             state-space sampler and private FS kernels
  core/             FitResult and numerical primitives
  diagnostics/      chain, engine and predictive diagnostics
  datasets/         Uccle data workflow
  io/               safe checksummed fit archives
  plotting/         plots consuming FitResult
  simulate/         simulation from Model
```

## Fit lifecycle

1. `fit()` resolves a declarative `Model` and preserves pandas dates/name.
2. `compile_model()` produces one linear state layout with semantic state and
   disturbance names.
3. `inference_plan()` validates family, engine, parameterization and ASIS.
4. `resolve_prior_spec()` selects a prior object compatible with that plan.
5. The state-space strategy or FS strategy runs one or more chains.
6. Strategy output is normalized immediately into `FitResult`.

The private `FSOutput` exists only as a five-field transport record between an
FS kernel and the normalizer. It is not exported, serialized, plotted or
returned to users.

## Stable semantic boundary

Every strategy stores centered semantic states in the same order. FS latent
states and signed coefficients are auxiliary/parameter draws. A consumer such
as forecasting, plotting, risk analysis or serialization therefore works on
`FitResult` once and does not branch on result type.

The resolved `InferencePlan` records:

- engine and parameterization;
- state-update description;
- whether the update targets the exact posterior;
- the named approximation, if any;
- the ASIS partner;
- warnings such as singular-transition PGAS support.

Unsupported combinations fail at planning or prior resolution, before MCMC.

## Parameterization strategies

- `centered` updates innovation scales conditional on centered state
  disturbances.
- `disturbance` reconstructs paths from standardized disturbances while
  updating scales. It supports every compiled component, including regression.
- `fruehwirth_schnatter` uses the historical signed-scale augmented regression,
  random sign switching, optional exact structural SSVS and ASIS restoration.

ASIS adds the complementary sweep but does not change the stored semantic
state representation.

## Persistence

`.bucex` archives contain JSON metadata and compressed NumPy arrays. Loading
uses `allow_pickle=False`, verifies a SHA-256 checksum, accepts only the two
archive members, and reconstructs only allowlisted prior dataclasses. The
compiled model is regenerated from the stored declarative model and data.

## Extension rule

A new component implements the component contract and is compiled into the
common state layout. A new engine consumes `CompiledModel` and returns the
standard state/diagnostic data. Neither should add a new public fit function or
result class.
