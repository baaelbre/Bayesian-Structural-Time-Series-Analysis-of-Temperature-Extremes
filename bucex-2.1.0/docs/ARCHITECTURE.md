# Architecture

The package is organized around one compiled state-space contract and one
inference contract. A univariate `Model` and a multichannel `FactorModel` are
two declarative grammars for that contract; neither creates a separate fitter
or result type. Parameterization-specific algebra remains internal.

```text
bucex/
  api/              fit, planning helpers, posterior prediction
  components/       trend, seasonal and regression components
  models/           Model, FactorModel and compilers
  observation/      Gaussian and GEV families
  priors/           general, factor and FS prior graphs
  inference/
    config.py       MCMC, Laplace and Particles
    plan.py         compatibility and exactness resolution
    state/           scalar/vector FFBS, iterated Laplace and PGAS kernels
    fit/             univariate/factor samplers and private FS kernels
  core/             FitResult and numerical primitives
  diagnostics/      chain, engine and predictive diagnostics
  datasets/         Uccle data workflow
  io/               safe checksummed fit archives
  plotting/         plots consuming FitResult
  simulate/         simulation from Model or FactorModel
```

## Fit lifecycle

1. `fit()` resolves a declarative `Model` or `FactorModel` and preserves pandas
   dates/names.
2. `compile_model()` produces one linear state layout with semantic state and
   disturbance names. A factor design has shape `(T, channels, state_dim)`.
3. `inference_plan()` validates family, engine, parameterization and ASIS.
4. `resolve_prior_spec()` or `resolve_factor_priors()` selects a prior graph
   compatible with that plan.
5. The state-space, factor, or FS strategy runs one or more chains.
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

For factor models the semantic namespace records ownership explicitly:
`factor.<name>.<state>` and `channel.<name>.<state>`. Observation and loading
parameters are similarly channel/factor scoped. Downstream consumers inspect
the compiled interface and `FitResult.is_factor_model`; they do not receive a
second result class.

## Factor compilation

Each `Factor` and each non-empty channel component tuple is first compiled as a
structural state block. The factor compiler then:

1. block-diagonalizes transitions, innovation loadings and initial priors;
2. namespaces state and disturbance names;
3. builds each channel design row by adding its individual block and the
   loading-weighted shared blocks;
4. records fixed/estimated loading constraints and per-channel reference
   scales;
5. exposes the same disturbance round-trip and affine-support projection as a
   scalar compiled model.

Estimated factor loadings are part of the design, so state kernels rebuild the
design from the current parameter draw. A fixed non-zero anchor is required
for every factor with estimated loadings, and the fixed-anchor matrix must have
full column rank across multiple factors.

All-Gaussian channel updates use one multivariate Kalman measurement step with
diagonal conditional observation covariance. Mixed-family Laplace and PGAS
multiply channel likelihood contributions at each time. Conditional
independence is an explicit model assumption and is recorded in plan warnings.

## Parameterization strategies

- `centered` updates innovation scales conditional on centered state
  disturbances.
- `disturbance` reconstructs paths from standardized disturbances while
  updating scales. It supports every compiled component, including regression.
- `fruehwirth_schnatter` uses signed-scale, unit-innovation augmented states,
  random sign switching, optional exact univariate structural SSVS, and ASIS
  restoration. The factor FS wrapper composes one augmented block for the
  shared trend and one for each channel's local level/seasonal block.

ASIS adds the complementary sweep but does not change the stored semantic
state representation.

All factor models support centered and disturbance strategies. The v2.1 FS
strategy additionally accepts the identified one-factor layout: one shared
dynamic local-linear trend with fixed initial level, one dynamic local level
per channel, and optional series-specific dynamic dummy seasonality. Other
factor graphs fail FS validation before sampling.

## Persistence

`.bucex` schema 2.1 archives contain JSON metadata and compressed NumPy arrays. Loading
uses `allow_pickle=False`, verifies a SHA-256 checksum, accepts only the two
archive members, and reconstructs only allowlisted prior dataclasses. The
compiled model is regenerated from the stored declarative model and data.
The v2.1 loader also accepts safe v1.2 and 2.0 archives.

## Extension rule

A new component implements the component contract and is compiled into the
common state layout. A new engine consumes the compiled transition/design
contract and returns standard state/diagnostic data. Neither should add a new
public fit function or result class. A new cross-channel dependence mechanism
must be explicit about whether it belongs in the latent transition, the design,
or the observation likelihood; it must not be smuggled into factor loadings.
