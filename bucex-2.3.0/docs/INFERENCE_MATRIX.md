# Inference matrix

## State engines

| Model/observation | Engine | State update | Posterior target |
| --- | --- | --- | --- |
| Gaussian | `ffbs` | Kalman filter/smoother plus FFBS | Exact conditional update |
| GEV | `laplace` | Iterated-Laplace pseudo-observations plus FFBS | Approximate |
| GEV | `pgas` | Conditional SMC with ancestor sampling | Exact-invariant kernel |
| All-Gaussian hierarchical panel | `ffbs` | Per-channel FS FFBS + joint hierarchy | Exact Gibbs kernel |
| Mixed/GEV hierarchical panel | `pgas` | FS PGAS + exact GEV model moves + joint hierarchy | Exact-invariant kernel |
| All-Gaussian factor | `ffbs` | Multivariate Kalman update plus FFBS | Exact conditional update |
| Mixed/GEV factor | `laplace` | Vector pseudo-observations plus FFBS | Approximate |
| Mixed/GEV factor | `pgas` | Joint conditional SMC/ancestor sampling | Exact-invariant kernel |

`engine="auto"` selects FFBS for all-Gaussian models, Laplace for a univariate
GEV model, and PGAS for a mixed/GEV panel or factor model. PGAS handles singular
structural transitions on their affine support and reports ESS, ancestor
diversity, and path-change diagnostics.

For univariate GEV SSVS, PGAS is exact-invariant for both the state path and
the zero/fixed/dynamic model allocation. A Gaussian/Laplace approximation is
used only as a full-support independence proposal. The trans-dimensional MH
ratio uses the exact GEV likelihood, normalized model/slab priors, and both
proposal densities; active coefficients then receive an exact
elliptical-slice update. With `engine="laplace"`, SSVS enumeration still uses
pseudo-observations and is explicitly approximate.

Hierarchical panels never use Laplace as their posterior target. Gaussian
channels use FFBS and any mixed/GEV panel uses PGAS. Their GEV SSVS move has the
same exact-likelihood correction as the univariate PGAS route.

## Parameterizations

| Parameterization | Scope | Scale representation | ASIS partner |
| --- | --- | --- | --- |
| `centered` | Any compiled model | Direct centered disturbances | disturbance |
| `disturbance` | Any compiled model | Standardized structural disturbances | centered |
| `fruehwirth_schnatter` | Eligible univariate, hierarchical panel, or v2.1 one-factor graph | Signed coefficients multiplying unit-innovation states | centered where compatible |

The v2.1 factor FS layout requires:

- exactly one factor with one dynamic local linear trend;
- a factor initial level fixed at zero;
- one dynamic local level per channel;
- at most one dynamic dummy-seasonal block per channel;
- no channel regressions in the FS graph.

For an identified smooth factor, declare both initial conditions explicitly:

```python
bx.LocalLinearTrend(
    initial_level=0.0,
    initial_level_sd=0.0,
    initial_slope=0.0,
    initial_slope_sd=0.0,
)
```

The slope remains dynamically stochastic when its process SD is non-zero;
only the initial coefficient is fixed.

Factor scale/sign anchoring is necessary but not sufficient to distinguish an
estimated loading times a persistent factor from an unrestricted persistent
channel deviation. Use `identified_factor_priors()` to declare smooth-factor,
pure-reference, or fixed-idiosyncratic restrictions, and inspect
`factor_identification_diagnostics()` after every decomposition fit.

The hierarchical panel layout requires every channel to declare one dynamic
`LocalLinearTrend` (trend may be switched off) plus at most one dynamic
`DummySeasonal`, with no regression. SSVS then selects the exact structural
state. It requires `asis=False`.

## Factor loading kernels

| Channel | Kernel | Path treatment |
| --- | --- | --- |
| Gaussian with estimated loading | Collapsed scale MH + exact augmented FFBS | Integrates and redraws intercept, loading, and idiosyncratic random walk |
| GEV with non-zero idiosyncratic SD | Predictor-preserving interweaving | Changes loading and deviation jointly; likelihood/support unchanged |
| GEV with fixed-zero idiosyncratic SD | Path-conditional Metropolis fallback | Loading changes the predictor directly |

These kernels are internal strategy blocks. They do not introduce another fit
function, result type, or parameterization.

`parameterization="auto"` chooses FS when that layout is satisfied and
disturbance otherwise. Aliases `fs`, `ncp`, `noncentered`, and `noncentred`
select the FS construction explicitly.

## Prior profiles

| Public profile | Univariate FS | Hierarchical panel | Centered/disturbance | Factor FS | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `manuscript_lasso` | Yes | No | No | No | Original shared Bayesian lasso |
| `regularized_horseshoe` | Yes | No | No | Yes | Univariate hierarchy, or factor idiosyncratic hierarchy |
| `triple_gamma` | Yes | No | No | Yes | Normal--gamma--gamma continuous shrinkage with stored shrinkage factors |
| `regularized_triple_gamma` | Yes | No | No | Yes | Triple gamma with an optional finite-variance slab cap |
| `pc` / factor `regularized` | Yes | No | Yes | Yes | Calibrated PC process-scale prior |
| `normal` | Yes | No | Yes | Yes | Signed normal or half-normal process scale |
| `ssvs` | Yes | via `hierarchical_ssvs` | Yes | No | Exact structural point masses under FS; continuous spike/slab elsewhere |

For factor models, `regularized_horseshoe` is also supported by centered and
disturbance parameterizations. It is restricted to namespaced dynamic channel
local-level innovations. Shared-factor and seasonal scales keep the process
priors in `FactorPriors.process`.

The two triple-gamma factor profiles have the same scope and are supported by
centered, disturbance, and eligible FS factor inference. Univariate
triple-gamma hierarchies require FS because their priors act on signed
innovation-scale coefficients. PC is a separate calibrated exponential prior,
not a finite triple-gamma member.

## Exactness and ASIS

Parameterization and ASIS do not alter the engine's exactness statement.
Gaussian FFBS and GEV PGAS are exact conditional/invariant kernels; iterated
Laplace is approximate in every parameterization. Read `fit.plan` for the
resolved contract.

All FS sign symmetries are randomized by flipping the signed coefficient and
its standardized path together. The sampler verifies the predictor/centered
path is invariant and stores the numerical error. `sd.*`, rather than the
unidentified signed coefficient, is the scientific scale.

## Deterministic FS components

The FS state grammar retains unit-innovation paths for every candidate block,
while structural coefficients determine whether they enter the predictor:

| SSVS state | Level | Trend | Seasonality |
| --- | --- | --- | --- |
| `zero` | not allowed | `beta0=0`, `sd.slope=0` | all static coefficients and `sd.seasonal` are zero |
| `fixed` | static `alpha0`, `sd.level=0` | static `beta0`, `sd.slope=0` | static initial-season vector, `sd.seasonal=0` |
| `dynamic` | `alpha0` plus signed level scale | `beta0` plus signed slope scale | static initial-season vector plus signed seasonal scale |

A fixed component is therefore not represented by a tiny innovation variance.
Its innovation coefficient is exactly zero and its deterministic contribution
is a separate static regression parameter.

The v2.2 seasonal regression design is generated from the same sum-to-zero
rotation used to reconstruct semantic states. This algebraic identity is
tested at every time point and replaces the inconsistent phase ordering in
2.1.x.

## Smoother covariance

FS backward sampling uses the Joseph-form conditional covariance
`(I-JG) C (I-JG)' + JQJ'`. It is algebraically equivalent to
`C - J R J'` but avoids catastrophic cancellation in long series with
singular transitions or nearly zero process scales. Valid zero eigenvalues are
kept at zero; only materially indefinite inputs raise an error.

For factors, `fit.plan.backend == "factor_state_space"`. The plan warning that
channels are conditionally independent given the latent states is part of the
model contract: the shared factor induces marginal dependence, but no residual
copula is present.

## Progress contract

All sampler/parameterization combinations honor `MCMC(progress=True)` and
`MCMC(progress_every=...)`. The common line reports chain, `it`, phase, saved
draws, current parameters, elapsed time, and ETA. Factor parameters are grouped
compactly by channel/process; PGAS adds particle ESS, distinct ancestors, and
path-change fraction, while Laplace adds inner-iteration convergence fields.
