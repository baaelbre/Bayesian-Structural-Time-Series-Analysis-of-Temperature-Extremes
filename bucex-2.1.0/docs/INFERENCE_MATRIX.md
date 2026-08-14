# Inference matrix

## State engines

| Model/observation | Engine | State update | Posterior target |
| --- | --- | --- | --- |
| Gaussian | `ffbs` | Kalman filter/smoother plus FFBS | Exact conditional update |
| GEV | `laplace` | Iterated-Laplace pseudo-observations plus FFBS | Approximate |
| GEV | `pgas` | Conditional SMC with ancestor sampling | Exact-invariant kernel |
| All-Gaussian factor | `ffbs` | Multivariate Kalman update plus FFBS | Exact conditional update |
| Mixed/GEV factor | `laplace` | Vector pseudo-observations plus FFBS | Approximate |
| Mixed/GEV factor | `pgas` | Joint conditional SMC/ancestor sampling | Exact-invariant kernel |

`engine="auto"` selects FFBS for all-Gaussian models, Laplace for a univariate
GEV model, and PGAS for a mixed/GEV factor model. PGAS handles singular
structural transitions on their affine support and reports ESS, ancestor
diversity, and path-change diagnostics.

## Parameterizations

| Parameterization | Scope | Scale representation | ASIS partner |
| --- | --- | --- | --- |
| `centered` | Any compiled model | Direct centered disturbances | disturbance |
| `disturbance` | Any compiled model | Standardized structural disturbances | centered |
| `fruehwirth_schnatter` | Eligible univariate or v2.1 one-factor graph | Signed coefficients multiplying unit-innovation states | centered |

The v2.1 factor FS layout requires:

- exactly one factor with one dynamic local linear trend;
- a factor initial level fixed at zero;
- one dynamic local level per channel;
- at most one dynamic dummy-seasonal block per channel;
- no channel regressions in the FS graph.

`parameterization="auto"` chooses FS when that layout is satisfied and
disturbance otherwise. Aliases `fs`, `ncp`, `noncentered`, and `noncentred`
select the FS construction explicitly.

## Prior profiles

| Public profile | Univariate FS | Centered/disturbance | Factor FS | Interpretation |
| --- | --- | --- | --- | --- |
| `manuscript_lasso` | Yes | No | No | Original shared Bayesian lasso |
| `regularized_lasso` | Yes | No | No | Componentwise Bayesian lasso |
| `regularized_horseshoe` | Yes | No | Yes | Univariate hierarchy, or factor idiosyncratic hierarchy |
| `pc` / factor `regularized` | Yes | Yes | Yes | Calibrated PC process-scale prior |
| `normal` | Yes | Yes | Yes | Signed normal or half-normal process scale |
| `ssvs` | Yes | Yes | No | FS exact states or general continuous spike/slab |

For factor models, `regularized_horseshoe` is also supported by centered and
disturbance parameterizations. It is restricted to namespaced dynamic channel
local-level innovations. Shared-factor and seasonal scales keep the process
priors in `FactorPriors.process`.

## Exactness and ASIS

Parameterization and ASIS do not alter the engine's exactness statement.
Gaussian FFBS and GEV PGAS are exact conditional/invariant kernels; iterated
Laplace is approximate in every parameterization. Read `fit.plan` for the
resolved contract.

For factors, `fit.plan.backend == "factor_state_space"`. The plan warning that
channels are conditionally independent given the latent states is part of the
model contract: the shared factor induces marginal dependence, but no residual
copula is present.
