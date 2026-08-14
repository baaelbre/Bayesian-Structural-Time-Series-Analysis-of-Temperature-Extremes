# Inference matrix

## Engines

| Model/observation | Engine | State update | Posterior target |
| --- | --- | --- | --- |
| Gaussian | `ffbs` | Kalman filter/smoother plus FFBS | Exact conditional update |
| GEV | `laplace` | Damped iterated-Laplace pseudo-observations plus FFBS | Approximate (`iterated_laplace`) |
| GEV | `pgas` | Conditional SMC with ancestor sampling | Exact-invariant Markov kernel |
| All-Gaussian factor | `ffbs` | Multivariate Kalman filter plus FFBS | Exact conditional update |
| Mixed/GEV factor | `laplace` | Vector pseudo-observations plus FFBS | Approximate (`iterated_laplace`) |
| Mixed/GEV factor | `pgas` | Global-state conditional SMC with ancestor sampling | Exact-invariant Markov kernel |

`engine="auto"` selects FFBS for Gaussian, Laplace for univariate GEV, and
PGAS for a mixed/GEV factor model. PGAS handles singular structural transitions
on their affine support and reports ESS, ancestor diversity and path-change
diagnostics.

## Parameterizations

| Parameterization | Model scope | Scale representation | ASIS partner |
| --- | --- | --- | --- |
| `centered` | Any univariate or factor compiled model | Direct centered disturbances | `disturbance` |
| `disturbance` | Any univariate or factor compiled model | Standardized/scaled disturbances | `centered` |
| `fruehwirth_schnatter` | One dynamic local-linear trend, optional dynamic dummy seasonality, no regression | Signed FS scale coefficients | `centered` |

`auto` chooses FS when its layout is supported and disturbance otherwise.
Aliases `fs`, `ncp`, `noncentered`, and `noncentred` mean the historical FS
parameterization. Use `disturbance` when that is what you mean.
Factor models always resolve `auto` to disturbance and reject FS explicitly.

## Prior profiles

| Public profile | FS | Centered/disturbance | Interpretation |
| --- | --- | --- | --- |
| `manuscript_lasso` | Yes | No | Original shared Bayesian-lasso hierarchy |
| `regularized_lasso` | Yes | No | Componentwise scale-aware Bayesian lasso |
| `regularized_horseshoe` | Yes | No | Regularized horseshoe hierarchy |
| `pc` | Yes | Yes | FS mixture/PC hierarchy or calibrated exponential process-SD prior |
| `normal` | Yes | Yes | Signed Normal FS scales or half-Normal process SDs |
| `ssvs` | Yes | Yes | Exact zero/fixed/dynamic FS selection or continuous process-SD spike/slab |
| factor `regularized`/`pc` | No | Factor only | PC process-SD priors plus channel-specific observation/shape priors |
| factor `normal` | No | Factor only | Half-Normal process SDs plus channel-specific observation/shape priors |

The two SSVS forms are deliberately labeled in `FitResult.meta`; they do not
pretend to be the same algorithm. `component_probabilities()` presents
zero/fixed/dynamic probabilities for FS SSVS and spike/slab probabilities for
the general process prior.

GEV FS structural SSVS is available with the Laplace state update and is marked
as Laplace-pseudo-observation model selection. It is rejected with PGAS rather
than mislabeled exact.

## Exactness and ASIS

Parameterization and ASIS do not change the engine's exactness statement.
Gaussian FFBS and GEV PGAS are exact conditional kernels; GEV iterated Laplace
is approximate in every parameterization. Read `fit.plan`, not a function name,
to determine the resolved contract.

For factors, `fit.plan.backend == "factor_state_space"`. The plan warning that
channels are conditionally independent is part of the statistical contract,
not merely an implementation note.
