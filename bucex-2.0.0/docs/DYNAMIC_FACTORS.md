# Dynamic factor models

## Statistical form

A `FactorModel` defines named observation channels and named shared structural
blocks. Its linear predictor is

\[
\eta_{i,t}
= Z^{(i)}_t u_{i,t}
+ \sum_{k=1}^K \lambda_{ik} Z^{(k)}_t f_{k,t},
\]

where each `u` or `f` block follows the same linear-Gaussian transition used by
a univariate `Model`. Observations may be Gaussian or GEV:

\[
y_{i,t}\mid\eta_{i,t}\sim
\begin{cases}
N(\eta_{i,t},\sigma_i^2), & \text{Gaussian channel},\\
\operatorname{GEV}(\eta_{i,t},\sigma_i,\xi_i), & \text{GEV channel}.
\end{cases}
\]

The likelihood factorizes across channels conditional on the full state path.
After integrating over shared states it does not factorize, so a factor model
represents shared trend uncertainty and cross-channel dependence. It does not
represent additional residual/copula dependence.

## Three model-building objects

### `Channel`

```python
bx.Channel(
    "TXx",
    bx.GEV(),
    components=(bx.LocalLevel(mode="static"),),
)
```

The component tuple is the channel's individual state block. It may be empty,
or it may contain one trend, optional dummy seasonality, and optional named
regression. A lower extreme is declared with `tail="lower"`; the internal sign
transform is reversed in stored observations, predictions and risk summaries.

### `Factor`

```python
bx.Factor(
    "climate",
    components=(
        bx.LocalLinearTrend(initial_level=0.0, initial_level_sd=0.0),
    ),
    loadings={
        "TXm": 1.0,
        "TXx": bx.Loading.estimated(0.8, mean=0.0, sd=1.0),
    },
)
```

A factor contains one trend and optional dummy seasonality. A plain numeric
loading is fixed. `Loading.estimated()` is a real-valued MCMC parameter with a
normal prior. Omitted channels have a fixed zero loading.

### `FactorModel`

```python
model = bx.FactorModel(
    channels=(bulk, maximum, minimum),
    factors=(climate,),
    name="shared temperature dynamics",
)
```

Channel and factor names must be unique and cannot contain dots. Dots are
reserved for compiler-generated semantic paths.

## Identification

The transformation \(f_{k,t}\mapsto c f_{k,t}\) and
\(\lambda_{ik}\mapsto\lambda_{ik}/c\) leaves the predictor unchanged. Its sign
is similarly arbitrary. Therefore every factor with an estimated loading must
have at least one fixed, non-zero loading. `bucex` rejects a model that lacks
this anchor before compilation.

With multiple factors, the matrix formed by all fixed loadings (including
implicit zero loadings for omitted channels) must have full column rank. A
lower-triangular anchor pattern or a full-rank fixed contrast matrix satisfies
this rule. Reusing the same anchor row for several otherwise estimated factors
does not, because a rotation can preserve that shared constraint.

This removes scale/sign indeterminacy, but it does not guarantee that a highly
flexible shared trend is empirically separable from highly flexible individual
trends. A useful staged specification is:

1. one shared trend and static channel offsets;
2. add shared seasonal structure if supported;
3. add strongly regularized individual local levels;
4. only then add individual slopes or further factors.

Compare posterior innovation scales, loading uncertainty, factor correlations
and predictive scores at every stage.

If channels contain individual intercept/level blocks, a nonstationary factor
also has a location invariance: a constant can move between the factor and all
channel intercepts. Fix the factor's initial level with
`initial_level=0.0, initial_level_sd=0.0` (or impose an equivalent intercept
constraint). The Uccle constructors do this. If channels have no individual
intercepts, an unconstrained initial factor level may instead carry the shared
baseline; document that choice.

## Fixed contrast factors for the six Uccle summaries

The Uccle series form a partly structured design: day maximum (`TX`) versus
night minimum (`TN`), and monthly mean (`m`) versus upper (`x`) or lower (`n`)
extreme. Rather than estimating an unrestricted rotation, fixed contrast
loadings can encode interpretable hypotheses:

| Series | common | day-night | extremes-mean | upper-lower |
| --- | ---: | ---: | ---: | ---: |
| TXm | 1 | 1 | -2 | 0 |
| TNm | 1 | -1 | -2 | 0 |
| TXx | 1 | 1 | 1 | 1 |
| TNx | 1 | -1 | 1 | 1 |
| TXn | 1 | 1 | 1 | -1 |
| TNn | 1 | -1 | 1 | -1 |

Each column can be one `Factor` with its own local level or local linear trend.
Scaling a contrast changes the factor's units, so choose and document a
normalization. Fixed contrasts avoid rotation ambiguity and directly expose
shared, day-night and tail-asymmetry trend paths. They do not require estimated
loadings; process-scale priors still regularize unnecessary factors toward
nearly static paths.

## Compilation

```python
compiled = bx.compile_model(model, observations)
```

`observations` may be:

- a DataFrame containing the channel names;
- a mapping from channel name to aligned arrays/Series;
- a `(T, channels)` array already in model channel order.

The compiler constructs one block-diagonal transition and innovation-loading
matrix. Its design has shape `(T, channels, state_dim)`. Names are stable:

```text
factor.climate.level
factor.climate.slope
channel.TXm.level
sd.factor.climate.level
sigma.TXm
xi.TXx
loading.climate.TXx
```

`compiled.eta(path, params=params)` maps one global state path to all channel
predictors. `to_disturbance()` and `from_disturbance()` work on the complete
path and preserve singular structural transition support.

Missing observations are handled channel by channel. Every channel must have
at least two finite values, and every time point must retain at least one
finite channel.

## Inference

All-Gaussian factor models use exact multivariate Kalman filtering and FFBS
with conditionally diagonal observation covariance. Models containing at least
one GEV channel use either:

- `engine="pgas"`: conditional SMC with ancestor sampling, exact-invariant for
  the declared posterior;
- `engine="laplace"`: an iterated local-Gaussian approximation, intended for
  screening and sensitivity analysis.

`engine="auto"` chooses FFBS for all-Gaussian factors and PGAS otherwise.
Centered and disturbance scale updates are supported, including ASIS between
them. The specialized Frühwirth-Schnatter augmented regression remains a
univariate strategy and is rejected for factor models.

PGAS operates on the full global state. Particle requirements can grow with
record length and state dimension. Report at least minimum ESS through time,
mean unique ancestors, path-change rate and changed-path fraction. Compare
Laplace and PGAS on an aligned subset before relying on Laplace for a final
tail claim.

## Priors

`default_factor_priors()` creates:

- PC or half-normal priors for every namespaced process SD;
- a half-Student-t observation SD prior per channel;
- a truncated normal shape prior centered at the Gumbel case for each GEV
  channel.

The calibration uses robust local-change and observation scales, not record
length. For complete control, pass `FactorPriors(process=...,
observation_sd=..., shape=...)`. Loading priors are stored in each
`Loading.estimated()` specification so model identification and persistence
remain self-contained.

## Results and downstream machinery

Factor fits still return `FitResult`:

```python
fit.factor_draws("climate", state="level")
fit.loading_draws("climate", "TXx")
fit.channel_eta_draws("TXx", original_scale=True)
fit.return_level_draws(100, channel="TXx")
fit.plot(kind="factor", factor="climate")
fit.plot(kind="channel", channel="TXx")
```

`fit.eta_draws()` has shape `(draws, T, channels)`. `fit.forecast(h)` returns
one `Forecast` whose observations have shape `(draws, h, channels)`; summaries
and scores can be computed for one channel or all channels. Safe `.bucex`
archives store the complete factor graph, loading constraints, priors, chain
draws, exactness plan and diagnostics.

## What v2 deliberately leaves separate

Dynamic factors address shared latent evolution. They do not yet include:

- unrestricted contemporaneous observation covariance;
- a Gaussian-copula residual link for mixed Gaussian/GEV channels;
- correlated process innovations outside shared components;
- a full daily process whose monthly means and extremes are induced
  functionals.

Those are distinct statistical assumptions and should be added as explicit
model components or observation layers, not hidden inside the factor fitter.
