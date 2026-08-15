# The v2.1 one-factor model

## Statistical model

For the six monthly temperature summaries, `bucex` implements

\[
\mu_{i,t}=c_i+\lambda_i f_t+S_{i,t}+\alpha_{i,t}.
\]

The observation model is

\[
y_{i,t}\mid\mu_{i,t}\sim
\begin{cases}
N(\mu_{i,t},\sigma_i^2), & i\in\{\mathrm{TXm},\mathrm{TNm}\},\\
\operatorname{GEV}(\mu_{i,t},\sigma_i,\xi_i),
& i\in\{\mathrm{TXx},\mathrm{TXn},\mathrm{TNx},\mathrm{TNn}\}.
\end{cases}
\]

`TXn` and `TNn` are lower extremes. `Channel(..., tail="lower")`
multiplies those observations by (-1) internally so one upper-tail GEV
implementation can be used. Predictions and summaries are mapped back to the
original temperature orientation.

The latent blocks are deliberately restricted:

- (f_t): one shared local linear trend;
- (\lambda_{\mathrm{TXm}}=1): fixed scale/sign anchor;
- (\lambda_i), (i\ne\mathrm{TXm}): estimated normal-prior loadings;
- (\alpha_{i,t}): independent channel local levels;
- (S_{i,t}): independent channel dummy-seasonal blocks.

Conditionally on all latent states, the six likelihood terms multiply. After
integrating over the shared (f_t), the channels are dependent. The model does
not add residual copula dependence, correlated observation errors, or
correlated idiosyncratic shocks.

## Construction

The complete Uccle graph is a single helper call:

```python
data = bx.load_uccle_factor_data()
model = bx.make_uccle_factor_model()
```

Its defaults are equivalent to:

```python
bx.make_uccle_factor_model(
    structure="estimated",
    individual="local_level",
    seasonal="series_specific",
    seasonal_mode="dynamic",
    period=12,
)
```

The modular objects are `Channel`, `Factor`, and `FactorModel`. An eligible FS
model has exactly one `Factor` containing one dynamic `LocalLinearTrend`, and
each channel contains exactly one dynamic `LocalLevel` plus an optional dynamic
`DummySeasonal`. The shared factor's initial level must be fixed to zero:

```python
common = bx.Factor(
    "common",
    components=(
        bx.LocalLinearTrend(
            initial_level=0.0,
            initial_slope=0.0,
            initial_level_sd=0.0,
            initial_slope_sd=0.0,
        ),
    ),
    loadings={
        "TXm": 1.0,
        "TNm": bx.Loading.estimated(1.0),
        "TXx": bx.Loading.estimated(1.0),
        "TXn": bx.Loading.estimated(-1.0),
        "TNx": bx.Loading.estimated(1.0),
        "TNn": bx.Loading.estimated(-1.0),
    },
)
```

The negative internal initial values for `TXn` and `TNn` correspond to
positive original-orientation loadings after the lower-tail sign transform.

The v2 compiler still accepts other identified factor graphs for backward
compatibility. They use centered or disturbance inference. The specialized FS
construction below is intentionally limited to the scientific one-factor
model rather than pretending to solve rotational identification for an
unrestricted DFM.

## Identification

The transformation (f_t\mapsto a f_t),
(\lambda_i\mapsto\lambda_i/a) leaves the predictor unchanged. A fixed
non-zero loading removes scale and sign invariance. Fixing (f_0=0) removes
the location invariance between the factor and the channel baselines (c_i).
With `initial_slope_sd=0`, the initial factor rate is also a fixed initial
condition rather than a static MCMC coefficient. Stochastic slope innovations
still allow the rate to evolve immediately after time zero.

Strongly regularized idiosyncratic innovations are also scientifically
important. Without them, a flexible shared random trend and six equally
flexible channel random trends can divide the same low-frequency signal in
many weakly identified ways.

Version 2.2 makes those restrictions explicit rather than hiding them in
example-specific prior edits:

```python
compiled = bx.compile_model(model, data)
priors = bx.identified_factor_priors(
    compiled,
    profile="regularized_triple_gamma",
    smooth_factor=True,          # fix the direct factor-level innovation
    reference_channel="TXm",    # make TXm a pure low-frequency reference
)
```

Alternatives include `fixed_idiosyncratic="all"` for a loading-only
sensitivity fit, `reference_channel=None` for smooth-factor-only
identification, and `smooth_factor=False` for an unrestricted stress test.
These are different models. Agreement in the complete reconstructed
predictors does not imply agreement in the shared/idiosyncratic allocation.

After fitting, `factor_identification_diagnostics()` reports correlations
between each estimated loading and both a factor-like projection and final
change of its idiosyncratic path. Large absolute correlations and
`ridge_flag=True` indicate posterior compensation. They are identification
diagnostics, not substitutes for R-hat, ESS, or simulation recovery.

## Frühwirth--Schnatter non-centred representation

The shared factor is written as

\[
f_t=f_0+t\beta_{f,0}
 +q_{f,\alpha}\widetilde\alpha_{f,t}
 +q_{f,\beta}A_{f,t},
\]

where

\[
\widetilde\alpha_{f,t}
=\widetilde\alpha_{f,t-1}+\varepsilon^{\alpha}_{f,t},\qquad
\widetilde\beta_{f,t}
=\widetilde\beta_{f,t-1}+\varepsilon^{\beta}_{f,t},
\]

\[
A_{f,t}=A_{f,t-1}+\widetilde\beta_{f,t-1},\qquad
\varepsilon^{\alpha}_{f,t},\varepsilon^{\beta}_{f,t}\sim N(0,1).
\]

For each channel,

\[
\alpha_{i,t}=q_{\alpha,i}\widetilde\alpha_{i,t},
\]

and the dynamic dummy-seasonal block has the corresponding unit-innovation
state (\widetilde\gamma_{i,t}) multiplied by (q_{\gamma,i}). Thus

\[
\mu_{i,t}=c_i+\lambda_i
\left(t\beta_{f,0}
+q_{f,\alpha}\widetilde\alpha_{f,t}
+q_{f,\beta}A_{f,t}\right)
+q_{\alpha,i}\widetilde\alpha_{i,t}
+S_{i,t}(\gamma_{0,i},q_{\gamma,i},\widetilde\gamma_{i,t}).
\]

All transition disturbances in this augmented graph have variance one. The
innovation scales are signed coefficients in the observation design; their
absolute values are the process standard deviations. This removes the usual
small-variance funnel from the latent-state transition. A scale near zero
simply suppresses that NCP contribution.

For mixed likelihoods, PGAS runs conditional SMC on this unit-innovation
graph. `proposal="bootstrap"` samples standard normal disturbances directly.
The default `proposal="guided"` tilts those disturbances using local
likelihood curvature and includes the exact prior/proposal weight correction.

The returned semantic state path is reconstructed from the NCP path. Both are
retained:

```python
fit.state_draws                       # centered semantic states
fit.auxiliary_draws["fs_state"]      # unit-innovation NCP states
fit.parameter("signed_sd.channel.TXx.level")
fit.parameter("sd.channel.TXx.level")
```

`asis=True` adds a centered scale update while keeping the semantic path fixed,
then maps it back to the FS coordinates.

## Disturbance parameterization

`parameterization="disturbance"` retains the ordinary centered state graph
but conditions scale updates on standardized disturbances (z_t):

\[
x_t=F x_{t-1}+R\operatorname{diag}(q)z_t,
\qquad z_t\sim N(0,I).
\]

It is useful for sensitivity checks and for factor graphs outside the exact FS
layout. It uses the same joint mixed likelihood, loading updates, observation
parameter updates, and factor horseshoe hierarchy.

## Idiosyncratic regularized horseshoe

Let (s_i=q_{\alpha,i}/r_i) be an idiosyncratic scale standardized by a
channel-specific reference (r_i). The default hierarchy is

\[
s_i\mid\lambda_i,\tau,c
\sim N(0,\tau^2\widetilde\lambda_i^2),
\qquad
\widetilde\lambda_i^2
=\frac{c^2\lambda_i^2}{c^2+\tau^2\lambda_i^2},
\]

with local and global half-Cauchy scales and an inverse-gamma prior on (c^2).
Only the six `channel.<name>.level` scales enter this hierarchy. Shared-factor
and seasonal innovation scales keep their calibrated process priors.

The horseshoe has a continuous spike, not a point mass. It supports the claim
that a channel remains tightly tethered when its idiosyncratic scale is near
zero, but it does not assign a literal posterior probability to
(q_{\alpha,i}=0). Report posterior scale intervals and prior/posterior
overlays rather than calling it exact variable selection.

## Idiosyncratic triple gamma

The full factor model also accepts `triple_gamma` and
`regularized_triple_gamma`. For each selected namespaced idiosyncratic scale,
the standardized signed coefficient follows

\[
u_i\mid r_i,d_i,\phi\sim N(0,\phi r_i/d_i),\quad
r_i\sim\operatorname{Gamma}(a,1),\quad
d_i\sim\operatorname{Gamma}(c,1).
\]

The hierarchy is restricted to `channel.<name>.level` processes, exactly like
the factor horseshoe; common-factor and seasonal process priors are unchanged.
Posterior output stores `triple_gamma.rho.<process>`, where values near one
mean strong shrinkage and values near zero mean little shrinkage. The
regularized profile replaces variance (v) by
(c_0^2v/(c_0^2+v)) and learns (c_0^2) under the configured inverse-gamma slab.

```python
priors = bx.identified_factor_priors(
    compiled,
    profile="regularized_triple_gamma",
    triple_gamma_options={"spike_shape": 0.1, "tail_shape": 0.1},
    smooth_factor=True,
    reference_channel="TXm",
)
```

Triple gamma is continuous shrinkage, not exact model selection. The
horseshoe, Bayesian-lasso, double-gamma, folded/half-t, and Gaussian cases are
members or limits of the unregularized family; the PC prior is kept separate
because its direct exponential-on-SD calibration is not the same finite
hierarchy.

## Likelihood and sampler blocks

At every time and particle, the mixed log weight is

\[
\log w_t=
\sum_{i\in\{\mathrm{TXm},\mathrm{TNm}\}}
\log p_N(y_{i,t}\mid\mu_{i,t})
+\sum_{i\in\{\mathrm{TXx},\mathrm{TXn},\mathrm{TNx},\mathrm{TNn}\}}
\log p_{\mathrm{GEV}}(y_{i,t}\mid\mu_{i,t}).
\]

One MCMC iteration updates the global latent path, static FS coefficients or
disturbance scales, channel observation parameters, estimated loadings, and
horseshoe or triple-gamma hyperparameters. In v2.2, an estimated
Gaussian-channel loading is
not conditioned on the previous idiosyncratic path: a marginal Kalman update
first moves its idiosyncratic scale and a three-state FFBS block then jointly
draws `(c_i, lambda_i, alpha_i[0:T])`. For a GEV channel, the interweaving move

\[
\lambda_i'=\lambda_i+\delta,\qquad
\alpha_{i,t}'=\alpha_{i,t}-\delta f_t
\]

leaves the observation predictor and GEV support exactly unchanged. A
path-conditional loading proposal remains as the fallback when the channel
deviation SD is fixed at zero. PGAS is the exact-invariant state kernel;
Laplace is explicitly marked approximate.

## Results and interpretation

```python
fit.factor("common")
fit.factor_rate_summary("common", 1950, 2022)
fit.factor_probabilities("common", start_year=1950, end_year=2022)
fit.loading_draws("common", "TXx", original_scale=True)
fit.loading_probability("common", "TXx", threshold=1.0)
fit.reconstructed_state("TXx")
fit.normalized_factor(slice(0, 30 * 12), "common")
parts = fit.channel_decomposition(
    "TXx", "common", baseline=slice(0, 30 * 12)
)
parts["baseline"], parts["shared"], parts["deviation"], parts["seasonal"]
parts["predictor"]
fit.channel_rate_draws("TXx", 1950, 2022)
fit.return_level_draws(100, channel="TXx")
fit.idiosyncratic_innovation_draws("TXx")
fit.loading_deviation_correlation("TXx", summary="factor_projection")
fit.factor_identification_diagnostics()

fit.plot("factor_decomposition", baseline=slice(0, 30 * 12))
fit.plot("parameter_density", parameters=["loading.common.TXx"])
fit.plot("traces")
fit.plot("acf")
fit.plot("loading_deviation", channel="TXx")
fit.plot("identification")
fit.plot("idiosyncratic_innovations", channel="TXx")
fit.plot("factor_decomposition", save="figures/decomposition.png")
```

A loading above one means that the `TXx` contribution associated with the
shared factor is more sensitive than the anchored `TXm` contribution. It is
not by itself proof that the complete `TXx` trend is faster, because the
idiosyncratic deviation may reinforce or offset it. Base total-channel claims
on reconstructed-state or channel-rate draws.

The semantic channel-level state is `c_i + alpha_i,t`, not `alpha_i,t` alone.
`channel_decomposition()` performs the subtraction, applies lower-tail
orientation when needed, compensates the intercept for baseline factor
centering, and checks that all returned pieces reconstruct the predictor.
The decomposition plot places credible bands on the predictor, shared, and
idiosyncratic panels. Simulation truths can be overlaid on all new recovery
plots.

## Scope

The release does not add:

- unrestricted contemporaneous observation covariance;
- residual Gaussian-copula dependence for mixed families;
- correlated idiosyncratic innovations;
- a daily latent process whose monthly means and extrema are induced
  functionals;
- trigonometric seasonality in the specialized FS factor layout.

The supported factor-seasonal block is independent dynamic dummy seasonality
for each series, matching the current monthly Uccle implementation.
