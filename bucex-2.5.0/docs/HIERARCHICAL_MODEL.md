# Hierarchical componentwise structural model

## Scientific target

The hierarchy asks whether related temperature summaries tend to use the same
kind of level, slope, and seasonal evolution. Every series keeps its own latent
path and observation model. Pooling is therefore about recurring *structure*
and, optionally, active innovation magnitudes—not a shared warming curve.

For channel `i`,

\[
y_{it}\mid\eta_{it},\vartheta_i\sim p_i(y_{it}\mid\eta_{it},\vartheta_i),
\qquad \eta_{it}=\mu_{it}+\gamma_{it},
\]

and

\[
\mu_{i,t+1}=\mu_{it}+\beta_{it}+s_{i,\mu}z^\mu_{i,t+1},\qquad
\beta_{i,t+1}=\beta_{it}+s_{i,\beta}z^\beta_{i,t+1}.
\]

Initial level, slope, and seasonal coefficients are posterior parameters.
Data-based regressions only initialize the chains.

## Componentwise SSVS

The v2.5 primary model treats decisions separately:

| Component | States | Meaning |
|---|---|---|
| level | fixed, dynamic | level innovation exactly zero or active |
| slope | zero, fixed, dynamic | no slope; constant estimated slope; evolving slope |
| season | fixed, dynamic | stable or evolving annual pattern |

This grammar directly represents the scientific alternatives discussed in the
TXx introduction. A zero slope is not confused with a fixed nonzero slope. For
monthly temperatures, seasonality is known to be present; its zero state is
excluded before seeing these observations.

The four-class joint level/slope innovation space remains available through
`model_space="joint_trend"`. It is a sensitivity model in v2.5.0.

## Shared selection probabilities

For component `k` and channel `i`,

\[
M_{ik}\mid\boldsymbol\pi_k\sim
\operatorname{Categorical}(\boldsymbol\pi_k),\qquad
\boldsymbol\pi_k\sim\operatorname{Dirichlet}(\boldsymbol a_k).
\]

There is one vector for level, one for slope, and one for seasonality. Given
these population vectors, channels still choose different states. With only six
channels, the posterior hyperprior uncertainty should remain visible rather
than being summarized as a hard population label.

The primary helper is:

```python
prior = bx.componentwise_hierarchical_prior(pool="selection")
```

Its explicit equivalent is:

```python
prior = bx.HierarchicalPrior(
    pool="selection",
    model_space="componentwise",
    level_states=("fixed", "dynamic"),
    trend_states=("zero", "fixed", "dynamic"),
    season_states=("fixed", "dynamic"),
    level_concentration=(1, 1),
    trend_concentration=(1, 1, 1),
    season_concentration=(1, 1),
    coefficient_scale={"level": 0.03, "trend": 0.0002, "season": 0.03},
)
```

## Optional shared slabs

Conditional on a dynamic allocation,

\[
s_{ik}\mid\tau_k\sim N(0,c_k^2\tau_k^2),\qquad
\tau_k\sim\operatorname{half\text{-}t}_{\nu}(0,A_k).
\]

`c_k` is component-specific because monthly slope innovations are on a very
different scale from level and seasonal innovations. Sharing `tau_k` can be
helpful only if the channels are exchangeable for active magnitudes after this
calibration.

- `pool="selection"`: share probabilities; fixed slab calibration;
- `pool="slab"`: force available innovations active; share slab multipliers;
- `pool="both"`: share both probabilities and slab multipliers.

The latter two are sensitivity analyses. A very diffuse slab can spuriously
favor the spike through Bartlett's paradox, so calibration and prior-predictive
simulation are part of the analysis.

## FS noncentring and signs

In the Frühwirth--Schnatter parameterization, each active innovation is written
as a signed coefficient times a unit-innovation path. Replacing both signs
leaves the predictor unchanged. The sampler performs paired sign switches and
stores the maximum reconstruction error. Interpret `abs(s)` or the reported
process SD, not the signed coefficient.

## What is and is not pooled

Shared:

- component allocation probabilities;
- optionally, one slab multiplier per component.

Not shared:

- latent level, slope, or seasonal paths;
- initial states;
- observation SD/GEV scale and shape;
- residual shocks or a copula;
- a factor or common trajectory.

The hierarchy therefore introduces prior dependence, not a multivariate
observation likelihood. Any residual/cross-series dependence extension must be
declared as a separate model.

## Summaries

```python
fit.component_probabilities()
fit.structural_model_probabilities()
fit.hierarchical_probabilities()
fit.hierarchical_slab_summary()
fit.component_transition_summary()
fit.channel_rate_summary("TXx")
```

Report marginal component probabilities and induced joint structures. The
former explain the model; the latter reveal whether marginal ambiguity is
concentrated in a small number of scientifically distinct combinations.

## Required sensitivity

1. independent versus pooled selection;
2. `selection` versus `slab`/`both`;
3. tighter and wider defensible slab scales;
4. Dirichlet concentration;
5. record start (1980 and the longest defensible record);
6. PGAS particle count;
7. held-out predictive scores and PIT.
