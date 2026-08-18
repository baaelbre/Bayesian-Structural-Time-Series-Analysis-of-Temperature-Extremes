# Hierarchical structural model

## Scientific target

The multiseries model asks whether related temperature summaries tend to use
the same *kind* of trend evolution, whether their annual cycles are stable or
changing, and—optionally—whether active innovations have a common magnitude.
Each series retains its own latent path and observation distribution. The model
therefore borrows strength without asserting a common monthly shock or a single
warming curve.

This distinction is central for the six Uccle summaries. Mean, minimum,
maximum, upper-tail, and lower-tail temperatures can evolve differently while
still informing a population statement about which structural mechanisms recur.

## Channel model

For channel `i`,

\[
y_{it}\mid\eta_{it},\vartheta_i
  \sim p_i(y_{it}\mid\eta_{it},\vartheta_i),
\qquad
\eta_{it}=\mu_{it}+\gamma_{it},
\]

where `p_i` is Gaussian or GEV. Lower-tail GEV series are sign-transformed
internally and mapped back on output. The local linear state equations are

\[
\begin{aligned}
\mu_{i,t+1} &= \mu_{it}+\beta_{it}+s_{i,\mu}z^\mu_{i,t+1},\\
\beta_{i,t+1} &= \beta_{it}+s_{i,\beta}z^\beta_{i,t+1},
\end{aligned}
\qquad z\sim N(0,1).
\]

Dummy seasonality follows the usual sum-to-zero recursion with a signed
innovation coefficient `s_i,gamma` when dynamic. Initial level, initial slope,
and the initial seasonal coefficients are posterior parameters. Data-informed
regressions provide only chain starting values.

## The four joint trend classes

The default `model_space="joint_trend"` always estimates a slope. It selects
which of the level and slope *innovation variances* are nonzero:

| Class | `s_level` | `s_trend` | State interpretation |
|---|---:|---:|---|
| `linear_trend` | 0 | 0 | constant slope, deterministic around initial state |
| `rw1_drift` | nonzero | 0 | random-walk level around constant drift |
| `rw2_smooth_trend` | 0 | nonzero | changing slope with smooth integrated level |
| `local_linear_trend` | nonzero | nonzero | level and slope both receive shocks |

Thus the model never equates “no slope innovation” with “no warming.” A linear
trend can forecast continued warming through its estimated initial slope. The
four classes distinguish how departures from that trend accumulate:

- an RW1 level shock changes the level once and is then carried forward;
- an RW2 slope shock changes the rate, so its effect on level accumulates;
- the full local linear trend permits both mechanisms.

The older componentwise `zero`/`fixed`/`dynamic` slope allocation remains an
explicit sensitivity model via `model_space="componentwise"`. It is not the
default scientific model because an exact no-slope state gives “no continued
warming” a special prior atom and complicates forecast interpretation.

## Seasonality is present

For monthly temperature, the annual cycle is established physical knowledge.
The default seasonal states are therefore only:

- `fixed`: eleven estimated seasonal coefficients and no seasonal innovation;
- `dynamic`: the same annual cycle plus stochastic seasonal innovations.

The model asks whether the seasonal pattern evolves, not whether seasonality
exists. Absence of seasonality is available only in the legacy componentwise
space for applications where it is scientifically plausible.

## Hierarchical SSVS

Let `M_i` be the joint trend class of series `i`. With pooled selection,

\[
M_i\mid\boldsymbol\pi_T
  \sim \operatorname{Categorical}(\boldsymbol\pi_T),
\qquad
\boldsymbol\pi_T\sim\operatorname{Dirichlet}(\boldsymbol a_T).
\]

The four entries of `pi_T` correspond to the four classes above. Seasonal
fixed/dynamic indicators have a separate beta/Dirichlet hierarchy. Conditional
on the population probabilities, every channel remains free to choose a
different class. Conditional on the channel allocations, the population
probabilities update from the counts across channels.

This is partial pooling of *model structure*. It can answer:

- Which trend mechanisms recur across the six summaries?
- Is dynamic level variation more prevalent than evolving slope variation?
- Is changing seasonality common or confined to particular summaries?
- How uncertain is the prevalence for a new exchangeable summary?

It does not posit a shared instantaneous shock. Dependence is introduced only
through the population-level structural probabilities and/or slab scales.

## Pooled normal slab

For active component `k`, the signed FS coefficient has a normal slab,

\[
s_{ik}\mid\tau_k,M_i \sim N(0,c_k^2\tau_k^2).
\]

`c_k` is a scientifically calibrated base scale. With `pool="slab"` or
`pool="both"`,

\[
\tau_k\sim\operatorname{half\text{-}t}_{\nu}(0,A_k).
\]

The shared multiplier pools innovation magnitudes rather than paths. The three
pooling modes are:

- `selection`: pool trend/season allocations; keep slab widths fixed;
- `slab`: keep available innovations active; pool their magnitudes;
- `both`: pool allocations and active magnitudes.

For the primary Uccle analysis, `selection` with a calibrated normal slab is
the clearest model. `slab` and `both` are valuable sensitivity analyses.

## Slab calibration and Bartlett's paradox

A slab should express plausible structural change. It should not be made very
wide in the hope of being “noninformative.” In model comparison, a very wide
slab spreads prior density across parameter values that the likelihood does
not support. The integrated likelihood of the dynamic class can then fall as
the slab widens, spuriously favoring the simpler class. This is Bartlett's
paradox.

Calibrate in observable units:

```python
scales = bx.calibrate_structural_scales(
    horizon_years=30,
    max_level_change=2.0,
    max_rate_change_per_decade=0.3,
    max_seasonal_innovation=0.2,
    probability=0.90,
)

implications = bx.structural_scale_implications(
    level_sd=scales["level"],
    slope_sd=scales["trend"],
    seasonal_sd=scales["season"],
    horizon_years=30,
    probability=0.90,
)
```

The first bound concerns accumulated RW1 level change; the second concerns
change in the warming rate over a decade; the seasonal bound is one-step and
must be followed by prior-predictive simulation of complete seasonal paths.
For a learned half-t multiplier, use
`half_student_t_scale_for_median(1, df=4)` so that multiplier one is the prior
median and `c_k` keeps its direct interpretation.

Document the elicited bounds, simulate complete predictors and forecasts, and
repeat the analysis under at least one tighter and one wider defensible scale.

## Signed coefficients and sign switching

In the FS parameterization, `s z_t` is unchanged when both `s` and the full
unit-innovation path `z` change sign. The sign is unidentified although
`abs(s)` is scientific. Random paired sign switches are therefore part of the
sampler. Every switch is checked by reconstructing the complete predictor;
non-invariance raises an error. Report `sd.channel.<series>.<component>`, not
the signed coefficient.

## Public API

```python
import bucex as bx

model = bx.MultiSeriesModel(
    channels=tuple(
        bx.Channel(
            name,
            bx.Gaussian(),
            (bx.LocalLinearTrend(), bx.DummySeasonal(12)),
        )
        for name in data.columns
    )
)

prior = bx.HierarchicalPrior(
    pool="selection",                 # "slab" or "both"
    model_space="joint_trend",
    trend_model_concentration=(1, 1, 1, 1),
    season_states=("fixed", "dynamic"),
    season_concentration=(1, 1),
    coefficient_scale=scales,
    slab_df=4,
    slab_prior_scale={"level": 1, "trend": 1, "season": 1},
)

fit = bx.fit(
    data,
    model,
    priors=prior,
    parameterization="fs",
    engine="auto",
    asis=False,
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, progress=True),
    particles=bx.Particles(n=512, proposal="guided"),
    hierarchical_sampler=bx.HierarchicalSampler(
        initializer="laplace",
        channel_workers=2,
    ),
)
```

Useful summaries include:

```python
fit.hierarchical_trend_model_probabilities()
fit.component_probabilities()
fit.structural_model_probabilities(channel="TXm")
fit.hierarchical_probabilities()
fit.hierarchical_slab_summary()
fit.component_transition_summary()
fit.channel_rate_summary("TXm")
```

## Exploratory Laplace and exact PGAS

All-Gaussian hierarchies use exact FFBS. A mixed or all-GEV hierarchy supports
two engines:

- `engine="laplace"`: fast exploratory iterated pseudo-Gaussian updates;
- `engine="pgas"`: exact-invariant conditional SMC with ancestor sampling.

Laplace is appropriate for debugging, rough sensitivity screening, and
initialization. It is not the publication posterior. A complete Laplace fit can
be passed directly as the PGAS start:

```python
screen = bx.fit(data, model, priors=prior, engine="laplace", ...)
exact = bx.fit(data, model, priors=prior, engine="pgas", init=screen, ...)
```

The exported start includes channel parameters, latent paths, population
probabilities, and slab scales. Compatibility of model and observations is
validated. PGAS then runs its full warmup and targets the same posterior it
would target from a data-based start.

GEV particle/path likelihood calculations are vectorized. Conditional channel
updates may run in threads with `channel_workers>1`; deterministic child seeds
preserve reproducibility for a fixed worker configuration. For publication,
independent chains in separate HPC processes remain preferable to one process
running many chains.

## Interpretation and limitations

The hierarchy assumes channels are exchangeable for the selected structural
features after family, tail orientation, observation scale, and latent path are
accounted for. With six channels, the Dirichlet and half-t hyperpriors remain
visible and must be reported.

Observation residuals are conditionally independent across channels. The model
does not contain contemporaneous residual correlation or a spatial process.
It also keeps observation scale static in the primary specification.

An allocation split is not a convergence failure when R-hat, ESS, traces,
switching, and ACFs are satisfactory. It is posterior uncertainty about model
structure. Report probabilities and model-averaged trajectories rather than
forcing a hard classification.

## Required checks

1. R-hat, bulk ESS, traces, and ACFs for all scientific and hierarchy parameters.
2. Transition counts for every non-degenerate allocation.
3. Sensitivity to record start, Dirichlet concentration, and slab calibration.
4. Comparison of independent, pooled-selection, pooled-slab, and pooled-both fits.
5. For PGAS: particle ESS, unique ancestors, path-update fraction, restorations,
   support margins, and stability after increasing particle count.
6. Leave-future-out log score, CRPS/tail-weighted scores, and PIT diagnostics.
7. Four independently seeded final chains, preferably as separate HPC jobs.
