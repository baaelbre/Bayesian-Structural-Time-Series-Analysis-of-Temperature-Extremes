# Hierarchical structural model

## Scientific target

The model asks whether several related temperature summaries tend to require
the same *types* and *magnitudes* of structural evolution. Each summary keeps
its own latent path. Pooling therefore supports statements such as:

- dynamic levels are common across the six summaries;
- evolving slopes are uncommon or weak;
- the annual cycle is present in every summary but may be time-varying in only
  some of them;
- when a component is dynamic, its typical innovation scale is similar across
  summaries.

It does not assert that two summaries share the same monthly shock or the same
warming curve. That distinction is physically useful: means, hot extremes,
cold extremes, upper tails, and lower tails can react differently while still
informing a population-level conclusion about structural change.

## Channel model

For channel `i`,

\[
y_{it}\mid\eta_{it},\vartheta_i
  \sim p_i(y_{it}\mid\eta_{it},\vartheta_i),
\]

where `p_i` is Gaussian or GEV. Lower-tail GEV series are sign-transformed
internally and mapped back on output. The predictor is

\[
\eta_{it}=\mu_{it}+\gamma_{it}.
\]

The local linear trend follows

\[
\begin{aligned}
\mu_{i,t+1} &= \mu_{it}+\beta_{it}+s_{i,\mu}z_{i,t+1}^{\mu},\\
\beta_{i,t+1} &= \beta_{it}+s_{i,\beta}z_{i,t+1}^{\beta},
\end{aligned}
\qquad z\sim N(0,1).
\]

Dummy seasonality uses the usual sum-to-zero transition and a signed seasonal
innovation coefficient `s_i,gamma` when dynamic.

Initial level, slope, and seasonal coefficients are posterior parameters. A
seasonally adjusted linear regression supplies only the chain start. The
posterior names are `initial.channel.<name>.level`,
`initial.channel.<name>.slope`, and `initial.channel.<name>.seasonal`.

## Structural selection

Let `M_ik` describe the state of component `k` in channel `i`. Available states
are:

| Component | Default states | Interpretation |
|---|---|---|
| level | fixed, dynamic | intercept-only level or stochastic level |
| trend | zero, fixed, dynamic | no trend, constant trend, or evolving trend |
| season | fixed, dynamic | stable or evolving annual cycle |

The level is always present. Monthly seasonality is also present by default;
`zero` is deliberately excluded because its existence is known physically.
The interesting question is whether the seasonal pattern changes through time.

With `pool="selection"` or `pool="both"`,

\[
M_{ik}\mid\boldsymbol\pi_k
  \sim \operatorname{Categorical}(\boldsymbol\pi_k),
\qquad
\boldsymbol\pi_k\sim\operatorname{Dirichlet}(\boldsymbol a_k).
\]

The channel allocations update conditional on `pi_k`; the population vector
then updates using counts across channels. This is partial pooling. It neither
forces identical allocations nor estimates each series in isolation.

The posterior `pi_k` has a direct interpretation: for a new exchangeable
summary from the same scientific collection, it is the population prevalence
of each structural state. With only six channels, the Dirichlet prior remains
visible and should be reported in sensitivity analyses.

## Pooled normal slab

When component `k` is dynamic, its signed FS coefficient has a normal slab:

\[
s_{ik}\mid\tau_k,M_{ik}=D
  \sim N(0,c_k^2\tau_k^2).
\]

With `pool="slab"` or `pool="both"`,

\[
\tau_k\sim \operatorname{half\text{-}t}_{\nu}(0,A_k).
\]

The base scale `c_k` sets units and plausible order of magnitude. The learned
positive multiplier `tau_k` says whether dynamic variation is generally
smaller or larger than that calibration. A half-t hyperprior regularizes the
small collection while retaining enough tail mass for genuinely dynamic
series.

The three pooling modes answer different questions:

- `selection`: Which structures recur? Slab widths remain fixed.
- `slab`: How large are innovations across related series? Every available
  component is dynamic.
- `both`: Which structures recur, and how large are their innovations when
  active?

For the main Uccle analysis, pooled selection with a clearly calibrated normal
slab is the simplest primary model. Pooled slab or both can be reported as
sensitivity analyses. This separates uncertainty about *whether* a component
is dynamic from uncertainty about *how dynamic* it is.

## Signed coefficients and sign switching

In the FS parameterization, `s z_t` is unchanged when both `s` and the full
unit-innovation path `z` change sign. The sign is therefore not identified,
although `abs(s)` is. Each iteration may perform a random paired sign switch.
The implementation reconstructs the complete predictor before and after the
move and raises if they differ beyond numerical tolerance. Counts and maximum
invariance errors are stored as diagnostics.

Report process standard deviations using `sd.channel.<series>.<component>`.
Signed values exist for computation and diagnostics, not physical
interpretation.

## API

```python
import bucex as bx

model = bx.MultiSeriesModel(
    channels=tuple(
        bx.Channel(
            name=name,
            observation=bx.Gaussian(),
            components=(bx.LocalLinearTrend(), bx.DummySeasonal(12)),
        )
        for name in data.columns
    )
)

prior = bx.HierarchicalPrior(
    pool="selection",
    level_states=("fixed", "dynamic"),
    trend_states=("zero", "fixed", "dynamic"),
    season_states=("fixed", "dynamic"),
    level_concentration=(1, 1),
    trend_concentration=(1, 1, 1),
    season_concentration=(1, 1),
    coefficient_scale={"level": 0.03, "trend": 0.0002, "season": 0.03},
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
)
```

Useful summaries:

```python
fit.component_probabilities()
fit.structural_model_probabilities(channel="TXm")
fit.most_probable_structure(channel="TXm")
fit.hierarchical_probabilities()
fit.hierarchical_slab_summary()
fit.component_transition_summary()
fit.channel_rate_summary("TXm")
```

## Interpretation and limitations

The hierarchy assumes channels are exchangeable for the selected structural
feature after their family, tail orientation, observation scale, and latent
path are accounted for. This is plausible for a carefully chosen set of
temperature summaries at one station, but it should not be automatic for an
arbitrary collection.

Conditional observation residuals are independent across channels. The model
does not include contemporaneous residual correlation. Posterior uncertainty
in population probabilities reflects only the declared hierarchy and channel
likelihoods.

An allocation near 0.5 is not a convergence failure when R-hat, ESS, chain
overlap, switching, and ACFs are satisfactory. For the 1980--2023 Uccle
window, that uncertainty is scientifically plausible: roughly 44 years can
identify average warming much more strongly than it can distinguish a very
small evolving slope from a fixed slope or a slowly evolving seasonal pattern
from a fixed one.

## Recommended checks

1. Verify all scientific R-hat values and ESS values.
2. Inspect traces and ACFs of process SDs, observation parameters, population
   probabilities, and slab multipliers.
3. Check that each non-degenerate SSVS allocation actually switches.
4. Compare 1980-start fits with the longest defensible record.
5. Vary Dirichlet concentrations and slab base scales.
6. Compare pooled selection, pooled slab, pooled both, and independent fits.
7. For GEV channels, inspect particle ESS, ancestor diversity, path changes,
   restored iterations, and support margins.
8. Validate prediction with leave-future-out scores and PITs.

Computational diagnostics establish that the posterior was explored. Prior and
record-length sensitivity establish how strongly the scientific conclusion is
supported.
