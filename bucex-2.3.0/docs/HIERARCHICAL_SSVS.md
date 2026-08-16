# Hierarchical SSVS for related series

## Scientific role

`MultiSeriesModel` is the middle design between six unrelated fits and a
dynamic factor. It is appropriate when the outcomes are scientifically
related and should inform one another about *which kinds of dynamics are
present*, but a single common trajectory is not assumed.

It can answer:

- how prevalent dynamic level, trend, and seasonal components are across the
  collection;
- which summaries are static, deterministic, or genuinely time-varying;
- whether a specific summary differs from the population tendency;
- how large dynamic innovations tend to be after partial pooling;
- complete-predictor rates and forecasts for every series.

It does not identify a common warming path, loadings, or synchronous shocks.
Those are `FactorModel` estimands. It also does not model residual/copula
dependence between channels.

## Structural model

Every channel declares a local-linear-trend grammar and optional dummy
seasonality. In the FS parameterization the complete predictor is a regression
on static coefficients and standardized latent paths. For component `k` in
series `i`, SSVS introduces allocation `M[i,k]`:

- level: `fixed` or `dynamic`;
- trend: `zero`, `fixed`, or `dynamic`;
- seasonality: `zero`, `fixed`, or `dynamic`.

The exact semantics are:

| Allocation | Static coefficient | Signed innovation coefficient |
|---|---|---|
| zero | zero | zero |
| fixed | sampled | exactly zero |
| dynamic | sampled | sampled |

The level cannot be absent because every channel needs a location. A fixed
level is a static intercept. A fixed trend is a deterministic linear slope. A
fixed seasonal component is a static sum-to-zero seasonal vector.

## Hierarchy

For each component, series allocations share a categorical probability vector:

```text
M[i,k] | pi[k] ~ Categorical(pi[k])
pi[k]          ~ Dirichlet(a[k])
```

Conditional on `dynamic`, the signed FS coefficient has a shared scale:

```text
s[i,k] | tau[k] ~ Normal(0, coefficient_scale[k]^2 * tau[k]^2)
tau[k]          ~ half-Student-t(df, slab_prior_scale[k])
```

The allocation probabilities receive conjugate Dirichlet updates. Each
positive slab multiplier receives an exact stepping-out slice update on the
log scale. These are population parameters, not post-processing summaries of
six separate fits.

Uniform Dirichlet concentrations are deliberately neutral defaults. The
component coefficient scales set units and should be calibrated to the
observation interval. For monthly temperature data the defaults are `0.03`
for level/seasonal innovation coefficients and `0.0002` for slope innovations.
The multipliers are learned. A sensitivity analysis should change
concentrations and coefficient scales separately so prevalence and magnitude
assumptions are not confounded.

## Inference

- All Gaussian channels use exact FS Gaussian FFBS plus exact enumerated
  Gaussian model-space updates.
- If any channel is GEV, the joint run uses PGAS. GEV structural moves use
  Laplace information only to form a full-support proposal; the exact GEV
  likelihood, normalized priors, and forward/reverse proposals determine
  acceptance. Active coefficients are refreshed by elliptical slice sampling.
- Gaussian and GEV nuisance parameters retain their channel-specific priors.
- Hierarchy updates use allocations from all eligible channels in the same
  Gibbs iteration.

The product likelihood is joint, while channels are conditionally independent
given their own state paths and the population hierarchy. The sampler stores
restoration, particle, model-move, and chain diagnostics.

Hierarchical SSVS requires `parameterization="fs"` and `asis=False`. Mixed/GEV
models require `engine="pgas"`; there is no approximate Laplace target for this
joint hierarchy. Aligned finite observations are currently required.

## Sign symmetry

The sign of an FS coefficient and its standardized path is unidentified:

```text
s * z == (-s) * (-z)
```

Every univariate, hierarchical, and eligible factor FS iteration independently
randomizes each available sign. The centered path/complete predictor is
computed before and after the move and must agree to numerical tolerance.
`sampler_diagnostics` stores switch counts and
`draws_aux["sign_invariance_error"]`; scientific innovation output is the
absolute `sd.*`, not `signed_sd.*`.

## API

```python
import bucex as bx

model = bx.MultiSeriesModel(
    channels=(
        bx.Channel("mean", bx.Gaussian(),
                   (bx.LocalLinearTrend(), bx.DummySeasonal(12))),
        bx.Channel("maximum", bx.GEV(),
                   (bx.LocalLinearTrend(), bx.DummySeasonal(12))),
        bx.Channel("minimum", bx.GEV(),
                   (bx.LocalLinearTrend(), bx.DummySeasonal(12)),
                   tail="lower"),
    )
)

fit = bx.fit(
    data,
    model,
    priors="hierarchical_ssvs",
    engine="pgas",
    parameterization="fs",
    asis=False,
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, progress=True),
    particles=bx.Particles(n=1_024),
)
```

Important result methods are:

```python
fit.component_probabilities()          # all series, or channel="mean"
fit.component_transition_summary()
fit.structural_model_probabilities(channel="mean")
fit.most_probable_structure(channel="mean")
fit.hierarchical_probabilities()
fit.hierarchical_slab_summary()
fit.channel_eta_draws("mean")
fit.channel_rate_summary("mean")
fit.forecast(12)
fit.save("results/hierarchy.bucex")
```

Every plot accepts `save=`:

```python
fit.plot("channel", channel="mean", save="figures/mean.png")
fit.plot("component_probabilities", save="figures/components.png")
fit.plot("hierarchy", save="figures/hierarchy.png")
fit.plot("process_sd", save="figures/process_sds.png")
fit.plot("traces", save="figures/traces.png")
fit.plot("acf", save="figures/acf.png")
```

## What to report

Report both series allocations and population parameters. A population mean
probability is not proof that every series has the same structure. For each
series/component, show posterior zero/fixed/dynamic probabilities and actual
switch counts. For continuous hierarchy parameters, show R-hat, ESS, traces,
and ACF. For mixed models, report particle ESS, ancestor diversity, changed
path fraction, GEV support/restoration counts, and model-move acceptance.

Constant SSVS allocations have undefined R-hat and ESS. A constant chain can
mean overwhelming posterior mass or failed switching; distinguish these using
multiple initial states, switching diagnostics, prior sensitivity, and
simulation recovery.

## Hierarchy, factor, or separate fits?

| Question | Preferred design |
|---|---|
| What happens in each summary without borrowing? | Separate `Model` fits |
| Which structural dynamics recur across summaries? | `MultiSeriesModel` |
| Is there a common warming path and how strongly does each summary respond? | `FactorModel` |

For heterogeneity *around a common warming signal*, use the factor as the
primary model and hierarchical SSVS as a structural sensitivity analysis. For
heterogeneity in the presence/absence of dynamics itself, hierarchical SSVS
can be primary. They are complementary models, not interchangeable samplers.
