# bucex 2.4.0

`bucex` fits Bayesian structural time-series models to one series or to several
related series. Version 2.4 has one public workflow and one scientific focus:

- `Model` fits a single Gaussian or GEV series;
- `MultiSeriesModel` gives every series its own latent level, slope, and
  seasonal path while partially pooling structural decisions, innovation-slab
  magnitudes, or both.

The multiseries model borrows strength without forcing physically different
temperature summaries to follow an identical trajectory. This makes it useful
for asking whether changes in location, rate, and seasonality recur across a
collection of summaries while preserving series-specific evolution.

## Install

```bash
python -m pip install .
```

Add plotting and test dependencies when needed:

```bash
python -m pip install ".[plot,test]"
```

## One series

```python
from pathlib import Path

import bucex as bx

model = bx.Model(
    bx.Gaussian(),
    (bx.LocalLinearTrend(), bx.DummySeasonal(12)),
    name="monthly temperature",
)

fit = bx.fit(
    temperature,
    model,
    priors="normal",
    parameterization="fruehwirth_schnatter",
    engine="auto",
    asis=True,
    mcmc=bx.MCMC(
        draws=2_000,
        warmup=2_000,
        chains=4,
        seed=101,
        progress=True,
    ),
)

print(fit.plan)
print(fit.diagnostics()["parameters"])
print(fit.rate_summary())
print(fit.forecast(12).summary())

fit.plot("predictor", save="figures/predictor.png")
fit.plot("level_slope", save="figures/level_slope.png")
fit.plot("process_sd", save="figures/process_sd.png")
fit.plot("traces", save="figures/traces.png")
fit.plot("acf", save="figures/acf.png")
```

For GEV observations, use `bx.GEV()` or `family="gev"`. `engine="auto"`
uses Laplace for a quick univariate GEV fit. Use `engine="pgas"` for an
exact-invariant particle MCMC kernel and inspect its particle diagnostics.

## Several related series

The model for channel `i` is

\[
y_{it}\mid\eta_{it},\vartheta_i \sim p_i(y_{it}\mid\eta_{it},\vartheta_i),
\qquad
\eta_{it}=\mu_{it}+\gamma_{it},
\]

with a local linear trend

\[
\mu_{i,t+1}=\mu_{it}+\beta_{it}+s_{i,\mu}z_{i,t+1}^{\mu},
\qquad
\beta_{i,t+1}=\beta_{it}+s_{i,\beta}z_{i,t+1}^{\beta}.
\]

The signed innovation coefficients are used in the Frühwirth--Schnatter
non-centred parameterization. Their signs are unidentified, so the sampler
performs random joint sign switches of each coefficient and its unit-variance
state path. Every move is checked numerically to preserve the complete linear
predictor.

Related channels can share two kinds of information:

1. `pool="selection"`: learn population probabilities for structural states;
2. `pool="slab"`: keep components dynamic and learn one normal-slab multiplier
   per component;
3. `pool="both"`: learn the structural probabilities and slab multipliers.

For component `k`, pooled selection uses

\[
M_{ik}\mid\boldsymbol\pi_k\sim
\operatorname{Categorical}(\boldsymbol\pi_k),\qquad
\boldsymbol\pi_k\sim\operatorname{Dirichlet}(\boldsymbol a_k).
\]

Conditional on `M_ik = dynamic`, pooled slab magnitude uses

\[
s_{ik}\mid\tau_k\sim N(0,c_k^2\tau_k^2),\qquad
\tau_k\sim\operatorname{half\text{-}t}_{\nu}(0,A_k).
\]

The default monthly hierarchy treats seasonality as known to exist: its states
are `fixed` and `dynamic`, not `zero`, `fixed`, and `dynamic`. This separates
the physically obvious annual cycle from the scientific question of whether
that cycle changes through time.

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
    ),
    name="related temperature summaries",
)

hierarchy = bx.HierarchicalPrior(
    pool="selection",       # or "slab" / "both"
    slab="normal",
    season_states=("fixed", "dynamic"),
    coefficient_scale={
        "level": 0.03,
        "trend": 0.0002,
        "season": 0.03,
    },
)

fit = bx.fit(
    data,
    model,
    priors=hierarchy,
    parameterization="fs",
    engine="auto",          # FFBS if all Gaussian; PGAS otherwise
    asis=False,
    mcmc=bx.MCMC(
        draws=2_000,
        warmup=2_000,
        chains=4,
        seed=240,
        progress=True,
    ),
    particles=bx.Particles(n=512, proposal="guided"),
)

print(fit.component_probabilities())
print(fit.hierarchical_probabilities())
print(fit.hierarchical_slab_summary())
print(fit.component_transition_summary())
print(fit.channel_rate_summary("TXm"))

fit.plot("channel", channel="TXm", save="figures/TXm.png")
fit.plot("component_probabilities", save="figures/allocations.png")
fit.plot("hierarchy", save="figures/hierarchy.png")
```

The string shortcuts are `pooled_selection`, `pooled_slab`, and `pooled_both`.
Passing `HierarchicalPrior` is preferred in publication scripts because it
makes all scientific assumptions visible.

## Initial level, slope, and season

Initial level and slope are posterior parameters, not fixed preprocessing
constants. Data-informed seasonally adjusted regressions provide chain starts;
the sampler subsequently updates `initial.level`, `initial.slope`, and the
initial seasonal coefficients. Multiseries names are scoped, for example
`initial.channel.TXm.level` and `initial.channel.TXm.slope`.

`fit.plot("level_slope")` compares the latent level with seasonally adjusted
observations. Plotting the raw monthly values against a deseasonalized level
would make the level appear artificially flat.

## Innovation priors

Univariate FS fits retain these profiles:

- `normal`: signed normal priors; a clear baseline and the slab underlying SSVS;
- `pc`: exponential priors on process standard deviations;
- `ssvs`: exact structural point masses with a normal dynamic slab;
- `regularized_horseshoe`;
- `triple_gamma` and `regularized_triple_gamma`.

Hierarchical fits deliberately use normal dynamic slabs. This keeps the pooled
model interpretable: selection answers whether a component is zero, fixed, or
dynamic, and the half-t hyperprior answers how large dynamic innovations tend
to be across series.

## Inference contract

| Data/model | `engine="auto"` | State update | Posterior contract |
|---|---|---|---|
| Univariate Gaussian | FFBS | exact Gaussian FFBS | exact |
| Univariate GEV | Laplace | iterated pseudo-Gaussian FFBS | approximate |
| Univariate GEV with `engine="pgas"` | PGAS | conditional SMC with ancestor sampling | exact-invariant |
| All-Gaussian multiseries | FFBS | channel FFBS inside one hierarchical Gibbs sampler | exact |
| Mixed or all-GEV multiseries | PGAS | channel PGAS inside one hierarchical Gibbs sampler | exact-invariant |

Laplace is a useful screening tool for a univariate GEV model. It is not a
drop-in replacement for mixed hierarchical inference; multiseries GEV models
use PGAS.

Version 2.4 hardens guided disturbance PGAS on singular transition support. A
conditioned trajectory is first projected onto the exact affine support, the
disturbance recovery uses the scaled transition loading, and an optional
ancestor move that has no valid candidate safely retains its conditioned
predecessor. This avoids the former `All particle weights are zero` failure
without changing the invariant target.

## Diagnostics and predictive validation

Always inspect:

- rank-normalized R-hat and bulk ESS for scientific parameters;
- trace plots and per-chain ACFs;
- SSVS transition counts and rates;
- population probabilities and slab-scale posteriors;
- for PGAS, minimum particle ESS, unique ancestors, changed path fraction, and
  restored iterations;
- sensitivity to record start, slab calibration, and pooling choice.

The package also provides expanding-window leave-future-out validation,
log-predictive scores, CRPS, threshold-weighted CRPS, quantile scores, and PIT
diagnostics. See `examples/13_leave_future_out.py` and
`docs/PREDICTIVE_VALIDATION.md`.

An R-hat near one and high ESS establish computational reliability. They do not
make an indecisive allocation scientifically decisive. With Uccle data starting
in 1980, posterior mass split between fixed and dynamic structures is a result
to report, together with sensitivity fits using longer records where available.

## Uccle workflow

```python
model = bx.make_uccle_hierarchical_model()
data = bx.load_uccle_multiseries(start="1980-01-01")

fit = bx.fit_uccle_hierarchical(
    model=model,
    pooling="selection",
    start="1980-01-01",
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=240),
    particles=bx.Particles(n=512, proposal="guided"),
)
```

The six channels keep separate trajectories. The hierarchy estimates which
types of evolution are common across summaries; it does not impose residual or
copula dependence. Independent fits remain available through
`fit_uccle_series` and `fit_uccle_all` as a sensitivity analysis.

## Examples

Run the scripts in numerical order. Start with `01`, use `04` and `05` for
prior and parameterization checks, then move to `06`--`07` for hierarchical
simulation and `09`--`10` for Uccle. Each script has editable constants at the
top, prints diagnostics, uses `progress=True`, and saves its figures through the
plot API. See [examples/README.md](examples/README.md).

## Release validation

Before publication:

1. run `python validation/run_release_validation.py`;
2. run the full test suite;
3. fit at least four chains for every reported model;
4. report prior, start-date, and pooling sensitivity;
5. verify PGAS diagnostics for every GEV channel;
6. archive the scripts, seeds, package version, tables, and figures.

Further details:

- [hierarchical model](docs/HIERARCHICAL_MODEL.md)
- [inference matrix](docs/INFERENCE_MATRIX.md)
- [Uccle workflow](docs/UCCLE.md)
- [architecture](docs/ARCHITECTURE.md)
- [validation](docs/VALIDATION.md)
- [migration to 2.4](docs/MIGRATION.md)
