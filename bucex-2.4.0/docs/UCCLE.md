# Uccle analysis

## Six monthly summaries

The packaged workflow supports:

| Name | Family | Orientation | Description |
|---|---|---|---|
| `TXm` | Gaussian | ordinary | monthly mean daily maximum |
| `TNm` | Gaussian | ordinary | monthly mean daily minimum |
| `TXx` | GEV | upper | monthly maximum daily maximum |
| `TXn` | GEV | lower | monthly minimum daily maximum |
| `TNx` | GEV | upper | monthly maximum daily minimum |
| `TNn` | GEV | lower | monthly minimum daily minimum |

Lower-tail series are sign-transformed internally so one GEV implementation
serves both orientations.

## Validate and load

```python
import bucex as bx

print(bx.validate_uccle_data())
data = bx.load_uccle_multiseries(start="1980-01-01")
```

The loader requires complete, aligned monthly series. Use
`derive_uccle_monthly()` to regenerate summaries from the optional daily file.

## Independent baseline

Fit each series independently before pooling:

```python
fit_txm = bx.fit_uccle_series(
    "TXm",
    priors="normal",
    start="1980-01-01",
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=901),
)

all_fits = bx.fit_uccle_all(
    priors="ssvs",
    start="1980-01-01",
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=902),
)
```

This answers which structure each series supports without borrowing information
from the other summaries.

## Hierarchical primary analysis

```python
model = bx.make_uccle_hierarchical_model()

prior = bx.HierarchicalPrior(
    pool="selection",
    level_states=("fixed", "dynamic"),
    trend_states=("zero", "fixed", "dynamic"),
    season_states=("fixed", "dynamic"),
    level_concentration=(1, 1),
    trend_concentration=(1, 1, 1),
    season_concentration=(1, 1),
    coefficient_scale={"level": 0.03, "trend": 0.0002, "season": 0.03},
)

fit = bx.fit_uccle_hierarchical(
    model=model,
    priors=prior,
    start="1980-01-01",
    mcmc=bx.MCMC(
        draws=2_000,
        warmup=2_000,
        chains=4,
        seed=903,
        progress=True,
    ),
    particles=bx.Particles(n=512, proposal="guided"),
)
```

Because four channels are GEV, this run uses PGAS. Start with a shorter smoke
run, but do not report it scientifically. Increase particles and chain length
until particle and MCMC diagnostics stabilise.

## What to report

For every series:

- posterior zero/fixed/dynamic probabilities;
- complete-predictor and average-rate summaries;
- innovation standard deviations conditional and unconditional on structural
  allocation;
- initial level and slope uncertainty;
- GEV scale and shape uncertainty for extreme summaries;
- predictor, seasonally adjusted level/slope, trace, and ACF plots.

For the hierarchy:

- posterior population allocation probabilities;
- pooled slab multipliers when used;
- Dirichlet concentrations and half-t hyperparameters;
- allocation switching diagnostics;
- sensitivity to `pool="selection"`, `"slab"`, and `"both"`.

For PGAS:

- minimum ESS distribution;
- unique ancestors;
- path-change rate and changed fraction;
- restored iterations and GEV support margins;
- sensitivity to particle count.

## Interpreting indecisive SSVS probabilities

The 1980--2023 record contains many monthly observations but only about 44
annual cycles. It can establish positive average warming while remaining
uncertain about whether a tiny slope innovation is truly dynamic or whether a
slowly changing seasonal pattern is distinguishable from a fixed one.

If R-hat is near one, ESS is adequate, chains overlap, allocations switch, and
ACFs decay, a fixed/dynamic split near 0.5 is posterior uncertainty rather than
a computational failure. Report it directly. Do not turn it into a hard state
selection without a declared decision rule.

## Recommended paper design

1. Independent normal-prior fits establish transparent per-series trajectories.
2. Independent SSVS fits show how much selection is data-driven before pooling.
3. Pooled-selection hierarchy is the primary joint analysis.
4. Pooled slab and pooled both are sensitivity analyses.
5. Start-date and slab-scale sensitivity address the short-record concern.
6. Leave-future-out log score, CRPS, tail-weighted CRPS, and PITs assess
   prediction rather than only in-sample fit.

The central joint estimand is recurrence of structural behavior across
temperature summaries. Each summary still has its own evolution, which keeps
the resulting scientific statements direct and reusable at other stations.
