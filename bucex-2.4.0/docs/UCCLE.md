# Uccle analysis

## Six monthly summaries

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
data = bx.load_uccle_multiseries(start="1892-01-01")
```

The loader requires complete, aligned monthly series. Use
`derive_uccle_monthly()` to regenerate summaries from the optional daily file.

## Independent baselines

Fit each series separately before pooling:

```python
fit_txm = bx.fit_uccle_series(
    "TXm",
    priors="normal",
    start="1892-01-01",
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=901),
)

all_fits = bx.fit_uccle_all(
    priors="ssvs",
    start="1892-01-01",
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=902),
)
```

The normal fits provide transparent trajectories. Independent SSVS shows how
much structural discrimination comes from each channel before borrowing.

## Primary hierarchical model

Use the four-class joint trend space and keep the annual cycle present:

```python
model = bx.make_uccle_hierarchical_model()

scales = bx.calibrate_structural_scales(
    horizon_years=30,
    max_level_change=2.0,
    max_rate_change_per_decade=0.3,
    max_seasonal_innovation=0.2,
    probability=0.90,
)

prior = bx.HierarchicalPrior(
    pool="selection",
    model_space="joint_trend",
    trend_model_concentration=(1, 1, 1, 1),
    season_states=("fixed", "dynamic"),
    season_concentration=(1, 1),
    coefficient_scale=scales,
)
```

The trend classes are deterministic linear trend, RW1 with drift, RW2 smooth
changing trend, and full local linear trend. Every class estimates an initial
slope, so an inactive slope innovation never means “force future warming to
zero.” Seasonality is fixed or dynamic, never absent.

## Fast exploratory screen

The mixed hierarchy can be screened with Laplace:

```python
screen = bx.fit_uccle_hierarchical(
    model=model,
    priors=prior,
    start="1892-01-01",
    engine="laplace",
    mcmc=bx.MCMC(draws=500, warmup=500, chains=1, seed=903),
    hierarchical_sampler=bx.HierarchicalSampler(channel_workers=2),
)
```

This is an approximation. Use it to catch data/model problems, inspect rough
allocations, compare defensible slab calibrations, and initialize PGAS. Do not
report its intervals as the final posterior.

## Exact-invariant PGAS analysis

```python
fit = bx.fit_uccle_hierarchical(
    model=model,
    priors=prior,
    start="1892-01-01",
    engine="pgas",
    init=screen,
    mcmc=bx.MCMC(
        draws=2_000,
        warmup=2_000,
        chains=4,
        seed=904,
        progress=True,
    ),
    particles=bx.Particles(n=512, proposal="guided"),
    hierarchical_sampler=bx.HierarchicalSampler(
        initializer="laplace",
        channel_workers=2,
    ),
)
```

The Laplace fit contributes only a compatible starting draw. PGAS still runs a
full warmup and targets its exact posterior. Because this model is expensive,
the preferred publication workflow runs four one-chain jobs independently;
see Examples 15 and 16 and `examples/hpc/slurm_four_chains.sh`.

## What to report

For every series:

- posterior probability of each of the four trend classes;
- fixed/dynamic seasonal probability;
- model-averaged complete-predictor, level, slope, and average-rate summaries;
- active innovation scales and unconditional point-mass mixtures;
- initial level, initial slope, and initial seasonal uncertainty;
- Gaussian observation SD or GEV scale/shape uncertainty;
- predictor, seasonally adjusted level/slope, trace, and ACF plots.

For the hierarchy:

- posterior population probabilities of the four trend classes;
- population fixed/dynamic seasonal probabilities;
- pooled slab multipliers when used;
- Dirichlet concentrations, slab scales, and half-t hyperparameters;
- channel allocation switching diagnostics;
- sensitivity to pooling mode, record start, and slab calibration.

For PGAS:

- minimum particle ESS distribution;
- unique ancestor count;
- path-change rate and mean path-update fraction;
- restored iterations and GEV support margins;
- agreement after increasing particle count.

## Interpreting structure and forecasts

Positive average warming and uncertain slope *innovation* are compatible. A
linear-trend or RW1-with-drift draw retains its estimated constant slope in the
forecast. RW2 and full local-linear draws additionally allow the future rate to
evolve. The posterior forecast averages these mechanisms and propagates model
uncertainty.

Do not condition the forecast on the modal class unless that is a separately
declared decision analysis. Model averaging is the coherent default.

## Recommended revision design

1. Independent normal-prior fits establish per-series trajectories.
2. Independent joint-space SSVS shows series-specific structural evidence.
3. Pooled-selection hierarchy is the primary joint analysis.
4. Pooled slab and pooled both are sensitivity analyses.
5. Longest defensible record is primary; a 1980-start analysis is sensitivity.
6. Slab widths are elicited by prior-predictive bounds, not vague defaults.
7. Leave-future-out log score, CRPS, tail-weighted CRPS, and PITs assess
   predictive usefulness.
8. Static observation scale is primary; time-varying scale is a separately
   motivated extension rather than an automatic complication.

The central estimand is recurrence of structural behavior across temperature
summaries. Every summary keeps its own evolution, making the method interpretable
and transferable to other stations or collections of related climate indices.
