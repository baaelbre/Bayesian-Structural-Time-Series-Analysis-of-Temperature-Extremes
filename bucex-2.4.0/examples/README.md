# bucex 2.4.0 examples

Each example is a genuine standalone script with editable constants at the
top—there is no shared configuration file. Run one from the project root:

```bash
python examples/01_gaussian_local_trend.py
```

| Order | Script | What it demonstrates |
|---:|---|---|
| 1 | `01_gaussian_local_trend.py` | Complete univariate Gaussian analysis with exact FFBS |
| 2 | `02_gev_local_trend.py` | Complete GEV analysis with guided PGAS and support checks |
| 3 | `03_diagnose_gev_pgas.py` | Particle-count and parameterization sensitivity after the guided-PGAS hotfix |
| 4 | `04_compare_priors.py` | Normal, PC, horseshoe, triple-gamma, regularized triple-gamma, and SSVS priors |
| 5 | `05_compare_parameterizations.py` | Centered, disturbance, and Frühwirth–Schnatter efficiency under matched priors |
| 6 | `06_hierarchical_gaussian.py` | Pool selection, slab magnitude, or both across Gaussian series |
| 7 | `07_hierarchical_mixed.py` | Joint Gaussian/GEV hierarchy and the guided disturbance PGAS regression test |
| 8 | `08_bulk_tail_independent.py` | Parallel unpooled Gaussian bulk and GEV tail analyses |
| 9 | `09_uccle_univariate.py` | Six separate Uccle analyses: the no-pooling comparator |
| 10 | `10_uccle_hierarchical.py` | Proposed six-summary Uccle analysis with pooled structure |
| 11 | `11_fixed_and_dynamic_components.py` | Exact SSVS semantics for zero, fixed, and dynamic components |
| 12 | `12_gev_ssvs_pgas.py` | Exact GEV SSVS with PGAS-corrected model moves |
| 13 | `13_leave_future_out.py` | Held-out log score, CRPS, tail scores, and PIT diagnostics |

## The unified API

A univariate analysis uses `Model`; a pooled analysis uses
`MultiSeriesModel`. Both go through `bx.fit`, return `FitResult`, support the
same MCMC controls, and use the same plotting and forecast conventions.

```python
# One series
fit = bx.fit(y, bx.Model(...), priors="normal", mcmc=bx.MCMC(...))

# Related series
fit = bx.fit(
    frame,
    bx.MultiSeriesModel(channels=(...)),
    priors=bx.HierarchicalPrior(pool="selection"),
    mcmc=bx.MCMC(...),
)
```

The hierarchical choices are deliberately simple:

- `pool="selection"`: learn population probabilities for structural states;
  keep the normal slab calibration fixed.
- `pool="slab"`: keep components dynamic and learn a shared normal-slab
  multiplier with a half-Student-t hyperprior.
- `pool="both"`: learn both allocation probabilities and slab multipliers.

Channels never share a latent path. They only share hyperparameters, so every
summary retains its own level, rate, seasonal evolution, observation SD, and
GEV shape where relevant. Monthly seasonality is physically present by
default and is therefore `fixed` versus `dynamic`, not `absent` versus
`present`.

## What to check before interpretation

1. Split rank-normalized R-hat (target approximately below 1.01) and bulk ESS
   for every scientific parameter and hierarchy hyperparameter.
2. Chain-specific traces and ACFs, not only pooled densities.
3. For SSVS, allocation probabilities *and* switching diagnostics. A constant
   state has undefined R-hat/ESS; it is not automatically evidence of perfect
   mixing.
4. For PGAS, minimum particle ESS, ancestor diversity, path-change rate,
   changed fraction, and zero unexplained/restored iterations.
5. Repeat a final mixed analysis with more particles. Agreement is more
   important than a single apparently healthy particle statistic.
6. Check the stored sign-invariance error. Signed FS innovation scales are
   not identified; random sign switching must leave the predictor unchanged.
7. Inspect initial-level and initial-slope posteriors. They are estimated in
   v2.4 rather than silently fixed.
8. Use held-out prediction for model comparisons. In-sample reconstruction is
   not predictive validation.

## Reading the Uccle results

Example 9 asks six separate questions. Example 10 asks a population question:
do fixed or dynamic levels, rates, and seasonal patterns recur across the six
summaries, and how large are their innovation scales? The complete-predictor
rate remains series-specific and is the primary physical summary.

The 1980–2023 record contains about 43 annual cycles. It can produce excellent
MCMC convergence while leaving `fixed` versus `dynamic` scientifically
uncertain. That is posterior uncertainty, not a convergence failure. Report
the probabilities rather than converting 0.55 versus 0.45 into a hard label,
then repeat the analysis on the longer homogenized record and under
`pool="selection"`, `"slab"`, and `"both"`.

## Figures

Every plot accepts `save=` and every diagnostic is available from the fit:

```python
fit.plot("traces", save="figures/traces.png")
fit.plot("acf", save={"path": "figures/acf.png", "dpi": 300})
fit.plot("level_slope", save="figures/level_slope.png")
fit.plot("hierarchy", save="figures/hierarchy.png")
```

The univariate `level_slope` plot compares the latent level with seasonally
adjusted observations. Raw monthly observations are intentionally not used in
that panel because their annual cycle makes a smooth level look misleadingly
flat.
