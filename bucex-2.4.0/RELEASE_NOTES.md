# bucex 2.4.1 release notes

Version 2.4.1 completes the hierarchical structural workflow introduced in
2.4. The package remains focused on univariate models and collections of
series-specific structural paths whose model choices and/or innovation scales
are partially pooled.

## Cleaner scientific model space

Hierarchical SSVS now defaults to four joint trend classes:

1. deterministic linear trend;
2. RW1 with drift;
3. RW2 smooth changing trend;
4. full local linear trend.

Every class estimates initial level and slope. Selection concerns level and
slope innovations, so an inactive slope innovation no longer means an exact
no-warming forecast. Monthly seasonality remains present and is selected as
fixed or dynamic. The old componentwise no-slope state is retained only through
the explicit `model_space="componentwise"` sensitivity option.

Posterior population probabilities of the four classes are available from
`fit.hierarchical_trend_model_probabilities()` and through
`fit.plot("trend_models")`.

## Hierarchical Laplace and exact PGAS

Mixed and all-GEV multiseries models now support an exploratory hierarchical
Laplace engine. Its `InferencePlan` clearly marks the fit approximate. A
complete screening fit can be passed directly to exact-invariant PGAS:

```python
screen = bx.fit(data, model, engine="laplace", ...)
exact = bx.fit(data, model, engine="pgas", init=screen, ...)
```

The validated warm start carries channel parameters, latent paths, hierarchy
probabilities, and slab scales. It changes only the start; PGAS still runs full
warmup and targets its declared posterior. `FitResult.warm_start()` exposes the
same conversion explicitly.

`HierarchicalSampler(initializer="laplace", channel_workers=n)` provides a
lighter internal Laplace initialization and optional concurrent channel
updates.

## PGAS performance and robustness

- Vectorized GEV particle observation weights and complete-path likelihoods.
- Vectorized Laplace pseudo-data calculations.
- Optional deterministic per-channel thread updates within a hierarchy sweep.
- Grouped `xi=(channel:value,...)` progress output for mixed models.
- Renamed the ambiguous changed-fraction diagnostic to
  `path_update_fraction`; the old name remains a deprecated compatibility
  alias.
- Retained the 2.4 singular-support guided disturbance fix, initial-slope
  correction, conditioned-predecessor fallback, and Joseph-form covariance
  handling.

## Priors and calibration

The primary hierarchy continues to use normal SSVS slabs, optionally with a
shared half-Student-t scale multiplier. New helpers translate expert bounds on
accumulated level change, change in decadal warming rate, and seasonal
innovations into coefficient scales:

- `calibrate_structural_scales()`;
- `structural_scale_implications()`;
- `half_student_t_scale_for_median()`.

This makes slab calibration reproducible and helps avoid Bartlett's paradox
from arbitrarily diffuse model-selection slabs.

## Reproducible examples and HPC

Three examples complete the workflow:

- Example 14: hierarchical Laplace screen followed by warm-started PGAS;
- Example 15: one independently seeded publication chain per HPC process;
- Example 16: combine four checksummed archives and run final diagnostics.

A Slurm array template is included in `examples/hpc/slurm_four_chains.sh`.

## Interpretation

The recommended Uccle analysis is pooled structural selection over the four
trend classes, with fixed/dynamic seasonality and a scientifically calibrated
normal slab. Pooled slab magnitude is a sensitivity analysis. Laplace is for
exploration; final mixed/GEV results use PGAS. Model probabilities and
model-averaged paths should be reported rather than forcing uncertain classes
into a hard decision.
