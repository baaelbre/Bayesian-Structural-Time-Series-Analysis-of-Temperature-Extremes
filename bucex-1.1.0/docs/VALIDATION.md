# Validation record for bucex 1.1.0

Validation date: 11 August 2026

The machine-readable fixed-seed record is
`validation/release_validation_2026-08-11.json`. Regenerate it from a source
checkout with:

```bash
python validation/run_release_validation.py --data-dir data
```

## Automated suite

All 71 tests pass. The suite combines the inherited 0.3 behavior checks, the
1.0 compiled-engine checks, and focused 1.1 integration tests. It covers:

- the 0.3 compact API, FS augmented non-centring, signed innovation scales,
  sign switching, ASIS, lasso updates, structural SSVS, and restoration
  diagnostics;
- regularized lasso, regularized horseshoe, and calibrated PC innovation
  priors, including their hierarchy variables and profile separation;
- general model compilation, static and dynamic regression, centred and
  disturbance-non-centred paths, singular systems, missing observations, and
  exact transform round trips;
- Kalman likelihoods, FFBS, iterated Laplace convergence and support, guided
  and bootstrap particle likelihoods, and PGAS ancestor sampling;
- the normalized-log-weight regression that prevents viable particle paths
  being lost through probability underflow;
- multi-chain storage and pooling, rank-normalized split R-hat and bulk ESS,
  forecasts, minima, risk summaries, scores, plots, and safe archive integrity;
- bundled Uccle integrity, exact reconstruction from daily data, both fitting
  interfaces, and full-record smoke fitting.

Run the suite with:

```bash
python -m pytest
```

## Fixed-seed release workflow

The release validation completed in 38.55 seconds on Python 3.12.13. These are
execution and invariant-kernel checks, not posterior-convergence studies.

| Check | Result |
| --- | --- |
| General dynamic-regression fit | 2 chains, 5 retained draws, finite states, exact-target plan |
| FS Gaussian profiles | manuscript lasso, regularized lasso, horseshoe, PC, normal, and SSVS all finite |
| FS GEV Laplace | converged on every update; median 3 iterations; no restorations or support rejections |
| FS safe archive | state arrays identical after round trip; model and prior types preserved |

The Laplace fit correctly records `exact_target=false` and identifies itself as
an iterated-Laplace approximation. Its median relative location change was
`1.03e-7`.

## Full-length TXx PGAS check

The required seed-40 run used all 1,572 TXx observations, the regularized
horseshoe profile, FS non-centring, ASIS, a guided 64-particle PGAS kernel, one
warmup sweep, and one retained sweep. It completed in 36.88 seconds with:

- finite states and zero restored iterations;
- path change rate `1.0` and mean changed fraction `0.0356`;
- mean unique ancestors `40.4`;
- median minimum particle ESS `17.18`.

The recorded `exact_target=true` claim concerns invariance of the PGAS and GEV
elliptical-slice kernels; it does not imply that this two-sweep smoke run has
converged. Production tail fits should use multiple long chains and increase
the particle count when path-refresh or ESS diagnostics are weak.

## Supplied Uccle data

Each of the six monthly files has 1,572 observations from January 1892 through
December 2022. Re-aggregation of the supplied daily `TX` and `TN` fields gives
zero mismatches at tolerance `1e-10`; the largest absolute floating-point
difference is `3.55e-15`.

## Gates before scientific publication

- Run production-length independent chains and review rank-normalized split
  R-hat, bulk ESS, trace behavior, acceptance rates, restorations, and engine
  diagnostics.
- Inspect resolved priors against posterior process-SD draws. Convenience
  profiles are not substitutes for domain justification.
- Compare Laplace results with a sufficiently large-particle PGAS analysis for
  every tail model used in conclusions.
- Report held-out CRPS, threshold-weighted CRPS, tail quantile, exceedance
  Brier, and log scores.
- Resolve the exact Uccle source record, citation, and redistribution license
  described in `data/README.md` before publishing the CSVs.
