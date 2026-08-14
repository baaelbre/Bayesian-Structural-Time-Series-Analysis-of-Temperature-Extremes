# Release validation

The v2 release check has four layers:

1. the retained 39-test v1.2.1 compatibility suite;
2. factor tests for identification, compiler namespaces, exact Gaussian
   likelihood, mixed PGAS, forecasting/risk, serialization and Uccle helpers;
3. `validation/run_factor_validation.py` for fixed-seed recovery and a
   near-zero-variance mixed-family stress chain;
4. wheel/source build, isolated install and installed-package smoke tests.

Run locally with:

```bash
python -m pytest -q
python validation/run_factor_validation.py
python validation/run_release_validation.py
python -m build
```

The factor validator writes `validation/factor_validation_2.0.0.json`. Its
acceptance requirements are:

- the multichannel Kalman likelihood agrees with a direct joint multivariate
  Normal density to absolute error below `1e-9`;
- an anchored estimated loading's fixed-seed 90% posterior interval contains
  the simulated truth;
- a genuinely positive slope innovation variance near zero remains an active
  transition direction rather than being numerically changed to a structural
  zero;
- mixed-family PGAS retains finite paths, changes its conditioned path and is
  labeled exact-invariant;
- the corresponding Laplace plan is labeled approximate;
- safe archive round-trip errors are exactly zero and multichannel forecasts
  are finite;
- the fixed Uccle contrast model compiles an aligned six-channel design.

## Recorded v2 result

The fixed-seed 2026-08-13 run passed every factor validation section. The
multichannel Kalman log-likelihood differed from the direct joint Normal value
by `1.78e-15`. The simulated loading truth `0.65` lay inside the retained 90%
interval `[0.622, 0.713]`; its posterior mean was `0.649`.

The mixed Gaussian/GEV stress fit used a local-linear factor with slope process
SD `2e-7` (variance `4e-14`). Both transition directions remained active. A
50-draw chain after 25 warmup iterations was finite, had PGAS path-change rate
`1.0`, mean changed-path fraction `0.82`, median minimum particle ESS `8.48`
with 32 particles, and non-zero terminal-slope draw variation. The test is
designed to catch accidental rank truncation or silent state freezing near the
zero-variance boundary; it is not a claim that 50 draws are sufficient for
scientific inference.

Archive state and predictor round-trip errors were both exactly zero. A
10-draw, four-step mixed forecast had shape `(10, 4, 2)` and contained only
finite values. The Uccle fixed-contrast constructor compiled 24 aligned months
into a design of shape `(24, 6, 14)` with four named factors.

For manuscript results, validation must be stronger than the release smoke
chain:

- run multiple long chains from overdispersed initial loading/state values;
- scale particle count and inspect ESS and ancestor diversity over the entire
  time axis, not only medians;
- compare Laplace and PGAS on identical data/prior specifications;
- perform fixed-seed recovery for factors, loadings, process scales and GEV
  shape/return-level functionals;
- compare models through aligned leave-future-out forecasts and tail-weighted
  scores;
- report sensitivity to factor identification, individual trend flexibility
  and loading priors.

## Retained v1.2.1 validation

The original 2026-08-11 record remains in
`validation/release_validation_2026-08-11.json`. It covered all three
univariate parameterizations, all six FS prior profiles, six full-length Uccle
fits, exact PGAS for `TXx`, data re-aggregation, forecasting and safe archives.
Version 2 retains those source tests and public contracts. The historical
validator still targets the univariate workflows and can be run independently.
