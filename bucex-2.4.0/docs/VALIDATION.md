# Release validation

Version 2.4 validation has four layers:

1. unit and integration tests for models, priors, samplers, results, plotting,
   prediction, and persistence;
2. fixed-seed release checks in `validation/run_release_validation.py`;
3. source compilation plus wheel and source-archive builds;
4. an installed-wheel smoke test in an isolated temporary directory.

Run:

```bash
python -m pytest
python validation/run_release_validation.py
python -m build
```

## Required 2.4 checks

The release validator checks:

- one public `fit()` path for `Model` and `MultiSeriesModel`;
- exact Gaussian FFBS and exact-invariant mixed PGAS plans;
- pooled selection, pooled slab, and pooled-both prior resolution;
- seasonality present by default with fixed/dynamic states only;
- initial level and slope stored as estimated posterior parameters;
- random FS sign switches preserve each complete predictor;
- long-series Gaussian FFBS remains finite with nearly singular process
  covariance;
- guided disturbance PGAS remains on affine support and does not produce
  all-zero ancestor weights;
- the FS PGAS ancestor fallback preserves the conditioned path;
- Gaussian and mixed hierarchical smoke fits return finite state and parameter
  draws;
- hierarchy, component allocation, trace, ACF, and save APIs work;
- `level_slope` uses seasonally adjusted observations;
- predictive scores and PITs remain finite and inside mathematical ranges;
- schema-2.4 archive round-trips preserve model, data, and draws;
- the Uccle helper preserves two Gaussian and four GEV channels and both
  lower-tail transformations.

## Smoke tests versus scientific inference

Release smoke tests use very short chains and small particle systems so they
run quickly. They establish software behavior, not posterior accuracy.

For manuscript results:

- use at least four chains and retain enough draws for all reported ESS values;
- use overdispersed starts and check between-chain agreement;
- increase PGAS particles until diagnostics and scientific summaries stabilise;
- report SSVS switching and constant-allocation diagnostics;
- compare normal, SSVS, and hierarchical results;
- vary hierarchy concentrations, slab scales, and record start;
- run leave-future-out scores and PIT diagnostics;
- preserve scripts, package version, seeds, data checks, tables, and figures.

An R-hat of one and high ESS are necessary computational checks. They do not
remove finite-sample uncertainty in structural allocation.
