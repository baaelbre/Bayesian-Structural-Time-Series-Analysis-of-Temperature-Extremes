# Release validation

The v1.2.1 release check has three layers:

1. the focused pytest suite for model grammar, numerical kernels, every
   parameterization, all named FS priors, serialization, forecasting and Uccle;
2. `validation/run_release_validation.py` for full-length Uccle and cross-engine
   smoke runs;
3. build, clean-install and installed-package smoke tests for the wheel and
   source distribution.

Run locally with:

```bash
python -m pytest -q
python validation/run_release_validation.py \
  --output validation/release_validation_2026-08-11.json
python -m build
```

The committed JSON record contains resolved inference plans, array shapes,
finite-value checks, data integrity values and archive/forecast results. It is
generated from the release source, not hand edited.

## Recorded 1.2.1 result

The 2026-08-11 release run passed all 39 tests and both clean-install checks.
It exercised all three parameterizations and all six named FS prior profiles.
All six full-length Uccle fits produced finite state arrays of shape
`(1, 1, 1573, 13)`: Gaussian `TXm`/`TNm` used exact FFBS, while the four GEV
series used iterated Laplace and converged on every recorded update. Separate
full-length checks covered centered `TXm` and exact-invariant PGAS/FS `TXx`;
the latter completed with 24 particles, no restored iterations, a path-change
rate of `1.0`, and median minimum particle ESS of `12.61`. The safe archive
round-trip was exact and its forecast was finite. Re-aggregation of the daily
source produced zero monthly mismatches; the largest floating-point difference
was `3.55e-15`.

Release acceptance requires:

- all pytest tests pass;
- imports expose version `1.2.1` and one `FitResult` class;
- centered, disturbance and FS Gaussian runs finish with canonical state names;
- GEV iterated Laplace is labeled approximate and GEV PGAS exact;
- every named FS prior profile resolves and produces finite draws;
- static/dynamic regression uses the disturbance backend;
- all six Uccle monthly series reproduce the daily aggregation;
- all six full-length Uccle smoke fits finish, including exact PGAS for `TXx`;
- a `.bucex` archive round-trips without pickle;
- the wheel and sdist install into empty environments and pass an import/fit
  smoke test.
