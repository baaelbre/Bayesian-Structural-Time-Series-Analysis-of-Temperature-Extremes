# Release and scientific validation

Version 2.6.0 has five software layers:

1. unit/integration tests for models, priors, engines, results, and archives;
2. fixed-seed numerical regression in `validation/run_release_validation.py`;
3. focused Laplace-to-PGAS workflow smoke validation;
4. source compilation plus wheel/source builds;
5. installed-wheel import and CLI checks.

```bash
python -m pytest
python validation/run_release_validation.py
python validation/run_presentation_smoke.py
python -m build
```

Short validation chains establish software behavior only. They are not
evidence for a scientific conclusion.

## Release gates

- `import bucex as bx` exposes the documented modelling and workflow API;
- Laplace plans remain explicitly approximate and PGAS plans exact-invariant;
- a univariate Laplace `FitResult` exports a compatible full-path warm start;
- PGAS metadata records that Laplace supplied the initializer;
- singular-support ancestor calculations remain finite and keep the
  conditioned predecessor available;
- the reference-ancestor change metric is stored, summarized, and exported;
- lower-tail Uccle fits round-trip observations in their original orientation;
- tail scenarios share their latent path and vary only shape over
  `-0.30/0/+0.30`;
- scale scenarios share their latent path and `xi=-0.30`, varying only
  `sigma=0.75/1.50/3.00`;
- structural scenarios share scale/shape and vary only component truth;
- workflow paths are deterministic and existing artifacts require explicit
  overwrite authorization;
- independently saved chains combine only when model, prior, plan, data, and
  dates agree;
- schema-2.6.0 archives round-trip and older supported archives remain readable;
- the seven Python examples are self-contained calls to the public API, while
  CLI stages and PBS jobs retain deterministic orchestration paths;
- PBS task IDs map to unique scenario/series/chain files and PGAS dependencies
  pass through a completed Laplace combine job.

## Manuscript and presentation gates

- at least four independent chains for each reported posterior;
- stable conclusions after increasing PGAS particles;
- satisfactory R-hat/ESS for continuous summaries;
- adequate structural allocation switching, not only continuous diagnostics;
- reported particle ESS, unique ancestors, path-update fraction, and
  reference-ancestor change;
- zero unexplained restoration failures and valid GEV support;
- prior predictive checks and defensible model odds/slab calibration;
- simulation recovery for every advertised structural state;
- sensitivity to record start, process slabs, shape bounds, and particle count;
- model-averaged trajectories and risk summaries rather than hard-selected
  post-fit models;
- archived config, version, seeds, manifest, fits, tables, and figures.

For selection recovery, inspect both the probability assigned to the true
state and whether chains actually move between plausible structures. A high
true-state probability from a chain that never switched is not sufficient
validation.
