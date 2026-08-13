# Package structure (v1.1)

Version 1.1 keeps the readable 0.3 package categories as the canonical layout.
The general compiled engine introduced in 1.0 lives in `_general` and is
re-exported through the same top-level package.

```text
bucex/
  api/                 FS and general fit/forecast entry points
  components/          explicit 0.3 structural components
  core/                FS posterior results
  datasets/            Uccle loaders, validation, and fit wrappers
  diagnostics/         PIT, residual, calibration, and MCMC helpers
  inference/
    fit/                centred/FS samplers and shrinkage priors
    state/              Kalman, FFBS, Laplace, and particle backends
  io/                   safe FS archives and simulation IO
  models/               explicit structural-model composition
  observation/          canonical Gaussian and GEV observations
  plotting/             posterior, process-SD, and risk plots
  risk/                 distributional risk helpers
  simulate/             simulation helpers
  _general/             retained 1.0 compiler and inference engine
  data/                 packaged monthly Uccle example series
docs/                   current API, inference, migration, and validation notes
examples/               local fitting, scoring, pooling, and plotting workflows
jobs/                   guarded production and quick-run scripts
tests/                  inherited and 1.1 regression suites
validation/             fixed-seed release workflow and JSON record
```

`bucex.observation` is the only canonical observation namespace. Public 0.3
types are exposed with explicit `Legacy*` names where they would otherwise
collide with the declarative component names. `fit_fs` / `fit_bayes` select the
FS engine; `fit` selects the compiled general engine.
