# Package structure (v0.2)

```text
bucex/
  api/                 high-level fit and forecast entry points
  components/          trend, seasonal and regression components
  core/                result objects
  datasets/            Uccle loaders and wrappers
  diagnostics/         PIT, calibration, residual and MCMC helpers
  inference/
    fit/                centred and non-centred samplers and priors
    state/              Kalman, FFBS, Laplace and particle backends
  models/               structural model composition
  observation/          Gaussian and GEV observation models
  plotting/             high-level posterior and risk plotting
  risk/                 low-level distributional risk helpers
  simulate/             model simulation
examples/
  uccle_manuscript_v02.py
tests/
  test_v02_api.py
```

The canonical public import path is `bucex.observation`; `bucex.obs` remains as a
compatibility namespace.
