# bucex 1.1

`bucex` fits Bayesian structural time-series models to Gaussian bulk series and
block extremes. Version 1.1 deliberately brings the research implementation
from 0.3 forward instead of replacing it: the Fruehwirth--Schnatter (FS)
augmented non-centred model, Bayesian lasso, exact structural SSVS, sign
switching, and restoration diagnostics now live alongside v1's PGAS,
forecasting, scoring, multi-chain diagnostics, safe archives, regression, and
compiled-model interface.

## Which fitting function should I use?

| Goal | Function | Parameterisation |
| --- | --- | --- |
| Reproduce or extend the Uccle/manuscript model | `fit_fs` / `fit_bayes` | Genuine FS augmented non-centring |
| Use Bayesian lasso, regularized lasso, horseshoe, PC, or exact structural SSVS | `fit_fs` / `fit_bayes` | Genuine FS augmented non-centring |
| Add static/dynamic regressors or compose a less standard model | `fit` | Compiler-generated disturbance non-centring |

The two parameterisations are not presented as synonyms. Every fit records its
engine, parameterisation, exact/approximate target claim, resolved priors,
initial values, and diagnostics.

## Install and test

```bash
python -m pip install -e ".[plot,test]"
python -m pytest
```

The source tree supports Python 3.10--3.12. NumPy, SciPy, and pandas are core
dependencies; matplotlib is optional.

## FS model: concise use

```python
import bucex as bx

fit = bx.fit_fs(
    y,
    family="gev",
    period=12,
    priors="horseshoe",
    state_method="pgas",            # exact-invariant; "laplace" is faster
    asis=True,
    chains=4,
    n_iter=4000,
    burn=2000,
    seed=40,
    state_kwargs={"particles": 256},
    dates=dates,
    name="TXx",
)

print(fit.static_summary())
print(fit.diagnostics()["parameters"])
forecast = fit.forecast(24, draws=4000, seed=41)
```

Available FS prior profiles are:

- `"manuscript"`: the shared hierarchical Bayesian lasso used by the older
  analysis. For Gaussian fits its scale mixture is tied to observation
  variance, exactly as in the research code.
- `"regularized_lasso"` (legacy alias `"regularized"`): component-specific
  lasso scales for level, monthly slope, and seasonality.
- `"horseshoe"` (alias `"regularized_horseshoe"`): a genuine regularized
  horseshoe with local half-Cauchy scales, a global half-Cauchy scale, and a
  finite inverse-gamma slab.
- `"pc"`: a PC prior declared through `P(|s_k| > upper_k) = alpha_k`, sampled
  through its exact normal--exponential mixture.
- `"normal"`: direct scale-aware Gaussian priors on signed innovation scales.
- `"ssvs"`: the 0.3 exact zero/fixed/dynamic structural model space. For GEV
  models this remains paired with the explicitly approximate Laplace engine.

The signed FS coefficients are retained as `s_level`, `s_trend`, and
`s_season`; process variances are their squares. Random sign switching handles
the two symmetric representations. `asis=True` interweaves centred signed-scale
updates while preserving the semantic centred trajectory.

## General compiled model

```python
import bucex as bx

model = bx.Model(
    bx.GEV(xi_bounds=(-0.5, 0.5)),
    [
        bx.LocalLinearTrend(),
        bx.DummySeasonal(12),
        bx.Regression(2, dynamic=True, name="x"),
    ],
)

fit = bx.fit(
    y,
    model=model,
    exog=x,
    priors="regularized",            # v1 PC-SD convenience profile
    engine="pgas",
    parameterization="noncentered", # disturbance non-centring
    asis=True,
    mcmc=bx.MCMC(draws=2000, warmup=2000, chains=4, seed=42),
    particles=bx.Particles(n=256),
)
```

The general engine retains v1's half-normal, half-Student-t, exponential/PC,
inverse-gamma-on-variance, fixed, and continuous spike-and-slab process-SD
priors. This continuous spike-and-slab is intentionally distinct from the FS
engine's exact structural SSVS.

## Inference claims

| Interface | Family | Engine | Claim |
| --- | --- | --- | --- |
| FS | Gaussian | `ffbs` | exact conditional state draw |
| FS | GEV | `pgas` | exact-invariant PGAS state kernel plus exact GEV FS elliptical-slice update |
| FS | GEV | `laplace` | converged, line-searched iterated-Laplace approximation |
| General | Gaussian | `ffbs` | exact conditional state draw |
| General | GEV | `pgas` | exact-invariant conditional-SMC/ancestor-sampling kernel |
| General | GEV | `laplace` | iterated-Laplace approximation |

Both PGAS implementations retain normalized log weights separately from
floating-point probabilities, so ancestor sampling does not lose a viable path
merely because a normalized probability underflowed to zero. Guided and
bootstrap proposals retain exact prior/proposal corrections.

Laplace is never labelled exact. FS restoration counts and reasons, convergence
flags, iteration counts, relative changes, support rejections, particle ESS,
ancestor diversity, and path-change rates are attached to the fit.

## Fits, forecasts, plots, and safe files

FS fits retain observations, dates, model, priors, configuration, initial
values, chain slices, centred and non-centred trajectories, static draws, and
engine metrics. The older flat draw arrays remain available, while
`fit.state_draws.shape` exposes `(chains, draws, time, state)` and
`fit.state_draws("alpha")` remains the convenient named selector.

```python
fit.plot("level_slope")
fit.plot("process_sd")       # prior versus posterior process SDs
fit.plot("endpoint")

fit.save("TXx.bucex")
restored = bx.PosteriorBundle.load("TXx.bucex")
```

`.bucex` files use JSON plus compressed NumPy arrays and a SHA-256 integrity
check. Loading never executes pickle. General `FitResult` archives use the same
safe design and preserve their own schema.

Posterior forecasts propagate retained parameter uncertainty, the terminal
state, future process disturbances, and observation noise. GEV helpers include
event probabilities, return periods, return levels, endpoints, CRPS,
threshold-weighted CRPS, tail quantile scores, and exceedance Brier/log scores.

## Uccle workflows

The six monthly series can be loaded and fitted directly:

```python
fit = bx.fit_uccle_series(
    "TXx",
    priors="horseshoe",
    engine="pgas",
    asis=True,
    mcmc=bx.MCMC(draws=2000, warmup=2000, chains=4, seed=40),
    particles=bx.Particles(n=256),
)
```

The monthly files are bundled. The repository also retains the supplied daily
source and validates exact daily-to-monthly reproduction. The upload did not
include an authoritative dataset identifier, citation, or redistribution
statement; `bucex/data/README.md` records this publication gate. The MIT
license covers the software, not an unverified right to redistribute the data.

## Source layout

The readable 0.3 package layout remains canonical:

```text
bucex/
  api/          high-level fit and forecast entry points
  components/   explicit structural blocks
  core/         result objects
  datasets/     Uccle workflows
  diagnostics/  MCMC, PIT, residual, and calibration tools
  inference/    centred, FS, Laplace, particle, and general samplers
  io/           safe archives and simulation IO
  models/       state-space composition
  observation/  Gaussian and GEV likelihoods
  plotting/     fit-only plotting interface
  risk/         return levels, periods, endpoints, exceedance risk
  simulate/     simulation helpers
```

The compact internal general engine preserves v1 model compatibility; its
public names are re-exported from the top level. There is only one public API
and one release version.

See `docs/INFERENCE_CONTRACT.md`, `docs/API.md`, `docs/UCCLE_WORKFLOW.md`, and
`docs/VALIDATION.md` for the detailed contracts.
