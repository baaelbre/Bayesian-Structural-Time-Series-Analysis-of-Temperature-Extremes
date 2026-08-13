# bucex 1.1 API

## FS structural models

The compact call stays intentionally close to 0.3:

```python
fit = bx.fit_bayes(
    y,
    family="gev",
    period=12,
    priors="manuscript",
    state_method="laplace",
    n_iter=4000,
    burn=2000,
    thin=1,
    chains=4,
    seed=40,
    asis=True,
    dates=dates,
    name="TXx",
    tail="max",
)
```

`fit_fs` is an explicit alias. `parameterization="noncentered"`, `"fs"`, and
`"fruehwirth_schnatter"` select the same genuine FS implementation.

GEV `state_method` accepts `"laplace"` and `"pgas"`; the old name
`"particle"` maps to PGAS. Gaussian FS fits use `"ffbs"`.

Useful `state_kwargs` include:

| Key | Meaning |
| --- | --- |
| `particles` | number of PGAS particles |
| `particle_proposal` | `"guided"` or `"bootstrap"` |
| `laplace_max_iterations` | maximum mode iterations |
| `laplace_tolerance` | relative location convergence tolerance |
| `draw_attempts` | pseudo-Gaussian support attempts |
| `max_state_tries` | complete transactional sweep attempts |
| `asis_step` | log-magnitude ASIS proposal scale |
| `horseshoe_step_local` | local horseshoe log-scale step |
| `horseshoe_step_global` | global horseshoe log-scale step |
| `horseshoe_step_slab` | slab log-scale step |

Resolved prior objects can be built directly with:

```python
bx.manuscript_gev_priors()
bx.regularized_gev_priors()             # regularized lasso
bx.regularized_horseshoe_gev_priors()
bx.pc_gev_priors()
bx.normal_gev_priors()
bx.ssvs_gev_priors()
```

Every builder has a Gaussian counterpart.

## General declarative models

```python
model = bx.Model(
    observation=bx.GEV(),
    components=[
        bx.LocalLinearTrend(),
        bx.DummySeasonal(12),
        bx.Regression(1, dynamic=True, name="x"),
    ],
)

fit = bx.fit(
    y,
    model=model,
    exog=x,
    priors="regularized",
    engine="pgas",
    parameterization="noncentered",
    asis=True,
    mcmc=bx.MCMC(draws=2000, warmup=2000, chains=4, seed=40),
    particles=bx.Particles(n=256),
)
```

Components are `LocalLevel`, `LocalLinearTrend`, `DummySeasonal`, and
`Regression`. `compile_model` validates dimensions, generates semantic state
and disturbance names, and provides centred/disturbance-NCP transforms.

General prior classes are `HalfNormalSD`, `HalfStudentTSD`, `ExponentialSD`,
`PCSD`, `InverseGammaVariance`, `FixedSD`, and `SpikeSlabSD`, plus static GEV
shape priors. `default_priors(compiled, profile=...)` exposes the resolved
calibration.

## Results

FS `PosteriorBundle` keeps backward-compatible fields:

```python
fit.draws_static
fit.draws_states                 # flat retained draws
fit.state_draws.shape            # chains, draws, time, state
fit.state_draws("alpha")         # named semantic draws, flat chains
fit.meta
fit.priors
fit.config
fit.initial_values
```

General `FitResult` uses chain-shaped `state_draws` and `parameter_draws`.
Both expose summaries, diagnostics, plotting, posterior forecasting, and safe
`.bucex` persistence.

```python
fit.static_summary()
fit.diagnostics()
fit.plot("process_sd")
future = bx.forecast(fit, 24, draws=2000, seed=41)
fit.save("fit.bucex")
```

Use `PosteriorBundle.load` for FS archives and `FitResult.load` for general
archives. The two formats are checked and never execute pickle.

## Tail and calendar helpers

GEV fits expose:

```python
fit.exceedance_probability_draws(threshold, annual=True)
fit.return_period_draws(threshold, annual=True)
fit.endpoint_draws()
fit.event_label(threshold)
```

Calendar-aware level summaries include `level_rate_draws`,
`period_rate_summary`, and `rate_contrast_summary`. For minima, use
`tail="min"`; bucex fits the sign-transformed maxima model and returns high-level
locations, events, and forecasts on the original scale.

## Uccle convenience layer

```python
series = bx.load_uccle_series("TXx")
fit = bx.fit_uccle_series(
    "TXx",
    priors="horseshoe",
    engine="pgas",
    mcmc=bx.MCMC(draws=2000, warmup=2000, chains=4, seed=40),
    particles=bx.Particles(n=256),
    asis=True,
)
```

Compact arguments (`n_iter`, `burn`, `thin`, `chains`, `seed`) remain valid.
`fit_uccle_all` returns a dictionary-like `UccleFitCollection`.

## Compatibility names

- `fit_bayes` and `fit_fs` are the FS path.
- `fit` is the general compiled path.
- `StructuralModel`, `StructuralSSM`, and `Legacy*` component/observation names
  expose explicit 0.3 construction.
- `Model`, `CompiledModel`, and the unprefixed declarative components expose
  the general path.
- `regularized` remains a compatibility alias for FS regularized lasso only in
  `fit_bayes`; in the general `fit` API it retains v1's PC-SD convenience
  profile. Prefer the explicit names in new confirmatory work.
