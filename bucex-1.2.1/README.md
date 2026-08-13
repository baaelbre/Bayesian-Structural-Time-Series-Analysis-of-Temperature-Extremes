# bucex 1.2.1

`bucex` fits Bayesian structural time-series models to Gaussian bulk series and
dynamic-location GEV extremes. Version 1.2.1 has one model grammar, one
`fit()` entry point and one `FitResult`. Parameterization, state-update engine,
prior profile and ASIS are choices inside that framework rather than separate
fitters.

## Quick start

```python
import bucex as bx

fit = bx.fit(
    y,
    family="gev",
    period=12,
    parameterization="fruehwirth_schnatter",
    engine="pgas",
    priors="regularized_horseshoe",
    asis=True,
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=42),
    particles=bx.Particles(n=256, proposal="guided"),
)

fit.state("level")
fit.parameter("sd.level")
fit.diagnostics()
forecast = fit.forecast(12, seed=43)
```

The equivalent centered fit changes only one option:

```python
centered = bx.fit(
    y,
    family="gev",
    period=12,
    parameterization="centered",
    engine="laplace",
    priors="normal",
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=42),
)
```

## The four independent choices

| Choice | Values | Meaning |
| --- | --- | --- |
| Parameterization | `centered`, `disturbance`, `fruehwirth_schnatter` | Direct states, standardized/scaled disturbances, or the FS signed-scale augmentation |
| Engine | `ffbs`, `laplace`, `pgas` | Exact Gaussian FFBS, approximate GEV iterated Laplace, or exact-invariant GEV PGAS |
| Prior | `manuscript_lasso`, `regularized_lasso`, `regularized_horseshoe`, `pc`, `normal`, `ssvs` | Innovation-scale prior profile; compatibility is checked before sampling |
| Interweaving | `asis=False` or `True` | Adds a second parameterization sweep and records its partner in the inference plan |

Gaussian models use FFBS. GEV models use either iterated Laplace or PGAS.
The FS parameterization currently supports one local linear trend, optional
dummy seasonality, and no regression. The disturbance parameterization covers
the general compiled model, including static and dynamic regression. See the
[inference matrix](docs/INFERENCE_MATRIX.md) for exact combinations and prior
semantics.

When `priors` is omitted, automatic FS fits use the regularized horseshoe;
centered and disturbance fits use calibrated PC innovation priors. Passing a
named profile always overrides that default.

## Declarative models

```python
model = bx.Model(
    bx.Gaussian(),
    [
        bx.LocalLinearTrend(),
        bx.DummySeasonal(period=12),
        bx.Regression(
            2,
            dynamic=True,
            name="climate",
            feature_names=("nao", "enso"),
        ),
    ],
)

fit = bx.fit(
    y,
    model,
    exog=climate_frame,
    parameterization="disturbance",
    priors="pc",
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=7),
)
```

Semantic names are stable across strategies: `level`, `slope`,
`seasonal[1]`, and `sd.level`, for example. FS-only signed coefficients and
algorithm diagnostics are retained separately without changing the public
state layout.

## One result type

Every fit returns `FitResult`, with chain-preserving arrays:

- `state_draws`: `(chains, draws, T + 1, state_dim)`;
- `parameter_draws[name]`: `(chains, draws, ...)`;
- `plan`: the resolved engine, parameterization, ASIS partner and exactness;
- `sampler_diagnostics`: acceptance, Laplace or particle diagnostics;
- `auxiliary_draws`: algorithm-specific draws such as FS latent states.

`FitResult.save()` writes a checksummed, non-pickle `.bucex` archive.
`FitResult.load()` validates its schema, checksum and class allowlist.

## Uccle workflow

The six monthly 1892–2022 Uccle temperature series are bundled:

```python
fit = bx.fit_uccle_series(
    "TXx",
    priors="manuscript_lasso",
    parameterization="fruehwirth_schnatter",
    engine="laplace",
    asis=True,
    mcmc=bx.MCMC(draws=2_000, warmup=1_000, chains=4, seed=40),
)
```

Minimum series (`TXn`, `TNn`) are sign-transformed internally and all summaries
are returned on their original orientation. See [the Uccle guide](docs/UCCLE.md)
for definitions, validation and the unresolved redistribution citation gate.

The installed package also provides one command-line workflow:

```bash
bucex-uccle validate-data

bucex-uccle fit TXx \
  --engine pgas \
  --parameterization fruehwirth_schnatter \
  --priors regularized_horseshoe \
  --asis \
  --draws 2000 \
  --warmup 2000 \
  --chains 4 \
  --particles 512 \
  --output results/TXx.bucex

bucex-uccle inspect results/TXx.bucex
```

## Installation and validation

```bash
python -m pip install .
python -m pytest -q
python validation/run_release_validation.py
```

Required dependencies are NumPy, SciPy and pandas. Matplotlib is optional via
`bucex[plot]`.

Further reading:

- [Architecture](docs/ARCHITECTURE.md)
- [Inference matrix](docs/INFERENCE_MATRIX.md)
- [Migration to 1.2.1](docs/MIGRATION.md)
- [Release validation](docs/VALIDATION.md)

`fit_bayes`, `fit_gaussian_structural`, `fit_gev_structural`,
`combine_fs_fits`, and `PosteriorBundle` remain thin compatibility names. They
delegate to the same framework; `fit_bayes` and `combine_fs_fits` are deprecated.

## License

The software is MIT licensed. That license does not establish the right to
redistribute the Uccle observations; consult `data/README.md` before publishing
the data files.
