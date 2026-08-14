# Uccle workflow

The package includes six monthly series from January 1892 through December
2022 (1,572 observations each).

| Name | Family | Direction | Definition |
| --- | --- | --- | --- |
| `TXm` | Gaussian | upper | Monthly mean daily maximum temperature |
| `TNm` | Gaussian | upper | Monthly mean daily minimum temperature |
| `TXx` | GEV | upper | Monthly maximum daily maximum temperature |
| `TXn` | GEV | lower | Monthly minimum daily maximum temperature |
| `TNx` | GEV | upper | Monthly maximum daily minimum temperature |
| `TNn` | GEV | lower | Monthly minimum daily minimum temperature |

## Load and validate

```python
import bucex as bx

txx = bx.load_uccle_series("TXx")
integrity = bx.validate_uccle_data("data", check_daily=True)
```

The source release includes `data/Uccle_24_10_23.csv`; the six monthly files
are stored once in `bucex/data` and are included in the installed package.
They are recreated by `derive_uccle_monthly()` and match the bundled values to
floating-point precision.

## Fit one series

```python
fit = bx.fit_uccle_series(
    "TXx",
    priors="manuscript_lasso",
    parameterization="fruehwirth_schnatter",
    engine="laplace",
    asis=True,
    mcmc=bx.MCMC(draws=2_000, warmup=1_000, chains=4, seed=40),
    laplace=bx.Laplace(max_iterations=30, tolerance=1e-5),
)
```

For an exact-invariant GEV path update, select `engine="pgas"` and provide
`Particles(...)`. For a general centered or scaled-disturbance analysis,
change `parameterization` and select a compatible process-SD prior.

`TXn` and `TNn` are multiplied by -1 internally. `observed`, forecasts,
endpoints, event labels, exceedance probabilities and return levels are mapped
back to the original lower-tail orientation.

## Fit a collection

```python
fits = bx.fit_uccle_all(
    series=("TXm", "TXx", "TNn"),
    priors="regularized_horseshoe",
    parameterization="fruehwirth_schnatter",
    mcmc=bx.MCMC(draws=2_000, warmup=1_000, chains=4, seed=40),
)

fits.summary()
fits.save("results/uccle")
```

Each series receives a deterministic child seed. Collection members are
ordinary `FitResult` objects.

## Fit one shared-factor model

Version 2 provides aligned data and model constructors for a genuinely joint
state analysis:

```python
data = bx.load_uccle_factor_data(
    start="1892-01-01",
    end="2022-12-01",
)
model = bx.make_uccle_factor_model(
    structure="estimated",
    individual="static",
)
joint = bx.fit(
    data,
    model,
    engine="pgas",
    parameterization="disturbance",
    priors="regularized",
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=50),
    particles=bx.Particles(n=1_024, proposal="guided"),
)
```

The shortcut `fit_uccle_factor(...)` loads and fits the same graph. The
`estimated` structure uses one common factor anchored at `TXm=1`; lower-tail
loading initial values are converted to the internal sign orientation. The
alternative `structure="contrasts"` creates four fixed-loading factors:
common, day-night, extremes-versus-mean and upper-versus-lower. See the
[dynamic factor guide](DYNAMIC_FACTORS.md) for the loading table and
identification discussion.

The default `individual="static"` gives every channel its own intercept while
the factor carries evolution. `individual="local_level"` is substantially
less identified and should be introduced only with shrinkage and recovery
checks. These helpers retain conditional channel independence; they do not add
a mean/max residual copula.

## Command line

The single `bucex-uccle` command delegates to the same functions as the Python
API. A small PGAS workflow check is:

```bash
bucex-uccle fit TXx \
  --engine pgas \
  --parameterization fruehwirth_schnatter \
  --priors regularized_horseshoe \
  --draws 20 --warmup 20 --chains 1 --particles 128 \
  --output results/TXx-smoke.bucex
```

Use `bucex-uccle combine --output results/TXx.bucex chain-1.bucex chain-2.bucex`
to pool independently run compatible chains without flattening chain identity.

## Provenance gate

The supplied archive did not identify the exact original dataset record,
download URL, access date, citation or license. The station and variables are
consistent with Royal Meteorological Institute of Belgium observations, but
that is not enough to assign redistribution rights. The MIT software license
does not cover these observations. See `data/README.md` and resolve the exact
source/license before public redistribution.
