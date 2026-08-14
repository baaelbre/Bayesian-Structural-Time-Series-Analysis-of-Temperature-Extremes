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
    mcmc=bx.MCMC(
        draws=2_000,
        warmup=1_000,
        chains=4,
        seed=40,
        progress=True,
    ),
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

## Fit the v2.1 shared-factor model

Version 2.1 provides the manuscript's one-factor graph directly:

```python
data = bx.load_uccle_factor_data(
    start="1892-01-01",
    end="2022-12-01",
)
model = bx.make_uccle_factor_model()
compiled = bx.compile_model(model, data)
priors = bx.identified_factor_priors(
    compiled,
    profile="regularized_horseshoe",
    smooth_factor=True,
    reference_channel="TXm",
)
joint = bx.fit(
    data,
    model,
    engine="pgas",
    parameterization="fruehwirth_schnatter",
    priors=priors,
    asis=True,
    mcmc=bx.MCMC(
        draws=2_000,
        warmup=2_000,
        chains=4,
        seed=50,
        progress=True,
    ),
    particles=bx.Particles(n=1_024, proposal="guided"),
)
```

The shortcut `fit_uccle_factor(...)` loads and fits the same graph. Its one
common local-linear trend is anchored at `TXm=1`. Each channel has an
independent dynamic local level and an independent dynamic dummy-seasonal
block. Lower-tail loading initial values are converted to the internal sign
orientation. The helper fixes both factor initial conditions at zero; the
factor remains dynamic through its level and slope innovation scales.

The default factor prior is a regularized horseshoe on the six idiosyncratic
local-level innovation scales only. This continuous shrinkage lets a channel
decouple when supported while strongly tethering weak idiosyncratic dynamics.
The shared-factor and seasonal innovations retain calibrated process priors.

Use `parameterization="disturbance"` as a sensitivity analysis or
`priors="regularized"` to replace the horseshoe with PC process-SD priors. The
older `structure="contrasts"` helper remains readable for v2.0 workflows, but
it is not the v2.1 manuscript model and is not eligible for the one-factor FS
layout.

Useful summaries are:

```python
joint.factor("common")
joint.factor_rate_summary("common", 1950, 2022)
joint.loading_probability("common", "TXx", threshold=1.0)
joint.factor_probabilities("common", start_year=1950, end_year=2022)
joint.reconstructed_state("TXx")
joint.normalized_factor(slice(0, 30 * 12), "common")
joint.channel_decomposition("TXx", baseline=slice(0, 30 * 12))
joint.channel_rate_draws("TXx", 1950, 2022)
joint.idiosyncratic_innovation_draws("TXx")
joint.factor_identification_diagnostics()
joint.plot("factor_decomposition", baseline=slice(0, 30 * 12))
joint.plot("identification", baseline=slice(0, 30 * 12))
```

A `TXx` loading above one concerns the shared-factor contribution; a claim
about the complete `TXx` rate must also include its idiosyncratic deviation.
The fixed `TXm` loading identifies scale/sign; the smooth-factor and pure-
reference restrictions above improve dynamic separation. They do not make an
unrestricted loading/random-walk split automatically identifiable, so report
the identification diagnostics and sensitivity fits with decomposition
claims.
The helper retains conditional channel independence and does not add a
mean/extreme residual copula.

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
