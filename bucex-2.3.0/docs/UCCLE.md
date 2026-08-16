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
    priors="ssvs",
    parameterization="fruehwirth_schnatter",
    mcmc=bx.MCMC(draws=2_000, warmup=1_000, chains=4, seed=40),
)

fits.summary()
fits.save("results/uccle")
```

Each series receives a deterministic child seed. Collection members are
ordinary `FitResult` objects.

`examples/09_uccle_univariate.py` runs all six summaries separately and
prints the zero/fixed/dynamic SSVS probabilities and switching diagnostics for
each. For GEV summaries its SSVS route uses the explicitly labelled Laplace
approximation only when `GEV_ENGINE="laplace"`. Its v2.2 default is exact-
invariant PGAS--SSVS; Laplace pseudo-information then proposes model moves but
the exact GEV likelihood determines acceptance.

## Fit the six-series hierarchical SSVS model

Version 2.3 adds a genuine joint alternative between six separate fits and the
common-factor model:

```python
model = bx.make_uccle_multiseries_model()
hierarchical = bx.fit_uccle_multiseries(
    model=model,
    start="1980-01-01",
    priors="hierarchical_ssvs",
    engine="pgas",
    parameterization="fs",
    asis=False,
    mcmc=bx.MCMC(
        draws=2_000,
        warmup=2_000,
        chains=4,
        seed=45,
        progress=True,
    ),
    particles=bx.Particles(n=1_024, proposal="guided"),
)

hierarchical.component_probabilities()
hierarchical.hierarchical_probabilities()
hierarchical.hierarchical_slab_summary()
hierarchical.component_transition_summary()
hierarchical.channel_rate_summary("TXx", 1950, 2022)
```

Each summary has its own local-linear-trend and seasonal predictor. The shared
Dirichlet probabilities learn how often level/trend/season states are
zero/fixed/dynamic, and the half-t hierarchy learns common dynamic-slab
multipliers. There is no latent common warming path and no loading. Lower-tail
series retain their original temperature orientation in results.

Because the graph contains GEV channels, `engine="auto"` resolves to PGAS. Its
Laplace information is used only for exact-likelihood-corrected model
proposals. Inspect particle diagnostics, restored iterations, allocation
switching, continuous-hyperparameter R-hat/ESS, and sign-invariance errors.

`examples/15_uccle_hierarchical_ssvs.py` is the full executable workflow.
`examples/16_hierarchy_or_factor.py` explains the estimand difference.

## Fit the shared-factor model

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
    profile="regularized_triple_gamma",
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

The package default remains a regularized horseshoe, while the example above
uses the new regularized triple gamma. Both act only on the six idiosyncratic
local-level innovation scales. Triple gamma additionally stores the
interpretable shrinkage factor `rho` for each selected process; its optional
slab caps extremely large local variances. Shared-factor and seasonal
innovations retain calibrated process priors.

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
joint.channel_rate_summary("TXx", 1950, 2022)
joint.idiosyncratic_innovation_draws("TXx")
joint.factor_identification_diagnostics()
joint.plot("factor_decomposition", baseline=slice(0, 30 * 12))
joint.plot("identification", baseline=slice(0, 30 * 12))
joint.plot("acf", save="figures/uccle_factor_acf.png")
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

Use the factor model as primary when the manuscript asks whether the six
summaries exhibit heterogeneous responses to a common warming signal. Use the
hierarchical model as primary when it asks which types of structural dynamics
recur across summaries. A strong design for the former question is the factor
as the main analysis, hierarchical SSVS as a structural sensitivity analysis,
and six separate SSVS fits as the no-pooling boundary.

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
