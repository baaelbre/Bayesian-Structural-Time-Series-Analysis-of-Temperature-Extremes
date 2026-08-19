# Uccle analysis

## Data

| Name | Family | Tail | Description |
|---|---|---|---|
| `TXm` | Gaussian | ordinary | monthly mean daily maximum |
| `TNm` | Gaussian | ordinary | monthly mean daily minimum |
| `TXx` | GEV | upper | monthly maximum daily maximum |
| `TXn` | GEV | lower | monthly minimum daily maximum |
| `TNx` | GEV | upper | monthly maximum daily minimum |
| `TNn` | GEV | lower | monthly minimum daily minimum |

Lower-tail series are sign-transformed internally and mapped back on output.
All loaders require complete consecutive months.

```python
import bucex as bx

print(bx.validate_uccle_data())
txx = bx.load_uccle_series("TXx", start="1980-01-01")
six = bx.load_uccle_multiseries(start="1980-01-01")
```

## Presentation sequence

The v2.5 workflow is a staged scientific argument, not merely a collection of
sampler demonstrations:

1. validate the data and show why nonstationary extremes matter;
2. introduce the full model using TXx only;
3. fit controlled stationary/linear/local-level/RW2/local-linear alternatives;
4. let componentwise SSVS represent structural uncertainty;
5. compare all TXx alternatives by leave-future-out scores and PIT;
6. establish six independent no-pooling results;
7. share selection probabilities across all six summaries;
8. use Laplace only to screen/initialize and PGAS for final mixed inference;
9. test shared slabs and rebuild results from archived fits.

```bash
bucex-presentation plan --profile pilot
bucex-presentation run txx-benchmarks --profile pilot --engine pgas
bucex-presentation run txx-ssvs --profile pilot --engine pgas
bucex-presentation run txx-validation --profile pilot --engine pgas
```

The literature-labelled TXx alternatives are structural analogues under a
common GEV likelihood and inference implementation. The comparison focuses on
where those restrictions matter: held-out prediction, PIT calibration, GEV
support, endpoint/shape behavior, and risk estimates. Poor mixing is a
computational failure, but good mixing alone does not validate a model.

## Six independent series

```python
workflow = bx.PresentationWorkflow(
    bx.PresentationConfig.for_profile("pilot")
)
workflow.run_six_univariate(gev_engine="pgas")
```

This is the no-pooling comparator. It reveals what each series identifies by
itself and whether the hierarchy later changes a conclusion through borrowing.

## Hierarchical selection

```python
prior = bx.componentwise_hierarchical_prior("selection")
screen = bx.fit_uccle_hierarchical(
    priors=prior,
    start="1892-01-01",
    engine="laplace",
    mcmc=bx.MCMC(draws=1_000, warmup=1_000, chains=4, seed=2501),
)

exact = bx.fit_uccle_hierarchical(
    priors=prior,
    start="1892-01-01",
    engine="pgas",
    init=screen,
    mcmc=bx.MCMC(draws=2_000, warmup=2_000, chains=4, seed=2502),
    particles=bx.Particles(n=512, proposal="guided"),
)
```

The screen is approximate. The PGAS result performs full warmup and has its own
exact-invariant target and diagnostics. A 1980 screen cannot initialize an
1892 fit because the stored paths and observations differ.

## What to report

For TXx and every later series:

- component and induced joint structural probabilities;
- model-averaged level, slope, predictor, and finite-period rate;
- process SD mixtures including exact mass at zero;
- observation scale and GEV shape/endpoint where relevant;
- exceedance probabilities/return summaries on the original orientation;
- R-hat, ESS, allocation switching, support and particle diagnostics;
- LFO score/PIT summaries for claims of predictive adequacy.

For the hierarchy:

- population component probabilities with intervals;
- channel-specific allocations and paths;
- slab multipliers only for models that estimate them;
- sensitivity to exchangeability, record length, concentration, and slabs;
- an explicit statement that residual/copula dependence is not modeled.

Use model-averaged summaries. Converting a probability such as 0.55 versus 0.45
into a hard state throws away the central posterior result.

## Artifact layout

The workflow writes `config.json`, a concurrency-safe `manifest.json`,
checksummed fits, tidy tables, and figures under one result directory. Run

```bash
bucex-presentation report --profile publication --figures
```

to regenerate presentation material without resampling.
