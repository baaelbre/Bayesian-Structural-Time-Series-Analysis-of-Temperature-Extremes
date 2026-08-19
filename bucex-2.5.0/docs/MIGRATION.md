# Migration to 2.5.0

The public `Model`, `MultiSeriesModel`, `fit`, `FitResult`, forecast, score, and
plot APIs remain available. The important default change is scientific:

```python
bx.HierarchicalPrior()
```

now means:

```python
bx.HierarchicalPrior(
    pool="selection",
    model_space="componentwise",
    level_states=("fixed", "dynamic"),
    trend_states=("zero", "fixed", "dynamic"),
    season_states=("fixed", "dynamic"),
)
```

Code relying on the 2.4.1 four-class joint trend default must request it:

```python
bx.HierarchicalPrior(
    pool="selection",
    model_space="joint_trend",
    trend_states=("fixed", "dynamic"),
    trend_model_concentration=(1, 1, 1, 1),
)
```

## New workflow API

```python
config = bx.PresentationConfig.for_profile(
    "pilot", output_dir="results/presentation"
)
workflow = bx.PresentationWorkflow(config)
workflow.run_txx_ssvs()
workflow.run_hierarchy_screen()
workflow.run_hierarchy_pgas()
workflow.report()
```

The equivalent CLI is `bucex-presentation`. Existing direct calls to
`fit_uccle_series()` and `fit_uccle_hierarchical()` continue to work.

## Archives

New archives use schema 2.5.0. Loading remains checksum-verified and pickle-free,
and readers accept earlier supported schemas including 2.4.1.

## Result directories

The workflow refuses to overwrite existing fits unless `overwrite=True` or
`--overwrite` is supplied. HPC array tasks save individual chain files; use the
`combine` command before interpreting multi-chain diagnostics.

## Laplace and PGAS

No contract changed: Laplace remains approximate; Gaussian FFBS is exact;
PGAS is exact-invariant. An external Laplace fit may initialize a compatible
hierarchical PGAS run but does not shorten required warmup or transfer its
posterior uncertainty.
