# Migration to 2.4

Version 2.4 narrows the public model surface to univariate structural models
and hierarchical collections of separate structural paths.

## Public model classes

Use `Model` for one series:

```python
model = bx.Model(
    bx.Gaussian(),
    (bx.LocalLinearTrend(), bx.DummySeasonal(12)),
)
fit = bx.fit(y, model, priors="normal")
```

Use `MultiSeriesModel` for related series:

```python
model = bx.MultiSeriesModel(
    tuple(
        bx.Channel(
            name,
            bx.Gaussian(),
            (bx.LocalLinearTrend(), bx.DummySeasonal(12)),
        )
        for name in data.columns
    )
)
fit = bx.fit(data, model, priors=bx.HierarchicalPrior(pool="selection"))
```

`PanelModel` aliases and specialised multichannel fitter names are not part of
the 2.4 API. The single entry point is `bx.fit`.

## Hierarchical prior names

Preferred:

```python
bx.HierarchicalPrior(pool="selection")
bx.HierarchicalPrior(pool="slab")
bx.HierarchicalPrior(pool="both")
```

Convenience strings are `pooled_selection`, `pooled_slab`, and `pooled_both`.
The result prefixes are now consistently `hierarchy.prob.*` and
`hierarchy.slab_scale.*`.

Monthly seasonality defaults to `season_states=("fixed", "dynamic")`. Add
`"zero"` explicitly only for applications where absence of seasonality is a
scientifically plausible state.

## Initial states

Initial level and slope are estimated. Canonical univariate names are
`initial.level` and `initial.slope`; multiseries names are
`initial.channel.<name>.level` and `initial.channel.<name>.slope`.

Data-based values initialise the chains but do not fix the posterior. If exact
custom starts are needed, pass:

```python
fit = bx.fit(
    y,
    model,
    init={"initial.level": 10.0, "initial.slope": 0.01},
)
```

For a multiseries model, use channel prefixes, for example
`channel.TXm.initial.level` and `channel.TXm.initial.slope`.

## Plot changes

`fit.plot("level_slope")` now compares latent level with seasonally adjusted
observations. The predictor plot still displays the original observations and
the complete level-plus-season predictor.

Every plot accepts either a path or save options:

```python
fit.plot("acf", save="figures/acf.png")
fit.plot("process_sd", save={"path": "figures/sd.png", "dpi": 200})
```

## PGAS changes

Guided disturbance PGAS now projects its conditioned path onto exact affine
support, recovers disturbances through the scaled transition loading, and
retains the conditioned predecessor if all optional ancestor candidates are
invalid. This fixes reproducible all-zero ancestor weights for singular
transitions.

The FS PGAS kernel applies the same conditioned-predecessor rule. Existing
scientific fits should be rerun because the old failure could terminate a chain
or encourage users to rely on an incomplete run.

## Archives

New hierarchical results use schema 2.4. Save and load with:

```python
fit.save("results/model.bucex")
restored = bx.FitResult.load("results/model.bucex")
```

Archives remain checksummed and pickle-free.
