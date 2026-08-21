# Migration to 2.6.2

The general `Model`, `MultiSeriesModel`, `fit`, `FitResult`, prediction,
diagnostic, plotting, Uccle-loader, and hierarchical APIs remain available.

## Presentation workflow removal

Version 2.6.2 removes `bucex.workflows`, `PresentationConfig`,
`PresentationWorkflow`, `WorkflowPaths`, the scenario factories, result
workflow helpers, and the `bucex-presentation` console command. These objects
hid analysis choices that are more useful when visible.

Replace a workflow call with the corresponding numbered script, or copy its
explicit public-API declarations into an analysis:

```python
import bucex as bx

model = bx.Model(
    bx.GEV(),
    (
        bx.LocalLinearTrend(level_mode="dynamic", trend_mode="dynamic"),
        bx.DummySeasonal(period=12, mode="dynamic"),
    ),
)
laplace = bx.fit(y, model=model, priors=priors, engine="laplace", ...)
pgas = bx.fit(y, model=model, priors=laplace.priors, engine="pgas", init=laplace, ...)
```

The seven complete examples are in `examples/`; matching PBS jobs are in
`examples/job_scripts/`.

## Result paths

Results are always indexed by script, timestamp, and identifying settings:

```text
results/<script>/<BUCEX_RUN_ID>__<settings-signature>/
  run_config.json
  simulations/               # simulation scripts
  fits/<scenario-or-series>/
  tables/<scenario-or-series>/
  figures/<scenario-or-series>/
```

Export one `BUCEX_RUN_ID` before running multiple scripts to share a timestamp
prefix. The settings signature is always present.

## New plots

`bx.loess_smooth()` replaces rolling-median smoothing in the Uccle descriptive
example. `fit.plot("season")` plots phase-specific posterior trajectories of
level plus the current seasonal effect.

## Warm starts and archives

The full-path Laplace-to-PGAS warm-start contract is unchanged. Compatibility
checks cover family, period, transformed observations, and state dimensions.
New archives use schema 2.6.2; loading remains checksum-verified and
pickle-free, with readers for all versions previously supported by 2.6.1.
