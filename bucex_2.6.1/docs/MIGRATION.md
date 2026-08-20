# Migration from 2.5.0 to 2.6.1

The general `Model`, `MultiSeriesModel`, `fit`, `FitResult`, prediction,
diagnostic, plotting, Uccle-loader, and hierarchical APIs remain available.
The intentional breaking change is the presentation workflow: 2.6 removes the
old benchmark/validation/hierarchy orchestration and replaces it with the
focused simulation-to-four-extremes sequence.

## Workflow mapping

Removed presentation methods include:

```text
run_txx_benchmarks
run_txx_ssvs
run_txx_validation
run_six_univariate
run_hierarchy_screen
run_hierarchy_pgas
run_sensitivity
```

Use:

```python
import bucex as bx

workflow = bx.PresentationWorkflow(
    bx.PresentationConfig.for_profile(
        "pilot", output_dir="results/presentation"
    )
)

workflow.run_data()
workflow.run_simulations(kind="tail")
workflow.run_simulations(kind="structure")
workflow.fit_simulations(engine="laplace")
workflow.fit_simulations(engine="pgas")
workflow.fit_uccle(engine="laplace")
workflow.fit_uccle(engine="pgas")
workflow.report(strict=True)
```

Direct hierarchical analysis is unaffected; only its old presentation wrapper
was removed.

## New univariate warm starts

In 2.5, a univariate `FitResult` could not be supplied as `init=`. In 2.6:

```python
laplace = bx.fit(y, family="gev", period=12, engine="laplace", ...)
pgas = bx.fit(y, family="gev", period=12, engine="pgas", init=laplace, ...)
```

Compatibility checks cover family, period, transformed observations, and
state-path dimensions. `warm_start()` includes the full centred path, signed
FS coefficients, and static parameters. An explicit conflicting
`state_kwargs["initial_centered_path"]` is rejected.

## Paths and chain combination

Simulation fits now live at:

```text
fits/simulations/<engine>/<scenario>/chain_XX.bucex
fits/simulations/<engine>/<scenario>/combined.bucex
```

Observed fits use the analogous `fits/uccle/<engine>/<series>/` layout. Run the
appropriate `combine simulation-fit` or `combine uccle-fit` command after HPC
array tasks. PGAS requires the combined Laplace path.

## Archives

New archives use schema 2.6.1. Loading remains checksum-verified and
pickle-free, and readers accept every schema previously supported by 2.5.0.
