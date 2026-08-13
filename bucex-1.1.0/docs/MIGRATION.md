# Migration to bucex 1.1

Version 1.1 reverses the most disruptive part of the 1.0 migration: the 0.3 FS
implementation is restored as a first-class engine rather than approximated by
the general disturbance-NCP machinery.

## From 0.3

Most compact scripts continue unchanged:

```python
fit = bx.fit_bayes(
    y,
    family="gev",
    priors="manuscript",
    state_method="laplace",
    n_iter=4000,
    burn=2000,
)
```

Recommended explicit updates:

| 0.3 name | 1.1 recommendation |
| --- | --- |
| `parameterization="noncentered"` | `parameterization="fruehwirth_schnatter"` |
| `priors="regularized"` | `priors="regularized_lasso"` |
| `state_method="particle"` | `state_method="pgas"` |
| separate chain files only | `chains=4`, or retain separate files and pool explicitly |
| `.pkl` fit files | checksummed `.bucex` archives |

The old aliases remain accepted. Loading arbitrary pickle was intentionally not
carried forward; it executes code and cannot satisfy the release safety
contract. Refit or migrate trusted historical objects in an isolated legacy
environment, then save arrays and metadata in the new format.

New FS profiles are `horseshoe` and `pc`. Existing manuscript lasso, normal,
regularized/componentwise lasso, and structural SSVS profiles remain.

`fit.state_draws("alpha")` still selects a named state. The same attribute now
also exposes chain-shaped array metadata:

```python
fit.draws_states.shape       # flat legacy view
fit.state_draws.shape        # (chains, draws, time, state)
```

## From 1.0

The declarative interface is retained:

```python
fit = bx.fit(y, model=model, mcmc=bx.MCMC(...))
```

`Model`, `compile_model`, dynamic regression, general process-SD priors,
disturbance NCP, ASIS, forecasts, scores, and `FitResult` archives remain. The
general PGAS implementation now retains normalized log weights for ancestor
sampling, fixing the rare all-weights-zero failure caused by probability
underflow.

The main semantic change is naming clarity:

- `fit` means the general compiled engine;
- `fit_fs` / `fit_bayes` means the FS augmented engine;
- general continuous `SpikeSlabSD` is not called exact SSVS;
- FS `ssvs` retains exact zero/fixed/dynamic model states;
- `regularized_lasso`, `horseshoe`, and `pc` are distinct FS prior profiles.

## Mixed projects

It is valid to use both interfaces in one project, for example FS models for
the six manuscript series and the general model for a regression sensitivity
analysis. Do not merge their raw parameter arrays without acknowledging the
different parameterisations. Compare semantic centred trajectories, forecasts,
and predictive scores instead.

## Output and automation

Quick and production runs should use different output paths. `fit.save` does
not choose or overwrite a destination implicitly. Production recommendations
remain four chains, rank-normalized R-hat and ESS review, prior-versus-posterior
process-SD inspection, PGAS/Laplace sensitivity for GEV models, and held-out
tail scoring.
