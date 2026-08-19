# bucex 2.5.0 examples

## Start here: the scientific sequence

The scripts in `presentation/` share one editable `settings.py` and correspond
directly to paper/presentation sections:

| Order | Script | Role |
|---:|---|---|
| 0 | `00_data_and_plan.py` | data integrity and resolved runtime |
| 1 | `01_txx_benchmark_models.py` | fixed TXx structural alternatives |
| 2 | `02_txx_componentwise_ssvs.py` | univariate componentwise selection |
| 3 | `03_txx_forecast_validation.py` | held-out proper scores and PIT |
| 4 | `04_six_univariate_series.py` | no-pooling baseline |
| 5 | `05_hierarchical_laplace_screen.py` | approximate screen/init |
| 6 | `06_hierarchical_exact_pgas.py` | final exact-invariant hierarchy |
| 7 | `07_pooling_sensitivity.py` | slab and both-pooling sensitivity |
| 8 | `08_build_report.py` | recreate output without refitting |

Run a script from the repository root:

```bash
python examples/presentation/00_data_and_plan.py
```

Or use the equivalent CLI:

```bash
bucex-presentation run txx-ssvs --profile pilot --engine pgas --figures
```

## Advanced feature examples

The numbered scripts at the root of `examples/` isolate package features such
as Gaussian FFBS, GEV PGAS diagnostics, prior comparison, parameterization
comparison, bulk/tail analyses, leave-future-out evaluation, Laplace warm
starts, and manual chain combination. They are useful after the main workflow
is understood; they no longer define the paper order.

## Exactness and diagnostics

Gaussian FFBS and PGAS fits target the declared posterior. Laplace fits are
approximations. For final GEV work inspect support failures, particle ESS,
ancestor diversity, path updates, restored iterations, allocation switching,
R-hat, ESS, and sensitivity to particle count. A constant SSVS indicator has
undefined R-hat/ESS and must be interpreted through allocation behavior and
between-chain agreement.

## HPC

`job_scripts/` contains PBS/Torque arrays for every presentation stage. Each
long chain is saved independently and the final job combines the four archives
before generating diagnostics. See `job_scripts/README.md` for dependencies and
environment overrides.
