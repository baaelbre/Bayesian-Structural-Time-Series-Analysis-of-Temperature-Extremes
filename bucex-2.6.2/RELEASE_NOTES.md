# bucex 2.6.2 release notes

Version 2.6.2 makes the COMPSTAT analysis a transparent demonstration of the
core package API.

## Examples and HPC

- The seven numbered scripts now live directly under `examples/`.
- Each script constructs every simulation and fitted model explicitly with
  `Model`, `GEV`, `LocalLevel`/`LocalLinearTrend`, and `DummySeasonal`.
- Scenario factories, workflow configuration objects, result exporters, and
  the `bucex-presentation` command were removed.
- Tables and figures are produced sequentially from `FitResult` methods, so
  every output is traceable to a public API call in the example.
- Results no longer contain a `presentation/` subdirectory.
- Results are always indexed as
  `results/<script>/<timestamp>__<settings-signature>/`; every run contains a
  complete `run_config.json`. `BUCEX_RUN_ID` shares the timestamp prefix across
  scripts and jobs.
- Every example has its own `run_*.sh` positional-argument runner and matching
  `submit_*.pbs` scheduler file. The PBS layer handles resources and logging,
  then passes named `qsub -v` settings to the runner in a documented order.

## Figures

- Uccle descriptive figures begin at 1892 and use a robust local-linear LOESS
  smoother exposed as `bucex.loess_smooth`.
- Shape and scale simulation figures are separate single-series plots, with a
  common y range inside each comparison group.
- `fit.plot("season")` now overlays phase-specific posterior trajectories of
  level plus the current seasonal effect, \(\mu_{ij}+\gamma_{ij}\), against
  cycle or calendar year.
- Structural simulation series remain separate; only the level/slope/seasonal
  decomposition is multi-panel.

## Reproducibility and compatibility

- Simulation scripts use period 4, explicit seeds, and adjacent CSV/JSON truth
  artifacts.
- Laplace and PGAS scripts use the same model, prior, data, and MCMC controls;
  PGAS creates or reuses its matching Laplace initializer.
- Uccle Laplace and PGAS prior settings are identical, including a
  zero-centred fixed-slope prior suitable for both upper tails and
  sign-transformed lower tails.
- Fit archives use schema 2.6.2. Readers retain support for all schema versions
  previously supported by 2.6.1.

The API removal is intentional: code using `bucex.workflows`, scenario
factories, or `bucex-presentation` should move the relevant declarations into
an analysis script, following the seven examples.
