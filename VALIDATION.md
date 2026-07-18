# bucex v0.2 validation

Validation performed for the v0.2 release candidate:

- Python byte-code compilation of the full `bucex` package;
- four API/regression tests (`python -m pytest -q`);
- wheel build and clean wheel import;
- short full-record smoke fits for all six Uccle series (1,572 monthly observations each):
  TXm, TNm, TXx, TXn, TNx and TNn;
- smoke plots for `level`, `slope`, `level_slope`, `exceedance`,
  `return_period`, and `endpoint`.

The short smoke fits verify execution and object consistency only. They are not
convergence runs and are not evidence that a 20,000-iteration chain reproduces
the manuscript tables exactly. Before publication-quality use, run the full
chains, inspect trace plots/effective sample sizes, and compare selected
posterior summaries against the original manuscript implementation.
