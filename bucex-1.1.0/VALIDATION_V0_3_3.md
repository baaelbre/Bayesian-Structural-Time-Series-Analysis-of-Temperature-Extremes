# Validation of bucex v0.3.3

Completed validation for this release:

1. all 17 automated tests pass;
2. the v0.3.2 covariance, regularisation, SSVS and finite-rate tests remain
   unchanged and pass;
3. a forced-restoration test verifies that the progress line prints the failed
   stage and attempt count;
4. DGEV fits save `restored_iterations`, `restored_fraction`,
   `attempt_failure_counts`, `restore_failure_counts`, and `max_state_tries`;
5. `jobs/run_uccle_series.sh`, `jobs/fit_uccle_array.pbs`, and
   `jobs/fit_uccle_single.pbs` pass `bash -n` syntax checking;
6. a short full-data TNx run through `run_uccle_series.sh` completes and writes
   the fit, metadata, summaries and figures;
7. a wheel built from the source imports as version 0.3.3 and completes short
   Gaussian and DGEV Uccle smoke fits;
8. the multi-chain pooling helpers concatenate compatible posterior arrays,
   omit duplicate non-centred trajectories from the pooled fit, and compute
   classical split Rhat for scalar static parameters.

The statistical Laplace update is unchanged from v0.3.2. Version 0.3.3 adds
observability around existing restoration behaviour; it does not claim to
eliminate GEV support failures.

The split Rhat written by `pool_uccle_chains.py` is the classical split version,
not rank-normalized Rhat. It is intended as a lightweight first diagnostic.
