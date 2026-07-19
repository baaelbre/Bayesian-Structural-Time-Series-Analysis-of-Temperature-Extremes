# Migration to bucex v0.3.3

Version 0.3.3 is a diagnostic and HPC-workflow release. The statistical
Laplace update, prior profiles and posterior parameterisation from v0.3.2 are
unchanged.

## DGEV restoration messages

A former message such as

```text
[it 500/8000] ... [restored]
```

now reports the failed stages:

```text
[it 500/8000] ... [restored attempts=25 reasons=(support_after_state_parameters:25)]
```

Aggregate counts are stored in the fitted object under:

```python
fit.meta["restored_iterations"]
fit.meta["restored_fraction"]
fit.meta["attempt_failure_counts"]
fit.meta["restore_failure_counts"]
```

No existing fitting call needs to change.

## HPC layout

`jobs/fit_uccle_array.pbs` now launches 18 tasks: six series times three
independent chains. The defaults are 8,000 iterations, 1,500 burn-in, no
thinning and `priors="regularized"`.

Outputs are written to:

```text
results/uccle_v033_<priors>/chains/chain_1/
results/uccle_v033_<priors>/chains/chain_2/
results/uccle_v033_<priors>/chains/chain_3/
```

Pool completed chains with:

```bash
python examples/pool_uccle_chains.py \
    --chains-dir results/uccle_v033_regularized/chains \
    --out-dir results/uccle_v033_regularized
```

The old one-series shell runner remains available and now defaults to 8,000
iterations, 1,500 burn-in and the regularized profile.
