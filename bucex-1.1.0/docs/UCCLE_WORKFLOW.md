# Uccle workflow

## 1. Verify the supplied data

```bash
python - <<'PY'
import bucex as bx
print(bx.validate_uccle_data("data", check_daily=True))
PY
```

The six canonical series are `TXm`, `TNm`, `TXx`, `TXn`, `TNx`, and `TNn`.
Monthly files are included in installed packages; the daily reconstruction
source is included only in the complete source release. `TXn` and `TNn` are
block minima and are fitted through the documented sign transformation.

## 2. Run a local fit

```bash
python examples/fit_uccle_series.py \
  --series TXx \
  --data-dir data \
  --out-dir results/uccle_v1_1 \
  --engine laplace \
  --priors regularized_lasso \
  --asis \
  --draws 2000 \
  --warmup 2000 \
  --chains 4 \
  --seed 40
```

This writes a self-contained `.bucex` fit plus posterior tables, diagnostics,
period-rate summaries, and figures. Use `--engine pgas --particles N` for the
exact-invariant GEV state kernel. Particle count is an accuracy/efficiency
choice; inspect minimum ESS, ancestor diversity, and path refresh.

## 3. Run four independent HPC chains

The PBS array has 24 tasks: four chains for each of six series.

```bash
qsub jobs/fit_uccle_array.pbs
```

Environment overrides include `BUCEX_DRAWS`, `BUCEX_WARMUP`, `BUCEX_THIN`,
`BUCEX_PRIORS`, `BUCEX_GEV_ENGINE`, `BUCEX_PARTICLES`, `BUCEX_DATA_DIR`, and
`BUCEX_OUT_ROOT`. Each task is single-core and writes a chain-specific safe
archive.

Pool compatible chain files without flattening chain identity:

```bash
python examples/pool_uccle_chains.py \
  results/uccle_v1_1/chains/chain_1/fits/TXx.bucex \
  results/uccle_v1_1/chains/chain_2/fits/TXx.bucex \
  results/uccle_v1_1/chains/chain_3/fits/TXx.bucex \
  results/uccle_v1_1/chains/chain_4/fits/TXx.bucex \
  --output results/uccle_v1_1/fits/TXx.bucex \
  --diagnostics results/uccle_v1_1/diagnostics/TXx_mcmc.csv
```

## 4. Score a chronological holdout

```bash
python examples/score_uccle_holdout.py \
  --series TXx \
  --data-dir data \
  --holdout 120 \
  --engine laplace \
  --draws 1000 \
  --warmup 1000 \
  --chains 4
```

The script saves the training fit, predictive summary, and proper-score table.
Thresholds are training-sample 90%, 95%, and 99% tail quantiles; minima use the
corresponding lower-tail quantiles.

## 5. Production review

For each series, review the retained `fit.priors`, `fit.plot("process_sd")`,
R-hat, ESS, acceptance, PIT, and engine-specific diagnostics. For GEV series,
compare the production Laplace result with at least one adequately tuned PGAS
run. Treat the
Gaussian and GEV analyses as independent even when wrapped in `BulkTailFit`.
Do not pair their draws or describe them as a joint posterior.
