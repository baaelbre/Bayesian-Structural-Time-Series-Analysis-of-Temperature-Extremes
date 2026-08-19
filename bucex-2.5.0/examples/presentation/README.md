# Uccle paper and presentation sequence

Edit `settings.py` once, then run the numbered scripts in order.  The same
sequence is available through the `bucex-presentation` command and the PBS
files in `examples/job_scripts/`.

| Stage | Question | Main output |
|---:|---|---|
| 0 | Are the six monthly summaries complete and aligned? | data integrity table |
| 1 | What do common fixed structural assumptions imply for TXx? | five benchmark fits |
| 2 | Which TXx components are supported when structure is uncertain? | componentwise SSVS fit |
| 3 | Which TXx models predict held-out extremes adequately? | LFO scores and PIT |
| 4 | What does each of the six summaries say without pooling? | six independent fits |
| 5 | Where is posterior mass in the pooled model? | approximate Laplace screen |
| 6 | What is the final pooled posterior? | exact-invariant PGAS fit |
| 7 | Do conclusions depend on sharing slab magnitudes? | pooling sensitivity |
| 8 | Can the presentation be rebuilt without sampling? | regenerated tables/figures |

The literature-labelled Stage 1 models are structural analogues in a common
GEV likelihood and sampler. They are not bit-for-bit replications of Huerta--
Sansó or Gaetan--Grigoletto. Stage 3 is therefore essential: comparison rests
on held-out predictive performance, calibration, support behavior, and risk
implications—not only on a smooth fitted curve.

For a software check:

```bash
bucex-presentation run all --profile smoke --output-dir results/smoke \
  --include-sensitivity
```

For final runs, use the PBS arrays. Every `.bucex` archive is checksummed and
the `manifest.json` file records which stage produced each artifact.
