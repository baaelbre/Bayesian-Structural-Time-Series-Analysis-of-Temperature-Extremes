# bucex 2.1.4 play scripts

These scripts are deliberately small experiments, not publication analyses.
They all use the same configuration in `play_config.py`, print the inference
plan, set `progress=True` by default, and finish with a small set of relevant
plots.

| Script | Question to play with |
|---|---|
| `quickstart.py` | Does the same structural model work for Gaussian and GEV observations? |
| `compare_parameterizations.py` | How do centered, disturbance, and Fruehwirth--Schnatter sampling compare? |
| `compare_priors.py` | How sensitive are the process innovation SDs to the prior profile? |
| `identified_gaussian_factor.py` | What is shared, what is idiosyncratic, and is that split identified? |
| `dynamic_factor.py` | Can a Gaussian series and a GEV extreme share one dynamic factor? |
| `combined_bulk_tail.py` | How do independent Gaussian bulk and GEV tail analyses look side by side? |
| `uccle_factor.py` | How do the six Uccle summaries enter the proposed factor analysis? |

Start with, for example:

```bash
python examples/identified_gaussian_factor.py
```

The default is a short exploratory run. A longer run requires no source-code
changes:

```bash
BUCEX_QUICK=0 BUCEX_SHOW=1 python examples/identified_gaussian_factor.py
```

Every setting can also be overridden separately:

```bash
BUCEX_DRAWS=500 BUCEX_WARMUP=750 BUCEX_CHAINS=4 \
BUCEX_PARTICLES=256 BUCEX_PROGRESS_EVERY=25 python examples/dynamic_factor.py
```

Set `BUCEX_SHOW=0` for a non-interactive run and `BUCEX_PROGRESS=0` to silence
progress. Short runs are useful for API checks only; use multiple long chains
and inspect R-hat, effective sample sizes, traces, and factor-identification
diagnostics before making scientific claims.
