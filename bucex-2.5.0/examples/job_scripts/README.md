# PBS/Torque jobs for the full Uccle presentation

The scheduler files use the same `bucex-presentation` stages as local runs.
`run_presentation_stage.sh` activates `${BUCEX_VENV}` (default
`${HOME}/venvs/bucex-2.5.0`) and invokes the package module. Submit from the
repository root.

## Job sequence

| File | Array | Output |
|---|---:|---|
| `00_submit_data.pbs` | no | integrity table |
| `01_submit_txx_benchmarks.pbs` | 5 models × 4 chains | TXx benchmark chains |
| `02_submit_txx_ssvs.pbs` | 4 chains | TXx SSVS chains |
| `03_submit_txx_validation.pbs` | 6 models | LFO score/PIT tables |
| `04_submit_six_univariate.pbs` | 6 series × 4 chains | independent chains |
| `05_submit_hierarchy_screen.pbs` | no | full-record Laplace screen |
| `06_submit_hierarchy_pgas.pbs` | 4 chains | exact hierarchy chains |
| `07_submit_sensitivity_screen.pbs` | slab/both | approximate sensitivity |
| `08_submit_combine_report.pbs` | no | combined fits, tables, figures |

The array throttles (for example `%2`) and wall times are templates. Change
them to match the local PBS flavor and timing pilot.

## Environment

```bash
qsub -v BUCEX_VENV=/path/to/venv,OUTPUT_DIR=/scratch/me/uccle \
  examples/job_scripts/00_submit_data.pbs
```

Every job limits BLAS/OpenMP libraries to one thread. `--workers 6` in the
hierarchical jobs uses the six allocated CPUs for conditionally independent
channel updates; it should never exceed `ppn`.

The shared runner also translates environment overrides `START`, `END`,
`DRAWS`, `WARMUP`, `CHAINS`, `PARTICLES`, `WORKERS`, `SEED`, `INITIAL`,
`HORIZON`, and `STEP` into CLI options. For a timing pilot:

```bash
qsub -v DRAWS=100,WARMUP=100,PARTICLES=128 \
  examples/job_scripts/06_submit_hierarchy_pgas.pbs
```

Set `OVERWRITE=true` only when replacing known failed/test artifacts.

## Dependencies

PBS dependency spelling varies. A typical Torque sequence is:

```bash
DATA=$(qsub examples/job_scripts/00_submit_data.pbs)
BENCH=$(qsub -W depend=afterok:${DATA} examples/job_scripts/01_submit_txx_benchmarks.pbs)
TXSSVS=$(qsub -W depend=afterok:${DATA} examples/job_scripts/02_submit_txx_ssvs.pbs)
LFO=$(qsub -W depend=afterok:${DATA} examples/job_scripts/03_submit_txx_validation.pbs)
UNI=$(qsub -W depend=afterok:${DATA} examples/job_scripts/04_submit_six_univariate.pbs)
SCREEN=$(qsub -W depend=afterok:${DATA} examples/job_scripts/05_submit_hierarchy_screen.pbs)
HIER=$(qsub -W depend=afterok:${SCREEN} examples/job_scripts/06_submit_hierarchy_pgas.pbs)
SENS=$(qsub -W depend=afterok:${DATA} examples/job_scripts/07_submit_sensitivity_screen.pbs)
```

Submit `08_submit_combine_report.pbs` only after all chain arrays have
finished. Array-dependency syntax is not portable enough to hard-code safely.

The older `submit_uccle_componentwise_*.pbs` files are retained as compact
compatibility templates for only the hierarchy stage. New work should use the
numbered full-sequence files.
