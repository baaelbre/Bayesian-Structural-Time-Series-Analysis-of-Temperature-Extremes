#!/usr/bin/env bash
# Submit all seven examples with one shared timestamp and dependency graph.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export BUCEX_RUN_ID="${BUCEX_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export BUCEX_TIMESTAMP_RESULTS="${BUCEX_TIMESTAMP_RESULTS:-1}"

# -V exports BUCEX_RUN_ID plus optional BUCEX_PROFILE, BUCEX_PYTHON,
# BUCEX_RESULTS_ROOT, BUCEX_DATA_DIR, and explicit sampler overrides.
UCCLE_RECORD_JOB="$(qsub -V 00_uccle_record.pbs)"
TAIL_JOB="$(qsub -V 01_tail_simulations.pbs)"
STRUCTURE_JOB="$(qsub -V 02_structural_simulations.pbs)"

SIM_LAPLACE_JOB="$(qsub -V -W "depend=afterok:$STRUCTURE_JOB" 03_simulation_laplace.pbs)"
SIM_PGAS_JOB="$(qsub -V -W "depend=afterok:$SIM_LAPLACE_JOB" 04_simulation_pgas.pbs)"

UCCLE_LAPLACE_JOB="$(qsub -V -W "depend=afterok:$UCCLE_RECORD_JOB" 05_uccle_laplace.pbs)"
UCCLE_PGAS_JOB="$(qsub -V -W "depend=afterok:$UCCLE_LAPLACE_JOB" 06_uccle_pgas.pbs)"

printf 'run ID:                 %s\n' "$BUCEX_RUN_ID"
printf 'Uccle record:           %s\n' "$UCCLE_RECORD_JOB"
printf 'tail/scale simulation:  %s\n' "$TAIL_JOB"
printf 'structural simulation:  %s\n' "$STRUCTURE_JOB"
printf 'simulation Laplace:     %s\n' "$SIM_LAPLACE_JOB"
printf 'simulation PGAS:        %s\n' "$SIM_PGAS_JOB"
printf 'Uccle Laplace:          %s\n' "$UCCLE_LAPLACE_JOB"
printf 'Uccle PGAS:             %s\n' "$UCCLE_PGAS_JOB"
