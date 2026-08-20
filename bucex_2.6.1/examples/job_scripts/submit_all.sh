#!/usr/bin/env bash
# Submit the complete PBS dependency graph.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PREPARE_JOB="$(qsub 00_prepare.pbs)"
SIM_LAPLACE_JOB="$(qsub -W "depend=afterok:$PREPARE_JOB" 01_simulation_laplace.pbs)"
SIM_LAPLACE_COMBINE_JOB="$(qsub -W "depend=afterok:$SIM_LAPLACE_JOB" 02_simulation_laplace_combine.pbs)"
SIM_PGAS_JOB="$(qsub -W "depend=afterok:$SIM_LAPLACE_COMBINE_JOB" 03_simulation_pgas.pbs)"
SIM_PGAS_COMBINE_JOB="$(qsub -W "depend=afterok:$SIM_PGAS_JOB" 04_simulation_pgas_combine.pbs)"

UCCLE_LAPLACE_JOB="$(qsub -W "depend=afterok:$PREPARE_JOB" 05_uccle_laplace.pbs)"
UCCLE_LAPLACE_COMBINE_JOB="$(qsub -W "depend=afterok:$UCCLE_LAPLACE_JOB" 06_uccle_laplace_combine.pbs)"
UCCLE_PGAS_JOB="$(qsub -W "depend=afterok:$UCCLE_LAPLACE_COMBINE_JOB" 07_uccle_pgas.pbs)"

FINAL_JOB="$(qsub -W "depend=afterok:$SIM_PGAS_COMBINE_JOB:$UCCLE_PGAS_JOB" 08_uccle_pgas_combine_report.pbs)"

printf 'prepare:                 %s\n' "$PREPARE_JOB"
printf 'simulation Laplace:      %s\n' "$SIM_LAPLACE_JOB"
printf 'simulation Laplace join: %s\n' "$SIM_LAPLACE_COMBINE_JOB"
printf 'simulation PGAS:         %s\n' "$SIM_PGAS_JOB"
printf 'simulation PGAS join:    %s\n' "$SIM_PGAS_COMBINE_JOB"
printf 'Uccle Laplace:           %s\n' "$UCCLE_LAPLACE_JOB"
printf 'Uccle Laplace join:      %s\n' "$UCCLE_LAPLACE_COMBINE_JOB"
printf 'Uccle PGAS:              %s\n' "$UCCLE_PGAS_JOB"
printf 'final report:            %s\n' "$FINAL_JOB"
