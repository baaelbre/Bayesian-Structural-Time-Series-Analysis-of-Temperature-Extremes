#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash examples/bash_scripts/run_01_tail_simulations.sh \
#   [N_TIME] [PERIOD] [RESULTS_ROOT] [RUN_ID] [OVERWRITE]

N_TIME="${1:-800}"
PERIOD="${2:-4}"
RESULTS_ROOT="${3:-results}"
RUN_ID="${4:-$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${5:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${BUCEX_PACKAGE_ROOT:-${PBS_O_WORKDIR:-${DEFAULT_ROOT}}}"
cd "${PROJECT_ROOT}"

VENV_DIR="${BUCEX_VENV_DIR:-${HOME}/venvs/bucex_env}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi
PYTHON_BIN="${BUCEX_PYTHON:-python}"

export BUCEX_N_TIME="${N_TIME}"
export BUCEX_PERIOD="${PERIOD}"
export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_RUN_ID="${RUN_ID}"
export BUCEX_OVERWRITE="${OVERWRITE}"
export BUCEX_PROGRESS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/bucex_mpl_${USER:-user}_${PBS_JOBID:-$$}}"
mkdir -p "${MPLCONFIGDIR}" "${RESULTS_ROOT}"

echo "Running tail simulations"
echo "n time      = ${N_TIME}"
echo "period      = ${PERIOD}"
echo "results     = ${RESULTS_ROOT}"
echo "run id      = ${RUN_ID}"
echo "workdir     = $(pwd)"
date
hostname

"${PYTHON_BIN}" -u examples/01_tail_simulations.py

echo "Finished tail simulations"
date
