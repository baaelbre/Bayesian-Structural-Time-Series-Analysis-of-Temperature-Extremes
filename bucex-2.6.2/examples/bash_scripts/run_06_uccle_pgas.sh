#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash examples/bash_scripts/run_06_uccle_pgas.sh \
#   [START] [END|latest] [DRAWS] [WARMUP] [CHAINS] [PARTICLES] \
#   [MCMC_SEED] [DATA_DIR] [RESULTS_ROOT] [RUN_ID] [OVERWRITE]

START="${1:-1892-01-01}"
END="${2:-latest}"
DRAWS="${3:-250}"
WARMUP="${4:-250}"
CHAINS="${5:-2}"
PARTICLES="${6:-128}"
MCMC_SEED="${7:-56000}"
DATA_DIR="${8:-data}"
RESULTS_ROOT="${9:-results}"
RUN_ID="${10:-$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${11:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${BUCEX_PACKAGE_ROOT:-${PBS_O_WORKDIR:-${DEFAULT_ROOT}}}"
cd "${PROJECT_ROOT}"

VENV_DIR="${BUCEX_VENV_DIR:-${HOME}/venvs/bucex_env}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi
PYTHON_BIN="${BUCEX_PYTHON:-python}"

export BUCEX_START="${START}"
export BUCEX_END="${END/latest/}"
export BUCEX_DRAWS="${DRAWS}"
export BUCEX_WARMUP="${WARMUP}"
export BUCEX_CHAINS="${CHAINS}"
export BUCEX_PARTICLES="${PARTICLES}"
export BUCEX_SEED="${MCMC_SEED}"
export BUCEX_DATA_DIR="${DATA_DIR}"
export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_RUN_ID="${RUN_ID}"
export BUCEX_OVERWRITE="${OVERWRITE}"
export BUCEX_PROGRESS=0
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/bucex_mpl_${USER:-user}_${PBS_JOBID:-$$}}"
mkdir -p "${MPLCONFIGDIR}" "${RESULTS_ROOT}"

echo "Running Uccle analysis with PGAS inference"
echo "start       = ${START}"
echo "end         = ${END}"
echo "draws       = ${DRAWS}"
echo "warmup      = ${WARMUP}"
echo "chains      = ${CHAINS}"
echo "particles   = ${PARTICLES}"
echo "MCMC seed   = ${MCMC_SEED}"
echo "data dir    = ${DATA_DIR}"
echo "results     = ${RESULTS_ROOT}"
echo "run id      = ${RUN_ID}"
echo "workdir     = $(pwd)"
date
hostname

"${PYTHON_BIN}" -u examples/06_uccle_pgas.py

echo "Finished Uccle analysis with PGAS inference"
date
