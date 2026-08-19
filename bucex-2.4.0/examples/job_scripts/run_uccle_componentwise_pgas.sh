#!/usr/bin/env bash
set -euo pipefail

CHAIN="${1:?Provide chain number 1--4}"
START="${2:-1892-01-01}"
END="${3:-}"
POOL="${4:-selection}"
DRAWS="${5:-2000}"
WARMUP="${6:-2000}"
PARTICLES="${7:-512}"
WORKERS="${8:-6}"
BASE_SEED="${9:-24100}"
SCREEN_PATH="${10:-results/componentwise_screen/full_record_laplace.bucex}"
OUTPUT_DIR="${11:-results/hpc_chains}"

PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

VENV_PATH="${BUCEX_VENV:-${HOME}/venvs/bucex}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment: ${VENV_PATH}" >&2
  echo "Set BUCEX_VENV to the correct environment before submitting." >&2
  exit 2
fi
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${SCREEN_PATH}" ]]; then
  echo "Missing full-record componentwise screen: ${SCREEN_PATH}" >&2
  exit 2
fi

mkdir -p logs "${OUTPUT_DIR}"

ARGS=(
  --chain "${CHAIN}"
  --screen "${SCREEN_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --start "${START}"
  --pool "${POOL}"
  --draws "${DRAWS}"
  --warmup "${WARMUP}"
  --particles "${PARTICLES}"
  --workers "${WORKERS}"
  --base-seed "${BASE_SEED}"
)
if [[ -n "${END}" ]]; then
  ARGS+=(--end "${END}")
fi

echo "Running one exact componentwise PGAS chain"
echo "chain       = ${CHAIN}"
echo "venv        = ${VENV_PATH}"
echo "workdir     = $(pwd)"
date
hostname

python -u examples/15_hpc_independent_chain.py "${ARGS[@]}"

echo "Finished exact componentwise PGAS chain"
date
