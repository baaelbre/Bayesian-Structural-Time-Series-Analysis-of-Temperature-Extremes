#!/usr/bin/env bash
set -euo pipefail

START="${1:-1892-01-01}"
END="${2:-}"
POOL="${3:-selection}"
DRAWS="${4:-1000}"
WARMUP="${5:-1000}"
CHAINS="${6:-4}"
WORKERS="${7:-6}"
SEED="${8:-1001}"
SCREEN_PATH="${9:-results/componentwise_screen/full_record_laplace.bucex}"

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

mkdir -p logs "$(dirname "${SCREEN_PATH}")" figures/componentwise_screen

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${PBS_JOBID:-interactive}"
mkdir -p "${MPLCONFIGDIR}"

ARGS=(
  --start "${START}"
  --engine laplace
  --pool "${POOL}"
  --draws "${DRAWS}"
  --warmup "${WARMUP}"
  --chains "${CHAINS}"
  --workers "${WORKERS}"
  --seed "${SEED}"
  --save-fit "${SCREEN_PATH}"
  --figure-dir figures/componentwise_screen
)
if [[ -n "${END}" ]]; then
  ARGS+=(--end "${END}")
fi

echo "Running componentwise hierarchical Laplace screen"
echo "venv        = ${VENV_PATH}"
echo "workdir     = $(pwd)"
date
hostname

python -u examples/10_uccle_hierarchical.py "${ARGS[@]}"

echo "Finished componentwise hierarchical Laplace screen"
date
