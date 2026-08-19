#!/usr/bin/env bash
set -euo pipefail

CHAIN_DIR="${1:-results/hpc_chains}"
FIGURE_DIR="${2:-figures/16_hpc_combined}"

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

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${PBS_JOBID:-interactive}"
mkdir -p "${MPLCONFIGDIR}" "${FIGURE_DIR}"

echo "Combining and diagnosing four exact PGAS chains"
echo "venv        = ${VENV_PATH}"
echo "workdir     = $(pwd)"
date
hostname

python -u examples/16_combine_hpc_chains.py \
  --chain-dir "${CHAIN_DIR}" \
  --figure-dir "${FIGURE_DIR}"

echo "Finished combining and diagnosing chains"
date
