#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash examples/bash_scripts/run_04_simulation_pgas.sh \
#   [N_TIME] [PERIOD] [SIMULATION_SEED] [DRAWS] [WARMUP] [CHAINS] \
#   [PARTICLES] [MCMC_SEED] [RESULTS_ROOT] [RUN_ID] [OVERWRITE]

N_TIME="${1:-1000}"
PERIOD="${2:-4}"
SIMULATION_SEED="${3:-13081997}"
DRAWS="${4:-400}"
WARMUP="${5:-100}"
CHAINS="${6:-4}"
PARTICLES="${7:-128}"
MCMC_SEED="${8:-13081997}"
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

export BUCEX_N_TIME="${N_TIME}"
export BUCEX_PERIOD="${PERIOD}"
export BUCEX_SIMULATION_SEED="${SIMULATION_SEED}"
export BUCEX_DRAWS="${DRAWS}"
export BUCEX_WARMUP="${WARMUP}"
export BUCEX_PARTICLES="${PARTICLES}"
export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_OVERWRITE="${OVERWRITE}"
export BUCEX_PROGRESS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
mkdir -p logs "${RESULTS_ROOT}"

if ! [[ "${CHAINS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CHAINS must be a positive integer; received ${CHAINS}." >&2
  exit 2
fi
MAX_PARALLEL_CHAINS="${BUCEX_MAX_PARALLEL_CHAINS:-${PBS_NP:-4}}"
if (( CHAINS > MAX_PARALLEL_CHAINS )); then
  echo "Requested ${CHAINS} chains but only ${MAX_PARALLEL_CHAINS} cores are available." >&2
  echo "Request at least ppn=${CHAINS}, or lower CHAINS." >&2
  exit 2
fi

echo "Running simulation study with PGAS inference"
echo "n time      = ${N_TIME}"
echo "period      = ${PERIOD}"
echo "sim seed    = ${SIMULATION_SEED}"
echo "draws       = ${DRAWS}"
echo "warmup      = ${WARMUP}"
echo "chains      = ${CHAINS} (parallel processes)"
echo "particles   = ${PARTICLES}"
echo "MCMC seed   = ${MCMC_SEED}"
echo "results     = ${RESULTS_ROOT}"
echo "run id      = ${RUN_ID}"
echo "workdir     = $(pwd)"
date
hostname

if (( CHAINS == 1 )); then
  export BUCEX_CHAINS=1
  export BUCEX_SEED="${MCMC_SEED}"
  export BUCEX_RUN_ID="${RUN_ID}"
  export BUCEX_CHAIN_ONLY=0
  unset BUCEX_COMBINE_RUNS || true
  export MPLCONFIGDIR="${TMPDIR:-/tmp}/bucex_mpl_${USER:-user}_${PBS_JOBID:-$$}_single"
  mkdir -p "${MPLCONFIGDIR}"
  "${PYTHON_BIN}" -u examples/04_simulation_pgas.py
  echo "Finished simulation study with PGAS inference"
  date
  exit 0
fi

chain_pids=()
chain_run_ids=()
chain_logs=()
for ((chain_index = 1; chain_index <= CHAINS; chain_index++)); do
  printf -v chain_label "%02d" "${chain_index}"
  chain_seed=$((MCMC_SEED + chain_index - 1))
  chain_run_id="${RUN_ID}_chain${chain_label}"
  chain_log="logs/04_simulation_pgas_${RUN_ID}_chain${chain_label}.log"
  chain_run_ids+=("${chain_run_id}")
  chain_logs+=("${chain_log}")

  echo "Launching chain ${chain_index}/${CHAINS}: seed=${chain_seed}, log=${chain_log}"
  (
    export BUCEX_CHAINS=1
    export BUCEX_SEED="${chain_seed}"
    export BUCEX_RUN_ID="${chain_run_id}"
    export BUCEX_CHAIN_ONLY=1
    unset BUCEX_COMBINE_RUNS || true
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/bucex_mpl_${USER:-user}_${PBS_JOBID:-$$}_chain${chain_label}"
    mkdir -p "${MPLCONFIGDIR}"
    "${PYTHON_BIN}" -u examples/04_simulation_pgas.py
  ) >"${chain_log}" 2>&1 &
  chain_pids+=("$!")
done

failed=0
for chain_offset in "${!chain_pids[@]}"; do
  if wait "${chain_pids[${chain_offset}]}"; then
    echo "Chain $((chain_offset + 1)) finished: ${chain_logs[${chain_offset}]}"
  else
    echo "Chain $((chain_offset + 1)) failed: ${chain_logs[${chain_offset}]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "At least one PGAS chain failed; the combine step was not run." >&2
  exit 1
fi

shopt -s nullglob
chain_dirs=()
for chain_run_id in "${chain_run_ids[@]}"; do
  matches=("${RESULTS_ROOT}/04_simulation_pgas/${chain_run_id}__"*)
  if (( ${#matches[@]} != 1 )); then
    echo "Expected one result directory for ${chain_run_id}; found ${#matches[@]}." >&2
    exit 1
  fi
  chain_dirs+=("$(cd "${matches[0]}" && pwd)")
done
COMBINE_RUNS="$(IFS=:; echo "${chain_dirs[*]}")"

echo "Combining ${CHAINS} independent chains and producing final figures"
export BUCEX_CHAINS="${CHAINS}"
export BUCEX_SEED="${MCMC_SEED}"
export BUCEX_RUN_ID="${RUN_ID}_combined"
export BUCEX_CHAIN_ONLY=0
export BUCEX_COMBINE_RUNS="${COMBINE_RUNS}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bucex_mpl_${USER:-user}_${PBS_JOBID:-$$}_combine"
mkdir -p "${MPLCONFIGDIR}"
"${PYTHON_BIN}" -u examples/04_simulation_pgas.py

echo "Finished simulation study with PGAS inference"
date
