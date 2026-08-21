#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash examples/bash_scripts/run_05_uccle_laplace.sh \
#   [START] [END|latest] [DRAWS] [WARMUP] [CHAINS] [MCMC_SEED] \
#   [DATA_DIR] [RESULTS_ROOT] [RUN_ID] [OVERWRITE]

START="${1:-1892-01-01}"
END="${2:-latest}"
DRAWS="${3:-250}"
WARMUP="${4:-250}"
CHAINS="${5:-4}"
MCMC_SEED="${6:-56000}"
DATA_DIR="${7:-data}"
RESULTS_ROOT="${8:-results}"
RUN_ID="${9:-$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${10:-0}"

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
export BUCEX_DATA_DIR="${DATA_DIR}"
export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_OVERWRITE="${OVERWRITE}"
export BUCEX_PROGRESS=0
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

echo "Running Uccle analysis with Laplace inference"
echo "start       = ${START}"
echo "end         = ${END}"
echo "draws       = ${DRAWS}"
echo "warmup      = ${WARMUP}"
echo "chains      = ${CHAINS} (parallel processes)"
echo "MCMC seed   = ${MCMC_SEED}"
echo "data dir    = ${DATA_DIR}"
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
  "${PYTHON_BIN}" -u examples/05_uccle_laplace.py
  echo "Finished Uccle analysis with Laplace inference"
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
  chain_log="logs/05_uccle_laplace_${RUN_ID}_chain${chain_label}.log"
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
    "${PYTHON_BIN}" -u examples/05_uccle_laplace.py
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
  echo "At least one Uccle Laplace chain failed; the combine step was not run." >&2
  exit 1
fi

shopt -s nullglob
chain_dirs=()
for chain_run_id in "${chain_run_ids[@]}"; do
  matches=("${RESULTS_ROOT}/05_uccle_laplace/${chain_run_id}__"*)
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
"${PYTHON_BIN}" -u examples/05_uccle_laplace.py

echo "Finished Uccle analysis with Laplace inference"
date
