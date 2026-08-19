#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

VENV_PATH="${BUCEX_VENV:-${HOME}/venvs/bucex-2.5.0}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment: ${VENV_PATH}" >&2
  echo "Set BUCEX_VENV when submitting the PBS job." >&2
  exit 2
fi
source "${VENV_PATH}/bin/activate"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bucex-matplotlib-${PBS_JOBID:-interactive}"
mkdir -p "${MPLCONFIGDIR}"

echo "bucex presentation command"
echo "workdir = $(pwd)"
echo "venv    = ${VENV_PATH}"
echo "args    = $*"
date
hostname

EXTRA_ARGS=()
if [[ -n "${START:-}" ]]; then EXTRA_ARGS+=(--start "${START}"); fi
if [[ -n "${END:-}" ]]; then EXTRA_ARGS+=(--end "${END}"); fi
if [[ -n "${DRAWS:-}" ]]; then EXTRA_ARGS+=(--draws "${DRAWS}"); fi
if [[ -n "${WARMUP:-}" ]]; then EXTRA_ARGS+=(--warmup "${WARMUP}"); fi
if [[ -n "${CHAINS:-}" ]]; then EXTRA_ARGS+=(--chains "${CHAINS}"); fi
if [[ -n "${PARTICLES:-}" ]]; then EXTRA_ARGS+=(--particles "${PARTICLES}"); fi
if [[ -n "${WORKERS:-}" ]]; then EXTRA_ARGS+=(--workers "${WORKERS}"); fi
if [[ -n "${SEED:-}" ]]; then EXTRA_ARGS+=(--seed "${SEED}"); fi
if [[ "${1:-}" == "run" && "${2:-}" == "txx-validation" ]]; then
  if [[ -n "${INITIAL:-}" ]]; then EXTRA_ARGS+=(--initial "${INITIAL}"); fi
  if [[ -n "${HORIZON:-}" ]]; then EXTRA_ARGS+=(--horizon "${HORIZON}"); fi
  if [[ -n "${STEP:-}" ]]; then EXTRA_ARGS+=(--step "${STEP}"); fi
fi
if [[ "${OVERWRITE:-false}" == "true" && "${1:-}" != "report" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "${FIGURES:-false}" == "true" ]]; then EXTRA_ARGS+=(--figures); fi

echo "overrides = ${EXTRA_ARGS[*]:-none}"
python -u -m bucex.workflows.cli "$@" "${EXTRA_ARGS[@]}"

echo "Finished bucex presentation command"
date
