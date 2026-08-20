#!/usr/bin/env bash
# Shared settings for every PBS job. Override any BUCEX_* variable at qsub time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PACKAGE_ROOT"

PROFILE="${BUCEX_PROFILE:-publication}"
OUTPUT_DIR="${BUCEX_OUTPUT_DIR:-$PACKAGE_ROOT/results/presentation}"
PYTHON_BIN="${BUCEX_PYTHON:-python}"

if [[ -n "${BUCEX_CHAINS:-}" ]]; then
  CHAINS="$BUCEX_CHAINS"
else
  case "$PROFILE" in
    smoke) CHAINS=1 ;;
    pilot) CHAINS=2 ;;
    publication) CHAINS=4 ;;
    *) echo "Unknown BUCEX_PROFILE=$PROFILE" >&2; exit 2 ;;
  esac
fi

MPL_ROOT="${TMPDIR:-/tmp}/bucex-matplotlib-${PBS_JOBID:-$$}"
mkdir -p "$MPL_ROOT" "$OUTPUT_DIR/logs"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$MPL_ROOT"

CLI_CONFIG=(
  --profile "$PROFILE"
  --output-dir "$OUTPUT_DIR"
  --chains "$CHAINS"
  --no-progress
  --formats pdf png
)
if [[ -n "${BUCEX_DATA_DIR:-}" ]]; then
  CLI_CONFIG+=(--data-dir "$BUCEX_DATA_DIR")
fi
if [[ -n "${BUCEX_DRAWS:-}" ]]; then CLI_CONFIG+=(--draws "$BUCEX_DRAWS"); fi
if [[ -n "${BUCEX_WARMUP:-}" ]]; then CLI_CONFIG+=(--warmup "$BUCEX_WARMUP"); fi
if [[ -n "${BUCEX_PARTICLES:-}" ]]; then CLI_CONFIG+=(--particles "$BUCEX_PARTICLES"); fi
if [[ -n "${BUCEX_SIMULATION_MONTHS:-}" ]]; then
  CLI_CONFIG+=(--simulation-months "$BUCEX_SIMULATION_MONTHS")
fi

OVERWRITE_ARGS=()
if [[ "${BUCEX_OVERWRITE:-0}" == "1" ]]; then OVERWRITE_ARGS+=(--overwrite); fi

bucex_run() {
  "$PYTHON_BIN" -m bucex.workflows.cli run "$@" "${CLI_CONFIG[@]}" "${OVERWRITE_ARGS[@]}"
}

bucex_combine() {
  "$PYTHON_BIN" -m bucex.workflows.cli combine "$@" "${CLI_CONFIG[@]}" "${OVERWRITE_ARGS[@]}"
}

bucex_report() {
  "$PYTHON_BIN" -m bucex.workflows.cli report "$@" "${CLI_CONFIG[@]}"
}

array_index() {
  local value="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}"
  if [[ -z "$value" ]]; then
    echo "This script must run as a PBS array job." >&2
    exit 2
  fi
  printf '%s\n' "$value"
}

