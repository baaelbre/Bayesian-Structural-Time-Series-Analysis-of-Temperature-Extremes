#!/usr/bin/env bash
# Shared runtime settings. Every PBS job executes one numbered Python example.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PACKAGE_ROOT"

PYTHON_BIN="${BUCEX_PYTHON:-python}"
PROFILE="${BUCEX_PROFILE:-publication}"
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PACKAGE_ROOT/results}"
export BUCEX_TIMESTAMP_RESULTS="${BUCEX_TIMESTAMP_RESULTS:-1}"
export BUCEX_RUN_ID="${BUCEX_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export BUCEX_PROGRESS="${BUCEX_PROGRESS:-0}"

# Explicit BUCEX_* values always win over the selected convenience profile.
case "$PROFILE" in
  smoke)
    export BUCEX_DRAWS="${BUCEX_DRAWS:-20}"
    export BUCEX_WARMUP="${BUCEX_WARMUP:-20}"
    export BUCEX_CHAINS="${BUCEX_CHAINS:-1}"
    export BUCEX_PARTICLES="${BUCEX_PARTICLES:-32}"
    ;;
  pilot)
    export BUCEX_DRAWS="${BUCEX_DRAWS:-250}"
    export BUCEX_WARMUP="${BUCEX_WARMUP:-250}"
    export BUCEX_CHAINS="${BUCEX_CHAINS:-2}"
    export BUCEX_PARTICLES="${BUCEX_PARTICLES:-128}"
    ;;
  publication)
    export BUCEX_DRAWS="${BUCEX_DRAWS:-2000}"
    export BUCEX_WARMUP="${BUCEX_WARMUP:-2000}"
    export BUCEX_CHAINS="${BUCEX_CHAINS:-4}"
    export BUCEX_PARTICLES="${BUCEX_PARTICLES:-512}"
    ;;
  *)
    echo "Unknown BUCEX_PROFILE=$PROFILE; choose smoke, pilot, or publication." >&2
    exit 2
    ;;
esac

MPL_ROOT="${TMPDIR:-/tmp}/bucex-matplotlib-${PBS_JOBID:-$$}"
case "${BUCEX_TIMESTAMP_RESULTS,,}" in
  0|false|no) RUN_OUTPUT="$BUCEX_RESULTS_ROOT" ;;
  *) RUN_OUTPUT="$BUCEX_RESULTS_ROOT/$BUCEX_RUN_ID" ;;
esac
mkdir -p "$MPL_ROOT" "$RUN_OUTPUT/logs"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$MPL_ROOT"

run_example() {
  local script="$1"
  echo "[$(date --iso-8601=seconds)] running examples/$script"
  "$PYTHON_BIN" "$PACKAGE_ROOT/examples/$script"
}
