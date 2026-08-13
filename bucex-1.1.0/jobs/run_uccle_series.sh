#!/usr/bin/env bash
# Run one independent Uccle chain from the repository root.
set -euo pipefail

SERIES="${1:-}"
SEED="${2:-40}"
DRAWS="${3:-2000}"
WARMUP="${4:-2000}"
THIN="${5:-1}"
PRIORS="${6:-regularized_lasso}"
ENGINE="${7:-auto}"
PARTICLES="${8:-256}"

case "${SERIES}" in
    TXm|TNm|TXx|TXn|TNx|TNn) ;;
    *)
        echo "Usage: $0 {TXm|TNm|TXx|TXn|TNx|TNn} [seed] [draws] [warmup] [thin] [priors] [engine] [particles]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${BUCEX_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_DIR="${BUCEX_DATA_DIR:-${REPO_DIR}/data}"
OUT_DIR="${BUCEX_OUT_DIR:-${REPO_DIR}/results/uccle_v1_1}"
PYTHON="${BUCEX_PYTHON:-python}"

cd "${REPO_DIR}"
if [[ -f "${BUCEX_VENV:-${REPO_DIR}/.venv}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${BUCEX_VENV:-${REPO_DIR}/.venv}/bin/activate"
fi

VERSION="$(${PYTHON} -c 'import bucex; print(bucex.__version__)')"
if [[ "${VERSION}" != "1.1.0" ]]; then
    echo "Expected bucex 1.1.0, but imported ${VERSION}. Run: python -m pip install -e '.[plot]'" >&2
    exit 3
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

"${PYTHON}" examples/fit_uccle_series.py \
    --series "${SERIES}" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --engine "${ENGINE}" \
    --priors "${PRIORS}" \
    --draws "${DRAWS}" \
    --warmup "${WARMUP}" \
    --thin "${THIN}" \
    --chains 1 \
    --seed "${SEED}" \
    --particles "${PARTICLES}" \
    --asis \
    --progress
