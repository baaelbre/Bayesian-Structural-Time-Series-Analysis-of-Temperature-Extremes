#!/usr/bin/env bash
#
# Run one Uccle series with bucex.
#
# Usage:
#   bash jobs/run_uccle_series.sh TXx
#   bash jobs/run_uccle_series.sh TXx 42 20000 5000 1 manuscript
#
# Positional arguments:
#   1  series   TXm, TNm, TXx, TXn, TNx or TNn
#   2  seed     default: 40
#   3  n_iter   default: 20000
#   4  burn     default: 5000
#   5  thin     default: 1
#   6  priors   default: manuscript
#
# Optional environment variables:
#   BUCEX_REPO_DIR   repository root; default: parent of this jobs directory
#   BUCEX_DATA_DIR   Uccle CSV directory; default: <repo>/data
#   BUCEX_OUT_DIR    output directory; default: <repo>/results/uccle_v02
#   BUCEX_VENV       virtual environment; default: <repo>/.venv
#   BUCEX_PYTHON     Python executable; default: python
#

set -euo pipefail

SERIES="${1:-}"
SEED="${2:-40}"
N_ITER="${3:-20000}"
BURN="${4:-5000}"
THIN="${5:-1}"
PRIORS="${6:-manuscript}"

case "${SERIES}" in
    TXm|TNm|TXx|TXn|TNx|TNn) ;;
    *)
        echo "Usage: $0 {TXm|TNm|TXx|TXn|TNx|TNn} [seed] [n_iter] [burn] [thin] [priors]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${BUCEX_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_DIR="${BUCEX_DATA_DIR:-${REPO_DIR}/data}"
OUT_DIR="${BUCEX_OUT_DIR:-${REPO_DIR}/results/uccle_v02}"
VENV_DIR="${BUCEX_VENV:-${REPO_DIR}/.venv}"

cd "${REPO_DIR}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
fi

PYTHON="${BUCEX_PYTHON:-python}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Python executable '${PYTHON}' was not found." >&2
    exit 3
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

mkdir -p \
    "${OUT_DIR}/fits" \
    "${OUT_DIR}/summaries" \
    "${OUT_DIR}/figures" \
    "${OUT_DIR}/metadata"

echo "============================================================"
echo "BUCEX Uccle fit"
echo "Series:       ${SERIES}"
echo "Seed:         ${SEED}"
echo "Iterations:   ${N_ITER}"
echo "Burn-in:      ${BURN}"
echo "Thinning:     ${THIN}"
echo "Priors:       ${PRIORS}"
echo "Repository:   ${REPO_DIR}"
echo "Data:         ${DATA_DIR}"
echo "Output:       ${OUT_DIR}"
echo "Host:         $(hostname)"
echo "Started:      $(date --iso-8601=seconds 2>/dev/null || date)"
echo "============================================================"

"${PYTHON}" - \
    "${SERIES}" \
    "${DATA_DIR}" \
    "${OUT_DIR}" \
    "${SEED}" \
    "${N_ITER}" \
    "${BURN}" \
    "${THIN}" \
    "${PRIORS}" <<'PY'
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from bucex import fit_uccle_series


series = sys.argv[1]
data_dir = Path(sys.argv[2])
out_dir = Path(sys.argv[3])
seed = int(sys.argv[4])
n_iter = int(sys.argv[5])
burn = int(sys.argv[6])
thin = int(sys.argv[7])
priors = sys.argv[8]

fit = fit_uccle_series(
    series=series,
    data_dir=data_dir,
    priors=priors,
    n_iter=n_iter,
    burn=burn,
    thin=thin,
    seed=seed,
    progress=True,
)

fit_path = out_dir / "fits" / f"{series}_bucex_v0.2.pkl"
fit.save(fit_path)

summary = pd.DataFrame(fit.static_summary()).T
summary.index.name = "parameter"
summary.to_csv(out_dir / "summaries" / f"{series}_posterior_summary.csv")

metadata = {
    "series": series,
    "seed": seed,
    "n_iter": n_iter,
    "burn": burn,
    "thin": thin,
    "priors": priors,
    "python": sys.version,
    "platform": platform.platform(),
    "fit_summary": fit.summary_dict(),
}
with (out_dir / "metadata" / f"{series}_run.json").open("w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, default=str)

fig, _ = fit.plot(type="level_slope", credible_interval=0.90)
fig.savefig(
    out_dir / "figures" / f"{series}_level_slope.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

if series in {"TXx", "TXn", "TNx", "TNn"}:
    fig, _ = fit.plot(type="endpoint", credible_interval=0.90)
    fig.savefig(
        out_dir / "figures" / f"{series}_endpoint.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

if series == "TXx":
    fig, _ = fit.plot(
        type="return_period",
        threshold=[36.8, 39.7],
        annual=True,
        credible_interval=0.90,
        max_return_period=10_000,
    )
    fig.savefig(
        out_dir / "figures" / "TXx_return_periods.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

if series == "TNx":
    fig, _ = fit.plot(
        type="exceedance",
        threshold=20.0,
        annual=True,
        credible_interval=0.90,
    )
    fig.savefig(
        out_dir / "figures" / "TNx_tropical_night_probability.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

print(f"Saved fitted object: {fit_path}", flush=True)
PY

echo "Finished: $(date --iso-8601=seconds 2>/dev/null || date)"
