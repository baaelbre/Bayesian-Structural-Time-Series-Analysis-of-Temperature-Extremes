#!/usr/bin/env bash
#SBATCH --job-name=bucex-uccle
#SBATCH --array=1-4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/bucex_chain_%A_%a.out
#SBATCH --error=logs/bucex_chain_%A_%a.err

set -euo pipefail

# Activate the environment containing the installed bucex 2.4.1 release before
# submitting, or uncomment and adapt the next line.
# source /path/to/bucex_env/bin/activate

python examples/15_hpc_independent_chain.py \
  --chain "${SLURM_ARRAY_TASK_ID}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --output-dir results/hpc_chains
