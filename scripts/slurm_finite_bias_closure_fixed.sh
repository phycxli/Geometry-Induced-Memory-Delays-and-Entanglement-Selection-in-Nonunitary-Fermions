#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
: "${REPRO_PROJECT:?absolute checkout path required}"
: "${REPRO_OUTPUT_DIR:?new scratch output path outside checkout required}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPRO_PROJECT"
.venv/bin/python scripts/reproduce_finite_bias_closure.py hpc-run \
  --output-dir "$REPRO_OUTPUT_DIR" --hpc-stage "${1:?stage required}" \
  --index "${SLURM_ARRAY_TASK_ID:?array index required}"
