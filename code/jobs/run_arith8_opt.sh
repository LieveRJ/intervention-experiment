#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=arith8_get_rw_cf_acc
#SBATCH --output=/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/logs/arith8_get_rw_cf_acc_%j.out
#SBATCH --error=/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/logs/arith8_get_rw_cf_acc_%j.err

set -euo pipefail

# Optional: load environment modules (uncomment and adjust for your cluster)
# module load python/3.10 cuda/12.1 poetry

# Determine project root (prefer GPFS when available)
if [ -d "/gpfs/home5/jholshuijsen/reasoning-reciting-probing" ]; then
  PROJECT_ROOT="/gpfs/home5/jholshuijsen/reasoning-reciting-probing"
else
  PROJECT_ROOT="/home/jholshuijsen/reasoning-reciting-probing"
fi

mkdir -p "$PROJECT_ROOT/outputs/logs"
mkdir -p "$PROJECT_ROOT/outputs/arithmetic-base8/tuning/counter_factual"

cd "$PROJECT_ROOT"

# Cache dirs (optional)
export HF_HOME="/home/jholshuijsen/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/jholshuijsen/.cache/huggingface/transformers"

python code/jobs/run_arithmetic_base8_optimal_indices.py \
  --project_root "$PROJECT_ROOT" \
  --chunk_size 1000 \
  --chunk_id 0 \
  --out_path "$PROJECT_ROOT/outputs/arithmetic-base8/tuning/counter_factual/optimal_sweep_indices_${SLURM_JOB_ID:-local}.json"


