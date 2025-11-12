#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=arith_variants
#SBATCH --output=/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/logs/arith_variants_%j.out
#SBATCH --error=/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/logs/arith_variants_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/home5/jholshuijsen/reasoning-reciting-probing"
mkdir -p "$PROJECT_ROOT/outputs/logs"

cd "$PROJECT_ROOT"

"$PROJECT_ROOT/.venv/bin/python" -u code/jobs/run_arithmetic_variant_directions.py \
  --project_root "$PROJECT_ROOT"


