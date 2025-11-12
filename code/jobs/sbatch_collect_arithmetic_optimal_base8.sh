#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=arith8_opt_collect
#SBATCH --output=/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/logs/arith8_collect_%j.out
#SBATCH --error=/gpfs/home5/jholshuijsen/reasoning-reciting-probing/outputs/logs/arith8_collect_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/home5/jholshuijsen/reasoning-reciting-probing"
mkdir -p "$PROJECT_ROOT/outputs/logs"

# User-tunable parameters
ALPHA=0.05
INJECT_LAYER=-1     # -1 => apply at all layers (with ZERO_FIRST_N zeroed)
ZERO_FIRST_N=3
CHUNK_SIZE=1000
CHUNK_ID=0

DIRECTION_PATH="$PROJECT_ROOT/outputs/optimal_directions/arithmetic-base8/optimal_direction.pth"
INPUT_PATH="$PROJECT_ROOT/inputs/arithmetic/data/base8.txt"
OUTPUT_DIR="$PROJECT_ROOT/outputs/arithmetic/intervention/optimal_base8_eval/"

cd "$PROJECT_ROOT"

echo "Starting arithmetic base8 optimal collection on $(date)"
echo "Alpha=$ALPHA inject_layer=$INJECT_LAYER zero_first_n=$ZERO_FIRST_N chunk_size=$CHUNK_SIZE chunk_id=$CHUNK_ID"

# Prefer Poetry environment (to ensure all Python deps like xxhash are available)
if command -v poetry >/dev/null 2>&1; then
  echo "Setting up Poetry environment"
  poetry config virtualenvs.in-project true --local || true
  poetry install --no-interaction --no-root
  echo "Using: $(poetry run python -c 'import sys; print(sys.executable)')"
  poetry run python -u code/jobs/collect_arithmetic_optimal_base8.py \
    --project_root "$PROJECT_ROOT" \
    --direction_path "$DIRECTION_PATH" \
    --alpha "$ALPHA" \
    --chunk_size "$CHUNK_SIZE" \
    --chunk_id "$CHUNK_ID" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --inject_layer "$INJECT_LAYER" \
    --zero_first_n "$ZERO_FIRST_N"
else
  echo "Poetry not found; falling back to system python ($(which python))"
  python -u code/jobs/collect_arithmetic_optimal_base8.py \
    --project_root "$PROJECT_ROOT" \
    --direction_path "$DIRECTION_PATH" \
    --alpha "$ALPHA" \
    --chunk_size "$CHUNK_SIZE" \
    --chunk_id "$CHUNK_ID" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --inject_layer "$INJECT_LAYER" \
    --zero_first_n "$ZERO_FIRST_N"
fi

echo "Done on $(date)"


