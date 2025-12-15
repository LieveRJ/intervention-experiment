#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=cross_task_eval
#SBATCH --output=/gpfs/home3/ljilesen/intervention-experiment/outputs/logs/cross_eval_%j.out
#SBATCH --error=/gpfs/home3/ljilesen/intervention-experiment/outputs/logs/cross_eval_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/home3/ljilesen/intervention-experiment"
mkdir -p "$PROJECT_ROOT/outputs/logs"

cd "$PROJECT_ROOT"

echo "Starting cross-task evaluation on $(date)"

python -u code/jobs/run_cross_task_eval.py \
  --project_root "$PROJECT_ROOT" \
  --tasks combined_balanced \
  --gsm_samples 200 \
  --gsm_test_set

echo "Done on $(date)"
