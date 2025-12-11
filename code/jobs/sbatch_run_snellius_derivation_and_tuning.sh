#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=derive_tune_dirs
#SBATCH --output=/gpfs/home5/ljilesen/intervention-experiment/outputs/logs/derive_tune_%j.out
#SBATCH --error=/gpfs/home5/ljilesen/intervention-experiment/outputs/logs/derive_tune_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/home5/ljilesen/intervention-experiment"
mkdir -p "$PROJECT_ROOT/outputs/logs"

cd "$PROJECT_ROOT"

echo "Starting derivation and tuning on $(date)"

python -u code/jobs/run_snellius_derivation_and_tuning.py \
  --project_root "$PROJECT_ROOT" \
  --tasks chess arithmetic-base8 programming \
  --gsm_samples 200

echo "Done on $(date)"


