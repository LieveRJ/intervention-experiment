#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --job-name=derive_tune_prog
#SBATCH --output=/gpfs/home3/ljilesen/intervention-experiment/outputs/logs/derive_tune_prog_%j.out
#SBATCH --error=/gpfs/home3/ljilesen/intervention-experiment/outputs/logs/derive_tune_prog_%j.err

set -euo pipefail

PROJECT_ROOT="/gpfs/home3/ljilesen/intervention-experiment"
mkdir -p "$PROJECT_ROOT/outputs/logs"

cd "$PROJECT_ROOT"

"$PROJECT_ROOT/.venv/bin/python" -u code/jobs/run_snellius_derivation_and_tuning.py \
  --project_root "$PROJECT_ROOT" \
  --tasks programming \
  --tune_target tasks \
  --gsm_samples 100
