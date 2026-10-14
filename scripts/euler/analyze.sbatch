#!/bin/bash
#SBATCH --job-name=material-vlm
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --output=logs/%x-%j.out
#
# Submit from the repository root; all arguments are passed to material-vlm:
#   sbatch scripts/euler/analyze.sbatch --images data/images --output outputs/run.json
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source scripts/euler/env.sh
source "$VENV_DIR/bin/activate"

nvidia-smi --query-gpu=name,memory.total --format=csv

material-vlm --config configs/default.yaml "$@"
