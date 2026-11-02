#!/bin/bash
#SBATCH --job-name=lev-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Run setup_env.sh on a compute node (pip install vLLM, Rust build).
cd "${SLURM_SUBMIT_DIR:-.}"
bash deploy/slurm/setup_env.sh
