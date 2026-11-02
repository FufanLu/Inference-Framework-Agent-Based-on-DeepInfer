#!/bin/bash
#SBATCH --job-name=lev-download
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Download DeepSeek-V4.1-Flash-1017 (~511 GB) to scratch. Resumable: rerun to continue.
source "$(dirname "$0")/common.sh" 2>/dev/null || source deploy/slurm/common.sh
lev_activate
mkdir -p "$MODEL_DIR"
df -h "$(dirname "$MODEL_DIR")"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download "$MODEL_ID" --local-dir "$MODEL_DIR" --max-workers 16
du -sh "$MODEL_DIR"
