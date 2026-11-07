#!/bin/bash
#SBATCH --job-name=lev-rl-h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# GRPO on Qwen3-1.7B, colocated on one H200: lev-engine rollouts, a
# transformers trainer, weight sync every step. Two runs that differ only
# in the rollout kernels, to measure training-inference mismatch:
#   invariant   lev-engine Triton, batch-invariant GEMMs
#   cublas      lev-engine Triton attention/MoE, cuBLAS GEMMs
#   sbatch deploy/slurm/rl_h200.sbatch
source "${SLURM_SUBMIT_DIR:-.}/deploy/slurm/common.sh"
lev_activate
lev_gpu_preflight
set +e
export PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=1
QWEN="${QWEN_MODEL_DIR:-$HF_HOME/models/Qwen3-1.7B}"
[[ -f "$QWEN/config.json" ]] || hf download Qwen/Qwen3-1.7B --local-dir "$QWEN" --max-workers 8
cd "$LEV_ROOT"
OUT="bench/results/rl-h200/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" logs
nvidia-smi --query-gpu=name,memory.total --format=csv > "$OUT/env.txt"
STEPS="${STEPS:-30}"
for run in invariant cublas; do
    inv=$([[ $run == invariant ]] && echo true || echo false)
    echo "== $run  $(date +%T)"
    (cd rl && PYTHONPATH=.:../worker timeout 4500 python -m lev_rl.loop --model "$QWEN" --out "../$OUT/$run" \
        --steps "$STEPS" --batch-invariant "$inv" --kernel-backend triton) > "$OUT/$run.log" 2>&1
    echo "   exit $?  $(date +%T)"
    tail -2 "$OUT/$run.log"
done
