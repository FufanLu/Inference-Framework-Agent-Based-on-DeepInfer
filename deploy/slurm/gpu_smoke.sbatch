#!/bin/bash
#SBATCH --job-name=lev-gpu-smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Ten-minute check that the env works on an H200 before queueing the long jobs:
#   1. CUDA, cuBLAS, Triton, NCCL               deploy/slurm/preflight_gpu.py
#   2. lev-engine's GPU tests (compiled kernels) engine/tests/test_triton_kernels.py
#   3. vLLM loads DeepSeek-V2-Lite and generates deploy/slurm/preflight_vllm.py
#   sbatch -A g.ji deploy/slurm/gpu_smoke.sbatch
source "${SLURM_SUBMIT_DIR:-.}/deploy/slurm/common.sh"
lev_activate
set +e
export PYTHONNOUSERSITE=1
V2="${V2_MODEL_DIR:-$HF_HOME/models/DeepSeek-V2-Lite-Chat}"
cd "$LEV_ROOT"
OUT="bench/results/gpu-smoke/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" logs
python -c "import torch, triton, vllm; print(torch.__version__, triton.__version__, vllm.__version__)"

step() {
    local name="$1" limit="$2"
    shift 2
    echo "== $name  $(date +%T)"
    timeout "$limit" "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    echo "   exit $rc  $(date +%T)"
    echo "$name $rc" >> "$OUT/steps.txt"
    tail -3 "$OUT/$name.log"
}

step preflight 120 python deploy/slurm/preflight_gpu.py
step triton-tests 600 bash -c "cd engine && PYTHONPATH=../worker python -m pytest -q -p no:cacheprovider -rs tests/test_triton_kernels.py"
step vllm 900 python deploy/slurm/preflight_vllm.py --model "$V2" --max-model-len 4096 --timeout 800 \
    --flags-out "$OUT/vllm-flags.txt"
echo "done $(date +%T)"
cat "$OUT/steps.txt"
