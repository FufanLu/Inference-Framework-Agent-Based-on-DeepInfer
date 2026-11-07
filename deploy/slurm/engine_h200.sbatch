#!/bin/bash
#SBATCH --job-name=lev-engine-h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# lev-engine on one H200 with DeepSeek-V2-Lite-Chat (run engine_prep.sbatch first):
#   1. engine tests on the GPU (compiled Triton kernels, bf16)
#   2. kernel microbenchmarks          bench/engine/kernels.py
#   3. logits parity vs transformers   bench/engine/parity.py
#   4. batch invariance                bench/engine/invariance.py
#   5. throughput: lev x3 kernel configs vs vLLM, same GPU
# Every step has its own log and JSON under bench/results/engine-h200/<job>/;
# a failing step does not stop the others.
#   sbatch deploy/slurm/engine_h200.sbatch
source "${SLURM_SUBMIT_DIR:-.}/deploy/slurm/common.sh"
lev_activate
lev_gpu_preflight
set +e  # run every step; each one records its own exit code
export PYTHONNOUSERSITE=1
MODEL="${V2_MODEL_DIR:-$HF_HOME/models/DeepSeek-V2-Lite-Chat}"
cd "$LEV_ROOT"
OUT="bench/results/engine-h200/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" logs
{
    hostname
    nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
    python -c "import torch, triton, vllm; print(torch.__version__, triton.__version__, vllm.__version__)"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

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

step gpu-tests 900 bash -c "cd engine && PYTHONPATH=../worker python -m pytest -q -p no:cacheprovider -rs"
step kernels 1800 python bench/engine/kernels.py --out "$OUT/kernels.json"
step parity 1800 python bench/engine/parity.py --model "$MODEL" --out "$OUT/parity.json"
step invariance 2700 python bench/engine/invariance.py --model "$MODEL" --out "$OUT/invariance.json"
for engine in lev:triton lev:triton-invariant vllm lev:reference; do
    name="throughput-${engine/:/-}"
    step "$name" 1800 python bench/engine/throughput.py --engine "$engine" --model "$MODEL" --out "$OUT/$name.json"
done
echo "done $(date +%T)"
cat "$OUT/steps.txt"
