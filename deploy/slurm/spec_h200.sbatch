#!/bin/bash
#SBATCH --job-name=lev-spec-h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
# Speculative decoding in lev-engine on one H200:
#   1. the speculative tests on the GPU (compiled kernels, verify graphs)
#   2. n-gram prompt lookup on DeepSeek-V2-Lite-Chat (MLA + MoE)
#   3. Qwen3-8B with a Qwen3-0.6B draft (coupled and greedy), and n-gram
# Each run is checked against plain decoding on the same kernels.
#   sbatch deploy/slurm/spec_h200.sbatch
source "${SLURM_SUBMIT_DIR:-.}/deploy/slurm/common.sh"
lev_activate
set +e
export PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=1
V2="${V2_MODEL_DIR:-$HF_HOME/models/DeepSeek-V2-Lite-Chat}"
Q8="$HF_HOME/models/Qwen3-8B"
Q06="$HF_HOME/models/Qwen3-0.6B"
[[ -f "$Q8/config.json" ]] || hf download Qwen/Qwen3-8B --local-dir "$Q8" --max-workers 8
[[ -f "$Q06/config.json" ]] || hf download Qwen/Qwen3-0.6B --local-dir "$Q06" --max-workers 8
cd "$LEV_ROOT"
OUT="bench/results/spec-h200/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" logs
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/env.txt"

step() {
    local name="$1" limit="$2"
    shift 2
    echo "== $name  $(date +%T)"
    timeout "$limit" "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    echo "   exit $rc  $(date +%T)"
    echo "$name $rc" >> "$OUT/steps.txt"
    tail -2 "$OUT/$name.log"
}

step gpu-tests 900 bash -c "cd engine && PYTHONPATH=../worker python -m pytest -q -p no:cacheprovider -rs \
    tests/test_speculative.py tests/test_cuda_graph.py tests/test_triton_kernels.py"
step spec-v2lite 2400 python bench/engine/speculative.py --model "$V2" --template v2 --out "$OUT/spec-v2lite.json"
step spec-qwen3 4200 python bench/engine/speculative.py --model "$Q8" --draft "$Q06" --template qwen3 \
    --out "$OUT/spec-qwen3.json"
echo "done $(date +%T)"
cat "$OUT/steps.txt"
