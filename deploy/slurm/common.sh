# Shared settings for the Explorer (Northeastern) H200 deployment.
# Source it: `source deploy/slurm/common.sh`.
set -euo pipefail

export LEV_ROOT="${LEV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export LEV_SCRATCH="${LEV_SCRATCH:-/scratch/$USER/leviathan}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
export MODEL_DIR="${MODEL_DIR:-$HF_HOME/models/DeepSeek-V4.1-Flash-1017}"
export LEV_ENV="${LEV_ENV:-$LEV_SCRATCH/envs/vllm}"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

lev_activate() {
    module load miniconda3/25.9.1 >/dev/null 2>&1 || true
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$LEV_ENV"
}

# vLLM engine arguments for DeepSeek-V4.1-Flash on 4x H200 (from the vLLM
# recipe at recipes.vllm.ai, H200 / single-node TP / text-only), minus the
# OpenAI-server-only flags (tool parser, auto tool choice): the worker gets
# token IDs from the gateway and never runs vLLM's chat frontend.
lev_vllm_args() {
    local extra="${1:-}"
    printf '%s' "$MODEL_DIR --tokenizer-mode deepseek_v41 --tensor-parallel-size 4 \
--engram-config '{\"cpu_offload\":true}' --reasoning-parser deepseek_v41 \
--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":true}' \
--language-model-only --max-model-len ${MAX_MODEL_LEN:-131072} ${extra}"
}

lev_wait_ready() {
    # Wait until the gateway reports N healthy workers.
    local want="$1" timeout="${2:-3600}" start
    start=$(date +%s)
    until curl -fsS "http://127.0.0.1:${LEV_PORT:-8080}/readyz" 2>/dev/null | grep -q "ready: ${want} healthy"; do
        if (( $(date +%s) - start > timeout )); then
            echo "workers not ready after ${timeout}s" >&2
            return 1
        fi
        sleep 10
    done
}
