#!/bin/bash
#SBATCH --job-name=lev-engine-prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# CPU job before the lev-engine GPU run: download DeepSeek-V2-Lite-Chat
# (~31 GB, resumable) and check that the environment imports everything the
# GPU job needs (login nodes kill processes that import torch + vLLM).
#   sbatch deploy/slurm/engine_prep.sbatch
source "${SLURM_SUBMIT_DIR:-.}/deploy/slurm/common.sh"
lev_activate
export PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=1
V2_MODEL_ID="${V2_MODEL_ID:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
V2_MODEL_DIR="${V2_MODEL_DIR:-$HF_HOME/models/DeepSeek-V2-Lite-Chat}"
cd "$LEV_ROOT"
python - <<'EOF'
import torch, triton, transformers, tokenizers, safetensors, xxhash, vllm
from transformers import DeepseekV2ForCausalLM  # native implementation
print("torch", torch.__version__, "triton", triton.__version__, "transformers", transformers.__version__,
      "vllm", vllm.__version__)
EOF
mkdir -p "$V2_MODEL_DIR"
hf download "$V2_MODEL_ID" --local-dir "$V2_MODEL_DIR" --max-workers 8
du -sh "$V2_MODEL_DIR"
python -c "import json; c = json.load(open('$V2_MODEL_DIR/config.json')); print(c['architectures'], c['num_hidden_layers'])"
