#!/bin/bash
# One-time setup on an Explorer login node:
#   conda env with vLLM >= 0.31 + worker deps, Rust toolchain, release binaries.
# Usage: bash deploy/slurm/setup_env.sh
source "$(dirname "$0")/common.sh"
mkdir -p "$LEV_SCRATCH/envs" "$HF_HOME"

module load miniconda3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ ! -d "$LEV_ENV" ]]; then
    conda create -y -p "$LEV_ENV" python=3.12
fi
conda activate "$LEV_ENV"
pip install -U "vllm>=0.31.0" "huggingface_hub[cli,hf_transfer]" \
    "grpcio>=1.70" "grpcio-tools>=1.70" "protobuf>=5.29" "tokenizers>=0.21" "xxhash>=3.5" pyzmq msgspec \
    openai anthropic matplotlib
python -c "import vllm; print('vllm', vllm.__version__)"

# protoc for the Rust build (no system package on the cluster).
if ! command -v protoc >/dev/null; then
    mkdir -p "$HOME/.local"
    curl -fsSL -o /tmp/protoc.zip \
        https://github.com/protocolbuffers/protobuf/releases/download/v29.3/protoc-29.3-linux-x86_64.zip
    unzip -o -q /tmp/protoc.zip -d "$HOME/.local" && rm /tmp/protoc.zip
fi
if ! command -v cargo >/dev/null; then
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
cd "$LEV_ROOT"
cargo build --release -p lev-gateway -p lev-bench
make proto-py PYTHON="$(command -v python)"
echo "environment ready: $LEV_ENV"
