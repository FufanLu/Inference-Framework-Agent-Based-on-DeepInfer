#!/bin/bash
#SBATCH --job-name=lev-engine-2gpu
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Two H200s on one node (NVLink): NCCL collectives and their tuning knobs,
# then DeepSeek-V2-Lite on one GPU vs TP2 vs EP2 (DP attention) vs AFD.
#   sbatch deploy/slurm/engine_2gpu.sbatch
source "${SLURM_SUBMIT_DIR:-.}/deploy/slurm/common.sh"
lev_activate
set +e
export PYTHONNOUSERSITE=1
MODEL="${V2_MODEL_DIR:-$HF_HOME/models/DeepSeek-V2-Lite-Chat}"
cd "$LEV_ROOT"
OUT="bench/results/engine-2gpu/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" logs
{
    hostname
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    nvidia-smi topo -m
    nvidia-smi nvlink --status -i 0 | head -20
} > "$OUT/topology.txt" 2>&1
cat "$OUT/topology.txt"

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
TR="torchrun --standalone --nproc-per-node 2"

# The gateway/lev-bench for step 4 (incremental; the V2 chat template is new).
step build 1500 cargo build --release --no-default-features -p lev-gateway -p lev-bench

# 1. Collectives: defaults (with NCCL's own view of the topology), then one knob at a time.
step nccl-default 600 env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING $TR bench/nccl/collectives.py --out "$OUT/nccl-default.json"
for proto in LL LL128 Simple; do
    step "nccl-proto-$proto" 600 env NCCL_PROTO=$proto $TR bench/nccl/collectives.py --ops all_reduce --out "$OUT/nccl-proto-$proto.json"
done
for algo in Ring Tree NVLS; do
    step "nccl-algo-$algo" 600 env NCCL_ALGO=$algo $TR bench/nccl/collectives.py --ops all_reduce --out "$OUT/nccl-algo-$algo.json"
done
step nccl-no-p2p 600 env NCCL_P2P_DISABLE=1 $TR bench/nccl/collectives.py --ops all_reduce all_to_all sendrecv --out "$OUT/nccl-no-p2p.json"
step nccl-1channel 600 env NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 $TR bench/nccl/collectives.py --ops all_reduce --out "$OUT/nccl-1channel.json"

# 2. The model: one GPU, TP2, EP2.
step parallel-single 1500 env CUDA_VISIBLE_DEVICES=0 python bench/engine/parallel.py --mode single --model "$MODEL" --out "$OUT/parallel-single.json"
step parallel-tp 1500 $TR bench/engine/parallel.py --mode tp --model "$MODEL" --out "$OUT/parallel-tp.json"
step parallel-ep 1500 $TR bench/engine/parallel.py --mode ep --model "$MODEL" --out "$OUT/parallel-ep.json"
# 3. Attention on GPU 0, FFN on GPU 1, two micro-batches in ping-pong (vs one: no overlap).
step afd 1500 $TR bench/engine/afd.py --model "$MODEL" --out "$OUT/afd.json"

# 4. Serving: gateway (V2 template, cache_aware) + one lev worker per GPU,
#    routing policies compared with lev-bench (binaries from the build step).
cat > "$OUT/gateway.toml" <<TOML
listen = "127.0.0.1:8080"
[[models]]
name = "DeepSeek-V2-Lite-Chat"
aliases = ["v2-lite"]
version = "v2"
tokenizer = "$MODEL/tokenizer.json"
context_limit = 8192
[router]
policy = "cache_aware"
index_mode = "exact"
[dispatch]
ready_timeout_ms = 600000
[[workers]]
endpoint = "http://127.0.0.1:51061"
[[workers]]
endpoint = "http://127.0.0.1:51062"
TOML
LEV_CFG='{"device":"cuda","dtype":"bfloat16","kernel_backend":"triton","block_size":16,"kv_cache_memory_bytes":60000000000,"max_model_len":8192,"max_num_batched_tokens":8192,"max_num_seqs":128}'
for i in 0 1; do
    (cd worker && CUDA_VISIBLE_DEVICES=$i PYTHONPATH=.:../engine python -m leviathan_worker.server --backend lev \
        --lev-model "$MODEL" --lev-config "$LEV_CFG" --model DeepSeek-V2-Lite-Chat --host 127.0.0.1 \
        --port $((51061 + i)) --worker-id "lev-$i" --log-format text) > "$OUT/worker-$i.log" 2>&1 &
done
target/release/lev-gateway --config "$OUT/gateway.toml" > "$OUT/gateway.log" 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT
if lev_wait_ready 2 900; then
    step serve-bench 1800 target/release/lev-bench --gateway http://127.0.0.1:8080 --model v2-lite \
        --workload agent-shared-prefix --requests 400 --concurrency 32 --max-tokens 64 \
        --policies round_robin,least_load,p2c,cache_aware --tag h200-lev-v2lite --out "$OUT/serve-bench"
    curl -s http://127.0.0.1:8080/metrics > "$OUT/metrics.prom"
else
    echo "serve: workers not ready" | tee -a "$OUT/steps.txt"
fi
echo "done $(date +%T)"
cat "$OUT/steps.txt"
