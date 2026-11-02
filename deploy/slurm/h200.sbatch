#!/bin/bash
#SBATCH --job-name=leviathan-h200
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#
# Two DeepSeek-V4.1-Flash replicas (TP4 each) behind the Leviathan gateway on
# one 8x H200 node.
#
#   MODE=bench  (default)  smoke test + routing benchmarks, then exit
#   MODE=serve             keep serving; tunnel from a laptop:
#                          ssh -L 8080:<node>:8080 explorer
#   MODE=pd                1 prefill + 1 decode replica with NIXL KV transfer
#
# sbatch --export=ALL,MODE=bench deploy/slurm/h200.sbatch
source "${SLURM_SUBMIT_DIR:-.}/deploy/slurm/common.sh"
lev_activate
cd "$LEV_ROOT"
MODE="${MODE:-bench}"
RUN="${LEV_SCRATCH}/runs/${SLURM_JOB_ID:-local}"
mkdir -p "$RUN" logs
echo "node $(hostname), mode $MODE, run dir $RUN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
free -g | head -2

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=1
cd worker

ROLES=(unified unified)
EXTRA=""
if [[ "$MODE" == "pd" ]]; then
    ROLES=(prefill decode)
    EXTRA="--kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}'"
fi
for i in 0 1; do
    gpus=$([[ $i == 0 ]] && echo 0,1,2,3 || echo 4,5,6,7)
    CUDA_VISIBLE_DEVICES=$gpus VLLM_NIXL_SIDE_CHANNEL_PORT=$((5600 + i)) \
        python -m leviathan_worker.server --backend vllm --host 127.0.0.1 --port $((50051 + i)) \
        --worker-id "h200-${SLURM_JOB_ID:-0}-$i" --role "${ROLES[$i]}" --model DeepSeek-V4.1-Flash \
        --kv-events-port $((5557 + i)) --vllm-args "$(lev_vllm_args "$EXTRA")" \
        > "$RUN/worker-$i.log" 2>&1 &
done
cd "$LEV_ROOT"
target/release/lev-gateway --config deploy/slurm/gateway-h200.toml --log-json > "$RUN/gateway.log" 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT

lev_wait_ready 2 3600
echo "both replicas ready after ${SECONDS}s"

case "$MODE" in
serve)
    echo "serving on $(hostname):8080 until the job ends"
    wait
    ;;
bench|pd)
    python scripts/smoke.py --base-url http://127.0.0.1:8080 | tee "$RUN/smoke.txt"
    for workload in agent-shared-prefix multi-turn-chat unique; do
        target/release/lev-bench --gateway http://127.0.0.1:8080 --workload "$workload" \
            --requests "${REQUESTS:-400}" --concurrency "${CONCURRENCY:-32}" --max-tokens 128 \
            --policies round_robin,least_load,p2c,cache_aware --tag "h200-vllm-$MODE" \
            --out "bench/results/h200-$MODE/$workload" | tee "$RUN/bench-$workload.md"
    done
    curl -s http://127.0.0.1:8080/metrics > "$RUN/metrics.prom"
    ;;
esac
