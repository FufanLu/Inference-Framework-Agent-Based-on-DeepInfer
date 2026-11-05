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
# then DeepSeek-V2-Lite on one GPU vs TP2 vs EP2 (DP attention).
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
echo "done $(date +%T)"
cat "$OUT/steps.txt"
