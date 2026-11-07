#!/bin/bash
#SBATCH --job-name=lev-engine-2node
#SBATCH --partition=gpu-short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# Two nodes, one H200 each: the same NCCL collectives as engine_2gpu.sbatch,
# now across the network, and EP2 (whose all_to_all crosses it every MoE
# layer). QoS allows it: gpu-short caps GPUs per job (2), not nodes.
#   sbatch deploy/slurm/engine_2node.sbatch
source "${SLURM_SUBMIT_DIR:-.}/deploy/slurm/common.sh"
lev_activate
lev_gpu_preflight all
set +e
export PYTHONNOUSERSITE=1
MODEL="${V2_MODEL_DIR:-$HF_HOME/models/DeepSeek-V2-Lite-Chat}"
cd "$LEV_ROOT"
OUT="bench/results/engine-2node/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" logs
MASTER="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
srun --ntasks-per-node=1 bash -c 'echo "== $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader;
    ls /sys/class/infiniband 2>/dev/null && ibv_devinfo 2>/dev/null | grep -E "hca_id|link_layer|state|rate" ;
    ip -br addr | grep -v "^lo"' > "$OUT/network.txt" 2>&1
cat "$OUT/network.txt"

# step NAME LIMIT "VAR=value ..." SCRIPT ARGS...   (the variables reach every rank via srun)
step() {
    local name="$1" limit="$2" vars="$3"
    shift 3
    echo "== $name  $(date +%T)"
    # shellcheck disable=SC2086 # $vars is a list of assignments
    timeout "$limit" env $vars srun --export=ALL --ntasks-per-node=1 torchrun --nnodes 2 --nproc-per-node 1 \
        --rdzv-backend c10d --rdzv-endpoint "$MASTER:29511" --rdzv-id "$SLURM_JOB_ID-$name" "$@" \
        > "$OUT/$name.log" 2>&1
    local rc=$?
    echo "   exit $rc  $(date +%T)"
    echo "$name $rc" >> "$OUT/steps.txt"
    tail -2 "$OUT/$name.log"
}

step nccl-default 900 "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH" bench/nccl/collectives.py --out "$OUT/nccl-default.json"
step nccl-socket 900 "NCCL_IB_DISABLE=1" bench/nccl/collectives.py --ops all_reduce all_to_all sendrecv --out "$OUT/nccl-socket.json"
step parallel-ep 1800 "" bench/engine/parallel.py --mode ep --model "$MODEL" --out "$OUT/parallel-ep.json"
step parallel-tp 1800 "" bench/engine/parallel.py --mode tp --model "$MODEL" --out "$OUT/parallel-tp.json"
echo "done $(date +%T)"
cat "$OUT/steps.txt"
