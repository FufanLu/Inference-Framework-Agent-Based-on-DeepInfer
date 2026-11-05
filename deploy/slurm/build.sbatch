#!/bin/bash
#SBATCH --job-name=lev-build
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Build the gateway and the load generator on a compute node (login nodes kill
# heavy processes). Without the kafka feature: no librdkafka/cmake needed on
# the cluster, and the H200 jobs use the in-process job queue.
#   sbatch deploy/slurm/build.sbatch
source "${SLURM_SUBMIT_DIR:-.}/deploy/slurm/common.sh"
lev_activate
cd "$LEV_ROOT"
cargo build --release --no-default-features -p lev-gateway -p lev-bench
ls -la target/release/lev-gateway target/release/lev-bench
