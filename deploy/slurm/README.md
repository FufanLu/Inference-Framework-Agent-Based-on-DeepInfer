# Explorer H200 deployment

Explorer has no SM100 GPUs, so DeepInfer cannot run there
(see [docs/deepinfer-status.md](../../docs/deepinfer-status.md)). The real-GPU
path is DeepSeek-V4.1-Flash on vLLM ≥ 0.31: two TP4 replicas on one 8× H200
node, behind the Leviathan gateway.

| Step | Command | Where |
| --- | --- | --- |
| 1. Environment (once) | `bash deploy/slurm/setup_env.sh` | login node |
| 2. Weights (~511 GB, once) | `sbatch deploy/slurm/download_model.sbatch` | `short` partition |
| 3. Benchmarks | `sbatch --export=ALL,MODE=bench deploy/slurm/h200.sbatch` | `gpu` partition, 8 h |
| 4. Live demo | `sbatch --export=ALL,MODE=serve deploy/slurm/h200.sbatch`, then `ssh -L 8080:<node>:8080 explorer` | |
| 5. PD disaggregation | `sbatch --export=ALL,MODE=pd deploy/slurm/h200.sbatch` | not officially verified on H200 |

Notes:

- The first model load takes a long time (`VLLM_ENGINE_READY_TIMEOUT_S=3600`). Each job
  runs every benchmark in one go instead of reloading.
- Two replicas each offload their Engram tables to pinned host memory; the node has
  512 GB. If that does not fit, drop `--engram-config '{"cpu_offload":true}'` from
  `lev_vllm_args` in `common.sh` (V4.1's KV cache is small, so GPU memory suffices).
- Workers publish vLLM KV events on ports 5557/5558; the worker translates them into
  Leviathan's block hash convention before the router sees them.
