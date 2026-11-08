# Illustrated 12: Two GPUs: NCCL, TP, EP and AFD

> Code: `engine/lev_engine/parallel.py` (TP and EP), `bench/nccl/collectives.py`, `bench/engine/parallel.py`, `bench/engine/afd.py`
> Tests: `engine/tests/test_parallel.py` (TP2 and EP2 against one process, gloo on CPU), CI job `engine`
> Real data: `deploy/slurm/engine_2gpu.sbatch`, job 11707602, two H200s on node d4055 (raw JSON: `bench/results/engine-2gpu/11707602/`; AFD rows parsed from its log into `afd-partial.json`); the AFD and serving steps rerun after two fixes in job 11721458 on node d4054 (`bench/results/engine-2gpu/11721458/`)
> Previous: [11 Speculative decoding](11-speculative-decoding.md) · Next: [13 RL with lev-engine rollouts](13-rl-rollouts.md)

DeepSeek-V2-Lite fits on one H200, so two GPUs here are not about memory. They are the smallest setup where every multi-GPU mechanism exists: NCCL collectives over NVLink, tensor parallelism (TP), expert parallelism with data-parallel attention (EP), and attention-FFN disaggregation (AFD). This chapter measures what each one costs before the 2-node job adds InfiniBand.

## The wire

```
d4055   GPU0 ══ NV18 ══ GPU1        18 NVLink links × 26.56 GB/s = 478 GB/s per direction
          └──── PCIe (NODE) ──── NIC0 mlx5_0
```

All-reduce, 1 GiB, bus bandwidth, one NCCL knob changed at a time (NCCL 2.29.7, 20 iterations after warm-up, CUDA events):

| setting | GB/s | |
| --- | --- | --- |
| default | 330 | NCCL's own choice |
| `NCCL_ALGO=Tree` | **345** | best; 4 % above default |
| `NCCL_ALGO=Ring` | 328 | |
| `NCCL_ALGO=NVLS` | 205 | NVLink SHARP multicast: no help with two GPUs |
| `NCCL_PROTO=Simple` | 330 | |
| `NCCL_PROTO=LL128` | 311 | |
| `NCCL_PROTO=LL` | 117 | half of every 8-byte word is a flag |
| `NCCL_P2P_DISABLE=1` | 205 | same as NVLS to the decimal: NCCL most likely switched to it |
| 1 channel | 21 | one SM pair cannot fill 18 links: 16× slower |

Point-to-point `sendrecv` reaches 371 GB/s at 1 GiB, 78 % of the links. With P2P disabled it drops to 37 GB/s and `all_to_all` to 23 GB/s bus: those go through host memory, not NVLink, and EP's dispatch is an `all_to_all`.

Small messages tell a different story. Every all-reduce up to 1 MiB takes 18–21 µs whatever the protocol (timed from Python, so launch included); at that size LL is fastest (17.7 µs vs Simple's 23.1 µs), at 64 MiB it is slowest (584 µs vs 248 µs). NCCL's default followed the best choice at every size within a few µs, so on this node there is nothing to tune for collectives. Decode messages are in the small regime: a TP2 all-reduce of a 128-request decode batch is 128 × 2048 × 2 B = 512 KiB.

## Three ways to use two GPUs

```
TP2   every rank: all requests, half the heads and half of each expert's columns
      layer: attn(½ heads) → all_reduce → MoE(½ columns) → all_reduce          2 all-reduces × 27 layers

EP2   rank r: its own half of the requests (DP attention), experts [32r, 32r+32)
      layer: attn(local) → all_to_all counts → .tolist() → all_to_all rows, ids → experts → all_to_all back

AFD   rank 0: embedding, attention, KV cache, norms, LM head   rank 1: every layer's MLP / MoE
      layer: rank 0 attn → send h → rank 1 FFN → send out → rank 0 next layer
```

DeepSeek-V2-Lite on one GPU vs TP2 vs EP2, 512-token random prompts, decode step median over 30 steps (AFD: 22):

| global batch | 1 GPU | TP2 | EP2 | AFD, 1 micro-batch | AFD, 2 micro-batches |
| --- | --- | --- | --- | --- | --- |
| 2 | 28.1 ms | 33.1 ms | 42.0 ms | – | – |
| 16 | 28.7 ms | 40.2 ms | 51.6 ms | **19.8 ms** | 39.3 ms |
| 64 | 28.9 ms | 40.3 ms | 51.6 ms | **22.2 ms** | 39.5 ms |
| 128 | 29.6 ms | 40.6 ms | 51.5 ms | **24.5 ms** | 39.9 ms (rerun; first run deadlocked, below) |
| 128: output tok/s | 4 324 | 3 154 | 2 483 | **5 214** | |

Prefill (64 and 128 prompts): 27.8 k tok/s on one GPU, 42–43 k with TP2 or EP2, 1.5× from two GPUs. AFD's prefill gives the same argmax as one GPU for every prompt (agreement 1.0), except batch 128 with two micro-batches in the rerun: 124 of 128 (0.969). These runs use the non-invariant kernels, and splitting the batch in two changes the GEMM shapes, so 4 near-tied argmaxes flip: chapter 10's invariance result seen from the other side.

The rerun on another node (d4054) measured AFD with one micro-batch at 23.3, 23.5 and 25.8 ms (batches 16, 64, 128) against 19.8, 22.2 and 24.5 ms in the first run: 5–18 % run-to-run and node-to-node spread, still below one GPU at every batch size.

## What the decode numbers say

**The step time does not depend on the batch.** One GPU takes 28.1 ms for 2 requests and 29.6 ms for 128. The GPU's work grew 64×; the time did not. Decode here is bound by the host launching kernels one by one from Python (eager mode, no CUDA graphs), the same 28 ms per token chapter 10 measured for a single request. Every column of the table follows from that:

- **TP2 adds 11 ms.** Its all-reduces move 54 × 512 KiB per step at batch 128, about 1 ms at the bandwidth measured above. The other 10 ms are on the host: each step makes 54 more collective calls through `torch.distributed` (about 190 µs each, if they account for all of it), and half-size GEMMs cost as many launches as full ones.
- **EP2 adds 22 ms.** Each MoE layer calls `send_counts.tolist()` to size its `all_to_all`, a device-to-host copy that waits for the GPU. 26 MoE layers give 26 points per step where the host stops launching until the GPU catches up, so host and GPU time add instead of overlapping.
- **AFD with one micro-batch is the fastest, 17–31 % below one GPU.** Neither side ever waits on the host: `send`/`recv` are queued on the stream like kernels. Rank 0 launches attention, rank 1 launches FFNs, and the two Python processes run at the same time, so each launches about half the kernels of a step.
- **Two micro-batches double the step (19.8 → 39.3 ms).** Ping-pong (MegaScale-Infer) hides FFN time behind the other micro-batch's attention, which helps when the GPU is the bottleneck. Here the host is, and two micro-batches mean twice the launches.

So on two GPUs, the parallelism that wins for this model is the one that splits host work, not GPU work. The bigger lever is removing the launches: CUDA graphs (chapter 10, `cuda_graph.py`) replay a whole step as one launch, and `spec_h200.sbatch` measures them on the same workloads.

**The profiler's NCCL share is mostly waiting.** `bench/engine/parallel.py` records one decode step under `torch.profiler` and reports the NCCL kernels' share of GPU time. For TP2 it varied from 7 % to 69 % between batch sizes with no trend, and at batch 64 it reported 28.7 ms of NCCL in a step whose transfers need about 1 ms. An NCCL kernel spins on the GPU until the other rank's kernel arrives, so when two host-bound ranks launch out of step, the kernel's "time" is the skew between them. Collective time in a real step has to be read as transfer plus wait.

## Serving on two workers

The same gateway and router as chapters 01–09, with one lev worker per GPU (V2-Lite, Triton kernels, 60 GB of KV each) instead of sim or vLLM workers. `lev-bench`, agent workload with shared prefixes (8 agents, ~1 800-token prompts), 400 requests at concurrency 32, 64 output tokens, each policy in turn on the same two workers:

| policy | errors | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | output tok/s | hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| round_robin (ran first) | 0/400 | 104 ms | 184 ms | 1 831 ms | 30.5 ms | 900 | 91.5 % |
| least_load | 0/400 | 110 ms | 196 ms | 891 ms | 30.5 ms | 920 | 90.9 % |
| p2c | 0/400 | 115 ms | 370 ms | 851 ms | 30.4 ms | 919 | 90.4 % |
| cache_aware | 0/400 | 105 ms | **127 ms** | 861 ms | 30.4 ms | **934** | 91.1 % |

- **The whole stack works on the engine.** 1 600 requests through the gateway (V2 chat template, tokenisation, routing on exact KV events from lev-engine's block hashes) to two lev workers and back, with no errors.
- **Every policy reaches ~91 % hits.** Chapter 04 predicted this from the sim: with 8 agents and two replicas that each hold all 8 prefixes, every replica soon caches every prompt and the policies converge. Real hardware gives the same answer. cache_aware's edge is in the tail of the warm-up: TTFT p90 127 ms vs 184–370 ms, and the most output throughput (934 tok/s).
- **p99 is the cold start, not the policy.** Each run has 19–24 requests with nothing cached, mostly among the first 32, prefilled together (several within a millisecond of each other: one shared step). round_robin ran first, on freshly started workers, so its 1.8 s p99 includes their warm-up and is not evidence against it.
- **TPOT is 30.4 ms under every policy**, the eager engine's step time from the table above: decode is host-bound here too.

## Across two nodes

Job 11722469 (`deploy/slurm/engine_2node.sbatch`, nodes d4054 and d4055, one H200 each; `bench/results/engine-2node/11722469/`) was meant to repeat the collectives and TP2/EP2 over InfiniBand. The nodes do not have it in service:

```
d4054, d4055   mlx5_0   InfiniBand  PORT_DOWN      (ibp188s0 DOWN)
               irdma0   Ethernet    PORT_ACTIVE    10 Gb/s RoCE (Intel), the only active RDMA device
               irdma1   Ethernet    PORT_DOWN
```

NCCL picked `irdma0` for its IB transport and failed the first all-reduce: `NET/IB: ncclIbCompletionEventProcess: Unknown completion (... wc.opcode=IBV_WC_RECV(128) ...)`, an internal error in NCCL 2.29.7 with this RoCE provider (it logged `provider=None`). The TP and EP steps used the same default and failed the same way. With `NCCL_IB_DISABLE=1` (TCP sockets over the `internal` interface) the collectives run:

| op, 1 GiB | 2 nodes, sockets | 1 node, NVLink | |
| --- | --- | --- | --- |
| all_reduce, bus | 1.23 GB/s | 330 GB/s | 270× |
| all_to_all, bus | 1.23 GB/s | 333 GB/s | |
| sendrecv | 1.24 GB/s | 371 GB/s | |

1.23 GB/s is the 10 Gb/s line rate (1.25 GB/s). At that rate a TP2 decode step's 54 all-reduces of 512 KiB (batch 128) take roughly 30 ms of transfer (a socket all-reduce took 0.32 ms at 256 KiB and 0.96 ms at 1 MiB). Job 11726646 (`LEV_STEPS=parallel-ep:parallel-tp`, the same nodes, `NCCL_IB_DISABLE=1`; `bench/results/engine-2node/11726646/`) measured TP2 and EP2 over the sockets:

| decode step, p50 | 1 GPU | TP2, NVLink | TP2, 2 nodes | EP2, NVLink | EP2, 2 nodes |
| --- | --- | --- | --- | --- | --- |
| batch 2 | 28.1 ms | 33.1 ms | 36.3 ms | 42.0 ms | 44.9 ms |
| batch 16 | 28.7 ms | 40.2 ms | 43.7 ms | 51.6 ms | 52.5 ms |
| batch 64 | 28.9 ms | 40.3 ms | 44.7 ms | 51.6 ms | 53.8 ms |
| batch 128 | 29.6 ms | 40.6 ms | **48.7 ms** | 51.5 ms | **208.1 ms** |
| batch 128, output tok/s | 4 324 | 3 154 | 2 626 | 2 483 | 615 |

Prefill across the nodes runs at 4.7 k tok/s with TP2 and 2.6–3.0 k with EP2, against 42–43 k on NVLink: a 1 024-token-per-rank prefill moves megabytes per layer, and there the 270× bandwidth gap shows. (The first prefill of each run also pays for NCCL opening its socket connections, 41 s for EP2, so these figures are from batch 16 up.)

**TP2 lost only 8 ms of the 30 estimated.** The estimate added the transfers to the one-node step. The step is host-bound, though: the GPU already idles between Python's launches, and most of the socket transfers most likely ran in that idle time. Only at batch 128, where each all-reduce is largest, does the gap reach 8 ms.

**EP2 holds until batch 64, then quadruples.** Through batch 64, EP2 across nodes is within 2.2 ms of EP2 on NVLink, because the per-layer host sync (`.tolist()`, above) dominates either way. At batch 128 the step jumps to 208 ms, and the profiled step spends 186 of its 203 ms of GPU time in NCCL kernels. The transfers alone do not explain it. Each rank sends about 0.8 MB per dispatch and per combine, so 26 layers take about 33 ms at line rate even if nothing overlaps. Something in the socket all-to-all costs more than bandwidth past that message size; this run did not profile further.

The conclusion for this cluster: across two nodes, 10 GbE is enough for TP2's small all-reduces in decode but not for prefill or for EP's all-to-all at batch 128. Cross-node runs here need InfiniBand in service.

## Two bugs the run found

**AFD deadlocked at 64 × 512 prefill rows.** Batch 128 with two micro-batches stopped in its prefill: both ranks waited 10 minutes in a `SEND` (sequence 9452, 67 M elements) until NCCL's watchdog aborted them. Both directions shared one communicator, and NCCL runs unbatched P2P operations of a communicator in issue order. Rank 0 had queued "send micro-batch B" before "receive A's FFN output"; rank 1 was in "send A's FFN output" before "receive B". Each send needs the other side's receive, which sits behind the other side's send. Batch 64 has the same issue order with 64 MiB messages and finished, so whether it hangs depends on message size and timing: a race, not a certain hang. The fix gives each direction its own communicator (`TO_FFN`, `TO_ATTN` in `afd.py`), so the two streams are ordered independently. The script also writes its results after every row now, so a failure keeps the rows before it.

**The serving step never started.** The two lev workers were up 25 s after launch and the gateway logged both as connected, but `lev_wait_ready` waited out its 900 s. Explorer sets `http_proxy` for every job, so `curl http://127.0.0.1:8080/readyz` went to the cluster's Squid proxy, which answered with an "Access Denied" page. `common.sh` now exports `no_proxy=127.0.0.1,localhost` for every job. Both steps were rerun with `LEV_STEPS=build:afd:serve-bench` (job 11721458): batch 128 with two micro-batches completed (39.9 ms per step) and the serving bench ran. A third, smaller bug came first: the rerun's first submission passed `--export=ALL,LEV_STEPS=build,afd,serve-bench`, which sbatch splits at the commas, so only the build ran (job 11719063). The step list is now read from the environment and separated by colons.

## Why it is built this way

**Why measure TP and EP on a model that fits one GPU?** Because the mechanics are the same as on V3-size models and the overheads are easier to isolate: any slowdown against one GPU is parallelism overhead, with no memory pressure mixed in. The result, that host-side launch work dominated, is the thing a bigger model would hide.

**Why does EP need `.tolist()` at all?** `all_to_all_single` takes Python lists for uneven split sizes, so the counts must reach the host. DeepEP's low-latency kernels avoid it with fixed-capacity buffers sized for the worst case (its normal-mode kernels also wait for the counts on the host); here the sync is kept because it makes the dispatch exact and simple, and its cost is now measured: 22 ms per step at this scale.

**Why AFD with a 1:1 ratio?** On two GPUs it can only be 1:1. AFD's real argument is choosing M attention GPUs for N FFN GPUs, since attention is memory-bound on the KV cache and the FFN compute-bound on weights. The 1:1 case tests the mechanism and its correctness (argmax equal to one GPU), and it turned up both the deadlock and the host-split speedup.
