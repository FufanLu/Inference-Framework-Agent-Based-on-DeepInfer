"""Mini attention-FFN disaggregation (AFD) on two GPUs, ping-pong pipelined.

rank 0 runs embedding, attention (it holds the KV cache), norms and the LM
head; rank 1 runs every layer's MLP / MoE. A decode batch is split into two
micro-batches so the two GPUs work at the same time (MegaScale-Infer's
ping-pong): while rank 1 computes micro-batch A's FFN for layer l, rank 0
computes micro-batch B's attention for layer l.

    rank 0:  attn(A,l) send │ attn(B,l) send │ recv A, attn(A,l+1) send │ recv B, attn(B,l+1) ...
    rank 1:                 │ ffn(A,l)  send │ ffn(B,l)  send           │ ffn(A,l+1) ...

Hidden states cross with NCCL send/recv (2 x 27 transfers per micro-batch
per step), one communicator per direction. On a single communicator NCCL runs
unbatched P2P ops in issue order, so rank 0's send of micro-batch B and rank
1's send of A's FFN output wait on each other once a message is too big to
buffer (deadlocked at 64 x 512 prefill rows, job 11707602). With two GPUs the ratio can only be 1:1; AFD's point at scale is
choosing M attention GPUs for N FFN GPUs.

    torchrun --nproc-per-node 2 bench/engine/afd.py --model $M --out r/afd.json
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from common import random_prompts, write
from parallel import BS

from lev_engine.batch import ForwardBatch
from lev_engine.config import ModelSpec
from lev_engine.models.deepseek_v2 import DeepseekV2, _Moe
from lev_engine.ops import get_ops
from lev_engine.weights import load_state_dict


TO_FFN: dist.ProcessGroup | None = None  # rank 0 -> rank 1: normed hidden states
TO_ATTN: dist.ProcessGroup | None = None  # rank 1 -> rank 0: FFN outputs


def ffn(model: DeepseekV2, layer, h: torch.Tensor) -> torch.Tensor:
    return model._moe(layer.mlp, h) if isinstance(layer.mlp, _Moe) else model._dense_mlp(layer.mlp, h)


class AttentionSide:
    def __init__(self, model: DeepseekV2, prompts: list[list[int]], max_len: int, micro: int):
        self.model = model
        self.micro = micro
        per_seq = (max_len + BS - 1) // BS
        n = len(prompts)
        self.cache = [torch.zeros(model.kv_cache_shape(per_seq * n, BS), dtype=model.dtype, device="cuda")
                      for _ in model.layers]  # fmt: skip
        self.tables = [list(range(i * per_seq, (i + 1) * per_seq)) for i in range(n)]
        self.lens = [len(p) for p in prompts]
        self.prompts = prompts

    def _batches(self, chunks, starts, idx):
        return ForwardBatch.build(chunks, starts, [self.tables[i] for i in idx], BS, [True] * len(idx), "cuda")

    def step(self, inputs: list[list[int]], starts: list[int]) -> torch.Tensor:
        """One forward over all sequences, in `micro` micro-batches, FFN remote."""
        m, s = self.model, self.model.spec
        n = len(inputs)
        groups = [list(range(n))[g :: self.micro] for g in range(self.micro)]
        batches = [self._batches([inputs[i] for i in g], [starts[i] for i in g], g) for g in groups]
        xs = [m.embed[b.input_ids] for b in batches]
        pending: list[torch.Tensor | None] = [None] * len(batches)  # FFN outputs in flight
        sends = []
        for layer, cache in zip(m.layers, self.cache, strict=True):
            for g, b in enumerate(batches):
                if pending[g] is not None:  # layer l-1's FFN for this micro-batch
                    dist.recv(pending[g], 1, group=TO_ATTN)
                    xs[g] = xs[g] + pending[g]
                # ...while rank 1 works on the other micro-batch's FFN.
                h = m.ops.rms_norm(xs[g], layer.input_norm, s.rms_norm_eps)
                xs[g] = xs[g] + m._attention(layer.attn, h, b, cache)
                h2 = m.ops.rms_norm(xs[g], layer.post_attention_norm, s.rms_norm_eps).contiguous()
                sends.append((dist.isend(h2, 1, group=TO_FFN), h2))
                pending[g] = torch.empty_like(xs[g])
        for g in range(len(batches)):
            dist.recv(pending[g], 1, group=TO_ATTN)
            xs[g] = xs[g] + pending[g]
        for work, _ in sends:
            work.wait()
        outs = []
        for g, b in enumerate(batches):
            x = m.ops.rms_norm(xs[g][b.logits_indices], m.norm, s.rms_norm_eps)
            outs.append((groups[g], m.ops.linear(x, m.lm_head).argmax(-1)))
        tokens = torch.empty(n, dtype=torch.int64, device="cuda")
        for g, t in outs:
            tokens[torch.tensor(g, device="cuda")] = t
        return tokens


def ffn_side(model: DeepseekV2, shapes: list[tuple[int, int]]) -> None:
    """Serve one forward: per layer, for each micro-batch: recv, FFN, send."""
    hidden = model.spec.hidden_size
    for layer in model.layers:
        for rows, _ in shapes:
            h = torch.empty((rows, hidden), dtype=model.dtype, device="cuda")
            dist.recv(h, 0, group=TO_FFN)
            dist.send(ffn(model, layer, h).contiguous(), 0, group=TO_ATTN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=24)
    ap.add_argument("--micro", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank = dist.get_rank()
    global TO_FFN, TO_ATTN
    TO_FFN, TO_ATTN = dist.new_group([0, 1]), dist.new_group([0, 1])
    spec = ModelSpec.from_pretrained(args.model)
    model = DeepseekV2(spec, load_state_dict(args.model, torch.bfloat16, "cuda"), get_ops("triton", False), 4096, "cuda")
    rows = []
    for B in args.batches:
        for micro in args.micro:
            prompts = random_prompts(B, args.prompt, spec.vocab_size, seed=B)
            groups = [list(range(B))[g::micro] for g in range(micro)]
            times = []
            if rank == 0:
                side = AttentionSide(model, prompts, args.prompt + args.decode_steps + 8, micro)
                tokens = side.step(prompts, [0] * B)  # prefill (FFN remote too)
                # The same prefill on rank 0 alone: does splitting change the argmax?
                alone = AttentionSide(model, prompts, args.prompt + 8, 1)
                b = ForwardBatch.build(prompts, [0] * B, alone.tables, BS, [True] * B, "cuda")
                local = model.forward(b, alone.cache).argmax(-1)
                agreement = float((local == tokens).float().mean())
                del alone
                lens = [len(p) for p in prompts]
                for step in range(args.decode_steps):
                    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    tokens = side.step([[t] for t in tokens.tolist()], lens)
                    ev1.record()
                    torch.cuda.synchronize()
                    lens = [n + 1 for n in lens]
                    if step >= 2:
                        times.append(ev0.elapsed_time(ev1))
                times.sort()
                row = {"global_batch": B, "micro_batches": micro, "prefill_argmax_agreement_vs_1gpu": agreement,
                       "decode_step_ms_p50": times[len(times) // 2],
                       "decode_tokens_per_s": B / (times[len(times) // 2] / 1e3)}  # fmt: skip
                rows.append(row)
                print("afd", row, flush=True)
                write(args.out, {"mode": "afd-1a1f", "model": args.model, "results": rows})  # keep partial rows
                del side
            else:
                ffn_side(model, [(len(g) * args.prompt, 0) for g in groups])  # prefill
                for _ in range(args.decode_steps):
                    ffn_side(model, [(len(g), 0) for g in groups])
            torch.cuda.empty_cache()
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
