"""CUDA graphs for decode steps.

A decode step of V2-Lite launches ~20 kernels per layer x 27 layers; at
small batch the GPU finishes each faster than Python launches the next. A
captured graph replays the whole forward pass with one launch.

Graphs are captured per batch-size bucket (1, 2, 4, ... max) on static
input buffers. A real decode batch of B sequences is copied into the
smallest bucket >= B; the padding rows decode position 0 of a reserved
scratch block, so they never touch a real sequence's cache, and their
logits are dropped. Only uniform batches use graphs: every sequence has
`query_len` tokens, all with logits; that is a plain decode (1) or a
speculative verify (k+1: the last token and k drafts). Prefill chunks run
eagerly. Requires the Triton backend: the reference kernels loop over
Python lists.
"""

from __future__ import annotations

import torch

from .batch import ForwardBatch


class DecodeGraphs:
    def __init__(
        self, model, kv_caches, block_size: int, max_blocks: int, scratch_block: int, max_batch: int, query_len: int = 1
    ):
        self.model = model
        self.kv_caches = kv_caches
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.scratch = scratch_block
        self.query_len = query_len
        self.buckets = []
        b = 1
        while b < max_batch:
            self.buckets.append(b)
            b *= 2
        self.buckets.append(max_batch)
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, ForwardBatch, torch.Tensor]] = {}
        self._capture()

    def _static(self, B: int) -> ForwardBatch:
        dev, q = self.model.device, self.query_len
        T = B * q
        return ForwardBatch(
            input_ids=torch.zeros(T, dtype=torch.int64, device=dev),
            positions=torch.zeros(T, dtype=torch.int64, device=dev),
            query_start_loc=torch.arange(B + 1, dtype=torch.int32, device=dev) * q,
            seq_lens=torch.ones(B, dtype=torch.int32, device=dev),
            block_tables=torch.full((B, self.max_blocks), self.scratch, dtype=torch.int32, device=dev),
            slot_mapping=torch.full((T,), self.scratch * self.block_size, dtype=torch.int64, device=dev),
            logits_indices=torch.arange(T, dtype=torch.int64, device=dev),
            token_seq=torch.arange(B, dtype=torch.int32, device=dev).repeat_interleave(q),
            query_lens_cpu=[q] * B,
            seq_lens_cpu=[1] * B,
        )

    def _capture(self) -> None:
        pool = None
        for B in sorted(self.buckets, reverse=True):  # largest first: later graphs reuse its memory
            static = self._static(B)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):  # warm up: Triton compilation, cuBLAS handles
                    self.model.forward(static, self.kv_caches)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = self.model.forward(static, self.kv_caches)
            pool = graph.pool()
            self.graphs[B] = (graph, static, out)
        torch.cuda.synchronize()

    def can_run(self, batch: ForwardBatch) -> bool:
        n, q = batch.num_seqs, self.query_len
        return (
            0 < n <= self.buckets[-1]
            and batch.num_tokens == n * q
            and all(k == q for k in batch.query_lens_cpu)
            and int(batch.logits_indices.numel()) == n * q
            and batch.block_tables.shape[1] <= self.max_blocks
        )

    def run(self, batch: ForwardBatch) -> torch.Tensor:
        n = batch.num_seqs
        T = n * self.query_len
        bucket = next(b for b in self.buckets if b >= n)
        graph, st, out = self.graphs[bucket]
        width = batch.block_tables.shape[1]
        st.input_ids[:T].copy_(batch.input_ids)
        st.positions[:T].copy_(batch.positions)
        st.slot_mapping[:T].copy_(batch.slot_mapping)
        st.seq_lens[:n].copy_(batch.seq_lens)
        st.block_tables[:n, :width].copy_(batch.block_tables)
        st.block_tables[:n, width:].fill_(self.scratch)
        if bucket > n:  # padding rows: position 0 of the scratch block
            st.input_ids[T:].zero_()
            st.positions[T:].zero_()
            st.slot_mapping[T:].fill_(self.scratch * self.block_size)
            st.seq_lens[n:].fill_(1)
            st.block_tables[n:].fill_(self.scratch)
        graph.replay()
        return out[:T]
