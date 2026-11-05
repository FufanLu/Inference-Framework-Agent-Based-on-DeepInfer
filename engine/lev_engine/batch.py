"""The flattened ("varlen") batch one forward step runs on.

Every scheduled token of every sequence is concatenated into one row of
`input_ids`: a decode contributes one token, a prefill chunk many. Per-sequence
metadata says where each sequence's tokens are and which KV blocks hold its
context::

    tokens:          [ s0 s0 s0 s0 | s1 | s2 s2 ]        T = 7
    query_start_loc: [0, 4, 5, 7]
    seq_lens:        [4, 37, 130]   context after this step (cached + new)
    block_tables:    [[3, 0, ...], [7, 9, 1, ...], ...]
    slot_mapping:    where each new token's KV is written (block * B + offset)
    logits_indices:  rows whose logits are sampled (last token of a sequence
                     whose prompt is complete; every token of a speculative
                     verify: the last real token and each draft)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForwardBatch:
    input_ids: torch.Tensor  # [T] int64
    positions: torch.Tensor  # [T] int64
    query_start_loc: torch.Tensor  # [B+1] int32
    seq_lens: torch.Tensor  # [B] int32
    block_tables: torch.Tensor  # [B, max_blocks] int32, padded with 0
    slot_mapping: torch.Tensor  # [T] int64
    logits_indices: torch.Tensor  # [S] int64
    token_seq: torch.Tensor  # [T] int32: the sequence each token belongs to
    #: Python copies of the per-sequence lengths, so that reference kernels do
    #: not synchronise with the device to read them.
    query_lens_cpu: list[int]
    seq_lens_cpu: list[int]

    @property
    def num_tokens(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_seqs(self) -> int:
        return len(self.seq_lens_cpu)

    @classmethod
    def build(
        cls,
        token_chunks: list[list[int]],
        start_positions: list[int],
        block_tables: list[list[int]],
        block_size: int,
        sample: list[bool] | list[int],
        device: torch.device | str = "cpu",
    ) -> ForwardBatch:
        """`token_chunks[i]` are the new tokens of sequence i, starting at
        position `start_positions[i]`; `block_tables[i]` covers its context.
        `sample[i]` is how many of its last tokens need logits (True = 1)."""
        ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        starts = [0]
        logits: list[int] = []
        owners: list[int] = []
        for seq_index, (chunk, start, table, wants) in enumerate(zip(token_chunks, start_positions, block_tables, sample, strict=True)):
            owners += [seq_index] * len(chunk)
            for offset, token in enumerate(chunk):
                pos = start + offset
                ids.append(token)
                positions.append(pos)
                slots.append(table[pos // block_size] * block_size + pos % block_size)
            starts.append(starts[-1] + len(chunk))
            if int(wants) > len(chunk):
                raise ValueError(f"sequence {seq_index}: {int(wants)} logits rows for {len(chunk)} tokens")
            logits += range(starts[-1] - int(wants), starts[-1])
        width = max((len(t) for t in block_tables), default=1) or 1
        padded = [t + [0] * (width - len(t)) for t in block_tables]
        seq_lens = [s + len(c) for s, c in zip(start_positions, token_chunks, strict=True)]

        def tensor(values, dtype):
            # Not torch.tensor(..., device=cuda): a blocking host-to-device
            # copy synchronises the stream, i.e. waits for every kernel
            # already queued, which would serialise building the next step
            # with running this one (overlap scheduling, draft loops).
            return torch.tensor(values, dtype=dtype).to(device, non_blocking=True)

        return cls(
            input_ids=tensor(ids, torch.int64),
            positions=tensor(positions, torch.int64),
            query_start_loc=tensor(starts, torch.int32),
            seq_lens=tensor(seq_lens, torch.int32),
            block_tables=tensor(padded, torch.int32),
            slot_mapping=tensor(slots, torch.int64),
            logits_indices=tensor(logits, torch.int64),
            token_seq=tensor(owners, torch.int32),
            query_lens_cpu=[len(c) for c in token_chunks],
            seq_lens_cpu=seq_lens,
        )
