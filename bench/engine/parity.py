"""lev-engine vs transformers on the real checkpoint: prompt logits on GPU.

For each text prompt, both compute the logits of every position (bf16
weights); we report the largest logit difference, KL(transformers || lev)
and how often the top-1 token agrees, per kernel backend.

    python bench/engine/parity.py --model /scratch/.../DeepSeek-V2-Lite-Chat --out results/parity.json
"""

from __future__ import annotations

import argparse
import gc

import torch

from common import DEVICE, TEXTS, tokenizer, write

from lev_engine.batch import ForwardBatch
from lev_engine.config import ModelSpec
from lev_engine.models.deepseek_v2 import DeepseekV2
from lev_engine.ops import get_ops
from lev_engine.weights import load_state_dict

BLOCK = 16


def lev_logits(model: DeepseekV2, tokens: list[int]) -> torch.Tensor:
    blocks = (len(tokens) + BLOCK - 1) // BLOCK
    cache = [torch.zeros(model.kv_cache_shape(blocks, BLOCK), dtype=model.dtype, device=DEVICE) for _ in model.layers]
    batch = ForwardBatch.build([tokens], [0], [list(range(blocks))], BLOCK, [True], DEVICE)
    batch.logits_indices = torch.arange(len(tokens), device=DEVICE)
    return model.forward(batch, cache)


def compare(ref: torch.Tensor, got: torch.Tensor) -> dict:
    ref, got = ref.float(), got.float()
    lp_ref, lp_got = ref.log_softmax(-1), got.log_softmax(-1)
    kl = (lp_ref.exp() * (lp_ref - lp_got)).sum(-1)
    return {
        "positions": ref.shape[0],
        "max_abs_logit_diff": float((ref - got).abs().max()),
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "top1_agreement": float((ref.argmax(-1) == got.argmax(-1)).float().mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = tokenizer(args.model)
    bos = ModelSpec.from_pretrained(args.model).bos_token_id
    prompts = [[bos] + tok.encode(t).ids for t in TEXTS]

    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(DEVICE).eval()
    with torch.no_grad():
        reference = [hf(torch.tensor([p], device=DEVICE)).logits[0].float().cpu() for p in prompts]
    del hf
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    spec = ModelSpec.from_pretrained(args.model)
    weights = load_state_dict(args.model, torch.bfloat16, DEVICE)
    results = {}
    for name, backend, invariant in (
        ("reference", "reference", False),
        ("triton", "triton", False),
        ("triton-invariant", "triton", True),
    ):
        model = DeepseekV2(spec, weights, get_ops(backend, invariant), 4096, DEVICE)
        per_prompt = [compare(ref, lev_logits(model, p).cpu()) for ref, p in zip(reference, prompts, strict=True)]
        total = sum(r["positions"] for r in per_prompt)
        results[name] = {
            "per_prompt": per_prompt,
            "positions": total,
            "max_abs_logit_diff": max(r["max_abs_logit_diff"] for r in per_prompt),
            "mean_kl": sum(r["mean_kl"] * r["positions"] for r in per_prompt) / total,
            "top1_agreement": sum(r["top1_agreement"] * r["positions"] for r in per_prompt) / total,
        }
        print(name, {k: v for k, v in results[name].items() if k != "per_prompt"}, flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    write(args.out, {"model": args.model, "dtype": "bfloat16", "prompt_lengths": [len(p) for p in prompts],
                     "results": results})  # fmt: skip


if __name__ == "__main__":
    main()
