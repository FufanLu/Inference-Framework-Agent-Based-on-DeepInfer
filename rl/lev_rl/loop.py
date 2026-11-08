"""The RL loop: lev-engine rollouts, a transformers trainer, weight sync.

    rollout (engine)  ──tokens, logp_rollout──▶  reward, advantages
          ▲                                          │
          │ update_weights (bf16)                    ▼
    trainer step  ◀── GRPO loss ◀── logp_old (trainer, no grad)

Colocated: the trainer and the engine share one process and GPU, and
weights move with a device copy. Every step also records the
training-inference mismatch on the sampled tokens:

- `trainer_vs_rollout`: |logp_old - logp_rollout| (different kernels: what TIS corrects)
- `rescore_vs_rollout`: the engine re-scoring its own samples as one prefill
  (`LLMEngine.score`); 0.0 exactly with batch-invariant kernels.

    python -m lev_rl.loop --model Qwen/Qwen3-1.7B --steps 50 --out results/rl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))

from lev_engine.config import EngineConfig  # noqa: E402
from lev_engine.engine import LLMEngine  # noqa: E402
from lev_engine.sequence import SamplingParams  # noqa: E402

from .grpo import group_advantages, grpo_loss, token_logprobs  # noqa: E402
from .task import Problem, make_problems, reward  # noqa: E402


@dataclass
class RLConfig:
    model: str
    out: str
    steps: int = 20
    prompts_per_step: int = 16
    group_size: int = 8
    max_new_tokens: int = 256
    lr: float = 1e-6
    micro_batch: int = 16
    clip_eps: float = 0.2
    tis_cap: float = 2.0
    seed: int = 0
    device: str = "cuda"
    engine_dtype: str = "bfloat16"
    kernel_backend: str = "triton"
    batch_invariant: bool = True
    digits: int = 2
    chat_template: bool = True
    rescore: bool = True


def render(tokenizer, problem: Problem, chat: bool) -> list[int]:
    if chat and getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem.question}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
        )
        # transformers 5 returns a BatchEncoding (a mapping, not a dict); older
        # versions a plain list. list() of the mapping gives its key names.
        if not isinstance(ids, list):
            ids = ids["input_ids"]
    else:
        ids = tokenizer(problem.question)["input_ids"]
    ids = [int(t) for t in ids]
    return ids


def pad(rows: list[list], value, device) -> torch.Tensor:
    width = max(len(r) for r in rows)
    return torch.tensor([r + [value] * (width - len(r)) for r in rows], device=device)


def trainer_logprobs(model, seqs: list[list[int]], plens: list[int], micro: int, device, grad: bool):
    """Per-sequence completion logprobs from the trainer, micro-batched."""
    outs = []
    for i in range(0, len(seqs), micro):
        chunk, pl = seqs[i : i + micro], plens[i : i + micro]
        ids = pad(chunk, 0, device)
        attn = pad([[1] * len(s) for s in chunk], 0, device)
        with torch.set_grad_enabled(grad), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
        lp = token_logprobs(logits, ids[:, 1:])
        outs.append((lp, pl, [len(s) for s in chunk]))
    return outs


def run(cfg: RLConfig, tokenizer=None) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = tokenizer or AutoTokenizer.from_pretrained(cfg.model)
    trainer = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32).to(device)
    trainer.gradient_checkpointing_enable()
    trainer.config.use_cache = False
    optim = torch.optim.AdamW(trainer.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
    engine = LLMEngine(
        cfg.model,
        EngineConfig(
            device=cfg.device, dtype=cfg.engine_dtype, block_size=16, max_num_batched_tokens=8192,
            max_num_seqs=cfg.prompts_per_step * cfg.group_size, max_model_len=2048,
            kv_cache_memory_bytes=8 << 30 if device.type == "cuda" else 0,
            num_blocks=0 if device.type == "cuda" else 4096,
            kernel_backend=cfg.kernel_backend, batch_invariant=cfg.batch_invariant,
        ),  # fmt: skip
    )
    rng = random.Random(cfg.seed)
    log: list[dict] = []
    metrics_file = (out_dir / "metrics.jsonl").open("w")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    for step in range(cfg.steps):
        t0 = time.perf_counter()
        problems = make_problems(cfg.prompts_per_step, seed=rng.randrange(1 << 30), digits=cfg.digits)
        prompts = [render(tok, p, cfg.chat_template) for p in problems]
        params = [
            SamplingParams(temperature=1.0, max_tokens=cfg.max_new_tokens, seed=rng.randrange(1 << 62), logprobs=True)
            for _ in range(len(prompts) * cfg.group_size)
        ]
        rid = [f"s{step}-{i}" for i in range(len(params))]
        for i, p in enumerate(params):
            engine.add_request(rid[i], prompts[i // cfg.group_size], p)
        while engine.has_unfinished:
            engine.step()
        samples = [engine.requests.pop(r) for r in rid]
        t_rollout = time.perf_counter() - t0

        texts = [tok.decode(s.output, skip_special_tokens=True) for s in samples]
        rewards = torch.tensor([reward(t, problems[i // cfg.group_size]) for i, t in enumerate(texts)], device=device)
        adv = group_advantages(rewards, cfg.group_size)
        seqs = [s.prompt + s.output for s in samples]
        plens = [len(s.prompt) for s in samples]
        rollout_lp = [s.logprobs for s in samples]

        # Trainer's view of the same tokens before the update.
        trainer.eval()
        old = trainer_logprobs(trainer, seqs, plens, cfg.micro_batch, device, grad=False)
        old_rows = [row[pl - 1 : ln - 1].tolist() for lp, pls, lns in old for row, pl, ln in zip(lp, pls, lns)]
        diffs = [abs(a - b) for o, r in zip(old_rows, rollout_lp) for a, b in zip(o, r)]
        stats = {
            "step": step,
            "reward_mean": float(rewards.mean()),
            "completion_tokens_mean": sum(len(s.output) for s in samples) / len(samples),
            "trainer_vs_rollout_mean_abs": sum(diffs) / max(len(diffs), 1),
            "trainer_vs_rollout_max_abs": max(diffs, default=0.0),
            "rollout_s": t_rollout,
        }
        if cfg.rescore:
            rescored = engine.score(seqs, plens)
            rd = [abs(a - b) for x, y in zip(rescored, rollout_lp) for a, b in zip(x, y)]
            stats["rescore_vs_rollout_max_abs"] = max(rd, default=0.0)
            stats["rescore_bitwise_equal_frac"] = sum(d == 0.0 for d in rd) / max(len(rd), 1)

        # One GRPO update over the batch (on-policy: ratio starts at 1).
        trainer.train()
        optim.zero_grad(set_to_none=True)
        n_tokens = sum(len(s.output) for s in samples)
        loss_total = 0.0
        tis_total = 0.0
        for i in range(0, len(seqs), cfg.micro_batch):
            sl = slice(i, i + cfg.micro_batch)
            (lp, pls, lns), = trainer_logprobs(trainer, seqs[sl], plens[sl], cfg.micro_batch, device, grad=True)
            mask = torch.zeros_like(lp)
            lp_old = torch.zeros_like(lp)
            lp_roll = torch.zeros_like(lp)
            for j, (pl, ln) in enumerate(zip(pls, lns)):
                mask[j, pl - 1 : ln - 1] = 1
                lp_old[j, pl - 1 : ln - 1] = torch.tensor(old_rows[i + j], device=device)
                lp_roll[j, pl - 1 : ln - 1] = torch.tensor(rollout_lp[i + j], device=device, dtype=lp.dtype)
            loss, s = grpo_loss(lp, lp_old, lp_roll, adv[sl], mask, cfg.clip_eps, cfg.tis_cap)
            # Token-level mean over the whole batch, accumulated across micro-batches.
            (loss * mask.sum() / max(n_tokens, 1)).backward()
            loss_total += s["loss"] * float(mask.sum()) / max(n_tokens, 1)
            tis_total += s["tis_mean"] * float(mask.sum()) / max(n_tokens, 1)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0))
        optim.step()

        t1 = time.perf_counter()
        weights = {k: v.detach().to(getattr(torch, cfg.engine_dtype)) for k, v in trainer.state_dict().items()}
        applied = engine.update_weights(weights)
        stats.update(
            loss=loss_total, tis_mean=tis_total, grad_norm=grad_norm, weights_synced=applied,
            sync_s=time.perf_counter() - t1, step_s=time.perf_counter() - t0,
        )  # fmt: skip
        if not math.isfinite(grad_norm):
            stats["warning"] = "non-finite gradient norm"
        log.append(stats)
        metrics_file.write(json.dumps(stats) + "\n")
        metrics_file.flush()
        print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()}), flush=True)
    metrics_file.close()
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    defaults = RLConfig(model="", out="")
    for name, value in asdict(defaults).items():
        if name in ("model", "out"):
            continue
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            ap.add_argument(flag, type=lambda v: v.lower() in ("1", "true", "yes"), default=value)
        else:
            ap.add_argument(flag, type=type(value), default=value)
    run(RLConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
