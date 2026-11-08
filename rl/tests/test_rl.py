"""The RL loop end to end on the tiny Qwen3 checkpoint (CPU, float64 engine)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


from lev_rl.grpo import group_advantages, grpo_loss  # noqa: E402
from lev_rl.loop import RLConfig, render, run  # noqa: E402
from lev_rl.task import make_problems, reward  # noqa: E402


class CharTokenizer:
    """Characters as token IDs 2..; enough for the loop's plumbing."""

    chat_template = None

    def __call__(self, text):
        return {"input_ids": [2 + (ord(c) % 300) for c in text[:40]]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 2) for i in ids if 2 <= i < 302)


def test_task_and_grpo_math():
    p = make_problems(3, seed=1)
    assert reward(f"so <answer>{p[0].answer}</answer>", p[0]) == 1.0
    assert reward("<answer>1</answer> no", p[0]) == (1.0 if p[0].answer == 1 else 0.0)
    adv = group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0]), 3)
    torch.testing.assert_close(adv[:3].sum(), torch.tensor(0.0), atol=1e-5, rtol=0)
    lp = torch.zeros(2, 3, requires_grad=True)
    loss, stats = grpo_loss(lp, lp.detach(), lp.detach(), torch.tensor([1.0, -1.0]), torch.ones(2, 3))
    loss.backward()
    assert stats["tis_mean"] == 1.0 and stats["clip_frac"] == 0.0
    # Positive advantage pushes logp up (negative gradient of the loss).
    assert lp.grad[0].lt(0).all() and lp.grad[1].gt(0).all()


def test_loop_runs_syncs_weights_and_measures_mismatch(tiny_qwen3, tmp_path):
    path, _ = tiny_qwen3
    cfg = RLConfig(model=str(path), out=str(tmp_path), steps=2, prompts_per_step=2, group_size=2,
                   max_new_tokens=6, lr=1e-2, micro_batch=4, device="cpu", engine_dtype="float64",
                   kernel_backend="reference", batch_invariant=False, chat_template=False)  # fmt: skip
    log = run(cfg, tokenizer=CharTokenizer())
    assert len(log) == 2 and all(s["weights_synced"] > 0 for s in log)
    # The trainer runs float32, the engine float64: tiny but nonzero mismatch.
    assert 0 < log[0]["trainer_vs_rollout_max_abs"] < 1e-3
    # The engine re-scoring its own samples as one prefill agrees closely.
    assert log[0]["rescore_vs_rollout_max_abs"] < 1e-9
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
    json.loads((tmp_path / "config.json").read_text())


def test_render_takes_input_ids_from_a_chat_template_mapping():
    """transformers 5 returns a BatchEncoding (a UserDict); job 11715386 fed its key names to the engine."""
    from collections import UserDict

    class ChatTokenizer(CharTokenizer):
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, **kwargs):
            return UserDict({"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]})

    (problem,) = make_problems(1, seed=0)
    assert render(ChatTokenizer(), problem, chat=True) == [7, 8, 9]
    assert all(isinstance(t, int) for t in render(CharTokenizer(), problem, chat=True))
