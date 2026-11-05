"""GRPO with a truncated importance-sampling correction for rollout mismatch.

For prompt i with G sampled completions and rewards r_ig:

    A_ig = (r_ig - mean_g r_i) / (std_g r_i + eps)         group-relative advantage

Per completion token t (token-level mean over the whole batch):

    ratio = exp(logp_theta - logp_old)                   trainer, new vs old policy
    tis   = min(exp(logp_old - logp_rollout), C)         trainer vs rollout engine
    loss  = -mean( tis * min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) )

`logp_rollout` comes from the inference engine that sampled the tokens,
`logp_old` from the trainer before the update. They differ when the two
compute the same policy with different kernels (training-inference
mismatch); `tis` keeps that difference from biasing the gradient.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """rewards [B * G] (grouped consecutively) -> advantages [B * G]."""
    r = rewards.view(-1, group_size)
    return ((r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True, unbiased=False) + eps)).view(-1)


def token_logprobs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """logits [B, L, V] predicting tokens [B, L] -> log p(token) [B, L] (fp32)."""
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def grpo_loss(
    logp: torch.Tensor,  # [B, L] trainer, with grad
    logp_old: torch.Tensor,  # [B, L] trainer before the update
    logp_rollout: torch.Tensor,  # [B, L] rollout engine
    advantages: torch.Tensor,  # [B]
    mask: torch.Tensor,  # [B, L] 1 for completion tokens
    clip_eps: float = 0.2,
    tis_cap: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    ratio = torch.exp(logp - logp_old)
    tis = torch.exp(logp_old - logp_rollout).clamp(max=tis_cap)
    a = advantages[:, None]
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip_eps, 1 + clip_eps) * a)
    n = mask.sum().clamp(min=1)
    loss = -(tis * surrogate * mask).sum() / n
    with torch.no_grad():
        stats = {
            "loss": float(loss),
            "tis_mean": float((tis * mask).sum() / n),
            "clip_frac": float((((ratio - 1).abs() > clip_eps).float() * mask).sum() / n),
        }
    return loss, stats
