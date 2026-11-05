"""Rotary embeddings as DeepSeek-V2 uses them: interleaved pairs, YaRN scaling.

DeepSeek checkpoints rotate adjacent pairs ``(x[2i], x[2i+1])`` (GPT-J
style), not the two halves. YaRN blends interpolated and extrapolated
frequencies with a linear ramp over the dimensions, and scales the softmax by
``mscale(factor, mscale_all_dim)^2`` (see `softmax_scale`).
"""

from __future__ import annotations

import math

import torch

from .config import RopeSpec


def _mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def inv_freq(spec: RopeSpec, dim: int) -> tuple[torch.Tensor, float]:
    """Inverse frequencies (fp32) and the factor applied to cos/sin."""
    pos_freqs = spec.theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if spec.kind == "default":
        return 1.0 / pos_freqs, 1.0
    if spec.kind != "yarn":
        raise ValueError(f"unsupported rope type {spec.kind!r}")

    def correction_dim(rotations: float) -> float:
        return (dim * math.log(spec.original_max_position / (rotations * 2 * math.pi))) / (2 * math.log(spec.theta))

    low = max(math.floor(correction_dim(spec.beta_fast)), 0)
    high = min(math.ceil(correction_dim(spec.beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extrapolation = 1 - ramp
    freqs = (1.0 / (spec.factor * pos_freqs)) * (1 - extrapolation) + (1.0 / pos_freqs) * extrapolation
    if spec.mscale and spec.mscale_all_dim:
        attention_factor = _mscale(spec.factor, spec.mscale) / _mscale(spec.factor, spec.mscale_all_dim)
    else:
        attention_factor = _mscale(spec.factor)
    return freqs, attention_factor


def softmax_scale(spec: RopeSpec, qk_head_dim: int) -> float:
    scale = qk_head_dim**-0.5
    if spec.kind != "default" and spec.mscale_all_dim:
        m = _mscale(spec.factor, spec.mscale_all_dim)
        scale *= m * m
    return scale


class RotaryCache:
    """cos/sin for every position up to `max_position`, computed once."""

    def __init__(
        self, spec: RopeSpec, dim: int, max_position: int, device: torch.device | str = "cpu", interleaved: bool = True
    ):
        self.interleaved = interleaved
        freqs, factor = inv_freq(spec, dim)
        t = torch.arange(max_position, dtype=torch.float32)
        angles = torch.outer(t, freqs)  # [P, dim/2]
        self.cos = (angles.cos() * factor).to(device)
        self.sin = (angles.sin() * factor).to(device)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate ``x[..., T, (heads,) dim]`` at ``positions[T]``: pairs
        (x[2i], x[2i+1]) when interleaved, else (x[i], x[i + dim/2])."""
        cos = self.cos[positions]
        sin = self.sin[positions]
        while cos.dim() < x.dim():  # broadcast over the head axis
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        xf = x.to(torch.promote_types(x.dtype, torch.float32))
        if self.interleaved:
            even, odd = xf[..., 0::2], xf[..., 1::2]
            out = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        else:
            half = xf.shape[-1] // 2
            a, b = xf[..., :half], xf[..., half:]
            out = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
        return out.to(x.dtype)
