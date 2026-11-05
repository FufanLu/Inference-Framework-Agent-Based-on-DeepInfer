"""Model implementations, chosen by the checkpoint's architecture."""

from __future__ import annotations

from ..config import ModelSpec


def model_class(spec: ModelSpec):
    if spec.architecture == "Qwen3ForCausalLM":
        from .qwen3 import Qwen3

        return Qwen3
    from .deepseek_v2 import DeepseekV2

    return DeepseekV2
