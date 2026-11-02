"""Synthetic model output for the simulated backend."""

from .logprobs import fake_logprobs
from .output import OutputSynthesizer, Synthesized, SynthesisError, derive_seed, render_invocation, request_seed

__all__ = [
    "OutputSynthesizer",
    "Synthesized",
    "SynthesisError",
    "derive_seed",
    "fake_logprobs",
    "render_invocation",
    "request_seed",
]
