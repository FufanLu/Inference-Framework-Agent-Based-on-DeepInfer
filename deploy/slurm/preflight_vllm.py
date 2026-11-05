"""Find vLLM engine flags that load a model on this GPU, before the real run.

Each attempt runs in a fresh subprocess (a failed engine can leave GPU memory
behind) with offline `vllm.LLM`: load, generate from token IDs, then send the
same prompt again and check that the prefix cache served it. Prints a JSON
report and writes the first working flags to --flags-out, ready for
`--vllm-args`.

    python deploy/slurm/preflight_vllm.py --model $MODEL_DIR --flags-out $RUN/vllm-flags.txt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

# Tried in order. Flags as for `vllm serve`; the Python dict is the same as
# keyword arguments to `vllm.LLM`.
ATTEMPTS: list[tuple[str, dict]] = [
    ("", {}),
    ("--kv-cache-dtype fp8", {"kv_cache_dtype": "fp8"}),
    ("--kv-cache-dtype fp8 --enforce-eager", {"kv_cache_dtype": "fp8", "enforce_eager": True}),
    ("--enforce-eager", {"enforce_eager": True}),
]

CHILD = r"""
import json, sys, time
from vllm import LLM, SamplingParams, TokensPrompt
model, max_len, kwargs = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
started = time.time()
llm = LLM(model=model, max_model_len=max_len, enable_prefix_caching=True,
          gpu_memory_utilization=0.85, **kwargs)
loaded = time.time() - started
tok = llm.get_tokenizer()
ids = tok.encode("According to all known laws of aviation, " * 40)
params = SamplingParams(temperature=0, max_tokens=32)
first = llm.generate([TokensPrompt(prompt_token_ids=ids)], params)[0]
second = llm.generate([TokensPrompt(prompt_token_ids=ids)], params)[0]
cfg = llm.llm_engine.vllm_config.cache_config
print("PREFLIGHT " + json.dumps({
    "load_s": round(loaded, 1),
    "prompt_tokens": len(ids),
    "text": tok.decode(first.outputs[0].token_ids),
    "cached_tokens_second": second.num_cached_tokens,
    "deterministic": first.outputs[0].token_ids == second.outputs[0].token_ids,
    "block_size": cfg.block_size,
    "gpu_blocks": cfg.num_gpu_blocks,
    "kv_cache_dtype": str(cfg.cache_dtype),
}))
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--flags-out", required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    report = []
    for flags, kwargs in ATTEMPTS:
        started = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", CHILD, args.model, str(args.max_model_len), json.dumps(kwargs)],
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            out, code = proc.stdout + proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as error:
            out, code = f"timeout after {args.timeout}s: {error}", -1
        result = next((line[10:] for line in out.splitlines() if line.startswith("PREFLIGHT ")), None)
        entry = {"flags": flags, "exit": code, "seconds": round(time.time() - started, 1)}
        if result:
            entry["result"] = json.loads(result)
        else:
            entry["tail"] = out.strip().splitlines()[-25:]
        report.append(entry)
        print(json.dumps(entry, indent=2), flush=True)
        if result:
            with open(args.flags_out, "w") as f:
                f.write(flags + "\n")
            break
    else:
        print("no flag set loaded the model", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
