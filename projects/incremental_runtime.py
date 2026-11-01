"""P2 reference: per-request real KV, explicit lifecycle, and online attention.

Synchronous NumPy CPU; contiguous per-layer arrays, not paged GPU storage.
The existing metadata Engine keeps its original API. This small runner makes
request identity explicit and verifies the incremental backend contract first.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import numpy as np

from labs.kernels.reference import online_attention
from labs.models.core import TinyDecoder, DecoderConfig, split_heads, merge_heads, rope
from projects.model_runtime import DecoderBackend, direct_generate


class OnlineDecoder(TinyDecoder):
    """Same full/step model; replace the attention computation in every layer.

Supported positions are full causal prefill or one last-position decode query.
This deliberately does not generalize to arbitrary rectangular chunk masks.
"""
    def _attn(self, x, block, position, previous=None):
        c = self.config
        q = rope(split_heads(x @ block["q"], c.heads), position)
        k = rope(split_heads(x @ block["k"], c.kv_heads), position)
        v = split_heads(x @ block["v"], c.kv_heads)
        if previous is not None:
            k = np.concatenate([previous[0], k], axis=1)
            v = np.concatenate([previous[1], v], axis=1)
        full = len(position) == k.shape[1] and np.array_equal(position, np.arange(k.shape[1]))
        last = len(position) == 1 and int(position[0]) == k.shape[1] - 1
        if not (full or last):
            raise ValueError("this adapter supports full prefill or one last-position query")
        group_size = c.heads // c.kv_heads
        heads = [online_attention(q[h], k[h // group_size], v[h // group_size],
                                  block=3, causal=full)
                 for h in range(c.heads)]
        return merge_heads(np.stack(heads)) @ block["o"], (k, v)


@dataclass
class RunningRequest:
    prompt: tuple[int, ...]
    max_new_tokens: int
    cache: list | None
    next_logits: np.ndarray | None
    output: list[int] = field(default_factory=list)
    status: str = "running"


class IncrementalRunner:
    def __init__(self, model=None):
        self.model = model or TinyDecoder()
        self.requests: dict[str, RunningRequest] = {}

    def start(self, rid, prompt, max_new_tokens=4):
        if not isinstance(rid, str) or not rid or rid in self.requests:
            raise ValueError("start requires a new nonempty request ID")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        prompt = tuple(prompt)
        if not prompt or any(type(t) is not int or not 0 <= t < self.model.config.vocab for t in prompt):
            raise ValueError("prompt must contain valid integer token IDs")
        if max_new_tokens == 0:
            self.requests[rid] = RunningRequest(prompt, 0, None, None, status="finished")
            return
        cache, logits = None, None
        for token in prompt:
            logits, cache = self.model.step(token, cache)
        # Publish a new request only after its prefill has completed successfully.
        self.requests[rid] = RunningRequest(prompt, max_new_tokens, cache, logits)

    def advance(self, rid):
        request = self.requests[rid]
        if request.status != "running":
            raise ValueError("only a running request can advance")
        token = int(np.argmax(request.next_logits))
        final = len(request.output) + 1 == request.max_new_tokens
        if final:
            new_logits, new_cache = None, None
        else:
            # TinyDecoder.step returns new cache arrays; it does not mutate the old cache.
            new_logits, new_cache = self.model.step(token, request.cache)
        request.output.append(token)
        request.next_logits, request.cache = new_logits, new_cache
        if final:
            request.status = "finished"
        return token

    def cancel(self, rid):
        request = self.requests[rid]
        if request.status != "running":
            return False
        request.cache = None
        request.next_logits = None
        request.status = "cancelled"
        return True

    def snapshot(self):
        report = {}
        for rid, request in self.requests.items():
            cache = request.cache or []
            report[rid] = {"status": request.status, "output": list(request.output),
                           "cache_lengths": [k.shape[1] for k, v in cache],
                           "cache_bytes": sum(k.nbytes + v.nbytes for k, v in cache)}
        return report

    def run(self, policy="round_robin", max_steps=10000):
        if policy not in {"fifo", "round_robin"}:
            raise ValueError("unknown policy")
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        trace, steps = [], 0
        ready = deque(rid for rid, request in self.requests.items()
                      if request.status == "running")
        while ready:
            if steps >= max_steps:
                raise RuntimeError("step budget exceeded")
            rid = ready.popleft()
            token = self.advance(rid)
            steps += 1
            trace.append({"request": rid, "token": token, "states": self.snapshot()})
            if self.requests[rid].status == "running":
                if policy == "fifo":
                    ready.appendleft(rid)
                else:
                    ready.append(rid)
        return trace


def demo(use_online=False, policy="round_robin"):
    model = OnlineDecoder() if use_online else TinyDecoder()
    runner = IncrementalRunner(model)
    prompts = {"A": (1, 2), "B": (2, 3, 4, 5, 6)}
    expected = {rid: direct_generate(DecoderBackend(model), prompt, 4) for rid, prompt in prompts.items()}
    for rid, prompt in prompts.items():
        runner.start(rid, prompt, 4)
    initial = runner.snapshot()
    trace = runner.run(policy)
    actual = {rid: list(request.output) for rid, request in runner.requests.items()}
    return {"model": type(model).__name__, "policy": policy, "initial": initial,
            "trace": trace, "expected": expected, "actual": actual,
            "matches_full_context_reference": expected == actual,
            "final": runner.snapshot(),
            "scope": "real per-request NumPy KV arrays; no paged pool, GPU timing, or trained-language claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--policy", choices=["fifo", "round_robin"], default="round_robin")
    args = parser.parse_args()
    print(json.dumps(demo(args.online, args.policy), indent=2))


if __name__ == "__main__":
    main()
