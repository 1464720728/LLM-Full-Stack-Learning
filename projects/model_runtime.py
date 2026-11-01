"""Cross-volume M+S+K project. Real NumPy decoder inside a metadata scheduler.

The adapter recomputes full context. The page pool holds token metadata, not
numeric KV tensors. Simulated times must never be reported as GPU benchmarks.
"""
import json
import numpy as np
from labs.models.core import TinyDecoder, attention, causal_mask
from labs.kernels.reference import online_attention
from labs.serving.engine import Engine, Request


class DecoderBackend:
    def __init__(self, model=None):
        self.model = model or TinyDecoder()
        self.vocab_size = self.model.config.vocab

    def logits(self, token_ids):
        return self.model.forward(np.asarray(token_ids, dtype=np.int64))[-1].tolist()


def direct_generate(backend, prompt, count):
    context = list(prompt)
    result = []
    for _ in range(count):
        token = int(np.argmax(backend.logits(context)))
        context.append(token)
        result.append(token)
    return result


def run(policy="round_robin"):
    backend = DecoderBackend()
    engine = Engine(backend=backend, token_budget=3, chunk_size=2, max_sequences=2, prefix_entries=0, policy=policy)
    inputs = {"short": (1, 2), "long": (2, 3, 4, 5, 6)}
    expected = {rid: direct_generate(backend, prompt, 4) for rid, prompt in inputs.items()}
    for rid, prompt in inputs.items():
        engine.submit(Request(rid, prompt, max_new_tokens=4))
    engine.run()
    generated = {rid: request.output for rid, request in engine.requests.items()}
    rng = np.random.default_rng(31)
    q, k, v = [rng.normal(size=(7, 4)) for _ in range(3)]
    model_result, _ = attention(q, k, v, causal_mask(7))
    kernel_result = online_attention(q, k, v, block=3, causal=True)
    return {"policy": policy, "direct": expected, "scheduled": generated,
            "matches_direct": generated == expected,
            "independent_attention_max_error": float(np.max(np.abs(model_result-kernel_result))),
            "scheduler_report": engine.report(),
            "timing_scope": "synthetic scheduler time; not measured latency",
            "memory_scope": "token metadata pages; numeric model context is recomputed"}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
