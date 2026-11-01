"""Complete tiny causal decoder with Block Attention Residual state flow.

Reference: MoonshotAI/Attention-Residuals README at commit
85e22310fe5ee860b4a023de312d791de8a5a5e6, the Block AttnRes pseudocode.
We reuse our independently written GQA/RoPE and SwiGLU sublayers. Blocks count
attention + MLP sublayers, must have even size, and close before attention.
Embedding is the initial partial source and is sealed exactly once at depth 0.
The LM head uses an explicit extra learned depth query followed by RMSNorm.
No training, checkpoint compatibility, norm gains, dropout, or performance claim.
"""
from __future__ import annotations

import json
import numpy as np
from .core import DecoderConfig, TinyDecoder, attention_residual, rms_norm, swiglu


def _snapshot(array):
    """Own immutable storage: later partial updates cannot rewrite history."""
    result = np.array(array, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


class TinyBlockAttnResDecoder(TinyDecoder):
    """A complete forward/step model, not merely a source aggregation helper.

    Zero-initialized queries are explicit mutable model parameters. At zero,
    every depth aggregation averages its sources. Set queries to finite
    nonzero vectors to study content-dependent depth selection.
    """

    def __init__(self, config=None, block_sublayers=4):
        if (isinstance(block_sublayers, (bool, np.bool_))
                or not isinstance(block_sublayers, (int, np.integer))
                or block_sublayers < 2 or block_sublayers % 2):
            raise ValueError("block_sublayers must be a positive even integer")
        super().__init__(config)
        self.block_sublayers = int(block_sublayers)
        self.depth_queries = np.zeros((2 * self.config.layers + 1, self.config.dim))

    def _ids(self, tokens):
        ids = np.asarray(tokens)
        if (ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= self.config.vocab)):
            raise ValueError("expected nonempty integer token ids in vocabulary")
        return ids.astype(np.int64, copy=False)

    def _execute(self, ids, positions, cache, return_trace):
        c = self.config
        if (self.depth_queries.shape != (2 * c.layers + 1, c.dim)
                or not np.isfinite(self.depth_queries).all()):
            raise ValueError("depth_queries must be finite and match model depth/dim")
        completed, labels, events, new_cache = [], [], [], []
        partial = self.embedding[ids].copy()
        partial_start = None  # Initial partial is the embedding, not block 0.

        def aggregate(sublayer, kind):
            current_label = ("embedding" if partial_start is None
                             else f"partial[{partial_start}:{sublayer}]")
            sources = np.stack(completed + [partial])
            mixed, weights = attention_residual(sources, self.depth_queries[sublayer])
            if return_trace:
                events.append({
                    "sublayer": sublayer, "kind": kind,
                    "source_labels": tuple(labels + [current_label]),
                    "sources": _snapshot(sources),
                    "weights": _snapshot(weights),
                    "mixed": _snapshot(mixed), "sealed_after_mix": None,
                })
            return mixed

        for layer, block in enumerate(self.blocks):
            sublayer = 2 * layer
            # Crucial ordering: read old partial BEFORE sealing/resetting it.
            mixed = aggregate(sublayer, "attention")
            if sublayer % self.block_sublayers == 0:
                label = ("embedding" if partial_start is None
                         else f"block[{partial_start}:{sublayer}]")
                completed.append(_snapshot(partial))
                labels.append(label)
                if return_trace:
                    events[-1]["sealed_after_mix"] = label
                partial = None
                partial_start = sublayer
            delta, state = self._attn(rms_norm(mixed), block, positions, cache[layer])
            partial = delta if partial is None else partial + delta
            new_cache.append(state)

            mixed = aggregate(sublayer + 1, "mlp")
            delta = swiglu(rms_norm(mixed), block["gate"], block["up"], block["down"])
            partial = partial + delta

        # A deliberate, explicit output closure: read every remaining source.
        # Do not take only the last partial, or ignore all completed blocks.
        mixed = aggregate(2 * c.layers, "head")
        logits = rms_norm(mixed) @ self.output
        audit = None
        if return_trace:
            audit = {
                "events": events, "completed_labels": tuple(labels),
                "completed_sources": tuple(_snapshot(x) for x in completed),
                "partial": _snapshot(partial),
                "block_sublayers": self.block_sublayers,
            }
        return logits, new_cache, audit

    def forward(self, tokens, return_trace=False):
        ids = self._ids(tokens)
        logits, _, audit = self._execute(ids, np.arange(len(ids)),
                                         [None] * self.config.layers, return_trace)
        return (logits, audit) if return_trace else logits

    def step(self, token, cache=None):
        ids = self._ids([token])
        c = self.config
        if cache is None:
            cache = [None] * c.layers
        if len(cache) != c.layers:
            raise ValueError("wrong number of cache layers")
        lengths = []
        for state in cache:
            if state is None:
                lengths.append(0)
                continue
            if not isinstance(state, (tuple, list)) or len(state) != 2:
                raise ValueError("each cache layer must contain K and V")
            key, value = map(np.asarray, state)
            expected = (c.kv_heads, c.dim // c.heads)
            if (key.ndim != 3 or key.shape != value.shape
                    or (key.shape[0], key.shape[2]) != expected
                    or not np.isfinite(key).all() or not np.isfinite(value).all()):
                raise ValueError("invalid K/V cache shape or values")
            lengths.append(key.shape[1])
        if len(set(lengths)) != 1:
            raise ValueError("cache layer lengths disagree")
        logits, new_cache, _ = self._execute(ids, np.array([lengths[0]]), cache, False)
        return logits[0], new_cache


def main():
    config = DecoderConfig(layers=3)
    model = TinyBlockAttnResDecoder(config, block_sublayers=4)
    ids = [1, 2, 3, 4]
    full, audit = model.forward(ids, return_trace=True)
    cached, state = [], None
    for token in ids:
        out, state = model.step(token, state)
        cached.append(out)
    changed = model.forward([1, 2, 3, 9])
    average_error = max(float(np.max(abs(event["mixed"] - event["sources"].mean(0))))
                        for event in audit["events"])
    cache_error = float(np.max(abs(full - cached)))
    causal_error = float(np.max(abs(full[:-1] - changed[:-1])))
    assert cache_error < 1e-10 and causal_error < 1e-10 and average_error < 1e-10
    print(json.dumps({
        "model": "TinyBlockAttnResDecoder", "trained": False,
        "block_sublayers": model.block_sublayers,
        "events": [{"sublayer": e["sublayer"], "kind": e["kind"],
                    "source_count": len(e["source_labels"]),
                    "labels": e["source_labels"], "sealed_after_mix": e["sealed_after_mix"]}
                   for e in audit["events"]],
        "full_vs_cached_max_error": cache_error,
        "future_token_invariance_max_error": causal_error,
        "zero_query_vs_mean_max_error": average_error,
        "logits_shape": list(full.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
