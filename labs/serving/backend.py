"""Deterministic reference backends; neither claims language-model quality."""
from typing import Protocol


class Backend(Protocol):
    vocab_size: int
    def logits(self, token_ids: list[int]) -> list[float]: ...


class BigramBackend:
    """A fixed 16-state transition model, suitable for exact scheduling tests."""
    vocab_size = 16
    def logits(self, token_ids):
        if not token_ids:
            raise ValueError("empty context")
        last = token_ids[-1]
        return [float(4 - abs(i - (last + 1) % self.vocab_size)) for i in range(self.vocab_size)]


class TinyAttentionBackend:
    """One attention head on CPU. Recomputes context; no numeric KV caching.

    It connects scheduler interfaces to real tensor calculations, while keeping
    paged *metadata* separate from the model's temporary NumPy tensors.
    """
    vocab_size = 16
    def __init__(self, seed=11, width=8):
        import numpy as np
        self.np = np
        g = np.random.default_rng(seed)
        self.embedding = g.normal(size=(self.vocab_size, width)) / width ** .5
        self.q = g.normal(size=(width, width)) / width ** .5
        self.k = g.normal(size=(width, width)) / width ** .5
        self.v = g.normal(size=(width, width)) / width ** .5
        self.out = g.normal(size=(width, self.vocab_size)) / width ** .5
    def logits(self, token_ids):
        np = self.np
        x = self.embedding[token_ids]
        q, k, v = x[-1] @ self.q, x @ self.k, x @ self.v
        scores = q @ k.T / x.shape[-1] ** .5
        p = np.exp(scores - scores.max())
        p /= p.sum()
        return ((p @ v + x[-1]) @ self.out).tolist()
