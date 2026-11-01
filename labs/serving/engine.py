"""An observable, bounded continuous-batching reference simulator.

Time is synthetic milliseconds, not measured GPU performance. The metadata KV
pool stores token IDs, whereas Backend.logits recomputes model context. This
separation is intentional and must not be described as a production KV engine.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import math
import random
from .backend import BigramBackend
from .kv import PagePool, PrefixCache
from .sampling import sample


@dataclass
class Request:
    rid: str
    prompt: tuple[int, ...]
    max_new_tokens: int = 4
    arrival_ms: float = 0.0
    seed: int = 0
    temperature: float = 0.0
    namespace: str = "model-v1:adapter-none:tenant-demo"
    output: list[int] = field(default_factory=list)
    emitted_ms: list[float] = field(default_factory=list)
    computed: int = 0
    state: str = "pending"
    finish_ms: float | None = None
    error: str | None = None
    cache_hit_tokens: int = 0
    preemptions: int = 0
    rng: random.Random = field(init=False, repr=False)
    _submitted: bool = field(init=False, default=False, repr=False)
    def __post_init__(self):
        self.prompt = tuple(self.prompt)
        if not isinstance(self.rid, str) or not self.rid or not self.prompt:
            raise ValueError("nonempty request ID and prompt required")
        if any(type(x) is not int or x < 0 for x in self.prompt):
            raise ValueError("invalid token ID")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 0:
            raise ValueError("invalid max_new_tokens")
        if not math.isfinite(self.arrival_ms) or self.arrival_ms < 0:
            raise ValueError("invalid arrival_ms")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("invalid temperature")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("nonempty namespace required")
        self.output = list(self.output)
        self.emitted_ms = list(self.emitted_ms)
        self.rng = random.Random(self.seed)
    @property
    def context(self):
        return list(self.prompt) + self.output
    def metrics(self):
        gaps = [b-a for a, b in zip(self.emitted_ms, self.emitted_ms[1:])]
        return {"id": self.rid, "state": self.state,
                "output_tokens": len(self.output),
                "ttft_ms": self.emitted_ms[0]-self.arrival_ms if self.emitted_ms else None,
                "e2e_ms": self.finish_ms-self.arrival_ms if self.finish_ms is not None else None,
                "mean_itl_ms": sum(gaps)/len(gaps) if gaps else None,
                "cache_hit_tokens": self.cache_hit_tokens,
                "preemptions": self.preemptions}


@dataclass(frozen=True)
class Work:
    rid: str
    start: int
    token_ids: tuple[int, ...]
    phase: str


class Engine:
    def __init__(self, token_budget=8, chunk_size=4, max_sequences=4,
                 pages=64, block_size=4, prefix_entries=16,
                 policy="round_robin", backend=None):
        if any(type(v) is not int or v < 1 for v in (token_budget, chunk_size, max_sequences)):
            raise ValueError("budgets must be positive")
        if policy not in {"fifo", "decode_first", "round_robin"}:
            raise ValueError("unknown policy")
        self.token_budget, self.chunk_size = token_budget, chunk_size
        self.max_sequences, self.policy = max_sequences, policy
        self.pool = PagePool(pages, block_size)
        self.cache = PrefixCache(self.pool, prefix_entries)
        self.backend = backend or BigramBackend()
        self.requests: dict[str, Request] = {}
        self.waiting: deque[str] = deque()
        self.running: list[str] = []
        self.events: list[dict] = []
        self.time_ms = 0.0
        self.iteration = 0
        self.cursor = 0

    def event(self, kind, **data):
        self.events.append({"iteration": self.iteration, "time_ms": round(self.time_ms, 6),
                            "kind": kind, **data})

    def submit(self, request: Request):
        if not isinstance(request, Request):
            raise ValueError("submit requires a Request")
        if (request._submitted or request.state != "pending" or request.output
                or request.emitted_ms or request.computed != 0
                or request.finish_ms is not None or request.error is not None
                or request.cache_hit_tokens != 0 or request.preemptions != 0):
            raise ValueError("submit requires a fresh pending request with no runtime history")
        if request.rid in self.requests:
            raise ValueError("request ID already exists")
        if request.arrival_ms < self.time_ms:
            raise ValueError("arrival cannot precede current simulator time")
        if max(request.prompt) >= self.backend.vocab_size:
            raise ValueError("token outside backend vocabulary")
        # This teaching scheduler never swaps a single request across the entire
        # pool. Reject impossible reservations before admitting it.
        if len(request.prompt) + max(0, request.max_new_tokens - 1) > len(self.pool.pages)*self.pool.block_size:
            raise ValueError("one request would exceed the entire KV pool")
        request._submitted = True
        self.requests[request.rid] = request
        self.event("submitted", rid=request.rid, arrival_ms=request.arrival_ms)

    def _finish(self, r, state, error=None):
        if r.rid in self.pool.tables:
            self.pool.release(r.rid)
        if r.rid in self.running:
            self.running.remove(r.rid)
        if r.rid in self.waiting:
            self.waiting.remove(r.rid)
        r.state, r.finish_ms, r.error = state, self.time_ms, error
        self.event(state, rid=r.rid, error=error)

    def cancel(self, rid):
        r = self.requests[rid]
        if r.state in {"finished", "cancelled", "error"}:
            return False
        # Cancelling a request not yet arrived is supported, but arrival must not
        # produce a negative duration in the synthetic metric report.
        if r.arrival_ms > self.time_ms:
            r.arrival_ms = self.time_ms
        self._finish(r, "cancelled")
        self.check()
        return True

    def _ingest(self):
        pending = sorted((r for r in self.requests.values()
                          if r.state == "pending" and r.arrival_ms <= self.time_ms),
                         key=lambda r: (r.arrival_ms, r.rid))
        for r in pending:
            if r.max_new_tokens == 0:
                self._finish(r, "finished")
            else:
                r.state = "waiting"
                self.waiting.append(r.rid)
                self.event("enqueued", rid=r.rid)

    def _admit(self):
        while self.waiting and len(self.running) < self.max_sequences:
            rid = self.waiting.popleft()
            r = self.requests[rid]
            self.pool.create(rid)
            # Recompute preempted output history too; only prompt prefixes are
            # published by this implementation.
            r.computed = self.cache.attach(rid, r.namespace, r.prompt)
            r.cache_hit_tokens += r.computed
            r.state = "running"
            self.running.append(rid)
            self.event("admitted", rid=rid, computed=r.computed)

    def _order(self):
        ids = list(self.running)
        if self.policy == "decode_first":
            ids.sort(key=lambda rid: self.requests[rid].computed < len(self.requests[rid].prompt))
        elif self.policy == "round_robin" and ids:
            i = self.cursor % len(ids)
            ids = ids[i:] + ids[:i]
            self.cursor += 1
        return ids

    def _preempt_for_progress(self):
        # Keep the oldest active request and release younger ones. The victim
        # retains generated IDs and its RNG state; recomputation must not emit
        # duplicate tokens. Each retry must have strictly fewer active tables.
        if len(self.running) < 2:
            return False
        rid = self.running[-1]
        r = self.requests[rid]
        self.pool.release(rid)
        self.running.remove(rid)
        r.state, r.computed = "waiting", 0
        r.preemptions += 1
        self.waiting.appendleft(rid)
        self.event("preempted", rid=rid)
        return True

    def step(self):
        self._ingest()
        if not self.waiting and not self.running:
            future = [r.arrival_ms for r in self.requests.values() if r.state == "pending"]
            if not future:
                return False
            self.time_ms = min(future)
            self._ingest()
        self._admit()
        self.iteration += 1
        budget = self.token_budget
        plan = []
        for rid in self._order():
            if budget == 0:
                break
            r = self.requests[rid]
            context = r.context
            gap = len(context)-r.computed
            phase = "prefill" if r.computed < len(r.prompt) else "decode"
            count = min(gap, budget, self.chunk_size if phase == "prefill" else 1)
            if count <= 0:
                continue
            while self.pool.needed(rid, count) > len(self.pool.free) and self.cache.evict_one():
                self.event("cache_evicted")
            if self.pool.needed(rid, count) > len(self.pool.free):
                self.event("allocation_blocked", rid=rid)
                continue
            values = context[r.computed:r.computed+count]
            start = r.computed
            self.pool.append(rid, values)
            r.computed += count
            plan.append(Work(rid, start, tuple(values), phase))
            budget -= count
        if not plan:
            if self._preempt_for_progress():
                # Do not immediately re-admit the released request in this
                # iteration; the caller will retry with reduced concurrency.
                saved_limit = self.max_sequences
                self.max_sequences = len(self.running)
                try:
                    return self.step()
                finally:
                    self.max_sequences = saved_limit
            if self.running:
                raise RuntimeError("no progress despite an individually feasible request")
            return any(r.state == "pending" for r in self.requests.values())
        prefill = sum(len(w.token_ids) for w in plan if w.phase == "prefill")
        decode = sum(len(w.token_ids) for w in plan if w.phase == "decode")
        # A transparent synthetic cost law, deliberately not hardware-calibrated.
        self.time_ms += 1.0 + .25*prefill + .10*decode
        self.event("batch", scheduled_tokens=sum(len(w.token_ids) for w in plan),
                   work=[{"rid": w.rid, "start": w.start, "count": len(w.token_ids), "phase": w.phase} for w in plan])
        for work in plan:
            r = self.requests[work.rid]
            if r.computed >= len(r.prompt):
                self.cache.publish(r.rid, r.namespace, len(r.prompt))
            # Catch-up after preemption is not a new generation step.
            if r.computed == len(r.context):
                try:
                    token = sample(self.backend.logits(r.context), r.rng, r.temperature)
                except Exception as exc:
                    self._finish(r, "error", type(exc).__name__)
                    continue
                r.output.append(token)
                r.emitted_ms.append(self.time_ms)
                self.event("token", rid=r.rid, token=token)
                if len(r.output) == r.max_new_tokens:
                    self._finish(r, "finished")
        self.check()
        return True

    def run(self, max_iterations=10000):
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        for _ in range(max_iterations):
            if not self.step():
                return self.report()
        raise RuntimeError("iteration budget exceeded")

    def check(self):
        self.pool.check(self.cache.refs())
        assert len(self.running) == len(set(self.running))
        for rid in self.running:
            r = self.requests[rid]
            assert r.state == "running"
            assert self.pool.tokens(rid) == r.context[:r.computed]
            assert 0 <= r.computed <= len(r.context)
        assert set(self.pool.tables) == set(self.running)
        for r in self.requests.values():
            assert len(r.output) == len(r.emitted_ms)
            assert len(r.output) <= r.max_new_tokens

    def report(self):
        return {"time_kind": "synthetic_ms_not_GPU_measurement", "time_ms": self.time_ms,
                "iterations": self.iteration, "requests": [r.metrics() for r in self.requests.values()],
                "outputs": {rid: list(r.output) for rid, r in self.requests.items()},
                "free_pages": len(self.pool.free), "cached_prefixes": len(self.cache.entries)}
