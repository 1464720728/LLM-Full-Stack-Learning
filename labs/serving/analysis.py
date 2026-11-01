"""Transparent unit-aware estimates; estimates are not admission guarantees."""
from dataclasses import dataclass, asdict
import math

GiB = 2 ** 30


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("nonempty observations and q in [0,1] required")
    data = sorted(values)
    index = (len(data)-1)*q
    lo, hi = math.floor(index), math.ceil(index)
    return data[lo] + (data[hi]-data[lo])*(index-lo)


@dataclass(frozen=True)
class Capacity:
    parameters: int
    weight_bits: int
    layers: int
    kv_heads: int
    head_dim: int
    kv_bytes: int
    context: int
    sequences: int = 1
    tp: int = 1
    gpu_gib: float = 80.0
    usable_fraction: float = .8
    runtime_gib_per_rank: float = 4.0
    weight_overhead_fraction: float = .05
    def estimate(self):
        positive = [self.parameters, self.weight_bits, self.layers, self.kv_heads,
                    self.head_dim, self.kv_bytes, self.context, self.sequences, self.tp]
        if min(positive) <= 0 or not 0 < self.usable_fraction <= 1:
            raise ValueError("invalid capacity inputs")
        if self.gpu_gib <= 0 or self.runtime_gib_per_rank < 0 or self.weight_overhead_fraction < 0:
            raise ValueError("invalid memory budget")
        if self.kv_heads % self.tp:
            raise ValueError("teaching estimator requires KV heads divisible by TP; replication needs another model")
        weights = self.parameters*self.weight_bits/8*(1+self.weight_overhead_fraction)/self.tp
        kv_per_token = 2*self.layers*(self.kv_heads//self.tp)*self.head_dim*self.kv_bytes
        kv = kv_per_token*self.context*self.sequences
        total = weights+kv+self.runtime_gib_per_rank*GiB
        available = self.gpu_gib*GiB*self.usable_fraction
        return {"assumptions": "uniform layer shape; weight and KV-head sharding; no MLA/SSM/EP/PP",
                "weights_gib_per_rank": weights/GiB, "kv_bytes_per_token_per_rank": kv_per_token,
                "kv_gib_per_rank": kv/GiB, "total_gib_per_rank": total/GiB,
                "budget_gib_per_rank": available/GiB, "margin_gib_per_rank": (available-total)/GiB,
                "arithmetic_fits": total <= available}


def ring_allreduce(bytes_per_rank, ranks, bandwidth_gbps, latency_us=5.0):
    if bytes_per_rank < 0 or ranks < 1 or bandwidth_gbps <= 0 or latency_us < 0:
        raise ValueError("invalid ring model inputs")
    if ranks == 1:
        return {"bytes_sent_per_rank": 0.0, "ideal_ms": 0.0}
    volume = 2*(ranks-1)/ranks*bytes_per_rank
    # GB/s is decimal bytes/s, explicitly not Gbit/s.
    ms = 2*(ranks-1)*latency_us/1000 + volume/(bandwidth_gbps*1e9)*1000
    return {"bytes_sent_per_rank": volume, "ideal_ms": ms}


def pd_break_even(prefill_ms, colocated_queue_ms, decode_queue_ms, kv_bytes,
                  bandwidth_gbps, setup_ms=0.0):
    if min(prefill_ms, colocated_queue_ms, decode_queue_ms, kv_bytes, setup_ms) < 0 or bandwidth_gbps <= 0:
        raise ValueError("invalid PD inputs")
    transfer_ms = setup_ms+kv_bytes/(bandwidth_gbps*1e9)*1000
    colocated_ttft = colocated_queue_ms+prefill_ms
    split_ttft = prefill_ms+decode_queue_ms+transfer_ms
    return {"transfer_ms": transfer_ms, "colocated_ttft_ms": colocated_ttft,
            "split_ttft_ms": split_ttft, "split_improves_ttft": split_ttft < colocated_ttft}


def symmetric_quantize(values, bits=8):
    if bits < 2 or bits > 16 or not values or any(not math.isfinite(x) for x in values):
        raise ValueError("invalid quantization inputs")
    qmax = (1 << (bits-1))-1
    scale = max(abs(x) for x in values)/qmax
    if scale == 0:
        return [0]*len(values), 1.0, [0.0]*len(values)
    quant = [max(-qmax, min(qmax, round(x/scale))) for x in values]
    return quant, scale, [q*scale for q in quant]
