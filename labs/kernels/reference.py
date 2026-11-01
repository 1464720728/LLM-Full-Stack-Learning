"""Readable NumPy reference algorithms. These are not performance kernels.

All indexing uses elements, except functions explicitly suffixed *_bytes.
Numeric references use float64 unless investigating low precision.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable
import numpy as np


def offset(index, shape, strides, base=0):
    if len(index) != len(shape) or len(shape) != len(strides):
        raise ValueError('rank mismatch')
    if any(i < 0 or i >= n for i, n in zip(index, shape)):
        raise IndexError('index outside logical tensor')
    return base + sum(i * s for i, s in zip(index, strides))


def c_strides(shape):
    if any(n < 0 for n in shape):
        raise ValueError('negative dimension')
    result, stride = [], 1
    for n in reversed(shape):
        result.append(stride)
        stride *= n
    return tuple(reversed(result))


def sectors(addresses_bytes: Iterable[int], width_bytes=4, sector_bytes=32):
    """Idealized touched sectors, not a complete hardware transaction predictor."""
    if width_bytes <= 0 or sector_bytes <= 0:
        raise ValueError('widths must be positive')
    touched = set()
    for addr in addresses_bytes:
        if addr < 0:
            raise ValueError('negative address')
        touched.update(range(addr // sector_bytes,
                             (addr + width_bytes - 1) // sector_bytes + 1))
    return len(touched)


def bank_conflicts(word_addresses, banks=32):
    """32-bit word model: same word is broadcast; distinct words contend."""
    if banks < 1:
        raise ValueError('banks must be positive')
    groups = [set() for _ in range(banks)]
    for address in word_addresses:
        groups[address % banks].add(address)
    return max(map(len, groups), default=0)


def launch_indices(n, block=128, grid=None):
    if n < 0 or block < 1:
        raise ValueError('invalid launch shape')
    grid = max(1, math.ceil(n / block)) if grid is None else grid
    if grid < 1:
        raise ValueError('grid must be positive')
    return [[i for i in range(b * block + t, n, grid * block)]
            for b in range(grid) for t in range(block)]


def tree_sum(x):
    a = np.asarray(x).copy().ravel()
    if a.size == 0:
        return 0.0
    while a.size > 1:
        if a.size % 2:
            a = np.concatenate((a, np.zeros(1, dtype=a.dtype)))
        a = a[::2] + a[1::2]
    return a[0]


def stable_softmax(x, mask=None, axis=-1):
    a = np.asarray(x, dtype=np.float64)
    if a.shape[axis] == 0:
        raise ValueError('empty reduction dimension')
    valid = np.ones(a.shape, dtype=bool) if mask is None else np.broadcast_to(np.asarray(mask, dtype=bool), a.shape)
    if not np.all(np.isfinite(a[valid])):
        raise ValueError('unmasked inputs must be finite')
    a = np.where(valid, a, -np.inf)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.where(valid, np.exp(a - m), 0.0)
    d = e.sum(axis=axis, keepdims=True)
    return np.divide(e, d, out=np.zeros_like(e), where=d > 0)


def softmax_backward(p, grad):
    p, grad = np.asarray(p), np.asarray(grad)
    if p.shape != grad.shape:
        raise ValueError('gradient shape mismatch')
    return p * (grad - (p * grad).sum(axis=-1, keepdims=True))


def rmsnorm(x, weight, eps=1e-5):
    x, weight = np.asarray(x, dtype=np.float64), np.asarray(weight, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] == 0 or weight.shape != (x.shape[-1],) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('invalid RMSNorm arguments')
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * inv * weight


def rmsnorm_backward(x, weight, grad, eps=1e-5):
    x, weight, grad = map(lambda a: np.asarray(a, dtype=np.float64), (x, weight, grad))
    if x.shape != grad.shape or weight.shape != (x.shape[-1],):
        raise ValueError('invalid shapes')
    inv = 1 / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)
    u = grad * weight
    dx = inv * u - x * inv**3 * np.mean(u*x, axis=-1, keepdims=True)
    axes = tuple(range(x.ndim - 1))
    dw = np.sum(grad*x*inv, axis=axes) if axes else grad*x*inv
    return dx, dw


def layernorm(x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered*centered).mean(axis=-1, keepdims=True) + eps)


def tiled_matmul(a, b, tile=4):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or tile < 1:
        raise ValueError('invalid GEMM')
    m, k = a.shape
    n = b.shape[1]
    out = np.zeros((m,n), dtype=np.result_type(a.dtype, b.dtype, np.float64))
    for i in range(0,m,tile):
        for j in range(0,n,tile):
            for p in range(0,k,tile):
                out[i:i+tile,j:j+tile] += a[i:i+tile,p:p+tile] @ b[p:p+tile,j:j+tile]
    return out


def dense_attention(q, k, v, causal=False):
    q, k, v = map(lambda a: np.asarray(a,dtype=np.float64), (q,k,v))
    _attention_shapes(q,k,v,causal)
    mask = np.arange(k.shape[0])[None,:] <= np.arange(q.shape[0])[:,None] if causal else None
    return stable_softmax(q @ k.T / np.sqrt(q.shape[1]), mask) @ v


def _attention_shapes(q,k,v,causal):
    if any(a.ndim != 2 for a in (q,k,v)) or q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError('attention shapes')
    if q.shape[1] == 0 or k.shape[0] == 0:
        raise ValueError('empty attention dimensions')
    if causal and q.shape[0] != k.shape[0]:
        raise ValueError('teaching causal convention requires equal query/key lengths')


def online_attention(q,k,v,block=4,causal=False,return_trace=False):
    """Stream key blocks, retaining m, l, and unnormalized weighted numerator.

    Self-attention causal convention: key index <= query index. No dropout.
    This CPU implementation verifies the algorithm, not HBM traffic or speed.
    """
    q,k,v = map(lambda a:np.asarray(a,dtype=np.float64),(q,k,v))
    _attention_shapes(q,k,v,causal)
    if block < 1:
        raise ValueError('block must be positive')
    rows, dim = q.shape
    m = np.full(rows, -np.inf)
    l = np.zeros(rows)
    numerator = np.zeros((rows,v.shape[1]))
    trace=[]
    for start in range(0,k.shape[0],block):
        scores=q @ k[start:start+block].T / np.sqrt(dim)
        valid=np.ones(scores.shape,dtype=bool)
        if causal:
            valid=np.arange(start,min(start+block,k.shape[0]))[None,:] <= np.arange(rows)[:,None]
        scores=np.where(valid,scores,-np.inf)
        block_m=np.max(scores,axis=1)
        new_m=np.maximum(m,block_m)
        safe_m=np.where(np.isfinite(new_m),new_m,0.0)
        alpha=np.exp(m-safe_m)
        p=np.where(valid,np.exp(scores-safe_m[:,None]),0.0)
        numerator=numerator*alpha[:,None]+p @ v[start:start+block]
        l=l*alpha+p.sum(axis=1)
        m=new_m
        if return_trace:
            trace.append({'start':start,'m':m.copy(),'l':l.copy(),'numerator':numerator.copy()})
    out=np.divide(numerator,l[:,None],out=np.zeros_like(numerator),where=l[:,None]>0)
    return (out,trace) if return_trace else out


def quantize_int8_rows(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not np.all(np.isfinite(x)):
        raise ValueError('finite matrix required')
    peak=np.max(np.abs(x),axis=1,keepdims=True)
    scale=np.where(peak>0,peak/127,1.0)
    q=np.clip(np.rint(x/scale),-127,127).astype(np.int8)
    return q,scale


def int8_matmul(a,b):
    """A scales per row; B scales per column by quantizing B.T."""
    qa,sa=quantize_int8_rows(a)
    qbt,sbt=quantize_int8_rows(np.asarray(b).T)
    # Cast BEFORE matmul: int8 dot could overflow in a narrow output type.
    accum=qa.astype(np.int32) @ qbt.astype(np.int32).T
    return accum*sa*sbt.T


def pack_int4(x):
    x=np.asarray(x)
    if not np.issubdtype(x.dtype,np.integer) or np.any(x < -8) or np.any(x > 7):
        raise ValueError('signed int4 range is [-8,7]')
    flat=x.ravel().astype(np.int8)
    codes=(flat.astype(np.int16)&15).astype(np.uint8)
    if codes.size%2:
        codes=np.pad(codes,(0,1))
    return codes[::2] | (codes[1::2] << 4),flat.size


def unpack_int4(packed,count):
    packed=np.asarray(packed,dtype=np.uint8).ravel()
    if count<0 or count>packed.size*2:
        raise ValueError('invalid logical count')
    codes=np.empty(packed.size*2,dtype=np.int8)
    codes[::2]=(packed & 15).astype(np.int8)
    codes[1::2]=(packed >> 4).astype(np.int8)
    return np.where(codes[:count]>=8,codes[:count]-16,codes[:count]).astype(np.int8)


def decode_e2m1(codes):
    """Finite E2M1 magnitudes, with signed zero; scaling is a SEPARATE contract."""
    codes=np.asarray(codes,dtype=np.int64)
    if np.any(codes<0) or np.any(codes>15):
        raise ValueError('four-bit codes only')
    values=np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float64)
    return values[codes & 7]*np.where(codes & 8,-1.,1.)


def grouped_experts(x,expert_ids,weights,gates=None):
    """One or several assigned experts/token; stable packing and scatter-add."""
    x=np.asarray(x,dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64)
    ids=np.asarray(expert_ids)
    if ids.ndim==1:ids=ids[:,None]
    if x.ndim!=2 or ids.ndim!=2 or weights.ndim!=3:
        raise ValueError('matrix inputs, rank-2 assignments, and rank-3 weights required')
    if ids.shape[0]!=x.shape[0] or x.shape[1]!=weights.shape[1]:
        raise ValueError('routing shape mismatch')
    if not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=weights.shape[0]):
        raise ValueError('invalid expert ID')
    gates=np.ones(ids.shape) if gates is None else np.asarray(gates)
    if gates.shape!=ids.shape:raise ValueError('gate shape mismatch')
    out=np.zeros((x.shape[0],weights.shape[2]))
    counts=[]
    for expert,w in enumerate(weights):
        token,slot=np.nonzero(ids==expert)
        counts.append(token.size)
        result=(x[token] @ w)*gates[token,slot,None]
        np.add.at(out,token,result)
    return out,np.array(counts)


@dataclass(frozen=True)
class Roofline:
    flops: float
    bytes: float
    peak_flops_per_s: float
    bandwidth_bytes_per_s: float
    def __post_init__(self):
        if self.flops<0 or self.bytes<0 or self.peak_flops_per_s<=0 or self.bandwidth_bytes_per_s<=0:
            raise ValueError('nonnegative work and positive capacities required')
    @property
    def arithmetic_intensity(self):
        return self.flops/self.bytes if self.bytes else math.inf
    @property
    def lower_bound_s(self):
        return max(self.flops/self.peak_flops_per_s,self.bytes/self.bandwidth_bytes_per_s)
    @property
    def bottleneck(self):
        return 'compute' if self.flops/self.peak_flops_per_s>self.bytes/self.bandwidth_bytes_per_s else 'memory'


def pipeline_time(tiles,load,compute):
    if tiles<0 or load<0 or compute<0:raise ValueError('negative work')
    return 0 if tiles==0 else load+compute+(tiles-1)*max(load,compute)


def ring_allreduce_seconds(ranks,bytes_per_rank,latency_s,bandwidth_bytes_s):
    if ranks<1 or bytes_per_rank<0 or latency_s<0 or bandwidth_bytes_s<=0:
        raise ValueError('invalid communication model')
    return 2*(ranks-1)*latency_s+2*(ranks-1)/ranks*bytes_per_rank/bandwidth_bytes_s


def finite_difference(fn,x,eps=1e-6):
    if eps<=0:raise ValueError('positive finite-difference step required')
    x=np.asarray(x,dtype=np.float64).copy()
    grad=np.empty_like(x)
    for idx in np.ndindex(x.shape):
        old=x[idx]
        x[idx]=old+eps;plus=fn(x)
        x[idx]=old-eps;minus=fn(x)
        x[idx]=old
        grad[idx]=(plus-minus)/(2*eps)
    return grad
