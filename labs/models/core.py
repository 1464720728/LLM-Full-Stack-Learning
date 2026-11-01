"""NumPy reference mechanisms. Shapes and deliberate simplifications are explicit.

All arithmetic defaults to float64 for diagnostic accuracy, not performance.
There are no downloads, model API calls, or GPU allocations in this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np


def softmax(x, axis=-1):
    """Stable softmax. An entirely -inf-masked row returns zeros, never NaN."""
    x = np.asarray(x, dtype=np.float64)
    if np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError("softmax accepts finite values and negative infinity only")
    maximum = np.max(x, axis=axis, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    e = np.exp(x - maximum)
    total = e.sum(axis=axis, keepdims=True)
    return np.divide(e, total, out=np.zeros_like(e), where=total > 0)


def cross_entropy(logits, targets):
    """Mean CE and gradient w.r.t. finite (N,V) logits; natural-log units."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2 or targets.shape != (len(logits),) or len(logits) == 0:
        raise ValueError("expected nonempty logits (N,V), targets (N,)")
    if not np.isfinite(logits).all() or np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("invalid logits or target index")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=1))
    loss = np.mean(log_z - shifted[np.arange(len(targets)), targets])
    grad = softmax(logits)
    grad[np.arange(len(targets)), targets] -= 1
    return float(loss), grad / len(targets)


class ByteTokenizer:
    """Reversible UTF-8 byte tokenizer. It is NOT a production BPE tokenizer."""
    vocab_size = 256
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))
    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="strict")


def bpe_train(text: str, merges: int = 4):
    """Tiny deterministic character BPE with no word boundaries/special tokens."""
    tokens = list(text)
    rules = []
    for _ in range(merges):
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        rules.append(pair)
        out, i = [], 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                out.append(tokens[i] + tokens[i + 1]); i += 2
            else:
                out.append(tokens[i]); i += 1
        tokens = out
    return tokens, rules


class Value:
    """Scalar reverse-mode autodiff: graph construction, accumulation, topology."""
    def __init__(self, data, children=(), label=""):
        self.data = float(data); self.grad = 0.0
        self.children = tuple(children); self.label = label
        self._backward: Callable[[], None] = lambda: None
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def backward():
            self.grad += out.grad; other.grad += out.grad
        out._backward = backward
        return out
    __radd__ = __add__
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out
    __rmul__ = __mul__
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), "tanh")
        def backward(): self.grad += (1 - t * t) * out.grad
        out._backward = backward
        return out
    def backward(self):
        order, seen = [], set()
        def visit(node):
            if node in seen: return
            seen.add(node)
            for c in node.children: visit(c)
            order.append(node)
        visit(self)
        for node in order: node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order): node._backward()


def finite_difference(fn, x, index, eps=1e-5):
    x = np.array(x, dtype=float, copy=True)
    old = x[index]
    x[index] = old + eps; plus = fn(x)
    x[index] = old - eps; minus = fn(x)
    return (plus - minus) / (2 * eps)


def _normalization_input(x,eps):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim<1 or x.shape[-1]==0 or not np.isfinite(x).all():
        raise ValueError("normalization needs finite values and a nonempty final axis")
    if isinstance(eps,(bool,np.bool_)) or not np.isscalar(eps) or not np.isfinite(eps) or eps<=0:
        raise ValueError("epsilon must be finite and positive")
    return x


def layer_norm(x, eps=1e-5):
    x = _normalization_input(x,eps)
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


def rms_norm(x, eps=1e-5):
    x = _normalization_input(x,eps)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def silu(x):
    x = np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError("SiLU expects finite values")
    # Clipping changes only vanishing tails in this float64 reference.
    return x / (1 + np.exp(-np.clip(x, -700, 700)))


def swiglu(x, w_gate, w_up, w_down):
    x,w_gate,w_up,w_down=map(lambda a:np.asarray(a,dtype=np.float64),(x,w_gate,w_up,w_down))
    if not all(np.isfinite(a).all() for a in (x,w_gate,w_up,w_down)):
        raise ValueError("SwiGLU expects finite inputs and weights")
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


def sinusoidal(length, dim):
    if dim % 2: raise ValueError("even dimension required")
    angles = np.arange(length)[:, None] / (10000 ** (np.arange(0, dim, 2) / dim))
    out = np.zeros((length, dim)); out[:, 0::2] = np.sin(angles); out[:, 1::2] = np.cos(angles)
    return out


def rope(x, positions, base=10000.0):
    """Interleaved-pair RoPE for (...,T,D); positions is (T,), D even."""
    x = np.asarray(x); positions = np.asarray(positions)
    if x.shape[-1] % 2 or positions.shape != (x.shape[-2],):
        raise ValueError("RoPE expects (...,T,even_D), positions (T,)")
    angles = positions[:, None] / (base ** (np.arange(0, x.shape[-1], 2) / x.shape[-1]))
    c, s = np.cos(angles), np.sin(angles)
    out = np.empty_like(x, dtype=float)
    out[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
    out[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
    return out


def causal_mask(q_len, k_len=None, offset=0):
    k_len = q_len if k_len is None else k_len
    return np.arange(k_len)[None, :] <= (np.arange(q_len) + offset)[:, None]


def attention(q, k, v, mask=None):
    """Scaled softmax attention, (...,Tq,D), (...,Tk,D), (...,Tk,Dv)."""
    q, k, v = map(np.asarray, (q, k, v))
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("incompatible attention dimensions")
    scores = q @ k.swapaxes(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None: scores = np.where(np.asarray(mask, dtype=bool), scores, -np.inf)
    weights = softmax(scores)
    return weights @ v, weights


def split_heads(x, heads):
    t, d = x.shape
    if d % heads: raise ValueError("hidden size must divide heads")
    return x.reshape(t, heads, d // heads).transpose(1, 0, 2)


def merge_heads(x):
    h, t, d = x.shape
    return x.transpose(1, 0, 2).reshape(t, h * d)


def grouped_attention(q, k, v, mask=None):
    """Head axis is 0. Repeat KV only as a readable correctness reference."""
    if q.shape[0] % k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("query head count must be divisible by KV head count")
    repeat = q.shape[0] // k.shape[0]
    return attention(q, np.repeat(k, repeat, axis=0), np.repeat(v, repeat, axis=0), mask)


@dataclass
class DecoderConfig:
    vocab: int = 32
    dim: int = 16
    heads: int = 4
    kv_heads: int = 2
    layers: int = 2
    hidden: int = 32
    seed: int = 7


class TinyDecoder:
    """Complete forward-only pre-RMSNorm GQA+RoPE+SwiGLU causal decoder.

    Random weights: this proves tensor/dataflow correctness, not language skill.
    Training is supplied separately in tiny_lm.py and torch_transformer.py.
    """
    def __init__(self, config=None):
        self.config = config or DecoderConfig(); c = self.config
        if min(c.vocab,c.dim,c.heads,c.kv_heads,c.layers,c.hidden) <= 0:
            raise ValueError("all dimensions must be positive")
        if c.dim % c.heads or c.heads % c.kv_heads or (c.dim // c.heads) % 2:
            raise ValueError("invalid GQA/RoPE head configuration")
        rng = np.random.default_rng(c.seed)
        def w(a,b): return rng.normal(0, 1/math.sqrt(a), (a,b))
        self.embedding = w(c.vocab,c.dim); self.output = w(c.dim,c.vocab)
        kv = c.kv_heads * (c.dim // c.heads)
        self.blocks = [dict(q=w(c.dim,c.dim), k=w(c.dim,kv), v=w(c.dim,kv),
                            o=w(c.dim,c.dim), gate=w(c.dim,c.hidden),
                            up=w(c.dim,c.hidden), down=w(c.hidden,c.dim)) for _ in range(c.layers)]
    def _attn(self, x, block, position, previous=None):
        c = self.config
        q = rope(split_heads(x @ block['q'],c.heads),position)
        k = rope(split_heads(x @ block['k'],c.kv_heads),position)
        v = split_heads(x @ block['v'],c.kv_heads)
        if previous is not None:
            k = np.concatenate([previous[0],k],axis=1)
            v = np.concatenate([previous[1],v],axis=1)
        mask = np.arange(k.shape[1])[None,:] <= position[:,None]
        out,_ = grouped_attention(q,k,v,mask)
        return merge_heads(out) @ block['o'], (k,v)
    def forward(self, tokens):
        tokens = np.asarray(tokens,dtype=int)
        if tokens.ndim != 1 or len(tokens)==0 or np.any(tokens<0) or np.any(tokens>=self.config.vocab):
            raise ValueError("expected a nonempty sequence of valid token ids")
        x=self.embedding[tokens]; pos=np.arange(len(tokens))
        for b in self.blocks:
            out,_ = self._attn(rms_norm(x),b,pos)
            x=x+out; x=x+swiglu(rms_norm(x),b['gate'],b['up'],b['down'])
        return rms_norm(x) @ self.output
    def step(self, token, cache=None):
        if not isinstance(token,(int,np.integer)) or not 0 <= token < self.config.vocab:
            raise ValueError("invalid token")
        if cache is None: cache = [None] * self.config.layers
        if len(cache)!=self.config.layers: raise ValueError("wrong number of cache layers")
        lengths = [0 if state is None else state[0].shape[1] for state in cache]
        if len(set(lengths)) != 1: raise ValueError("cache layer lengths disagree")
        pos=np.array([lengths[0]]); x=self.embedding[[token]]; new=[]
        for b,previous in zip(self.blocks,cache):
            out,state=self._attn(rms_norm(x),b,pos,previous)
            x=x+out; x=x+swiglu(rms_norm(x),b['gate'],b['up'],b['down']); new.append(state)
        return (rms_norm(x)@self.output)[0],new


class OriginalTransformer:
    """One original-style post-LayerNorm encoder and decoder, no dropout.

    Explicit encoder self-attention, causal decoder self-attention and cross
    attention. Training weights/scales/biases/dropout are omitted intentionally.
    """
    def __init__(self, vocab=16,dim=8,heads=2,seed=4):
        if dim%heads or dim%2: raise ValueError("invalid dimensions")
        self.dim=dim; self.heads=heads; r=np.random.default_rng(seed)
        def w(a,b): return r.normal(0,1/math.sqrt(a),(a,b))
        self.e=w(vocab,dim); self.lm=w(dim,vocab)
        self.proj=[tuple(w(dim,dim) for _ in range(4)) for _ in range(3)]
        self.ff=[(w(dim,2*dim),w(2*dim,dim)) for _ in range(2)]
    def _mha(self,qx,kvx,p,mask=None):
        q,k,v=[split_heads(x@w,self.heads) for x,w in zip((qx,kvx,kvx),p[:3])]
        a,_=attention(q,k,v,mask)
        return merge_heads(a)@p[3]
    def forward(self,source,target_input):
        source=np.asarray(source,dtype=int); target_input=np.asarray(target_input,dtype=int)
        if not len(source) or not len(target_input): raise ValueError("nonempty sequences required")
        enc=self.e[source]*math.sqrt(self.dim)+sinusoidal(len(source),self.dim)
        enc=layer_norm(enc+self._mha(enc,enc,self.proj[0]))
        enc=layer_norm(enc+np.maximum(enc@self.ff[0][0],0)@self.ff[0][1])
        dec=self.e[target_input]*math.sqrt(self.dim)+sinusoidal(len(target_input),self.dim)
        dec=layer_norm(dec+self._mha(dec,dec,self.proj[1],causal_mask(len(dec))))
        dec=layer_norm(dec+self._mha(dec,enc,self.proj[2]))
        dec=layer_norm(dec+np.maximum(dec@self.ff[1][0],0)@self.ff[1][1])
        return dec@self.lm


def latent_attention(q, latent, up_k, up_v, mask=None):
    """Exact factorized attention without RoPE: K=C Uk, V=C Uv.

    Keep scaling by original key dimension, not latent rank. This is an algebra
    demonstration related to MLA absorption, NOT a DeepSeek MLA reproduction.
    """
    scores=(q@up_k.T)@latent.T/math.sqrt(q.shape[-1])
    if mask is not None: scores=np.where(mask,scores,-np.inf)
    return (softmax(scores)@latent)@up_v


def moe(x,router,experts,top_k=2,capacity=None):
    """Token-choice top-k with deterministic ties and explicit overflow dropping.

    experts is list of (up,down) ReLU FFNs. Surviving weights are not renormalized
    after overflow. Returned audit makes the chosen semantics inspectable.
    """
    x=np.asarray(x); probs=softmax(x@router); e=len(experts)
    if not 1<=top_k<=e or router.shape[1]!=e: raise ValueError("invalid top_k/experts")
    if capacity is not None and capacity<0: raise ValueError("capacity must be nonnegative")
    chosen=np.argsort(-probs,axis=1,kind='stable')[:,:top_k]
    weights=np.take_along_axis(probs,chosen,axis=1); weights/=weights.sum(1,keepdims=True)
    counts=np.zeros(e,dtype=int); dropped=np.zeros_like(chosen,dtype=bool); out=np.zeros_like(x,dtype=float)
    for t in range(len(x)):
        for slot,expert in enumerate(chosen[t]):
            if capacity is not None and counts[expert]>=capacity:
                dropped[t,slot]=True; continue
            counts[expert]+=1
            up,down=experts[expert]
            out[t]+=weights[t,slot]*(np.maximum(x[t]@up,0)@down)
    return out,dict(chosen=chosen,weights=weights,counts=counts,dropped=dropped,probabilities=probs)


def load_balance_loss(probabilities, chosen):
    """Simple E*sum(mean_prob * assignment_fraction), not a full training recipe."""
    e=probabilities.shape[1]; fractions=np.bincount(chosen.ravel(),minlength=e)/chosen.size
    return float(e*np.dot(probabilities.mean(0),fractions))


def local_global_mask(length,window=3,global_stride=None):
    if window<1: raise ValueError("window must include at least the current token")
    i=np.arange(length)[:,None]; j=np.arange(length)[None,:]
    out=(j<=i)&(j>i-window)
    if global_stride is not None:
        if global_stride<1: raise ValueError("positive stride required")
        out|=(j<=i)&(j%global_stride==0)
    return out


def sparse_attention(q,k,v,mask):
    """Gather each row's allowed keys, avoiding a dense score matrix; CPU loop."""
    q,k,v=map(np.asarray,(q,k,v)); mask=np.asarray(mask,dtype=bool)
    if mask.shape!=(len(q),len(k)): raise ValueError("mask shape mismatch")
    out=np.zeros((len(q),v.shape[1])); visited=0
    for i in range(len(q)):
        idx=np.flatnonzero(mask[i]); visited+=len(idx)
        if len(idx): out[i]=softmax(q[i]@k[idx].T/math.sqrt(q.shape[1]))@v[idx]
    return out,visited


def positive_features(x):
    """ELU(x)+1 without evaluating overflow-prone exp for positive elements."""
    x=np.asarray(x); out=np.empty_like(x,dtype=float); pos=x>=0
    out[pos]=x[pos]+1; out[~pos]=np.exp(x[~pos]); return out


def _sequence_inputs(q,k,v):
    q,k,v=map(lambda x:np.asarray(x,dtype=float),(q,k,v))
    if q.ndim!=2 or k.ndim!=2 or v.ndim!=2 or q.shape!=k.shape or len(q)!=len(v):
        raise ValueError("expected Q/K (T,Dk) with matching shape and V (T,Dv)")
    if len(q)==0 or q.shape[1]==0 or v.shape[1]==0 or not all(np.isfinite(x).all() for x in (q,k,v)):
        raise ValueError("nonempty finite sequences and feature dimensions required")
    return q,k,v


def linear_attention_dense(q,k,v):
    q,k,v=_sequence_inputs(q,k,v)
    q,k=positive_features(q),positive_features(k)
    scores=q@k.T*causal_mask(len(q))
    den=scores.sum(1,keepdims=True)
    return np.divide(scores@v,den,out=np.zeros((len(q),v.shape[1])),where=den>0)


def linear_attention_recurrent(q,k,v,state=None):
    q,k,v=_sequence_inputs(q,k,v)
    q,k=positive_features(q),positive_features(k)
    s,z=(np.zeros((k.shape[1],v.shape[1])),np.zeros(k.shape[1])) if state is None else (np.array(state[0],dtype=float,copy=True),np.array(state[1],dtype=float,copy=True))
    if s.shape!=(k.shape[1],v.shape[1]) or z.shape!=(k.shape[1],) or not np.isfinite(s).all() or not np.isfinite(z).all():
        raise ValueError("state shape or finite-value contract violated")
    out=[]
    for qi,ki,vi in zip(q,k,v):
        s+=np.outer(ki,vi); z+=ki
        denominator=float(qi@z)
        # Match the dense definition exactly: divide any positive denominator;
        # an epsilon clamp would change the function for tiny valid features.
        out.append((qi@s)/denominator if denominator>0 else np.zeros(v.shape[1]))
    return np.asarray(out), (s,z)


def delta_recurrence(q,k,v,beta,decay=None):
    """Minimal delta-memory update; no conv/norm/gating stack, NOT full KDA."""
    q,k,v=_sequence_inputs(q,k,v)
    beta=np.asarray(beta,dtype=float)
    decay=np.ones(len(q)) if decay is None else np.asarray(decay,dtype=float)
    if beta.shape!=(len(q),) or decay.shape!=(len(q),) or not np.isfinite(beta).all() or not np.isfinite(decay).all():
        raise ValueError("beta and decay must be finite vectors of length T")
    s=np.zeros((k.shape[1],v.shape[1])); out=[]
    for qi,ki,vi,b,a in zip(q,k,v,beta,decay):
        s=a*s
        error=vi-ki@s
        s+=b*np.outer(ki,error)
        out.append(qi@s)
    return np.asarray(out),s


def diagonal_ssm(u,a,b,c,initial=None):
    """Discrete diagonal state: h[t]=a*h[t-1]+b*u[t]; y[t]=c dot h[t]."""
    a,b,c=map(np.asarray,(a,b,c)); h=np.zeros_like(a,dtype=float) if initial is None else initial.copy()
    ys=[]
    for ut in u:
        h=a*h+b*ut; ys.append(c@h)
    return np.asarray(ys),h


def ssm_convolution(u,a,b,c):
    kernel=np.array([np.sum(c*(a**i)*b) for i in range(len(u))])
    return np.convolve(u,kernel)[:len(u)]


def selective_ssm(u,rate=-1.0):
    """Scalar exact-ZOH discretization, input-dependent step, pedagogical only."""
    if rate>=0: raise ValueError("negative continuous decay rate required")
    h=0.; ys=[]
    for x in u:
        delta=float(np.logaddexp(0,x))
        a=math.exp(delta*rate); b=math.expm1(delta*rate)/rate
        h=a*h+b*x; ys.append(h)
    return np.asarray(ys)


def attention_residual(sources,query,eps=1e-6):
    """Full depth-attention aggregation: sources (L,T,D), learned query (D,).

    RMS-normalize sources for keys; values retain original amplitudes. Softmax
    over L, independently per token. A primitive, not the official block system.
    """
    sources=np.asarray(sources); query=np.asarray(query)
    if sources.ndim!=3 or query.shape!=(sources.shape[-1],): raise ValueError("expected (L,T,D), (D,)")
    logits=np.einsum('ltd,d->lt',rms_norm(sources,eps),query)
    weights=softmax(logits,axis=0)
    return np.einsum('lt,ltd->td',weights,sources),weights


def sinkhorn(logits,iterations=30):
    """Log-domain normalization towards doubly-stochastic; toy mHC aid.

    Normalization before exponentiation avoids all-zero underflow columns.
    Finite iterations only approximately enforce both axes simultaneously.
    """
    logits=np.asarray(logits,dtype=np.float64)
    if logits.ndim!=2 or logits.shape[0]!=logits.shape[1] or logits.shape[0]==0 or not np.isfinite(logits).all():
        raise ValueError("nonempty finite square matrix required")
    if isinstance(iterations,(bool,np.bool_)) or not isinstance(iterations,(int,np.integer)) or iterations<1:
        raise ValueError("positive integer iterations required")
    log_x=logits-logits.max()
    for _ in range(iterations):
        for axis in (1,0):
            maximum=log_x.max(axis=axis,keepdims=True)
            log_sum=maximum+np.log(np.exp(log_x-maximum).sum(axis=axis,keepdims=True))
            log_x-=log_sum
    return np.exp(log_x)


def symmetric_quantize(x,bits=8,axis=None):
    if not 2<=bits<=8: raise ValueError("2..8 bits in this reference")
    x=np.asarray(x,dtype=float); limit=2**(bits-1)-1
    maximum=np.max(np.abs(x),axis=axis,keepdims=True)
    scale=np.where(maximum==0,1.0,maximum/limit)
    q=np.clip(np.rint(x/scale),-limit,limit).astype(np.int8)
    return q,scale,q.astype(float)*scale


def sample_token(logits,rng,temperature=1.,top_k=None,top_p=1.):
    logits=np.asarray(logits,dtype=float)
    if logits.ndim!=1 or not np.isfinite(logits).all(): raise ValueError("finite 1D logits required")
    if temperature<0 or not 0<top_p<=1: raise ValueError("bad temperature/top_p")
    if top_k is not None and not 1<=top_k<=len(logits): raise ValueError("bad top_k")
    if temperature==0: return int(np.argmax(logits))
    scores=logits/temperature
    order=np.argsort(-scores,kind='stable')
    if top_k is not None: order=order[:top_k]
    probs=softmax(scores[order]); cumulative=np.cumsum(probs)
    count=min(len(order),int(np.searchsorted(cumulative,top_p,side='left'))+1)
    probs=probs[:count]; probs/=probs.sum()
    return int(rng.choice(order[:count],p=probs))


def lora(x,w,a,b,alpha=1.):
    return x@w+(alpha/a.shape[1])*(x@a)@b


def dpo_loss(policy_chosen,policy_rejected,ref_chosen,ref_rejected,beta=0.1):
    margin=beta*((policy_chosen-policy_rejected)-(ref_chosen-ref_rejected))
    return float(np.logaddexp(0,-margin))


def patchify(image,patch):
    image=np.asarray(image); h,w,c=image.shape
    if patch<1 or h%patch or w%patch: raise ValueError("image dimensions must divide patch size")
    return image.reshape(h//patch,patch,w//patch,patch,c).transpose(0,2,1,3,4).reshape(-1,patch*patch*c)
