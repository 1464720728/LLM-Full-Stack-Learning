"""Optional Triton kernels. No import until explicitly requested on a CUDA host.

Teaching contract: float32 contiguous tensors, no gradients, bounded shapes.
These kernels emphasize correctness and expose algorithmic structure, not SOTA.
"""
import math

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit('SKIP: install a mutually compatible CUDA PyTorch and Triton in an isolated environment; GPU kernels were not run') from exc

@triton.jit
def _add(A,B,C,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(A+i,i<N,other=0);b=tl.load(B+i,i<N,other=0)
    tl.store(C+i,a+b,i<N)

@triton.jit
def _row_softmax(X,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK)
    x=tl.load(X+row*N+col,col<N,other=-float('inf')).to(tl.float32)
    p=tl.exp(x-tl.max(x,0));p=p/tl.sum(p,0)
    tl.store(Y+row*N+col,p,col<N)

@triton.jit
def _rmsnorm(X,W,Y,N:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK)
    x=tl.load(X+row*N+col,col<N,other=0).to(tl.float32)
    w=tl.load(W+col,col<N,other=0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/N+EPS)
    tl.store(Y+row*N+col,x*inv*w,col<N)

@triton.jit
def _partial_sum(X,P,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<N,other=0).to(tl.float32)
    tl.store(P+tl.program_id(0),tl.sum(x,0))

@triton.jit
def _matmul(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    accum=tl.full((BM,BN),0,tl.float32)
    for start in range(0,tl.cdiv(K,BK)):
        k=start*BK+kk
        a=tl.load(A+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        b=tl.load(B+k[:,None]*N+cols[None,:],(k[:,None]<K)&(cols[None,:]<N),other=0)
        accum += tl.dot(a,b,input_precision='ieee')
    tl.store(C+rows[:,None]*N+cols[None,:],accum,(rows[:,None]<M)&(cols[None,:]<N))

@triton.jit
def _attention(Q,K,V,O,N:tl.constexpr,D:tl.constexpr,DV:tl.constexpr,
               CAUSAL:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,BV:tl.constexpr):
    # One query per program: transparent streaming reference, not FA throughput.
    row=tl.program_id(0);d=tl.arange(0,BD);dv=tl.arange(0,BV)
    q=tl.load(Q+row*D+d,d<D,other=0).to(tl.float32)
    maximum=tl.full((),-float('inf'),tl.float32)
    denom=tl.full((),0,tl.float32)
    numerator=tl.full((BV,),0,tl.float32)
    for block in range(0,tl.cdiv(N,BK)):
        token=block*BK+tl.arange(0,BK)
        keys=tl.load(K+token[:,None]*D+d[None,:],(token[:,None]<N)&(d[None,:]<D),other=0).to(tl.float32)
        score=tl.sum(keys*q[None,:],1)*tl.rsqrt(float(D))
        valid=token<N
        if CAUSAL: valid=valid & (token<=row)
        score=tl.where(valid,score,-float('inf'))
        new_max=tl.maximum(maximum,tl.max(score,0))
        # Every self-attention query sees key zero in block zero, so finite here.
        alpha=tl.exp(maximum-new_max)
        p=tl.exp(score-new_max)
        values=tl.load(V+token[:,None]*DV+dv[None,:],(token[:,None]<N)&(dv[None,:]<DV),other=0).to(tl.float32)
        numerator=numerator*alpha+tl.sum(p[:,None]*values,0)
        denom=denom*alpha+tl.sum(p,0)
        maximum=new_max
    tl.store(O+row*DV+dv,numerator/denom,dv<DV)


def _check(*tensors):
    for t in tensors:
        if not t.is_cuda or t.dtype!=torch.float32 or not t.is_contiguous():
            raise ValueError('teaching kernels require contiguous CUDA float32 tensors')
    if len({t.device for t in tensors})!=1:raise ValueError('same device required')


def add(a,b):
    _check(a,b)
    if a.shape!=b.shape:raise ValueError('equal shapes required')
    out=torch.empty_like(a)
    if a.numel():_add[(triton.cdiv(a.numel(),256),)](a,b,out,a.numel(),256)
    return out


def softmax(x):
    _check(x)
    if x.ndim!=2 or not 1<=x.shape[1]<=4096:raise ValueError('matrix with 1..4096 columns required')
    out=torch.empty_like(x)
    if x.shape[0]:_row_softmax[(x.shape[0],)](x,out,x.shape[1],triton.next_power_of_2(x.shape[1]))
    return out


def rmsnorm(x,w,eps=1e-5):
    _check(x,w)
    if x.ndim!=2 or w.shape!=(x.shape[1],) or not 1<=x.shape[1]<=4096 or not math.isfinite(eps) or eps<=0:raise ValueError('RMSNorm contract')
    out=torch.empty_like(x)
    if x.shape[0]:_rmsnorm[(x.shape[0],)](x,w,out,x.shape[1],eps,triton.next_power_of_2(x.shape[1]))
    return out


def reduce_sum(x):
    _check(x)
    if not x.numel():return torch.zeros((),device=x.device,dtype=x.dtype)
    partial=torch.empty((triton.cdiv(x.numel(),256),),device=x.device,dtype=x.dtype)
    _partial_sum[(partial.numel(),)](x,partial,x.numel(),256)
    # Framework second stage is intentional; not falsely presented as all fused.
    return partial.sum()


def matmul(a,b):
    _check(a,b)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or min(a.shape+b.shape)<=0:raise ValueError('nonempty matrices required')
    m,k=a.shape;n=b.shape[1];out=torch.empty((m,n),device=a.device,dtype=a.dtype)
    _matmul[(triton.cdiv(m,16),triton.cdiv(n,16))](a,b,out,m,n,k,16,16,32)
    return out


def attention(q,k,v,causal=False):
    _check(q,k,v)
    if q.ndim!=2 or k.shape!=q.shape or v.ndim!=2 or v.shape[0]!=q.shape[0]:raise ValueError('self-attention shape contract')
    n,d=q.shape;dv=v.shape[1]
    if not 1<=n<=512 or not 1<=d<=64 or not 1<=dv<=64:raise ValueError('bounded lab: N<=512, D,DV<=64')
    out=torch.empty((n,dv),device=q.device,dtype=q.dtype)
    _attention[(n,)](q,k,v,out,n,d,dv,causal,32,triton.next_power_of_2(d),triton.next_power_of_2(dv))
    return out
