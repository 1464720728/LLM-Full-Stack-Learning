"""Actual NumPy K/V pages with metadata COW and reference paged attention.

This is a synchronous CPU correctness reference: one attention layer, ordinary
full causal history, no RoPE, no GPU kernels. Physical arrays stay allocated
when metadata pages are freed; valid lengths govern what may be read.
"""
import argparse
import json
import numpy as np
from .kv import PagePool,PrefixCache
from .backend import TinyAttentionBackend


class TensorKV:
    def __init__(self,pages=16,block_size=4,heads=1,dim=8,dtype=np.float64):
        if heads<1 or dim<1:raise ValueError('positive heads and dimension required')
        self.pool=PagePool(pages,block_size)
        self.keys=np.zeros((pages,block_size,heads,dim),dtype=dtype)
        self.values=np.zeros_like(self.keys)
        self.heads,self.dim=heads,dim
    def create(self,rid):self.pool.create(rid)
    def fork(self,parent,child):self.pool.fork(parent,child)
    def release(self,rid):self.pool.release(rid)
    def append(self,rid,tokens,k,v):
        k=np.asarray(k,dtype=self.keys.dtype);v=np.asarray(v,dtype=self.values.dtype)
        expected=(len(tokens),self.heads,self.dim)
        if k.shape!=expected or v.shape!=expected:raise ValueError(f'K/V shape must be {expected}')
        if not np.isfinite(k).all() or not np.isfinite(v).all():raise ValueError('finite K/V required')
        table=self.pool.tables[rid];old_size=len(self.pool.tokens(rid))
        old_tail=table[-1] if table else None
        valid_tail=old_size%self.pool.block_size
        self.pool.append(rid,list(tokens))
        # The metadata operation may have split a shared partial tail.
        if old_tail is not None and valid_tail:
            new_tail=self.pool.tables[rid][old_size//self.pool.block_size]
            if new_tail!=old_tail:
                self.keys[new_tail,:valid_tail]=self.keys[old_tail,:valid_tail]
                self.values[new_tail,:valid_tail]=self.values[old_tail,:valid_tail]
        for offset in range(len(tokens)):
            position=old_size+offset;logical,slot=divmod(position,self.pool.block_size)
            bid=self.pool.tables[rid][logical]
            self.keys[bid,slot]=k[offset];self.values[bid,slot]=v[offset]
    def blocks(self,rid):
        for bid in self.pool.tables[rid]:
            count=len(self.pool.pages[bid].data)
            if count:yield self.keys[bid,:count],self.values[bid,:count]
    def gather(self,rid):
        blocks=list(self.blocks(rid))
        if not blocks:raise ValueError('empty KV history')
        return np.concatenate([x[0] for x in blocks]),np.concatenate([x[1] for x in blocks])
    def _query(self,q):
        q=np.asarray(q,dtype=self.keys.dtype)
        if q.shape!=(self.heads,self.dim) or not np.isfinite(q).all():raise ValueError('invalid query')
        return q
    def attention(self,rid,q):
        q=self._query(q);k,v=self.gather(rid)
        scores=np.einsum('hd,thd->ht',q,k)/self.dim**.5
        weights=np.exp(scores-scores.max(axis=1,keepdims=True))
        weights/=weights.sum(axis=1,keepdims=True)
        return np.einsum('ht,thd->hd',weights,v)
    def online_attention(self,rid,q):
        q=self._query(q)
        maximum=np.full(self.heads,-np.inf,dtype=self.keys.dtype)
        denominator=np.zeros(self.heads,dtype=self.keys.dtype)
        numerator=np.zeros((self.heads,self.dim),dtype=self.keys.dtype)
        seen=False
        for k,v in self.blocks(rid):
            seen=True
            scores=np.einsum('hd,thd->ht',q,k)/self.dim**.5
            new_max=np.maximum(maximum,scores.max(axis=1))
            alpha=np.exp(maximum-new_max)
            p=np.exp(scores-new_max[:,None])
            numerator=numerator*alpha[:,None]+np.einsum('ht,thd->hd',p,v)
            denominator=denominator*alpha+p.sum(axis=1)
            maximum=new_max
        if not seen:raise ValueError('empty KV history')
        return numerator/denominator[:,None]


class PagedTinyDecoder:
    """Same one-layer math as TinyAttentionBackend, but K/V are really reused."""
    def __init__(self,pages=32,block_size=4,seed=11):
        self.model=TinyAttentionBackend(seed=seed)
        self.kv=TensorKV(pages,block_size,1,self.model.embedding.shape[1])
    def create(self,rid):self.kv.create(rid)
    def fork(self,parent,child):self.kv.fork(parent,child)
    def append(self,rid,token):
        if type(token) is not int or not 0<=token<self.model.vocab_size:raise ValueError('invalid token')
        x=self.model.embedding[token]
        self.kv.append(rid,[token],(x@self.model.k)[None,None,:],(x@self.model.v)[None,None,:])
        value=self.kv.online_attention(rid,(x@self.model.q)[None,:])[0]
        return (value+x)@self.model.out


def demo():
    decoder=PagedTinyDecoder(block_size=4);decoder.create('A')
    prefix=[1,2,3];errors=[]
    for i,token in enumerate(prefix):
        actual=decoder.append('A',token);expected=decoder.model.logits(prefix[:i+1])
        errors.append(float(np.max(np.abs(actual-expected))))
    decoder.fork('A','B')
    a=decoder.append('A',4);b=decoder.append('B',9)
    errors.extend([float(np.max(np.abs(a-decoder.model.logits(prefix+[4])))),
                   float(np.max(np.abs(b-decoder.model.logits(prefix+[9]))))])
    result={'kind':'actual_NumPy_KV_CPU_reference_not_GPU_kernel','max_abs_errors':errors,
            'A_tokens':decoder.kv.pool.tokens('A'),'B_tokens':decoder.kv.pool.tokens('B'),
            'tables':decoder.kv.pool.tables.copy()}
    decoder.kv.pool.check()
    decoder.kv.release('A');decoder.kv.release('B');decoder.kv.pool.check()
    result['all_pages_returned']=len(decoder.kv.pool.free)==len(decoder.kv.pool.pages)
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(demo(),indent=2))
if __name__=='__main__':main()
