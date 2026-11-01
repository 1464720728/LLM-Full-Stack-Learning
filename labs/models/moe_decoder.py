"""Incremental structural project: replace dense FFNs with routed expert FFNs.

Forward-only random-weight model; no capacity dropping, no shared expert,
no production EP, and no claim of matching any named model checkpoint.
"""
import json
import math
import numpy as np
from .core import TinyDecoder,DecoderConfig,rms_norm,moe

class TinyMoEDecoder(TinyDecoder):
    def __init__(self,config=None,experts=3,top_k=2):
        super().__init__(config)
        if not 1<=top_k<=experts: raise ValueError('invalid expert selection')
        self.top_k=top_k; r=np.random.default_rng(self.config.seed+100)
        c=self.config; self.expert_blocks=[]
        def w(a,b): return r.normal(0,1/math.sqrt(a),(a,b))
        for block in self.blocks:
            # Dense FFN parameters are no longer part of this model.
            del block['gate']; del block['up']; del block['down']
            self.expert_blocks.append({'router':w(c.dim,experts),'experts':[(w(c.dim,c.hidden),w(c.hidden,c.dim)) for _ in range(experts)]})
    def feed_forward(self,x,index):
        block=self.expert_blocks[index]
        return moe(x,block['router'],block['experts'],self.top_k)
    def forward(self,tokens):
        tokens=np.asarray(tokens,dtype=int)
        if tokens.ndim!=1 or len(tokens)==0 or np.any(tokens<0) or np.any(tokens>=self.config.vocab): raise ValueError('valid nonempty ids required')
        x=self.embedding[tokens]; pos=np.arange(len(tokens))
        for i,b in enumerate(self.blocks):
            delta,_=self._attn(rms_norm(x),b,pos); x=x+delta
            delta,_=self.feed_forward(rms_norm(x),i); x=x+delta
        return rms_norm(x)@self.output
    def step(self,token,cache=None):
        if not isinstance(token,(int,np.integer)) or not 0<=token<self.config.vocab: raise ValueError('invalid token')
        if cache is None: cache=[None]*self.config.layers
        if len(cache)!=self.config.layers: raise ValueError('wrong cache layers')
        lengths=[0 if s is None else s[0].shape[1] for s in cache]
        if len(set(lengths))!=1: raise ValueError('cache lengths disagree')
        x=self.embedding[[token]]; pos=np.array([lengths[0]]); new=[]
        for i,(b,state) in enumerate(zip(self.blocks,cache)):
            delta,state=self._attn(rms_norm(x),b,pos,state); x=x+delta
            delta,_=self.feed_forward(rms_norm(x),i); x=x+delta; new.append(state)
        return (rms_norm(x)@self.output)[0],new

def main():
    model=TinyMoEDecoder(); ids=[1,2,3,4]; full=model.forward(ids); out=[]; state=None
    for token in ids:
        logits,state=model.step(token,state); out.append(logits)
    print(json.dumps({'variant':'GQA/RoPE + token-choice MoE ReLU FFN','experts':3,'top_k':2,'full_vs_step_max_error':float(np.max(abs(full-out))),'logits_shape':list(full.shape),'trained':False},indent=2))

if __name__=='__main__': main()
