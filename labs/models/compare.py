"""Project 2: compare mechanism equivalence, state size and computation counts.

No tokens/s or GPU claims: these are exact element counts and CPU correctness.
"""
import json
import numpy as np
from .core import (attention,causal_mask,local_global_mask,sparse_attention,
                  linear_attention_dense,linear_attention_recurrent,diagonal_ssm,
                  ssm_convolution,TinyDecoder)

def compare(length=24,dim=8,seed=17):
    if not 1<=length<=2048: raise ValueError('length outside bounded teaching range')
    r=np.random.default_rng(seed); q=r.normal(size=(length,dim)); k=r.normal(size=q.shape); v=r.normal(size=q.shape)
    mask=local_global_mask(length,window=4,global_stride=8)
    dense,_=attention(q,k,v,mask); sparse,visits=sparse_attention(q,k,v,mask)
    lin=linear_attention_dense(q,k,v); rec,state=linear_attention_recurrent(q,k,v)
    u=r.normal(size=length); a=np.array([.4,.9]); b=np.array([1.,.5]); c=np.array([1.,-1.])
    ys,_=diagonal_ssm(u,a,b,c)
    model=TinyDecoder(); tokens=r.integers(0,model.config.vocab,size=length)
    full=model.forward(tokens); cache=None; steps=[]
    for token in tokens:
        logits,cache=model.step(int(token),cache); steps.append(logits)
    return {
        'length':length,'head_dimension':dim,
        'max_error_sparse_vs_same_dense_mask':float(np.max(np.abs(dense-sparse))),
        'max_error_linear_parallel_vs_recurrent':float(np.max(np.abs(lin-rec))),
        'max_error_ssm_recurrence_vs_convolution':float(np.max(np.abs(ys-ssm_convolution(u,a,b,c)))),
        'max_error_decoder_full_vs_cache':float(np.max(np.abs(full-np.array(steps)))),
        'dense_causal_score_pairs':length*(length+1)//2,'sparse_score_pairs':visits,
        'one_head_dense_kv_state_elements':2*length*dim,
        'one_head_kernel_linear_state_elements':int(state[0].size+state[1].size),
        'note':'Different attention families are different functions; only equivalent formulations above are compared.'}

if __name__=='__main__': print(json.dumps(compare(),indent=2))
