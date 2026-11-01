"""A deterministic 2-token, 4-channel decoder with every intermediate exposed."""
import argparse
import json
import numpy as np
from .core import (TinyDecoder,DecoderConfig,rms_norm,split_heads,rope,
                   grouped_attention,merge_heads,silu,softmax)


def fixture(active_ffn=False):
    m=TinyDecoder(DecoderConfig(vocab=5,dim=4,heads=2,kv_heads=1,layers=1,hidden=6,seed=7))
    m.embedding=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,1.,0.,0.],[0.,0.,1.,1.],[0.,0.,0.,0.]])
    m.output=np.concatenate([np.eye(4),np.array([[1.],[1.],[-1.],[-1.]])],axis=1)
    b=m.blocks[0]; b['q']=np.eye(4); b['o']=np.eye(4)
    b['k']=np.array([[1.,0.],[0.,1.],[0.,0.],[0.,0.]])
    b['v']=b['k'].copy(); b['gate']=np.zeros((4,6)); b['up']=np.zeros((4,6)); b['down']=np.zeros((6,4))
    if active_ffn:
        b['gate'][:,:4]=np.eye(4); b['up'][:,:4]=np.eye(4); b['down'][:4,:]=.1*np.eye(4)
    return m


def trace(active_ffn=False):
    m=fixture(active_ffn); b=m.blocks[0]; ids=np.array([0,1]); stages={}
    def save(name,value): stages[name]=np.asarray(value).copy(); return value
    x=save('01_embedding',m.embedding[ids]); n=save('02_rmsnorm',rms_norm(x))
    q0=save('03_q_projection',n@b['q']); k0=save('04_k_projection',n@b['k']); v0=save('05_v_projection',n@b['v'])
    q=save('06_q_heads',split_heads(q0,2)); k=save('07_k_heads',split_heads(k0,1)); v=save('08_v_heads',split_heads(v0,1))
    q=save('09_q_rope',rope(q,[0,1])); k=save('10_k_rope',rope(k,[0,1]))
    expanded_k=save('11_gqa_k_reference_repeat',np.repeat(k,2,axis=0))
    save('12_scores_before_mask',q@expanded_k.swapaxes(-1,-2)/np.sqrt(2))
    mask=save('13_causal_mask',np.array([[True,False],[True,True]]))
    out,prob=grouped_attention(q,k,v,mask); save('14_probability',prob); save('15_head_outputs',out)
    mixed=save('16_merged_output',merge_heads(out)@b['o']); x=save('17_attention_residual',x+mixed)
    fn=save('18_ffn_rmsnorm',rms_norm(x)); gate=save('19_gate_projection',fn@b['gate']); up=save('20_up_projection',fn@b['up'])
    product=save('21_silu_times_up',silu(gate)*up); delta=save('22_down_projection',product@b['down']); x=save('23_ffn_residual',x+delta)
    final=save('24_final_rmsnorm',rms_norm(x)); logits=save('25_logits',final@m.output); save('26_next_token_distribution',softmax(logits))
    np.testing.assert_allclose(logits,m.forward(ids),atol=1e-12)
    return stages


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--active-ffn',action='store_true'); p.add_argument('--json',action='store_true'); args=p.parse_args()
    stages=trace(args.active_ffn)
    if args.json: print(json.dumps({k:{'shape':list(v.shape),'value':v.tolist()} for k,v in stages.items()},indent=2))
    else:
        np.set_printoptions(precision=6,suppress=True)
        for name,value in stages.items(): print(name,'shape=',value.shape,'\n',value)

if __name__=='__main__': main()
