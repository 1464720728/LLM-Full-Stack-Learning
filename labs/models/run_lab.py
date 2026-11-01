"""Run one of 31 bounded, offline CPU mechanism laboratories."""
import argparse
import json
import math
import numpy as np
from . import core as c

LABS={}
def lab(name):
    def register(fn): LABS[name]=fn; return fn
    return register

@lab('tokenization')
def tokenization():
    tok=c.ByteTokenizer(); text='猫 eats.'; ids=tok.encode(text)
    print('UTF-8 bytes:',ids,'roundtrip:',tok.decode(ids)); print('BPE:',c.bpe_train('ababab abab',3))
@lab('tensor_shapes')
def tensor_shapes():
    x=np.arange(24).reshape(3,8); h=c.split_heads(x,2)
    print('X:',x.shape,'heads:',h.shape,'roundtrip:',np.array_equal(c.merge_heads(h),x))
@lab('autograd')
def autograd():
    x=c.Value(2.); y=x*x+x; y.backward(); print('y=',y.data,'dy/dx=',x.grad,'expected=5')
@lab('softmax')
def softmax():
    print(c.softmax([1000,1001,999])); print('all masked:',c.softmax([-np.inf,-np.inf]))
    loss,grad=c.cross_entropy([[0.,0.,0.]],[2]); print('CE=',loss,'gradient=',grad)
@lab('attention')
def attention():
    q=np.array([[1.,0.],[0.,1.]]); k=q.copy(); v=np.array([[10.,0.],[0.,20.]])
    out,w=c.attention(q,k,v,c.causal_mask(2)); print('weights=',w,'output=',out,sep='\n')
@lab('transformer')
def transformer():
    model=c.OriginalTransformer(); a=model.forward([1,2,3],[0,4,5]); b=model.forward([1,2,3],[0,4,9])
    print('logits shape',a.shape,'earlier-prefix invariant',np.max(abs(a[:2]-b[:2])))
@lab('decoder')
def decoder():
    m=c.TinyDecoder(); print('logits shape',m.forward([1,2,3,4]).shape)
    print('parameters',sum(a.size for a in [m.embedding,m.output])+sum(a.size for b in m.blocks for a in b.values()))
@lab('rope')
def rope():
    x=np.array([[1.,2.,3.,4.]]); a=c.rope(x,[3]); print('norm before/after',np.linalg.norm(x),np.linalg.norm(a)); print(a)
@lab('kv_cache')
def kv_cache():
    m=c.TinyDecoder(); ids=[1,2,3,4,5]; full=m.forward(ids); cache=None; out=[]
    for token in ids:
        z,cache=m.step(token,cache); out.append(z)
    print('max full-vs-step error',np.max(abs(full-np.array(out)))); print('K shape per layer',cache[0][0].shape)
@lab('gqa')
def gqa():
    q=np.ones((4,2,2)); k=np.ones((2,2,2)); v=np.arange(8).reshape(2,2,2)
    out,_=c.grouped_attention(q,k,v,c.causal_mask(2)); print('4 Q heads / 2 KV heads',out); print('KV elements',k.size+v.size)
@lab('mla')
def mla():
    r=np.random.default_rng(0); q=r.normal(size=(4,6)); latent=r.normal(size=(4,2)); uk=r.normal(size=(2,6)); uv=r.normal(size=(2,3)); mask=c.causal_mask(4)
    exact,_=c.attention(q,latent@uk,latent@uv,mask); absorbed=c.latent_attention(q,latent,uk,uv,mask)
    print('factorization error',np.max(abs(exact-absorbed))); print('full K+V elements',4*(6+3),'latent elements',latent.size)
@lab('moe')
def moe():
    x=np.array([[2.,1.],[1.,2.],[3.,1.]]); router=np.eye(2); experts=[(np.eye(2),np.eye(2)),(np.eye(2),2*np.eye(2))]
    out,audit=c.moe(x,router,experts,top_k=1,capacity=1); print('output',out,'chosen',audit['chosen'],'dropped',audit['dropped'],sep='\n')
@lab('sparse')
def sparse():
    r=np.random.default_rng(0); q=r.normal(size=(8,4)); k=r.normal(size=q.shape); v=r.normal(size=q.shape); mask=c.local_global_mask(8,3,4)
    ref,_=c.attention(q,k,v,mask); out,n=c.sparse_attention(q,k,v,mask)
    print(mask.astype(int)); print('pairs',n,'error',np.max(abs(ref-out)))
@lab('linear')
def linear():
    r=np.random.default_rng(2); q=r.normal(size=(6,3)); k=r.normal(size=q.shape); v=r.normal(size=(6,2)); full=c.linear_attention_dense(q,k,v); out,state=c.linear_attention_recurrent(q,k,v)
    print('error',np.max(abs(full-out)),'state shapes',state[0].shape,state[1].shape)
@lab('delta')
def delta():
    k=np.array([[1.,0.],[1.,0.]]); v=np.array([[3.],[7.]]); out,s=c.delta_recurrence(k,k,v,[1.,1.])
    print('same key overwrite outputs',out.ravel(),'state',s)
@lab('ssm')
def ssm():
    u=np.array([1.,0.,0.,0.]); a=np.array([.5]); b=np.ones(1); cc=np.ones(1)
    ys,_=c.diagonal_ssm(u,a,b,cc); print('recurrence',ys,'convolution',c.ssm_convolution(u,a,b,cc))
@lab('selective_ssm')
def selective_ssm():
    print('input-dependent exact-ZOH scalar state',c.selective_ssm([1.,0.,2.,0.]))
@lab('attnres')
def attnres():
    sources=np.array([[[1.,0.]],[[0.,2.]],[[3.,3.]]]); out,w=c.attention_residual(sources,np.array([1.,0.]))
    print('depth weights',w.ravel(),'sum',w.sum(0),'output',out)
@lab('hyperconnections')
def hyperconnections():
    w=c.sinkhorn(np.array([[1.,2.],[3.,1.]])); print('mixing matrix',w,'row sums',w.sum(1),'col sums',w.sum(0),sep='\n')
@lab('quantize')
def quantize():
    x=np.array([[-1.,-.3,0.,.8],[0.,0.,0.,0.]])
    for bits in [4,8]:
        q,s,y=c.symmetric_quantize(x,bits,axis=1); print('bits',bits,'q',q,'scale',s.ravel(),'max error',np.max(abs(x-y)))
@lab('sampling')
def sampling():
    r=np.random.default_rng(1); logits=np.array([2.,1.,0.]); print('greedy',c.sample_token(logits,r,temperature=0)); print('top-k=2',[c.sample_token(logits,r,top_k=2) for _ in range(12)])
@lab('lora')
def lora():
    x=np.array([[1.,2.]]); w=np.eye(2); a=np.array([[1.],[0.]]); b=np.array([[2.,3.]])
    print('LoRA output',c.lora(x,w,a,b),'merged',x@(w+a@b))
@lab('dpo')
def dpo():
    print('equal margin',c.dpo_loss(-3,-4,-3,-4),'improved margin',c.dpo_loss(-2,-5,-3,-4))
    derivative=c.finite_difference(lambda z:c.dpo_loss(float(z[0]),-4,-3,-4),np.array([-3.]),(0,))
    print('d(loss)/d(chosen_logprob)',derivative,'expected',-.1/2)
@lab('multimodal')
def multimodal():
    image=np.arange(4*4*3).reshape(4,4,3); patches=c.patchify(image,2)
    print('image',image.shape,'patches',patches.shape,'first patch',patches[0])
@lab('loss_mask')
def loss_mask():
    logits=np.zeros((4,3)); targets=np.array([0,1,2,1]); active=np.array([False,False,True,True]); loss,g=c.cross_entropy(logits[active],targets[active]); full=np.zeros_like(logits); full[active]=g
    print('assistant-only target loss',loss,'gradient',full,sep='\n')

@lab('compressed')
def compressed():
    from .advanced import compressed_sparse_attention
    r=np.random.default_rng(8); q=r.normal(size=(8,4)); k=r.normal(size=q.shape); v=r.normal(size=q.shape)
    out,selected=compressed_sparse_attention(q,k,v)
    print('output shape',out.shape,'selected compressed block starts',selected)
@lab('ced')
def ced():
    from .advanced import causal_encoder_decoder
    x=np.arange(12,dtype=float).reshape(4,3)/10
    out,encoded=causal_encoder_decoder(x,np.eye(3),np.eye(3),np.eye(3))
    print('encoder',encoded,'decoder',out,sep='\n')
@lab('conditional_memory')
def conditional_memory():
    from .advanced import conditional_ngram_memory
    out,slots=conditional_ngram_memory([1,2,1,2],np.arange(32).reshape(8,4))
    print('hash slots',slots,'retrieved rows',out,sep='\n')
@lab('block_attnres')
def block_attnres():
    from .advanced import block_depth_aggregate
    traces=block_depth_aggregate(np.array([[1.,0.]]),[np.array([[0.,1.]])]*4,np.zeros(2),2)
    for i,t in enumerate(traces): print('layer',i,'candidate count',len(t['weights']),'output',t['output'])
@lab('grpo')
def grpo():
    from .advanced import group_advantages,clipped_policy_objective
    a=group_advantages([0.,1.,2.]); loss,g=clipped_policy_objective(np.log([.2,.3,.5]),np.log([.2,.3,.5]),a)
    print('group advantages',a,'loss',loss,'gradient wrt logprob',g)
@lab('crossmodal')
def crossmodal():
    from .advanced import cross_modal_read
    patches=c.patchify(np.arange(16).reshape(4,4,1),2)/16
    projection=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    print('text reads visual tokens',cross_modal_read(np.eye(2),patches,projection))

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('name',choices=sorted(LABS)); args=p.parse_args()
    np.set_printoptions(precision=6,suppress=True); LABS[args.name]()
if __name__=='__main__': main()
