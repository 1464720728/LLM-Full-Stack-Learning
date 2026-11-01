"""Explicit single-GPU correctness check. Exits cleanly with SKIP if unavailable."""
import argparse
import importlib.util
import json
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true',help='opt in to actual GPU work')
    a=p.parse_args()
    if not a.run:print('Not run. Pass --run after selecting CUDA_VISIBLE_DEVICES.');return
    if any(importlib.util.find_spec(m) is None for m in ['torch','triton']):
        print('SKIP: torch or triton unavailable; no GPU validation');return
    import torch
    if not torch.cuda.is_available():print('SKIP: no CUDA GPU');return
    from . import triton_ops as op
    torch.manual_seed(7);start=time.monotonic();device='cuda:0'
    cases=0
    for n in [1,31,257,1003]:
        x=torch.randn(n,device=device);y=torch.randn(n,device=device)
        torch.testing.assert_close(op.add(x,y),x+y,rtol=1e-6,atol=1e-6);cases+=1
        torch.testing.assert_close(op.reduce_sum(x),x.sum(),rtol=1e-5,atol=1e-4);cases+=1
    for m,n in [(1,1),(3,31),(17,257)]:
        x=torch.randn(m,n,device=device);w=torch.randn(n,device=device)
        torch.testing.assert_close(op.softmax(x),torch.softmax(x,dim=-1),rtol=1e-5,atol=1e-6);cases+=1
        ref=x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)*w
        torch.testing.assert_close(op.rmsnorm(x,w),ref,rtol=1e-5,atol=1e-5);cases+=1
    for m,k,n in [(1,1,1),(19,23,17),(33,65,31)]:
        x=torch.randn(m,k,device=device);y=torch.randn(k,n,device=device)
        torch.testing.assert_close(op.matmul(x,y),x@y,rtol=1e-4,atol=1e-4);cases+=1
    for n,d,dv in [(1,1,1),(17,7,5),(65,16,9)]:
        q=torch.randn(n,d,device=device);k=torch.randn_like(q);v=torch.randn(n,dv,device=device)
        for causal in [False,True]:
            score=q@k.T/(d**.5)
            if causal:score.masked_fill_(~torch.ones(n,n,device=device,dtype=torch.bool).tril(),float('-inf'))
            ref=torch.softmax(score,-1)@v
            torch.testing.assert_close(op.attention(q,k,v,causal),ref,rtol=1e-4,atol=1e-4);cases+=1
    torch.cuda.synchronize()
    print(json.dumps({'status':'passed','cases':cases,'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'torch':torch.__version__,'seconds_including_compilation':time.monotonic()-start,'performance_claim':False},ensure_ascii=False))
if __name__=='__main__':main()
