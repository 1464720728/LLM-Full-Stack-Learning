"""Bounded opt-in GPU event benchmark; correctness first, no automatic tuning."""
import argparse
import importlib.util
import json
import statistics
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--op',choices=['add','matmul'],default='add')
    p.add_argument('--size',type=int,default=1024);p.add_argument('--repeats',type=int,default=20)
    args=p.parse_args()
    if not 1<=args.repeats<=100:raise SystemExit('repeats must be 1..100')
    if not 1<=args.size<=2048:raise SystemExit('size must be 1..2048')
    if not args.run:print('Not run; pass --run only after selecting an available GPU.');return
    if any(importlib.util.find_spec(m) is None for m in ['torch','triton']):print('SKIP: GPU dependencies unavailable');return
    import torch
    if not torch.cuda.is_available():print('SKIP: CUDA unavailable');return
    from . import triton_ops as ops
    torch.manual_seed(4)
    if args.op=='add':
        a=torch.randn(args.size*256,device='cuda');b=torch.randn_like(a);fn=lambda:ops.add(a,b);baseline=lambda:a+b
    else:
        a=torch.randn(args.size,args.size,device='cuda');b=torch.randn_like(a);fn=lambda:ops.matmul(a,b);baseline=lambda:a@b
    t=time.perf_counter();got=fn();torch.cuda.synchronize();cold=time.perf_counter()-t
    expected=baseline();torch.testing.assert_close(got,expected,rtol=1e-4,atol=1e-3)
    err=float((got-expected).abs().max().item())
    for _ in range(3):fn();baseline()
    torch.cuda.synchronize()
    def measure(call):
        samples=[]
        for _ in range(args.repeats):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();out=call();end.record();end.synchronize();samples.append(start.elapsed_time(end))
        return {'median_ms':statistics.median(samples),'min_ms':min(samples),'max_ms':max(samples),'samples':len(samples),'samples_ms':samples}
    result={'op':args.op,'shape':list(a.shape),'dtype':str(a.dtype),'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'torch':torch.__version__,'max_abs_error':err,'first_call_wall_s':cold,'triton':measure(fn),'torch_baseline':measure(baseline),'scope':'GPU event interval around allocation+launch wrapper; no host-to-device copies, includes any device work in interval; not end-to-end service latency','note':'shared-GPU contention and allocator/cache effects are not controlled'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
