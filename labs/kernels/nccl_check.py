"""Opt-in 2-rank NCCL correctness, small buffers and timeout, no benchmarks."""
import argparse
from datetime import timedelta
import importlib.util
import os

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:print('Not run. Use documented torchrun command and --run.');return
    if importlib.util.find_spec('torch') is None:print('SKIP: torch unavailable');return
    import torch
    import torch.distributed as dist
    size=int(os.environ.get('WORLD_SIZE','1'))
    if size!=2:raise SystemExit('This bounded exercise requires exactly 2 ranks.')
    rank=int(os.environ['LOCAL_RANK'])
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:raise SystemExit('Need 2 visible GPUs; no process group created.')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=60))
    try:
        x=torch.full((16,),float(dist.get_rank()+1),device=f'cuda:{rank}')
        dist.all_reduce(x)
        torch.testing.assert_close(x,torch.full_like(x,3))
        torch.cuda.synchronize()
        print(f'PASS rank={dist.get_rank()} allreduce sum=3, 16 float32 values')
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
