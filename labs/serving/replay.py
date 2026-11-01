"""Capstone A: replay a bounded request trace and compare scheduler policies.

Run without arguments for the built-in workload. Input JSON is a list of
{id,prompt:[token IDs],output,arrival}. All timings are simulated.
"""
import argparse
import json
from pathlib import Path
from .engine import Engine, Request
from .analysis import percentile

DEFAULT=[{"id":"chat-1","prompt":[1,2,3],"output":8,"arrival":0},
         {"id":"document","prompt":list(range(16))*2,"output":4,"arrival":1},
         {"id":"chat-2","prompt":[1,2,3,4],"output":8,"arrival":3},
         {"id":"followup","prompt":[1,2,3,4,5],"output":4,"arrival":20}]


def compare(trace, budgets=(2,8,16), policies=('fifo','decode_first','round_robin')):
    if not isinstance(trace,list) or not 1<=len(trace)<=1000:
        raise ValueError('trace must contain 1..1000 requests')
    reports=[]
    for budget in budgets:
        for policy in policies:
            e=Engine(token_budget=budget,chunk_size=min(budget,4),policy=policy,pages=256)
            for row in trace:
                e.submit(Request(row['id'],tuple(row['prompt']),row['output'],row['arrival'],seed=7))
            e.run();report=e.report()
            ttft=[r['ttft_ms'] for r in report['requests'] if r['ttft_ms'] is not None]
            total=sum(r['output_tokens'] for r in report['requests'])
            reports.append({'policy':policy,'budget':budget,'p50_ttft_ms':percentile(ttft,.5) if ttft else None,
                            'p95_ttft_ms':percentile(ttft,.95) if ttft else None,
                            'synthetic_output_tokens_per_s':1000*total/report['time_ms'] if report['time_ms'] else 0,
                            'detail':report})
    reference=reports[0]['detail']['outputs']
    if not all(r['detail']['outputs']==reference for r in reports):
        raise AssertionError('scheduling changed the reference model outputs')
    return {'kind':'synthetic_cost_model_not_real_GPU_benchmark','runs':reports,'output_equivalence':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path)
    args=p.parse_args();trace=json.loads(args.trace.read_text()) if args.trace else DEFAULT
    print(json.dumps(compare(trace),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
