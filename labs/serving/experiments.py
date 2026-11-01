"""Run: python -m labs.serving.experiments --lab lifecycle (all labs are CPU/offline)."""
import argparse
import json
import random
from .engine import Engine, Request
from .kv import PagePool, PrefixCache, OutOfPages
from .backend import TinyAttentionBackend
from .sampling import probabilities, sample
from .analysis import Capacity, ring_allreduce, pd_break_even, symmetric_quantize, percentile
from .streaming import SSEDecoder


def lifecycle():
    e = Engine(token_budget=4, chunk_size=2, prefix_entries=0)
    e.submit(Request("a", (1,2,3), 3)); e.run()
    return {"report":e.report(), "events":e.events}


def token_budget():
    reports = {}
    for budget in [1, 4, 8]:
        e = Engine(token_budget=budget, prefix_entries=0)
        for i in range(4): e.submit(Request(str(i), (1,2,3,4,5), 3))
        e.run(); reports[str(budget)] = e.report()
    return reports


def chunks():
    reports = {}
    for size in [1, 4, 16]:
        e = Engine(token_budget=8, chunk_size=size, max_sequences=2, prefix_entries=0)
        e.submit(Request("long", tuple(range(12)), 2))
        e.submit(Request("short", (1,2), 2, arrival_ms=2))
        e.run(); reports[str(size)] = e.report()
    return reports


def continuous():
    e = Engine(max_sequences=2, prefix_entries=0)
    for i in range(5): e.submit(Request(str(i), (1,2,3), i+1, arrival_ms=float(i)))
    e.run(); return {"report":e.report(),"batches":[x for x in e.events if x["kind"]=="batch"]}


def prefix():
    e = Engine(block_size=2)
    e.submit(Request("cold", (1,2,3,4,5), 2)); e.run()
    e.submit(Request("warm", (1,2,3,4,6), 2, arrival_ms=e.time_ms)); e.run()
    return e.report()


def namespace():
    e = Engine(block_size=2)
    e.submit(Request("a", (1,2,3,4,5), 1, namespace="adapter-A")); e.run()
    e.submit(Request("b", (1,2,3,4,5), 1, arrival_ms=e.time_ms, namespace="adapter-B")); e.run()
    return e.report()


def cow():
    p = PagePool(4,4); p.create("a"); p.append("a",[1,2,3]); p.fork("a","b")
    before = {k:list(v) for k,v in p.tables.items()}
    p.append("b",[9]); p.check()
    return {"tables_before":before,"tables_after":p.tables,"a":p.tokens("a"),"b":p.tokens("b"),"refs":[x.refs for x in p.pages]}


def allocation_atomic():
    p=PagePool(1,4); p.create("a"); p.append("a",[1,2,3]); p.fork("a","b")
    try:p.append("b",[4])
    except OutOfPages as exc: message=str(exc)
    p.check(); return {"error":message,"a":p.tokens("a"),"b":p.tokens("b")}


def cancel():
    e=Engine(block_size=2,prefix_entries=0); e.submit(Request("a",(1,2,3,4),8)); e.step()
    once=e.cancel("a"); twice=e.cancel("a");e.run()
    return {"first_cancel":once,"second_cancel":twice,"report":e.report()}


def preemption():
    e=Engine(pages=4,block_size=2,token_budget=4,chunk_size=2,max_sequences=2,prefix_entries=0)
    e.submit(Request("a",(1,2,3,4),4));e.submit(Request("b",(5,6,7,8),4));e.run()
    return {"report":e.report(),"events":[x for x in e.events if x["kind"] in {"preempted","allocation_blocked"}]}


def fairness():
    result={}
    for policy in ["fifo","decode_first","round_robin"]:
        e=Engine(policy=policy,token_budget=2,chunk_size=2,prefix_entries=0)
        e.submit(Request("long",tuple(range(12)),4));e.submit(Request("short",(1,),4))
        e.run();result[policy]=e.report()
    return result


def runner_swap():
    result={}
    for name,backend in [("bigram",None),("attention",TinyAttentionBackend())]:
        e=Engine(backend=backend);e.submit(Request("a",(1,2,3),4));e.run();result[name]=e.report()
    return result


def sampling():
    return {str(t):probabilities([1.,2.,3.],temperature=t,top_p=.9) for t in [0.,.5,1.,2.]}


def stream_decoder():
    wire='data: {"text":"你好"}\r\n\r\ndata: [DONE]\r\n\r\n'.encode()
    d=SSEDecoder();events=[]
    for byte in wire:events.extend(d.feed(bytes([byte])))
    events.extend(d.feed(b"",final=True));return {"bytes":len(wire),"events":events}


def memory_capacity():
    c=Capacity(parameters=8_000_000_000,weight_bits=16,layers=32,kv_heads=8,head_dim=128,kv_bytes=2,context=32768)
    return c.estimate()


def communication():
    return {str(n):ring_allreduce(32*1024*1024,n,50.0) for n in [1,2,4,8]}


def pd_transfer():
    return {str(b):pd_break_even(100,300,30,2**30,b,1) for b in [1,10,50]}


def deterministic():
    outputs={}
    for budget in [1,4,16]:
        e=Engine(token_budget=budget,prefix_entries=0)
        for i in range(3):e.submit(Request(str(i),(i,1,2),8,seed=17+i,temperature=.8))
        e.run();outputs[str(budget)]=e.report()["outputs"]
    return {"outputs":outputs,"equal":len({json.dumps(x,sort_keys=True) for x in outputs.values()})==1}


def quantization():
    values=[-2.,-.3,.1,.8,1.7]
    result={}
    for bits in [4,8]:
        q,s,x=symmetric_quantize(values,bits)
        result[str(bits)]={"q":q,"scale":s,"reconstructed":x,"max_abs_error":max(abs(a-b) for a,b in zip(values,x))}
    return result


def metrics():
    r=Request("a",(1,),3,arrival_ms=2);r.output=[2,3,4];r.emitted_ms=[7,9,12];r.finish_ms=12;r.state="finished"
    return {"request":r.metrics(),"p50":percentile([1,2,3,100],.5),"p95":percentile([1,2,3,100],.95)}


def trace_replay():
    workload=[{"id":"interactive","prompt":[1,2],"output":4,"arrival":0},
              {"id":"document","prompt":list(range(15)),"output":2,"arrival":1},
              {"id":"followup","prompt":[1,2,3],"output":3,"arrival":8}]
    e=Engine(chunk_size=3,max_sequences=2)
    for row in workload:e.submit(Request(row["id"],tuple(row["prompt"]),row["output"],row["arrival"]))
    e.run();return {"workload":workload,"result":e.report()}


def fault_injection():
    class Failing:
        vocab_size=16
        def logits(self,ids):
            if ids[0]==9:raise ValueError("injected")
            return [float(i) for i in range(16)]
    e=Engine(backend=Failing(),prefix_entries=0)
    e.submit(Request("bad",(9,),3));e.submit(Request("good",(1,),3));e.run();e.check();return e.report()


def cache_eviction():
    p=PagePool(4,2);c=PrefixCache(p,max_entries=1)
    for rid,ids in [("a",[1,2]),("b",[3,4])]:
        p.create(rid);p.append(rid,ids);c.publish(rid,"m",2);p.release(rid);p.check(c.refs())
    result={"cached":len(c.entries),"free_before_clear":len(p.free)}
    c.clear();p.check(c.refs());result["free_after_clear"]=len(p.free);return result


def batch_equivalence():
    b=TinyAttentionBackend();contexts=[[1,2,3],[2,4],[8,3,2,1]]
    forward={str(x):b.logits(x) for x in contexts}
    reverse={str(x):b.logits(x) for x in reversed(contexts)}
    return {"same_per_request":forward==reverse,"first_logits":forward[str(contexts[0])][:4]}


LABS={f.__name__:f for f in [lifecycle,token_budget,chunks,continuous,prefix,namespace,cow,allocation_atomic,cancel,preemption,fairness,runner_swap,sampling,stream_decoder,memory_capacity,communication,pd_transfer,deterministic,quantization,metrics,trace_replay,fault_injection,cache_eviction,batch_equivalence]}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab",choices=sorted(LABS));parser.add_argument("--list",action="store_true")
    args=parser.parse_args()
    if args.list:print("\n".join(sorted(LABS)));return
    if not args.lab:parser.error("provide --lab NAME or --list")
    print(json.dumps(LABS[args.lab](),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
