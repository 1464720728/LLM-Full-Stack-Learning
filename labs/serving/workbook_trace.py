"""Dump exact ownership/state snapshots for the advanced workbook."""
import argparse
import json
from .engine import Engine,Request
from .kv import PagePool,PrefixCache


def pool_snapshot(pool,cache=None):
    return {'tables':{k:list(v) for k,v in pool.tables.items()},
            'pages':[{'id':i,'tokens':list(p.data),'refs':p.refs} for i,p in enumerate(pool.pages) if p.refs],
            'free_pages':sorted(pool.free),
            'cache':[] if cache is None else [
                {'namespace':key[0],'prefix':list(key[1]),'pages':list(bids)} for key,bids in cache.entries.items()]}


def engine_snapshot(engine,label,event_start=0):
    return {'label':label,'iteration':engine.iteration,'time_ms':engine.time_ms,
            'requests':{rid:{'state':r.state,'computed':r.computed,'context':r.context,
                'output':list(r.output),'emitted_ms':list(r.emitted_ms),'preemptions':r.preemptions}
                for rid,r in engine.requests.items()},
            'memory':pool_snapshot(engine.pool,engine.cache),
            'events':engine.events[event_start:]}


def prefix_trace():
    e=Engine(token_budget=4,chunk_size=4,pages=8,block_size=2)
    e.submit(Request('A',(1,2,3,4,5),2))
    result=[engine_snapshot(e,'A submitted')]
    while e.requests['A'].state!='finished':
        n=len(e.events);e.step();result.append(engine_snapshot(e,'A step',n))
    e.submit(Request('B',(1,2,3,4,6),2,arrival_ms=e.time_ms))
    while e.requests['B'].state!='finished':
        n=len(e.events);e.step();result.append(engine_snapshot(e,'B step',n))
    e.cache.clear();e.check();result.append(engine_snapshot(e,'cache cleared',len(e.events)))
    return result


def cow_trace():
    p=PagePool(4,4);result=[]
    p.create('A');p.append('A',[1,2,3]);result.append({'label':'A partial page',**pool_snapshot(p)})
    p.fork('A','B');result.append({'label':'B shares A',**pool_snapshot(p)})
    p.append('B',[9]);result.append({'label':'B copy-on-write',**pool_snapshot(p)})
    p.append('A',[4]);result.append({'label':'A keeps original page',**pool_snapshot(p)})
    p.release('A');p.release('B');p.check();result.append({'label':'both released',**pool_snapshot(p)})
    return result


def preempt_trace():
    e=Engine(pages=4,block_size=2,token_budget=4,chunk_size=2,max_sequences=2,prefix_entries=0)
    e.submit(Request('A',(1,2,3,4),4));e.submit(Request('B',(5,6,7,8),4));result=[]
    for _ in range(100):
        n=len(e.events)
        if not e.step():break
        result.append(engine_snapshot(e,'iteration boundary',n))
    else:raise RuntimeError('trace exceeded bound')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['prefix','cow','preempt'],required=True);a=p.parse_args()
    fn={'prefix':prefix_trace,'cow':cow_trace,'preempt':preempt_trace}[a.case]
    print(json.dumps(fn(),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
