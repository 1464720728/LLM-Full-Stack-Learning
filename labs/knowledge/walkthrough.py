"""Eight transparent checkpoints for the long source-reading workbook."""
import argparse
from collections import Counter
import json
from .store import CorpusStore,Context,ConflictError
from .text import tokenize,chunk_text
from .retrieval import BM25,TfidfVectors,rrf,hybrid
from .rag import Claim,Citation,validate_claim,answer_report
from .memory import MemoryStore
from .graph import EvidenceGraph

CTX=Context('workshop','learner',('readers',))


def build_case():
    store=CorpusStore()
    for key,title,text,acl in [
        ('d0','First','cat cat dog',('*',)),
        ('d1','Second','cat fish',('*',)),
        ('d2','Third','dog fish',('*',)),
        ('private','Restricted','cat canary secret',('group:admins',)),
    ]:
        store.ingest('workshop',key,title,text,acl,max_chars=80,overlap=8)
    return store


def stage1():
    with build_case() as store:
        return {'tables':store.counts(),'visible_docs':[c.doc_id for c in store.visible(CTX)],
                'principals':CTX.principals,'tokens':tokenize('cat cat dog')}


def stage2():
    with build_case() as store:
        chunks=store.visible(CTX);bm=BM25(chunks)
        return {'tf':[dict(x) for x in bm.tf],'df':dict(bm.df),'lengths':bm.lengths,
                'avgdl':bm.avgdl,'contributions':[bm.contributions('cat',i) for i in range(len(chunks))],
                'ranking':[(h.chunk.doc_id,h.score) for h in bm.search('cat')]}


def stage3():
    with build_case() as store:
        chunks=store.visible(CTX);bm=BM25(chunks);vector=TfidfVectors(chunks)
        a=bm.search('cat');b=vector.search('cat')
        return {'bm25':[(h.chunk.doc_id,h.score) for h in a],
                'tfidf':[(h.chunk.doc_id,h.score) for h in b],
                'rrf':[(h.chunk.doc_id,h.score) for h in rrf([a,b],5)],
                'query_vector':vector.encode('cat')}


def stage4():
    with build_case() as store:
        good=answer_report(store,CTX,'cat')
        no_answer=answer_report(store,CTX,'elephant')
        denied=answer_report(store,CTX,'canary')
        return {'answer':good,'unknown_abstains':no_answer['abstain'],
                'restricted_abstains':denied['abstain']}


def stage5():
    with build_case() as store:
        chunk=next(c for c in store.visible(CTX) if c.doc_id=='d0')
        old=Claim(chunk.text,Citation(chunk.chunk_id,chunk.text))
        first=validate_claim(store,CTX,old)
        new_rev=store.ingest('workshop','d0','First','cat dog updated',('*',),expected_revision=1)
        stale=validate_claim(store,CTX,old)
        try:
            store.ingest('workshop','d0','First','lost update',('*',),expected_revision=1)
        except ConflictError:
            conflict=True
        else:
            conflict=False
        return {'before':first,'revision':new_rev,'old_after_update':stale,'stale_write_rejected':conflict}


def stage6():
    with MemoryStore() as memory:
        memory.remember('workshop','learner','learning.mode','short','fictional message 1',expected_revision=0)
        memory.remember('workshop','learner','learning.mode','step by step','fictional correction 2',expected_revision=1)
        active=[m.__dict__ for m in memory.recall('workshop','learner',now=1)]
        states=[r['state'] for r in memory.history('workshop','learner','learning.mode')]
        memory.forget('workshop','learner','learning.mode')
        return {'active':active,'history_states':states,'after_forget':memory.history('workshop','learner','learning.mode')}


def stage7():
    with build_case() as store:
        store.ingest('workshop','relations','Relations','Agent uses retrieval. Retrieval uses index.',('*',))
        c=next(c for c in store.visible(CTX) if c.doc_id=='relations')
        graph=EvidenceGraph(store)
        graph.add(CTX,'agent','uses','retrieval',c.chunk_id,'Agent uses retrieval.')
        graph.add(CTX,'retrieval','uses','index',c.chunk_id,'Retrieval uses index.')
        path=graph.path(CTX,'agent','index')
        return {'path':[(e.subject,e.predicate,e.object,e.quote) for e in path],
                'depth_one':graph.path(CTX,'agent','index',max_depth=1)}


def stage8():
    with build_case() as store:
        store.ingest('workshop','relations','Relations','Agent uses retrieval.',('*',))
        c=next(c for c in store.visible(CTX) if c.doc_id=='relations')
        graph=EvidenceGraph(store);graph.add(CTX,'agent','uses','retrieval',c.chunk_id,c.text)
        before=store.counts()
        store.delete('workshop','relations')
        invisible=len(graph.visible_edges(CTX))
        after_soft=store.counts()
        store.purge('workshop','relations')
        after_purge=store.counts()
        return {'before':before,'visible_edges_after_soft_delete':invisible,
                'after_soft_delete':after_soft,'after_purge':after_purge}


STAGES={str(i):globals()[f'stage{i}'] for i in range(1,9)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all',*STAGES],default='all')
    args=p.parse_args(argv)
    result={name:fn() for name,fn in STAGES.items()} if args.stage=='all' else STAGES[args.stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
