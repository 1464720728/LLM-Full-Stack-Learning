"""Twenty-eight offline micro-labs; inspect each function before running it."""
import argparse
from collections import Counter
import json
import math
import tempfile
from pathlib import Path
from .text import tokenize,chunk_text,digest
from .store import CorpusStore,Context,Chunk,ConflictError
from .retrieval import BM25,TfidfVectors,cosine,rrf,hybrid,lexical_rerank,mmr,Hit
from .evaluation import metrics,paired_delta
from .fixtures import seed,DEMO
from .rag import Citation,Claim,validate_claim,pack_context,answer_report,bounded_research
from .memory import MemoryStore,select_memory
from .graph import EvidenceGraph


def _store():
    s=CorpusStore();seed(s);return s


def tokenize_lab():
    text="KV Cache 缓存：A100 与 a100!"
    tokens=tokenize(text)
    assert tokens==['kv','cache','缓','存','a100','与','a100']
    return {'text':text,'lexical_tokens':tokens,'warning':'Not LLM tokenizer tokens.'}


def chunks_lab():
    text="alpha beta gamma. delta epsilon zeta. eta theta iota."
    spans=chunk_text(text,24,6)
    assert all(text[s.start:s.end]==s.text for s in spans)
    covered=set(i for s in spans for i in range(s.start,s.end))
    assert covered==set(range(len(text)))
    return [s.__dict__ for s in spans]


def identity_lab():
    a=digest('tenant-a','doc',1,0,4,'text');b=digest('tenant-a','doc',2,0,4,'text')
    assert a!=b
    return {'revision_1':a,'revision_2':b,'stable':a==digest('tenant-a','doc',1,0,4,'text')}


def bm25_lab():
    cs=[Chunk(str(i),'t',str(i),1,'','',0,len(t),t) for i,t in enumerate(['cat cat dog','cat fish','dog fish'])]
    index=BM25(cs)
    scores=[sum(index.contributions('cat',i).values()) for i in range(3)]
    assert scores[0]>scores[1]>scores[2]
    return {'df':dict(index.df),'avgdl':index.avgdl,'idf_cat':index.idf('cat'),'scores':scores}


def bm25_parameters_lab():
    cs=[Chunk(str(i),'t',str(i),1,'','',0,len(t),t) for i,t in enumerate(['cat','cat cat cat '+('dog '*15)])]
    return {str(b):[sum(BM25(cs,b=b).contributions('cat',i).values()) for i in range(2)] for b in [0,0.75,1]}


def vectors_lab():
    assert abs(cosine({'x':3,'y':4},{'x':6,'y':8})-1)<1e-12
    assert cosine({}, {'x':1})==0
    return {'same_direction':cosine({'x':3,'y':4},{'x':6,'y':8}),'orthogonal':cosine({'x':1},{'y':1})}


def tfidf_lab():
    with _store() as s:
        index=TfidfVectors(s.visible(DEMO))
        return {'query_vector':index.encode('cache'),'hits':[(h.chunk.doc_id,h.score) for h in index.search('cache')]}


def ann_lab():
    # Deliberately crude partition pruning, NOT HNSW/FAISS. Exhibit a missed neighbor.
    points={'a':0.0,'b':0.49,'c':0.51,'d':1.0};query=0.505
    exact=sorted(points,key=lambda i:abs(points[i]-query))[:2]
    coarse=[i for i,v in points.items() if v>=0.5]
    approx=sorted(coarse,key=lambda i:abs(points[i]-query))[:2]
    return {'exact':exact,'pruned':approx,'recall_at_2':len(set(exact)&set(approx))/2}


def fusion_lab():
    with _store() as s:
        cs=s.visible(DEMO)[:3]
        a=[Hit(cs[0],100),Hit(cs[1],90)];b=[Hit(cs[1],0.9),Hit(cs[2],0.8)]
        out=rrf([a,b],3,60)
        assert out[0].chunk.chunk_id==cs[1].chunk_id
        return [(h.chunk.doc_id,h.score) for h in out]


def rerank_lab():
    with _store() as s:
        hits=hybrid(s,DEMO,'KV cache',5)
        return {'before':[(h.chunk.doc_id,h.score) for h in hits],
                'after':[(h.chunk.doc_id,h.score) for h in lexical_rerank('KV cache',hits)]}


def metrics_lab():
    result=metrics(['b','x','a'],{'a':2,'b':1},3)
    assert result['recall']==1 and result['precision']==2/3 and result['mrr']==1
    return result


def compare_lab():
    return paired_delta([1,0,0.5,1],[1,1,0.25,1])


def acl_lab():
    with _store() as s:
        student=hybrid(s,DEMO,'ORCHID-731',5)
        admin=hybrid(s,Context('course','admin',('admins',)),'ORCHID-731',5)
        assert not student and len(admin)==1
        return {'learner_ids':[h.chunk.doc_id for h in student],'admin_ids':[h.chunk.doc_id for h in admin]}


def revision_lab():
    with _store() as s:
        before=next(c for c in s.visible(DEMO) if c.doc_id=='cache')
        rev=s.ingest('course','cache','KV cache','New cache policy.',('*',),expected_revision=1)
        ok,why=validate_claim(s,DEMO,Claim(before.text,Citation(before.chunk_id,before.text)))
        assert rev==2 and not ok
        return {'new_revision':rev,'old_citation_valid':ok,'reason':why}


def idempotency_lab():
    with CorpusStore() as s:
        a=s.ingest('t','d','Doc','content',('*',));b=s.ingest('t','d','Doc','content',('*',))
        assert a==b==1 and s.counts()['revisions']==1
        return {'first':a,'retry':b,'counts':s.counts()}


def deletion_lab():
    with _store() as s:
        s.delete('course','cache',1)
        assert not hybrid(s,DEMO,'Decode',5)
        logical=s.counts()
        s.purge('course','cache')
        return {'after_soft_delete':logical,'after_purge':s.counts()}


def packing_lab():
    with _store() as s:
        result=pack_context(hybrid(s,DEMO,'cache request',5),500)
        assert len(result)<=500
        return {'characters':len(result),'payload':json.loads(result)}


def citation_lab():
    with _store() as s:
        c=s.visible(DEMO)[0]
        good=Claim(c.text,Citation(c.chunk_id,c.text))
        bad=Claim('invented conclusion',Citation(c.chunk_id,c.text))
        return {'exact':validate_claim(s,DEMO,good),'unsupported':validate_claim(s,DEMO,bad)}


def rag_lab():
    with _store() as s:return answer_report(s,DEMO,'How does KV cache help decode?')


def refusal_lab():
    with _store() as s:
        result=answer_report(s,DEMO,'unicornmarshmallow')
        assert result['abstain']
        return result


def multihop_lab():
    with _store() as s:
        hits,events=bounded_research(s,DEMO,['Agent','retrieval','index','corpus'],3)
        assert len([e for e in events if e['event']=='retrieve'])==3
        return {'documents':sorted({h.chunk.doc_id for h in hits}),'events':events}


def memory_lab():
    with MemoryStore() as m:
        r=m.remember('t','alice','study.style','numeric examples','message:1',expected_revision=0)
        m.remember('t','alice','study.style','code first','message:2 correction',expected_revision=r)
        assert not m.recall('t','bob',now=1)
        return {'active':[x.__dict__ for x in m.recall('t','alice',now=1)],'history_states':[x['state'] for x in m.history('t','alice','study.style')]}


def ttl_lab():
    with MemoryStore() as m:
        m.remember('t','a','temporary.goal','finish lab','message:1',expires_at=10)
        assert len(m.recall('t','a',now=9))==1 and not m.recall('t','a',now=10)
        return {'before_expiry':len(m.recall('t','a',now=9)),'at_expiry':len(m.recall('t','a',now=10))}


def forget_lab():
    with MemoryStore() as m:
        m.remember('t','a','color','blue','message:1');m.remember('t','a','color','green','message:2')
        m.forget('t','a','color');assert not m.history('t','a','color')
        return {'all_revisions_removed':True}


def graph_lab():
    with _store() as s:
        graph=EvidenceGraph(s);c=next(x for x in s.visible(DEMO) if x.doc_id=='graph')
        graph.add(DEMO,'agent','uses','retrieval',c.chunk_id,'Agent uses retrieval.')
        graph.add(DEMO,'retrieval','uses','index',c.chunk_id,'Retrieval uses index.')
        path=graph.path(DEMO,'agent','index')
        assert len(path)==2
        return [e.__dict__ for e in path]


def graph_stale_lab():
    with _store() as s:
        graph=EvidenceGraph(s);c=next(x for x in s.visible(DEMO) if x.doc_id=='graph')
        graph.add(DEMO,'agent','uses','retrieval',c.chunk_id,'Agent uses retrieval.')
        s.delete('course','graph')
        assert graph.path(DEMO,'agent','retrieval') is None
        return {'visible_edges_after_source_delete':len(graph.visible_edges(DEMO))}


def diversity_lab():
    with _store() as s:
        hits=hybrid(s,DEMO,'cache request retrieval',6)
        return {'before':[h.chunk.doc_id for h in hits],'selected':[h.chunk.doc_id for h in mmr('cache request retrieval',hits,3)]}


def persistence_lab():
    with tempfile.TemporaryDirectory(prefix='knowledge-lab-') as d:
        p=Path(d)/'corpus.db'
        with CorpusStore(p) as s:seed(s)
        with CorpusStore(p) as s:
            answer=answer_report(s,DEMO,'cache')
            assert not answer['abstain']
        return {'reopened_database':True,'claim_count':len(answer['claims'])}


LABS={
 '01-tokenize':tokenize_lab,'02-chunks':chunks_lab,'03-identity':identity_lab,
 '04-bm25':bm25_lab,'05-bm25-parameters':bm25_parameters_lab,'06-vectors':vectors_lab,
 '07-tfidf':tfidf_lab,'08-ann':ann_lab,'09-fusion':fusion_lab,'10-rerank':rerank_lab,
 '11-metrics':metrics_lab,'12-compare':compare_lab,'13-acl':acl_lab,'14-revision':revision_lab,
 '15-idempotency':idempotency_lab,'16-deletion':deletion_lab,'17-packing':packing_lab,
 '18-citation':citation_lab,'19-rag':rag_lab,'20-refusal':refusal_lab,'21-multihop':multihop_lab,
 '22-memory':memory_lab,'23-ttl':ttl_lab,'24-forget':forget_lab,'25-graph':graph_lab,
 '26-graph-stale':graph_stale_lab,'27-diversity':diversity_lab,'28-persistence':persistence_lab,
}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('lab',choices=['all',*LABS])
    args=p.parse_args(argv)
    result={name:fn() for name,fn in LABS.items()} if args.lab=='all' else LABS[args.lab]()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
