import math
from pathlib import Path
import tempfile
import unittest
from labs.knowledge.text import tokenize,chunk_text
from labs.knowledge.store import CorpusStore,Context,ConflictError,Chunk
from labs.knowledge.retrieval import BM25,cosine,rrf,Hit,hybrid,TfidfVectors,mmr
from labs.knowledge.evaluation import metrics
from labs.knowledge.memory import MemoryStore
from labs.knowledge.graph import EvidenceGraph
from labs.knowledge.rag import Citation,Claim,validate_claim,pack_context,answer_report,bounded_research
from labs.knowledge.fixtures import seed,DEMO
from labs.knowledge.lab import LABS


class TextTests(unittest.TestCase):
    def test_case_and_cjk(self):
        self.assertEqual(tokenize('CACHE缓存'),['cache','缓','存'])
    def test_chunk_coverage_many_sizes(self):
        text='abc。def\n'+('中文 xyz '*20)
        for size in [1,2,7,30,1000]:
            for overlap in {0,size-1}:
                spans=chunk_text(text,size,overlap)
                self.assertEqual(set(range(len(text))),{i for s in spans for i in range(s.start,s.end)})
                self.assertTrue(all(s.text==text[s.start:s.end] and 0<len(s.text)<=size for s in spans))
                self.assertTrue(all(a.start<b.start for a,b in zip(spans,spans[1:])))
    def test_empty(self):self.assertEqual(chunk_text(''),[])
    def test_invalid_chunk_parameters(self):
        for args in [(0,0),(3,3),(3,-1)]:
            with self.assertRaises(ValueError):chunk_text('abc',*args)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.cs=[Chunk(str(i),'t',str(i),1,'','',0,len(t),t) for i,t in enumerate(['cat cat dog','cat fish','dog fish'])]
    def test_bm25_manual(self):
        b=BM25(self.cs)
        expected=math.log1p(1.5/2.5)*2*2.2/(2+1.2*(0.25+0.75*3/(7/3)))
        self.assertAlmostEqual(b.contributions('cat',0)['cat'],expected)
    def test_query_repeat_contract(self):
        b=BM25(self.cs)
        self.assertEqual(b.search('cat'),b.search('cat cat'))
    def test_unknown_and_empty(self):
        self.assertEqual(BM25(self.cs).search('unknown'),[])
        self.assertEqual(BM25([]).search('cat'),[])
    def test_length_norm(self):
        b=BM25(self.cs,b=0)
        self.assertAlmostEqual(b.contributions('cat',1)['cat'],b.idf('cat'))
    def test_cosine(self):
        self.assertAlmostEqual(cosine({'a':3,'b':4},{'a':6,'b':8}),1)
        self.assertEqual(cosine({},{}),0)
    def test_tfidf_unknown(self):self.assertEqual(TfidfVectors(self.cs).search('unknown'),[])
    def test_rrf_dedup(self):
        h=Hit(self.cs[0],3)
        self.assertAlmostEqual(rrf([[h,h]],2)[0].score,1/61)
    def test_rrf_consensus(self):
        out=rrf([[Hit(self.cs[0],99),Hit(self.cs[1],3)],[Hit(self.cs[1],0.2)]],3)
        self.assertEqual(out[0].chunk.chunk_id,'1')
    def test_metrics_duplicates(self):
        m=metrics(['a','a','b'],{'a':1,'b':1},2)
        self.assertEqual(m['recall'],1)
        self.assertEqual(m['ndcg'],1)
    def test_mmr_unique(self):
        selected=mmr('cat',[Hit(c,1) for c in self.cs],3)
        self.assertEqual(len({h.chunk.chunk_id for h in selected}),3)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store=CorpusStore();seed(self.store)
    def tearDown(self):self.store.close()
    def test_tenant_isolation(self):
        self.assertTrue(all(c.tenant=='course' for c in self.store.visible(DEMO)))
    def test_acl_before_search(self):
        self.assertEqual(hybrid(self.store,DEMO,'ORCHID-731'),[])
        self.assertEqual(hybrid(self.store,Context('course','a',('admins',)),'ORCHID-731')[0].chunk.doc_id,'restricted')
    def test_deny_default(self):
        self.store.ingest('course','deny','Hidden','canary')
        self.assertEqual(hybrid(self.store,DEMO,'canary'),[])
    def test_sql_special_characters(self):
        ctx=Context("course' OR 1=1 --",'x')
        self.assertEqual(self.store.visible(ctx),[])
    def test_idempotency(self):
        seed(self.store)
        self.assertEqual(self.store.counts()['revisions'],8)
    def test_revision_guard(self):
        with self.assertRaises(ConflictError):
            self.store.ingest('course','cache','x','x',expected_revision=0)
        self.assertEqual(self.store.counts()['revisions'],8)
    def test_acl_change_is_revision(self):
        self.store.ingest('course','cache','KV cache','new',('user:other',),expected_revision=1)
        self.assertNotIn('cache',[c.doc_id for c in self.store.visible(DEMO)])
    def test_deleted_not_visible(self):
        self.store.delete('course','cache')
        self.assertNotIn('cache',[c.doc_id for c in self.store.visible(DEMO)])
    def test_purge_cascades(self):
        self.store.purge('course','restricted')
        self.assertEqual(self.store.conn.execute("SELECT COUNT(*) FROM grants WHERE doc_id='restricted'").fetchone()[0],0)
    def test_delete_cas_checks_revision_at_update(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'race.db'
            with CorpusStore(path) as first,CorpusStore(path) as second:
                first.ingest('t','d','Doc','before',('*',))
                connection=first.conn
                class InterleavedConnection:
                    def __enter__(self):connection.__enter__();return self
                    def __exit__(self,*args):return connection.__exit__(*args)
                    def execute(self,sql,params=()):
                        if sql.startswith('UPDATE documents SET deleted=1'):
                            second.ingest('t','d','Doc','concurrent update',('*',),expected_revision=1)
                        return connection.execute(sql,params)
                first.conn=InterleavedConnection()
                try:
                    with self.assertRaises(ConflictError):first.delete('t','d',expected_revision=1)
                finally:first.conn=connection
                row=first.conn.execute("SELECT current_revision,deleted FROM documents").fetchone()
                self.assertEqual(tuple(row),(2,0))
    def test_purge_recreate_does_not_revive_old_citation(self):
        ctx=Context('t','reader')
        self.store.ingest('t','d','Doc','same body',('*',),source='old-source')
        old=self.store.visible(ctx)[0]
        claim=Claim(old.text,Citation(old.chunk_id,old.text))
        self.store.purge('t','d')
        revision=self.store.ingest('t','d','Doc','same body',('*',),source='new-source',expected_revision=0)
        self.assertEqual(revision,2)
        self.assertNotEqual(old.chunk_id,self.store.visible(ctx)[0].chunk_id)
        self.assertFalse(validate_claim(self.store,ctx,claim)[0])
        with self.assertRaises(ConflictError):self.store.delete('t','d',expected_revision=1)
    def test_purge_counter_survives_reopen_and_legacy_migration(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'legacy.db'
            with CorpusStore(path) as store:
                store.ingest('t','d','Doc','same',('*',))
                store.conn.execute('DROP TABLE document_revision_counters')
                store.conn.commit()
            with CorpusStore(path) as store:store.purge('t','d')
            with CorpusStore(path) as store:
                self.assertEqual(store.ingest('t','d','Doc','same',('*',),expected_revision=0),2)
    def test_delete_missing_and_repeated(self):
        self.assertFalse(self.store.delete('t','missing',expected_revision=7))
        self.assertTrue(self.store.delete('course','cache',expected_revision=1))
        self.assertTrue(self.store.delete('course','cache',expected_revision=1))
    def test_transaction_rolls_back_partial_write(self):
        import sqlite3
        self.store.conn.execute("CREATE TRIGGER fail_bad BEFORE INSERT ON chunks WHEN NEW.doc_id='bad' BEGIN SELECT RAISE(ABORT,'planned failure'); END")
        before=self.store.counts()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.ingest('course','bad','Bad','some text',('*',))
        self.assertEqual(self.store.counts(),before)
        self.assertIsNone(self.store.conn.execute("SELECT * FROM documents WHERE doc_id='bad'").fetchone())
    def test_chunk_span_matches(self):
        for c in self.store.visible(DEMO):
            body=self.store.conn.execute('SELECT body FROM revisions WHERE tenant=? AND doc_id=? AND revision=?',(c.tenant,c.doc_id,c.revision)).fetchone()[0]
            self.assertEqual(body[c.start:c.end],c.text)
    def test_stale_citation(self):
        c=self.store.visible(DEMO)[0]
        self.store.ingest(c.tenant,c.doc_id,c.title,'replacement',('*',))
        self.assertFalse(validate_claim(self.store,DEMO,Claim(c.text,Citation(c.chunk_id,c.text)))[0])
    def test_unsupported_claim(self):
        c=self.store.visible(DEMO)[0]
        self.assertFalse(validate_claim(self.store,DEMO,Claim('false',Citation(c.chunk_id,c.text)))[0])
    def test_character_budget(self):
        payload=pack_context(hybrid(self.store,DEMO,'cache'),20)
        self.assertEqual(payload,'[]')
    def test_abstention(self):self.assertTrue(answer_report(self.store,DEMO,'zzunknown')['abstain'])
    def test_bounded_queries(self):
        _,events=bounded_research(self.store,DEMO,['cache','cache','agent','retrieval'],2)
        self.assertEqual(sum(e['event']=='retrieve' for e in events),2)
    def test_graph_stale_and_depth(self):
        g=EvidenceGraph(self.store)
        c=next(x for x in self.store.visible(DEMO) if x.doc_id=='graph')
        g.add(DEMO,'a','uses','b',c.chunk_id,'Agent uses retrieval.')
        g.add(DEMO,'b','uses','c',c.chunk_id,'Retrieval uses index.')
        self.assertIsNone(g.path(DEMO,'a','c',max_depth=1))
        self.assertEqual(len(g.path(DEMO,'a','c',max_depth=2)),2)
        self.store.delete('course','graph')
        self.assertIsNone(g.path(DEMO,'a','c'))
    def test_graph_bad_quote(self):
        c=self.store.visible(DEMO)[0]
        with self.assertRaises(ValueError):EvidenceGraph(self.store).add(DEMO,'a','uses','b',c.chunk_id,'fake quote')


class MemoryTests(unittest.TestCase):
    def setUp(self):self.m=MemoryStore()
    def tearDown(self):self.m.close()
    def test_owner_isolation(self):
        self.m.remember('t','a','x','v','m1')
        self.assertEqual(self.m.recall('t','b'),[])
    def test_correction(self):
        self.m.remember('t','a','x','old','m1',expected_revision=0)
        self.m.remember('t','a','x','new','m2',expected_revision=1)
        self.assertEqual([m.value for m in self.m.recall('t','a')],['new'])
        self.assertEqual([m['state'] for m in self.m.history('t','a','x')],['superseded','active'])
    def test_cas(self):
        self.m.remember('t','a','x','one','m1',expected_revision=0)
        with self.assertRaises(ConflictError):self.m.remember('t','a','x','two','m2',expected_revision=0)
        self.assertEqual(self.m.recall('t','a')[0].value,'one')
    def test_ttl(self):
        self.m.remember('t','a','x','v','m1',expires_at=10)
        self.assertTrue(self.m.recall('t','a',now=9))
        self.assertFalse(self.m.recall('t','a',now=10))
    def test_forget_all_versions(self):
        self.m.remember('t','a','x','old','m1');self.m.remember('t','a','x','new','m2')
        self.m.forget('t','a','x')
        self.assertEqual(self.m.history('t','a','x'),[])
    def test_forget_recreate_rejects_stale_revision(self):
        self.m.remember('t','a','x','old','m1',expected_revision=0)
        self.m.forget('t','a','x')
        self.assertEqual(self.m.remember('t','a','x','new','m2',expected_revision=0),2)
        with self.assertRaises(ConflictError):
            self.m.remember('t','a','x','stale overwrite','m3',expected_revision=1)
        self.assertEqual(self.m.recall('t','a')[0].value,'new')
    def test_forget_counter_survives_reopen_and_legacy_migration(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'legacy.db'
            with MemoryStore(path) as memory:
                memory.remember('t','a','x','old','m1')
                memory.conn.execute('DROP TABLE memory_revision_counters')
                memory.conn.commit()
            with MemoryStore(path) as memory:memory.forget('t','a','x')
            with MemoryStore(path) as memory:
                self.assertEqual(memory.remember('t','a','x','new','m2',expected_revision=0),2)
    def test_retract(self):
        self.m.remember('t','a','x','v','m1');self.m.retract('t','a','x')
        self.assertFalse(self.m.recall('t','a'))
        self.assertEqual(self.m.history('t','a','x')[0]['state'],'retracted')
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):self.m.remember('t','a','x',float('nan'),'m1')
    def test_disk_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.db'
            with MemoryStore(p) as m:m.remember('t','a','x','v','m1')
            with MemoryStore(p) as m:self.assertEqual(m.recall('t','a')[0].value,'v')


class DemoTests(unittest.TestCase):
    def test_workbook_stages(self):
        from labs.knowledge.walkthrough import STAGES
        for name, fn in STAGES.items():
            with self.subTest(stage=name):self.assertIsNotNone(fn())

    def test_all_28(self):
        for name,fn in LABS.items():
            with self.subTest(lab=name):self.assertIsNotNone(fn())


if __name__=='__main__':unittest.main()
