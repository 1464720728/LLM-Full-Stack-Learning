import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from labs.agents.advanced_solutions import *
from labs.agents.contracts import object_schema,Tool
from labs.agents.runtime import Harness,ScriptedModel,Turn
from labs.agents.workflows import Workflow,GraphState,approval_node,action_digest

class BeforeInsertConnection:
    """Deterministically commit a competing connection before the first INSERT.

    A SELECT inside `with sqlite3_connection` has not acquired a writer lock.
    This hook reproduces that interleaving without threads or timing assertions.
    """
    def __init__(self,connection,callback):self.connection=connection;self.callback=callback
    def execute(self,sql,*args):
        if sql.startswith('INSERT') and self.callback is not None:
            callback=self.callback;self.callback=None;callback()
        return self.connection.execute(sql,*args)
    def __enter__(self):self.connection.__enter__();return self
    def __exit__(self,*args):return self.connection.__exit__(*args)
    def __getattr__(self,name):return getattr(self.connection,name)

class AdvancedSolutionTests(unittest.TestCase):
    def test_validate_state(self):
        h=Harness(ScriptedModel([Turn('done')]),[]);s=h.run(h.new('r','goal'));self.assertTrue(validate_state(s))
        s.events[0]['seq']=2
        with self.assertRaises(ContractError):validate_state(s)
    def test_output_contract(self):
        t=Tool('bad','test',object_schema({}),lambda a:'not an integer')
        wrapped=output_checked(t,{'type':'integer'})
        with self.assertRaises(RuntimeError):wrapped.function({})
    def test_pinned_goal(self):
        h=[{'role':'system','content':'system'},{'role':'user','content':'old '*500},{'role':'assistant','content':'latest'}]
        result=pinned_context(h,'repair',['no network'],350)
        self.assertEqual(result[1]['role'],'user');self.assertIn('repair',result[1]['content']);self.assertEqual(result[-1]['content'],'latest')
    def test_duplicate_delta(self):
        a=SequencedAssembler();e={'index':0,'call_id':'c','name':'add','arguments':'{}'}
        self.assertTrue(a.feed(0,e));self.assertFalse(a.feed(0,e));self.assertEqual(a.finish()[0].arguments,'{}')
        with self.assertRaises(ContractError):a.feed(0,dict(e,arguments='changed'))
    def test_sequence_gap(self):
        with self.assertRaises(ContractError):SequencedAssembler().feed(2,{'index':0,'arguments':'{}'})
    def test_atomic_budget(self):
        async def run():
            b=AsyncBudget(3);r=await asyncio.gather(*(b.reserve() for _ in range(20)));return sum(r),b.used
        self.assertEqual(asyncio.run(run()),(3,3))
    def test_resource_serialization(self):
        async def run():
            active=set();overlap=[]
            async def worker(job):
                key=job['resource_key'];overlap.append(key in active);active.add(key);await asyncio.sleep(0);active.remove(key);return job['id']
            result=await resource_fanout([{'id':i,'resource_key':i%2} for i in range(8)],worker,4)
            return result,overlap
        result,overlap=asyncio.run(run());self.assertEqual(result,list(range(8)));self.assertFalse(any(overlap))
    def test_resource_fanout_rejects_noninteger_limits(self):
        calls=[]
        async def worker(job):calls.append(job['id'])
        async def run():
            for limit in [0,-1,True,1.5,float('nan'),float('inf'),'2']:
                with self.subTest(limit=limit),self.assertRaises(ValueError):
                    await resource_fanout([{'id':1,'resource_key':'a'}],worker,limit)
        asyncio.run(run());self.assertEqual(calls,[])
    def test_persistent_graph_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'g.db';g=Workflow({'r':approval_node},{'r':'END'});action={'x':1}
            s=g.run(GraphState('r',{'proposed_action':action}));db=GraphSnapshots(p);db.save('run',s,0);db.close()
            db=GraphSnapshots(p);s,v=db.load('run');g.approve(s,action_digest(action),True);g.run(s);db.save('run',s,v);self.assertEqual(s.status,'finished');db.close()
    def test_outbox_crash_replay(self):
        box=Outbox();box.create('a','title')
        with self.assertRaises(RuntimeError):box.deliver_one(True)
        self.assertEqual(box.consumer.execute('SELECT count(*) FROM inbox').fetchone()[0],1)
        box.deliver_one();self.assertEqual(box.consumer.execute('SELECT count(*) FROM inbox').fetchone()[0],1);self.assertFalse(box.deliver_one());box.close()
    def test_outbox_conflict(self):
        b=Outbox();b.create('a','x')
        with self.assertRaises(ContractError):b.create('a','y')
        b.close()
    def test_outbox_concurrent_business_key_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            producer=Path(directory)/'p.db';consumer=Path(directory)/'c.db'
            first=Outbox(producer,consumer);second=Outbox(producer,consumer)
            try:
                first.producer=BeforeInsertConnection(first.producer,lambda:second.create('task','competing title'))
                with self.assertRaisesRegex(ContractError,'business key conflict'):first.create('task','original title')
                self.assertEqual(first.producer.execute('SELECT title FROM tasks').fetchall(),[('competing title',)])
                self.assertEqual(first.producer.execute('SELECT count(*) FROM outbox').fetchone()[0],1)
            finally:first.close();second.close()
    def test_outbox_concurrent_inbox_collision_never_marks_sent(self):
        with tempfile.TemporaryDirectory() as directory:
            producer=Path(directory)/'p.db';consumer=Path(directory)/'c.db'
            first=Outbox(producer,consumer);second=Outbox(producer,consumer)
            try:
                first.create('task','original title')
                def competing_delivery():
                    with second.consumer:second.consumer.execute('INSERT INTO inbox VALUES(?,?)',('created:task','different payload'))
                first.consumer=BeforeInsertConnection(first.consumer,competing_delivery)
                with self.assertRaisesRegex(ContractError,'event identity collision'):first.deliver_one()
                self.assertEqual(first.consumer.execute('SELECT payload FROM inbox').fetchone()[0],'different payload')
                self.assertEqual(first.producer.execute('SELECT sent FROM outbox').fetchone()[0],0)
            finally:first.close();second.close()
    def test_outbox_existing_event_collision_rolls_back_new_task(self):
        box=Outbox()
        try:
            with box.producer:box.producer.execute('INSERT INTO outbox(event_id,payload) VALUES(?,?)',('created:task','different payload'))
            with self.assertRaisesRegex(ContractError,'event identity collision'):box.create('task','original title')
            self.assertEqual(box.producer.execute('SELECT count(*) FROM tasks').fetchone()[0],0)
            self.assertEqual(box.producer.execute('SELECT payload FROM outbox').fetchone()[0],'different payload')
        finally:box.close()
if __name__=='__main__':unittest.main()
