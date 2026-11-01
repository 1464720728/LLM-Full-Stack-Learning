import io
import json
from contextlib import redirect_stdout
import unittest
from labs.foundations.workshop import step01_loop, step02_function, step03_json
from labs.foundations.workshop import step04_pipeline, step05_cli, step06_memory_tasks
from labs.foundations.workshop import step07_persistence, step08_routing

class WorkshopTests(unittest.TestCase):
    def test_state_trace(self):
        total,trace=step01_loop.total_with_trace([3,5,0])
        self.assertEqual(total,8)
        self.assertEqual(trace,[(0,3,0,3),(1,5,3,8),(2,0,8,8)])

    def test_function_contract(self):
        self.assertEqual(step02_function.summarize_tokens([]),dict(count=0,tokens=0,mean_tokens=None))
        with self.assertRaises(ValueError): step02_function.summarize_tokens([True])

    def test_parse(self):
        self.assertEqual(step03_json.parse('{"request_id":"x","tokens":4}').tokens,4)
        with self.assertRaises(ValueError): step03_json.parse('[]')

    def test_pipeline_handles_blank_and_duplicates(self):
        line='{"request_id":"a","tokens":3}'
        report=step04_pipeline.analyze(['',line,line])
        self.assertEqual(report['count'],1)
        self.assertEqual(report['tokens'],3)
        self.assertEqual(report['errors'][0]['line'],3)

    def test_cli_default(self):
        stream=io.StringIO()
        with redirect_stdout(stream): self.assertEqual(step05_cli.main([]),0)
        self.assertEqual(json.loads(stream.getvalue())['tokens'],8)

    def test_memory_copies_protect_state(self):
        store=step06_memory_tasks.MemoryTasks()
        task,_=store.add('Learn','a'); task['done']=True
        self.assertFalse(store.list()[0]['done'])
        rows=store.list(); rows[0]['title']='changed'
        self.assertEqual(store.list()[0]['title'],'Learn')

    def test_memory_idempotency(self):
        store=step06_memory_tasks.MemoryTasks()
        self.assertTrue(store.add('Learn','a')[1])
        self.assertFalse(store.add('Learn','a')[1])
        with self.assertRaises(ValueError): store.add('Other','a')
        self.assertEqual(len(store.list()),1)

    def test_restart(self):
        self.assertEqual(step07_persistence.restart_demo(),[dict(id=1,title='Persist this task',done=True)])

    def test_routing_sequence(self):
        self.assertEqual([row[2] for row in step08_routing.requests_demo()],[200,201,200,200,200,404])

if __name__=='__main__': unittest.main()
