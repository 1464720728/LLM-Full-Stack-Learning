import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from labs.foundations import log_project as logs
from labs.foundations import task_service as tasks

class LogProjectTests(unittest.TestCase):
    def test_reference_report(self):
        report=logs.analyze(logs.sample_lines())
        self.assertEqual((report['valid'],report['invalid'],report['tokens']), (3,1,8))
        self.assertEqual(report['latency_ms'],dict(mean=20,p50_nearest_rank=20,p95_nearest_rank=30))

    def test_empty_input(self):
        report=logs.analyze(['','  '])
        self.assertEqual(report['valid'],0)
        self.assertIsNone(report['latency_ms']['mean'])

    def test_duplicate_rejected(self):
        line=logs.sample_lines()[0]
        report=logs.analyze([line,line])
        self.assertEqual(report['valid'],1)
        self.assertEqual(report['invalid'],1)

    def test_malformed_shapes_are_data_errors(self):
        for row in [[],{}, dict(request_id='x', latency_ms=-1,tokens=0,status='ok'),
                    dict(request_id='x',latency_ms=float('nan'),tokens=1,status='ok'),
                    dict(request_id='x',latency_ms=1,tokens=True,status='ok')]:
            with self.subTest(row=row):
                with self.assertRaises(logs.InvalidRecord): logs.parse_record(json.dumps(row))

    def test_strict_includes_line(self):
        with self.assertRaisesRegex(logs.InvalidRecord,'line 2'):
            logs.analyze([logs.sample_lines()[0],'broken'],strict=True)

    def test_percentile_definition(self):
        self.assertEqual(logs.nearest_rank([10,20,30,40],.5),20)
        self.assertEqual(logs.nearest_rank([10,20,30,40],.95),40)
        with self.assertRaises(ValueError): logs.nearest_rank([1],0)

class TaskProjectTests(unittest.TestCase):
    def setUp(self): self.store=tasks.TaskStore()
    def tearDown(self): self.store.close()
    def payload(self,title='Learn',key='k1'):
        return json.dumps(dict(title=title,idempotency_key=key)).encode()

    def test_create_retry_and_conflict(self):
        first,payload=tasks.dispatch(self.store,'POST','/tasks',self.payload())
        self.assertEqual(first,201)
        self.assertEqual(payload,dict(id=1,title='Learn',done=False))
        self.assertEqual(tasks.dispatch(self.store,'POST','/tasks',self.payload())[0],200)
        self.assertEqual(tasks.dispatch(self.store,'POST','/tasks',self.payload('different'))[0],409)
        self.assertEqual(len(self.store.list()),1)

    def test_complete_is_idempotent(self):
        self.store.add('Learn','k1')
        a=self.store.complete(1); b=self.store.complete(1)
        self.assertEqual(a,b)
        self.assertTrue(a['done'])

    def test_missing_task(self):
        self.assertEqual(tasks.dispatch(self.store,'POST','/tasks/99/complete')[0],404)

    def test_invalid_body(self):
        for body in [b'broken', b'[]', b'{"title":true,"idempotency_key":"k"}', b'\xff']:
            with self.subTest(body=body):
                self.assertEqual(tasks.dispatch(self.store,'POST','/tasks',body)[0],400)
        self.assertEqual(tasks.dispatch(self.store,'POST','/tasks',b'a'*(tasks.MAX_BODY+1))[0],413)

    def test_sql_metacharacters_are_literal(self):
        title="Robert'); DROP TABLE tasks;--"
        self.store.add(title,'safe')
        self.assertEqual(self.store.list()[0]['title'],title)

    def test_query_and_completion_body_contract(self):
        self.assertEqual(tasks.dispatch(self.store,'GET','/tasks?x=1')[0],400)
        self.assertEqual(tasks.dispatch(self.store,'POST','/tasks/1/complete',b'{}')[0],400)

    def test_persistence(self):
        with TemporaryDirectory() as d:
            path=str(Path(d)/'tasks.db')
            a=tasks.TaskStore(path); a.add('Persist','k'); a.close()
            b=tasks.TaskStore(path)
            try: self.assertEqual(b.list(),[dict(id=1,title='Persist',done=False)])
            finally: b.close()

    def test_demo_status_sequence(self):
        self.assertEqual([row[0] for row in tasks.demo()],[201,200,200,200])

if __name__=='__main__': unittest.main()
