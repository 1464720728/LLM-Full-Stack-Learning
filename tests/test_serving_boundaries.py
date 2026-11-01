"""Independent-review regressions: validation, transactions and aliasing."""
import unittest
from labs.serving.kv import PagePool,PrefixCache
from labs.serving.engine import Engine,Request

class BoundaryTests(unittest.TestCase):
    def test_negative_publish_never_creates_prefix(self):
        p=PagePool(4,2);c=PrefixCache(p);p.create('a');p.append('a',[1,2,3,4])
        for value in [-1,True,1.5]:
            with self.assertRaises(ValueError):c.publish('a','ns',value)
            self.assertEqual(len(c.entries),0);p.check(c.refs())
    def test_integer_resource_contracts(self):
        for value in [True,1.5,0,-1]:
            for key in ['token_budget','chunk_size','max_sequences','pages','block_size']:
                with self.subTest(key=key,value=value):
                    with self.assertRaises(ValueError):Engine(**{key:value})
        for value in [True,1.5,-1]:
            with self.assertRaises(ValueError):Engine(prefix_entries=value)
    def test_report_is_not_mutable_engine_state(self):
        e=Engine();e.submit(Request('a',(1,),2));e.run();expected=e.requests['a'].output.copy()
        e.report()['outputs']['a'].append(7)
        self.assertEqual(e.requests['a'].output,expected);e.check()
    def test_atomic_adopt_and_retain(self):
        p=PagePool(3,2);p.create('a');p.append('a',[1,2]);p.create('s')
        with self.assertRaises(AssertionError):p.adopt('s',(0,1))
        self.assertEqual(p.tables['s'],[]);self.assertEqual(p.pages[0].refs,1);p.check()
        with self.assertRaises(ValueError):p.retain([0,99])
        self.assertEqual(p.pages[0].refs,1);p.check()
    def test_duplicate_page_adopt_rejected(self):
        p=PagePool(3,2);p.create('a');p.append('a',[1,2]);p.create('s')
        with self.assertRaises(ValueError):p.adopt('s',(0,0))
        p.check()
    def test_failed_fork_leaves_no_child(self):
        p=PagePool(2,2)
        with self.assertRaises(KeyError):p.fork('missing','child')
        self.assertNotIn('child',p.tables);p.check()
    def test_runtime_history_cannot_be_submitted(self):
        for kwargs in [{'state':'finished'},{'output':[2]},{'computed':1},{'finish_ms':0},{'cache_hit_tokens':1},{'preemptions':1},{'emitted_ms':[1.]}]:
            e=Engine()
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):e.submit(Request('a',(1,),**kwargs))
            self.assertEqual(e.requests,{})
    def test_same_request_object_cannot_enter_two_engines(self):
        r=Request('a',(1,));Engine().submit(r)
        with self.assertRaises(ValueError):Engine().submit(r)
    def test_unpin_failure_is_atomic(self):
        p=PagePool(3,2);p.create('a');p.append('a',[1,2]);p.create('b');p.append('b',[3,4])
        with self.assertRaises(AssertionError):p.unpin([0,1,1])
        self.assertEqual([p.pages[i].refs for i in (0,1)],[1,1]);p.check()

if __name__=='__main__':unittest.main()
