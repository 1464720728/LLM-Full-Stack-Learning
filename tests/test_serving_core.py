import math
import random
import unittest
from labs.serving.kv import PagePool, PrefixCache, OutOfPages
from labs.serving.engine import Engine, Request
from labs.serving.sampling import probabilities
from labs.serving.analysis import Capacity, percentile, ring_allreduce, symmetric_quantize
from labs.serving.streaming import SSEDecoder


class PageTests(unittest.TestCase):
    def test_cow_partial_tail_and_release(self):
        p=PagePool(3,4);p.create('a');p.append('a',[1,2,3]);p.fork('a','b');p.append('b',[9])
        self.assertEqual(p.tokens('a'),[1,2,3]);self.assertEqual(p.tokens('b'),[1,2,3,9]);p.check()
        p.release('a');p.release('b');p.check();self.assertEqual(len(p.free),3)
    def test_atomic_failure(self):
        p=PagePool(1,4);p.create('a');p.append('a',[1,2,3]);p.fork('a','b')
        with self.assertRaises(OutOfPages):p.append('b',[9])
        self.assertEqual(p.tokens('a'),p.tokens('b'));p.check()
    def test_full_pages_shared_new_tail_private(self):
        p=PagePool(3,2);p.create('a');p.append('a',[1,2]);p.fork('a','b');p.append('b',[3])
        self.assertEqual(p.tables['a'][0],p.tables['b'][0]);self.assertEqual(len(p.free),1);p.check()
    def test_prefix_retains_after_request_free(self):
        p=PagePool(5,2);c=PrefixCache(p);p.create('a');p.append('a',[1,2,3,4]);c.publish('a','m',4);p.release('a')
        p.create('b');self.assertEqual(c.attach('b','m',(1,2,3,4,5)),4);p.check(c.refs())
        p.release('b');c.clear();p.check();self.assertEqual(len(p.free),5)
    def test_partial_prefix_and_namespace(self):
        p=PagePool(5,2);c=PrefixCache(p);p.create('a');p.append('a',[1,2,3,4]);c.publish('a','m',4)
        p.create('b');self.assertEqual(c.attach('b','m',(1,2,9)),2)
        p.create('c');self.assertEqual(c.attach('c','other',(1,2,3,4,5)),0);p.check(c.refs())
    def test_cache_leaves_logit_token(self):
        p=PagePool(5,2);c=PrefixCache(p);p.create('a');p.append('a',[1,2,3,4]);c.publish('a','m',4)
        p.create('b');self.assertEqual(c.attach('b','m',(1,2,3,4)),2)
    def test_invalid_tokens_do_not_mutate(self):
        p=PagePool(2,2);p.create('a')
        with self.assertRaises(ValueError):p.append('a',[True])
        self.assertEqual(p.tokens('a'),[]);p.check()
    def test_eviction_does_not_free_active_page(self):
        p=PagePool(2,2);c=PrefixCache(p);p.create('a');p.append('a',[1,2]);c.publish('a','m',2);c.clear()
        self.assertEqual(p.tokens('a'),[1,2]);p.check(c.refs())


class EngineTests(unittest.TestCase):
    def run_requests(self,budget=4,policy='round_robin'):
        e=Engine(token_budget=budget,policy=policy,prefix_entries=0)
        for i in range(5):e.submit(Request(str(i),(i,1,2),6,seed=i,temperature=.7))
        e.run();return e
    def test_batch_budget_and_completeness(self):
        e=self.run_requests()
        for event in e.events:
            if event['kind']=='batch':self.assertLessEqual(event['scheduled_tokens'],4)
        self.assertTrue(all(r.state=='finished' and len(r.output)==6 for r in e.requests.values()))
        self.assertEqual(len(e.pool.free),64)
    def test_per_request_seed_batch_invariant_in_reference(self):
        a=self.run_requests(1);b=self.run_requests(16)
        self.assertEqual(a.report()['outputs'],b.report()['outputs'])
    def test_arrival_and_metrics(self):
        e=Engine(prefix_entries=0);e.submit(Request('later',(1,),2,arrival_ms=9));e.run()
        r=e.requests['later'];self.assertGreaterEqual(r.emitted_ms[0],9);self.assertGreater(r.metrics()['ttft_ms'],0)
    def test_zero_generation(self):
        e=Engine();e.submit(Request('z',(1,),0));e.run();self.assertEqual(e.requests['z'].output,[])
        self.assertEqual(e.requests['z'].state,'finished');e.check()
    def test_cancel_idempotent(self):
        e=Engine(prefix_entries=0);e.submit(Request('x',(1,2,3),20));e.step()
        self.assertTrue(e.cancel('x'));self.assertFalse(e.cancel('x'));e.run();self.assertEqual(len(e.pool.free),64)
    def test_cache_hit_preserves_output(self):
        e=Engine(block_size=2);e.submit(Request('a',(1,2,3,4,5),3));e.run()
        e.submit(Request('b',(1,2,3,4,5),3,arrival_ms=e.time_ms));e.run()
        self.assertEqual(e.requests['b'].cache_hit_tokens,4);self.assertEqual(e.requests['a'].output,e.requests['b'].output)
    def test_preemption_progress_and_no_duplicate_output(self):
        e=Engine(pages=4,block_size=2,token_budget=4,chunk_size=2,max_sequences=2,prefix_entries=0)
        for rid,start in [('a',1),('b',5)]:e.submit(Request(rid,tuple(range(start,start+4)),4))
        e.run(max_iterations=500);self.assertTrue(any(r.preemptions for r in e.requests.values()))
        self.assertEqual(e.requests['a'].output,[5,6,7,8]);self.assertEqual(e.requests['b'].output,[9,10,11,12])
    def test_backend_failure_isolated(self):
        class B:
            vocab_size=16
            def logits(self,x):
                if x[0]==9:raise RuntimeError('injected')
                return [float(i) for i in range(16)]
        e=Engine(backend=B(),prefix_entries=0);e.submit(Request('a',(9,),3));e.submit(Request('b',(1,),3));e.run()
        self.assertEqual(e.requests['a'].state,'error');self.assertEqual(e.requests['b'].state,'finished');e.check()
    def test_impossible_request_rejected(self):
        e=Engine(pages=1,block_size=2)
        with self.assertRaises(ValueError):e.submit(Request('x',(1,2),2))
    def test_randomized_workloads_invariants(self):
        g=random.Random(19)
        for _ in range(30):
            e=Engine(token_budget=g.randint(1,8),chunk_size=g.randint(1,5),pages=32,block_size=2,max_sequences=4)
            for i in range(5):e.submit(Request(str(i),tuple(g.randrange(16) for _ in range(g.randint(1,9))),g.randint(1,5)))
            e.run(max_iterations=1000);e.check();self.assertTrue(all(r.state=='finished' for r in e.requests.values()))


class NumericalTests(unittest.TestCase):
    def test_softmax_extreme_finite(self):
        p=probabilities([10000.,9999.]);self.assertAlmostEqual(sum(p),1);self.assertTrue(all(math.isfinite(x) for x in p))
    def test_masks_and_invalid(self):
        self.assertEqual(probabilities([-math.inf,3.],temperature=0),[0.,1.])
        with self.assertRaises(ValueError):probabilities([-math.inf])
        with self.assertRaises(ValueError):probabilities([math.nan])
    def test_topk_ties_stable(self):
        self.assertEqual(probabilities([3.,3.,2.],top_k=1),[1.,0.,0.])
    def test_top_p_keeps_crossing_token(self):
        p=probabilities([math.log(.6),math.log(.3),math.log(.1)],top_p=.8)
        self.assertAlmostEqual(p[0],2/3);self.assertAlmostEqual(p[1],1/3);self.assertEqual(p[2],0)
    def test_capacity_units(self):
        c=Capacity(1,8,2,4,8,2,10,tp=2)
        self.assertEqual(c.estimate()['kv_bytes_per_token_per_rank'],128)
        with self.assertRaises(ValueError):Capacity(1,8,2,3,8,2,10,tp=2).estimate()
    def test_communication_single_and_two_ranks(self):
        self.assertEqual(ring_allreduce(1024,1,10)['ideal_ms'],0)
        self.assertEqual(ring_allreduce(1024,2,10)['bytes_sent_per_rank'],1024)
    def test_quantization_error_bound(self):
        values=[-3.,-.2,0.,1.4,2.]
        q,s,x=symmetric_quantize(values,4)
        self.assertLessEqual(max(abs(a-b) for a,b in zip(values,x)),s/2+1e-12)
    def test_metric_definition(self):
        r=Request('a',(1,),3,arrival_ms=2);r.output=[1,2,3];r.emitted_ms=[7,9,12];r.finish_ms=12
        self.assertEqual(r.metrics()['ttft_ms'],5);self.assertEqual(r.metrics()['mean_itl_ms'],2.5)
        self.assertEqual(percentile([1,2,3,100],.5),2.5)
    def test_sse_all_byte_boundaries(self):
        wire='data: {"text":"你好"}\r\n\r\ndata: [DONE]\r\n\r\n'.encode()
        for cut in range(len(wire)+1):
            d=SSEDecoder();events=d.feed(wire[:cut])+d.feed(wire[cut:])+d.feed(b'',True)
            self.assertEqual(events,['{"text":"你好"}','[DONE]'])
    def test_sse_truncation(self):
        with self.assertRaises(ValueError):SSEDecoder().feed(b'data: x',True)

if __name__=='__main__':unittest.main()
