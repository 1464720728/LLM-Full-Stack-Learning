import unittest
import numpy as np
from labs.serving.paged_attention import TensorKV,PagedTinyDecoder
from labs.serving.kv import OutOfPages,PrefixCache

class TensorKVTests(unittest.TestCase):
    def test_online_matches_gather_varied_pages_heads(self):
        g=np.random.default_rng(42)
        for block in [1,2,4,7]:
            for length in [1,3,8,13]:
                kv=TensorKV(20,block,heads=3,dim=5);kv.create('r')
                k=g.normal(size=(length,3,5));v=g.normal(size=(length,3,5));q=g.normal(size=(3,5))
                kv.append('r',list(range(length)),k,v)
                np.testing.assert_allclose(kv.online_attention('r',q),kv.attention('r',q),rtol=1e-12,atol=1e-12)
    def test_cow_copies_tensor_prefix(self):
        kv=TensorKV(4,4,1,2);kv.create('a')
        k=np.arange(6.).reshape(3,1,2);v=k+10
        kv.append('a',[1,2,3],k,v);kv.fork('a','b')
        kv.append('b',[9],np.array([[[99.,98.]]]),np.array([[[97.,96.]]]))
        np.testing.assert_array_equal(kv.gather('a')[0],k)
        np.testing.assert_array_equal(kv.gather('b')[0][:3],k)
        self.assertNotEqual(kv.pool.tables['a'][-1],kv.pool.tables['b'][-1]);kv.pool.check()
    def test_failed_append_keeps_numeric_and_metadata_state(self):
        kv=TensorKV(1,4,1,2);kv.create('a');x=np.ones((3,1,2));kv.append('a',[1,2,3],x,x);kv.fork('a','b')
        before=kv.keys.copy()
        with self.assertRaises(OutOfPages):kv.append('b',[4],np.zeros((1,1,2)),np.zeros((1,1,2)))
        np.testing.assert_array_equal(kv.keys,before);self.assertEqual(kv.pool.tokens('b'),[1,2,3]);kv.pool.check()
    def test_stale_unused_rows_are_not_read(self):
        kv=TensorKV(2,4,1,2);kv.create('a');kv.append('a',[1],np.ones((1,1,2)),np.array([[[2.,3.]]]))
        kv.keys[0,1:]=1e20;kv.values[0,1:]=-1e20
        np.testing.assert_allclose(kv.online_attention('a',np.ones((1,2))),[[2.,3.]])
    def test_prefix_cache_reuses_actual_tensor_pages(self):
        kv=TensorKV(8,2,1,2);cache=PrefixCache(kv.pool);kv.create('a')
        x=np.arange(8.).reshape(4,1,2);kv.append('a',[1,2,3,4],x,x+2)
        cache.publish('a','model',4);kv.release('a');kv.create('b')
        self.assertEqual(cache.attach('b','model',(1,2,3,4,5)),4)
        np.testing.assert_array_equal(kv.gather('b')[0],x);kv.pool.check(cache.refs())
    def test_incremental_logits_match_full_context(self):
        d=PagedTinyDecoder(block_size=3);d.create('r');tokens=[1,2,3,4,9,2,8]
        for i,t in enumerate(tokens):
            actual=d.append('r',t);expected=d.model.logits(tokens[:i+1])
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    def test_invalid_shape_no_allocation(self):
        kv=TensorKV(2,2,1,3);kv.create('a')
        with self.assertRaises(ValueError):kv.append('a',[1],np.zeros((1,3)),np.zeros((1,3)))
        self.assertEqual(len(kv.pool.free),2)
    def test_released_page_reuse_is_correct(self):
        kv=TensorKV(1,4,1,2);kv.create('a');kv.append('a',[1,2,3],np.ones((3,1,2)),np.ones((3,1,2))*99)
        kv.release('a');kv.create('b');kv.append('b',[8],np.ones((1,1,2)),np.ones((1,1,2))*7)
        np.testing.assert_allclose(kv.online_attention('b',np.ones((1,2))),[[7,7]])

if __name__=='__main__':unittest.main()
