import unittest
import numpy as np
from labs.models.advanced import (block_depth_aggregate,compressed_sparse_attention,
    causal_encoder_decoder,conditional_ngram_memory,group_advantages,clipped_policy_objective,cross_modal_read)
from labs.models.core import finite_difference

class AdvancedModelTests(unittest.TestCase):
    def test_compression_no_future_leak(self):
        r=np.random.default_rng(9); q=r.normal(size=(9,4)); k=r.normal(size=q.shape); v=r.normal(size=q.shape)
        a,selected=compressed_sparse_attention(q,k,v); k[-1]+=100; v[-1]-=100
        b,_=compressed_sparse_attention(q,k,v); np.testing.assert_allclose(a[:-1],b[:-1])
        for i,starts in enumerate(selected):
            for start in starts: self.assertLessEqual(start+2,max(0,i-1))
    def test_compression_first_token(self):
        q=np.array([[1.,2.]]); v=np.array([[3.,4.]])
        out,_=compressed_sparse_attention(q,q,v); np.testing.assert_array_equal(out,v)
    def test_ced_no_future_leak(self):
        r=np.random.default_rng(9); x=r.normal(size=(7,3)); w=np.eye(3); a,_=causal_encoder_decoder(x,w,w,w); x[-1]+=50; b,_=causal_encoder_decoder(x,w,w,w); np.testing.assert_allclose(a[:-1],b[:-1])
    def test_memory_repeat_ngram(self):
        table=np.arange(32).reshape(8,4); out,slots=conditional_ngram_memory([1,2,1,2],table); self.assertEqual(slots[1],slots[3]); np.testing.assert_array_equal(out[1],out[3])
    def test_block_sources_grow_at_boundaries(self):
        t=block_depth_aggregate(np.ones((1,2)),[np.ones((1,2))]*4,np.zeros(2),2); self.assertEqual([len(x['weights']) for x in t],[1,2,2,3])
    def test_group_centering_and_equal_rewards(self):
        a=group_advantages([0,1,2]); self.assertAlmostEqual(float(a.mean()),0); np.testing.assert_allclose(group_advantages([1,1,1]),0)
    def test_policy_surrogate_gradient_and_clipping(self):
        lp=np.log([.2,.3,.5]); old=lp.copy(); a=group_advantages([0,1,2]); _,g=clipped_policy_objective(lp,old,a)
        for i in [0,2]: self.assertAlmostEqual(g[i],finite_difference(lambda z:clipped_policy_objective(z,old,a)[0],lp,(i,)),places=7)
        high=lp.copy(); high[2]+=np.log(1.5); _,g2=clipped_policy_objective(high,old,a); self.assertEqual(g2[2],0)
    def test_cross_modal_shape_and_image_effect(self):
        q=np.eye(2); patches=np.eye(2); a=cross_modal_read(q,patches,np.eye(2)); b=cross_modal_read(q,patches*2,np.eye(2)); self.assertEqual(a.shape,(2,2)); self.assertGreater(float(abs(a-b).max()),.1)

if __name__=='__main__': unittest.main()
