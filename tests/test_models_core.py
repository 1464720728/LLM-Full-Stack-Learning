import unittest
import tempfile
from pathlib import Path
import numpy as np
from labs.models import core as c
from labs.models.tiny_lm import TinyLanguageModel,train
from labs.models.compare import compare

class ModelMechanismTests(unittest.TestCase):
    def setUp(self): self.r=np.random.default_rng(123)
    def test_softmax_shift_invariance(self):
        x=np.array([[1000.,1001.,999.]])
        np.testing.assert_allclose(c.softmax(x),c.softmax(x-1000)); np.testing.assert_allclose(c.softmax(x).sum(-1),1)
    def test_fully_masked_row_zero(self):
        np.testing.assert_array_equal(c.softmax([-np.inf,-np.inf]),[0.,0.])
    def test_bad_softmax_rejected(self):
        with self.assertRaises(ValueError): c.softmax([np.inf,0])
    def test_ce_finite_extreme(self):
        loss,g=c.cross_entropy([[1000,-1000]],[1]); self.assertAlmostEqual(loss,2000); self.assertTrue(np.isfinite(g).all())
    def test_ce_gradient(self):
        x=self.r.normal(size=(3,5)); targets=np.array([0,2,1]); _,g=c.cross_entropy(x,targets)
        for index in [(0,0),(1,2),(2,4)]:
            self.assertAlmostEqual(g[index],c.finite_difference(lambda a:c.cross_entropy(a,targets)[0],x,index),places=7)
    def test_utf8_roundtrip(self):
        t=c.ByteTokenizer(); self.assertEqual(t.decode(t.encode('猫🙂 café')),'猫🙂 café')
    def test_bpe_preserves_text(self):
        tokens,_=c.bpe_train('ababa ababa',7); self.assertEqual(''.join(tokens),'ababa ababa')
    def test_shared_autograd_nodes_accumulate(self):
        x=c.Value(2); z=x*x; y=z+z+x; y.backward(); self.assertEqual(x.grad,9)
    def test_autograd_tanh(self):
        x=c.Value(.3); y=(x*x+2*x).tanh(); y.backward(); expected=(1-y.data*y.data)*(2*.3+2); self.assertAlmostEqual(x.grad,expected)
    def test_head_roundtrip(self):
        x=self.r.normal(size=(5,12)); np.testing.assert_array_equal(c.merge_heads(c.split_heads(x,3)),x)
    def test_attention_mask_and_rows(self):
        q=self.r.normal(size=(4,3)); v=self.r.normal(size=(4,2)); _,p=c.attention(q,q,v,c.causal_mask(4))
        np.testing.assert_allclose(p.sum(1),1); self.assertEqual(float(np.triu(p,1).sum()),0.)
    def test_attention_no_future_leak(self):
        q=self.r.normal(size=(4,3)); v=self.r.normal(size=(4,2)); a,_=c.attention(q,q,v,c.causal_mask(4)); v[-1]+=999; b,_=c.attention(q,q,v,c.causal_mask(4)); np.testing.assert_allclose(a[:-1],b[:-1])
    def test_rope_norm(self):
        x=self.r.normal(size=(2,7,8)); np.testing.assert_allclose(np.linalg.norm(c.rope(x,np.arange(7)),axis=-1),np.linalg.norm(x,axis=-1),atol=1e-12)
    def test_rope_relative_inner_product(self):
        q=self.r.normal(size=(1,8)); k=self.r.normal(size=(1,8)); a=c.rope(q,[4])@c.rope(k,[9]).T; b=c.rope(q,[0])@c.rope(k,[5]).T; np.testing.assert_allclose(a,b,atol=1e-12)
    def test_gqa_matches_manual_head_mapping(self):
        q=self.r.normal(size=(4,3,2)); k=self.r.normal(size=(2,3,2)); v=self.r.normal(size=(2,3,2)); a,_=c.grouped_attention(q,k,v)
        for h in range(4): np.testing.assert_allclose(a[h],c.attention(q[h],k[h//2],v[h//2])[0])
    def test_decoder_cached_matches_full(self):
        for kv in [1,2,4]:
            m=c.TinyDecoder(c.DecoderConfig(kv_heads=kv)); ids=[2,1,2,4,6]; full=m.forward(ids); state=None; actual=[]
            for t in ids:
                z,state=m.step(t,state); actual.append(z)
            np.testing.assert_allclose(full,actual,rtol=1e-10,atol=1e-10)
    def test_decoder_prefix_invariance(self):
        m=c.TinyDecoder(); np.testing.assert_allclose(m.forward([1,2,3])[:2],m.forward([1,2,9])[:2],atol=1e-12)
    def test_cache_no_cross_request_leak(self):
        m=c.TinyDecoder(); _,cache=m.step(1); a,_=m.step(2); b,_=m.step(2,None); np.testing.assert_array_equal(a,b); self.assertEqual(cache[0][0].shape[1],1)
    def test_original_transformer_prefix(self):
        m=c.OriginalTransformer(); a=m.forward([1,2,3],[0,2,3]); b=m.forward([1,2,3],[0,2,8]); np.testing.assert_allclose(a[:2],b[:2])
    def test_original_transformer_source_matters(self):
        m=c.OriginalTransformer(); a=m.forward([1,2,3],[0,2]); b=m.forward([4,2,3],[0,2]); self.assertGreater(np.max(abs(a-b)),1e-5)
    def test_latent_absorption_correct_scale(self):
        q=self.r.normal(size=(5,6)); latent=self.r.normal(size=(5,2)); uk=self.r.normal(size=(2,6)); uv=self.r.normal(size=(2,4)); mask=c.causal_mask(5)
        np.testing.assert_allclose(c.latent_attention(q,latent,uk,uv,mask),c.attention(q,latent@uk,latent@uv,mask)[0],atol=1e-12)
    def test_moe_top1_exact_expert(self):
        x=np.array([[2.,1.],[1.,2.]]); experts=[(np.eye(2),np.eye(2)),(np.eye(2),2*np.eye(2))]
        out,audit=c.moe(x,np.eye(2),experts,1); np.testing.assert_allclose(out,[[2,1],[2,4]]); np.testing.assert_array_equal(audit['counts'],[1,1])
    def test_moe_capacity_explicit_drop(self):
        x=np.array([[3.,1.],[2.,1.]]); e=[(np.eye(2),np.eye(2))]*2
        out,a=c.moe(x,np.eye(2),e,1,1); np.testing.assert_array_equal(out[1],[0,0]); self.assertTrue(a['dropped'][1,0])
    def test_moe_ties_deterministic(self):
        e=[(np.eye(2),np.eye(2))]*3; _,a=c.moe(np.ones((2,2)),np.zeros((2,3)),e,2); np.testing.assert_array_equal(a['chosen'],[[0,1],[0,1]])
    def test_sparse_equivalence_and_empty_row(self):
        q=self.r.normal(size=(9,4)); v=self.r.normal(size=(9,2)); mask=c.local_global_mask(9,3,4); mask[0]=False
        out,visits=c.sparse_attention(q,q,v,mask); np.testing.assert_allclose(out,c.attention(q,q,v,mask)[0],atol=1e-12); self.assertEqual(visits,int(mask.sum()))
    def test_linear_recurrent_and_chunk_equivalence(self):
        q=self.r.normal(size=(11,3)); k=self.r.normal(size=q.shape); v=self.r.normal(size=(11,2)); all_,_=c.linear_attention_recurrent(q,k,v)
        a,state=c.linear_attention_recurrent(q[:4],k[:4],v[:4]); b,_=c.linear_attention_recurrent(q[4:],k[4:],v[4:],state)
        np.testing.assert_allclose(all_,c.linear_attention_dense(q,k,v),atol=1e-12); np.testing.assert_allclose(all_,np.concatenate([a,b]),atol=1e-12)
    def test_linear_tiny_positive_denominator_preserves_definition(self):
        q=np.array([[-350.]])
        dense=c.linear_attention_dense(q,q,np.ones((1,1)))
        recurrent,_=c.linear_attention_recurrent(q,q,np.ones((1,1)))
        np.testing.assert_allclose(dense,1.)
        np.testing.assert_allclose(recurrent,dense)
    def test_linear_underflow_zero_row_consistent(self):
        q=np.array([[-1000.]])
        dense=c.linear_attention_dense(q,q,np.ones((1,1)))
        recurrent,_=c.linear_attention_recurrent(q,q,np.ones((1,1)))
        np.testing.assert_array_equal(dense,[[0.]])
        np.testing.assert_array_equal(recurrent,dense)
    def test_linear_rejects_silent_sequence_truncation(self):
        for fn in [c.linear_attention_dense,c.linear_attention_recurrent]:
            with self.assertRaises(ValueError): fn(np.ones((3,2)),np.ones((2,2)),np.ones((2,1)))
            with self.assertRaises(ValueError): fn(np.ones((3,2)),np.ones((3,2)),np.ones((2,1)))
    def test_delta_rejects_short_gate_and_sequence(self):
        with self.assertRaises(ValueError): c.delta_recurrence(np.ones((3,2)),np.ones((2,2)),np.ones((2,1)),np.ones(3))
        with self.assertRaises(ValueError): c.delta_recurrence(np.ones((3,2)),np.ones((3,2)),np.ones((3,1)),np.ones(1))
        with self.assertRaises(ValueError): c.delta_recurrence(np.ones((3,2)),np.ones((3,2)),np.ones((3,1)),np.ones(3),np.ones(2))
    def test_delta_overwrites_unit_key(self):
        k=np.array([[1.,0.],[1.,0.]]); out,_=c.delta_recurrence(k,k,np.array([[3.],[7.]]),[1.,1.]); np.testing.assert_allclose(out.ravel(),[3,7])
    def test_ssm_convolution_equivalent(self):
        u=self.r.normal(size=20); a=np.array([.2,.8]); b=np.array([1.,2.]); cc=np.array([.7,-.4]); np.testing.assert_allclose(c.diagonal_ssm(u,a,b,cc)[0],c.ssm_convolution(u,a,b,cc),atol=1e-12)
    def test_ssm_chunk_state(self):
        u=self.r.normal(size=12); args=(np.array([.5]),np.ones(1),np.ones(1)); a,s=c.diagonal_ssm(u[:4],*args); b,_=c.diagonal_ssm(u[4:],*args,initial=s); np.testing.assert_allclose(np.r_[a,b],c.diagonal_ssm(u,*args)[0])
    def test_attention_residual_uniform_query(self):
        src=self.r.normal(size=(4,3,6)); out,w=c.attention_residual(src,np.zeros(6)); np.testing.assert_allclose(out,src.mean(0)); np.testing.assert_allclose(w,.25)
    def test_sinkhorn_stochastic(self):
        w=c.sinkhorn(self.r.normal(size=(3,3)),100); np.testing.assert_allclose(w.sum(0),1,atol=1e-10); np.testing.assert_allclose(w.sum(1),1,atol=1e-10)
    def test_sinkhorn_extreme_columns_stay_finite(self):
        matrix=c.sinkhorn(np.array([[1000.,0.],[1000.,0.]]))
        self.assertTrue(np.isfinite(matrix).all())
        np.testing.assert_allclose(matrix,np.full((2,2),.5),atol=1e-12)
    def test_rmsnorm_integer_input_casts_before_square(self):
        x=np.array([[300,400]],dtype=np.int16)
        np.testing.assert_allclose(c.rms_norm(x),c.rms_norm(x.astype(np.float64)),atol=1e-12)
        np.testing.assert_allclose(c.rms_norm(x),[[.8485281374,1.1313708499]],atol=1e-9)
    def test_norm_rejects_invalid_epsilon_and_nonfinite_input(self):
        for fn in [c.rms_norm,c.layer_norm]:
            for eps in [0,-1,np.nan,True]:
                with self.assertRaises(ValueError): fn([1,2],eps)
            with self.assertRaises(ValueError): fn([1,np.inf])
    def test_silu_int_min_and_swiglu_integer_arithmetic(self):
        x=np.array([-32768],dtype=np.int16)
        self.assertTrue(np.isfinite(c.silu(x)).all())
        np.testing.assert_allclose(c.silu(x),c.silu(x.astype(float)))
        xx=np.array([[300,400]],dtype=np.int16); w=np.eye(2,dtype=np.int16)*200
        np.testing.assert_allclose(c.swiglu(xx,w,w,w),c.swiglu(xx.astype(float),w.astype(float),w.astype(float),w.astype(float)))
    def test_quantizer_zero_and_error_bound(self):
        x=np.array([[0.,0.,0.],[-.7,.3,1.]])
        for bits in [4,8]:
            q,s,y=c.symmetric_quantize(x,bits,axis=1); self.assertTrue(np.isfinite(y).all()); self.assertTrue(np.all(abs(x-y)<=s/2+1e-12))
    def test_sampling_filter_and_greedy(self):
        r=np.random.default_rng(3); logits=np.array([3.,2.,1.]); self.assertEqual(c.sample_token(logits,r,0),0)
        for _ in range(50): self.assertIn(c.sample_token(logits,r,top_k=2),[0,1]); self.assertEqual(c.sample_token(logits,r,top_p=.1),0)
    def test_lora_merged_equivalence(self):
        x=self.r.normal(size=(4,6)); w=self.r.normal(size=(6,5)); a=self.r.normal(size=(6,2)); b=self.r.normal(size=(2,5)); np.testing.assert_allclose(c.lora(x,w,a,b,4),x@(w+2*a@b))
    def test_dpo_margin_direction(self): self.assertLess(c.dpo_loss(-2,-5,-3,-4),c.dpo_loss(-3,-4,-3,-4))
    def test_patchify_order(self):
        image=np.arange(16).reshape(4,4,1); patches=c.patchify(image,2); np.testing.assert_array_equal(patches[0],[0,1,4,5]); self.assertEqual(patches.shape,(4,4))

class LanguageProjectTests(unittest.TestCase):
    def test_all_parameter_gradient_types(self):
        m=TinyLanguageModel(4,context=2,embedding=3,hidden=5); x=np.array([[1,1],[2,1],[1,3]]); y=np.array([2,3,0]); _,grads=m.loss_and_grads(x,y)
        for name in m.params:
            parameter=m.params[name]; index=tuple(0 for _ in parameter.shape)
            if name=='E': index=(1,1)
            def loss(candidate):
                old=m.params[name]; m.params[name]=candidate
                try: return m.loss_and_grads(x,y)[0]
                finally: m.params[name]=old
            numerical=c.finite_difference(loss,parameter,index)
            self.assertAlmostEqual(grads[name][index],numerical,places=7,msg=name)
    def test_learns_nontrivial_signal(self):
        _,_,metrics=train(160); self.assertLess(metrics['final_train_ce'],metrics['initial_train_ce']-.5)
    def test_checkpoint_roundtrip(self):
        m=TinyLanguageModel(4,context=2); x=np.array([[1,2]])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.npz'; m.save(path,'abcd'); restored,alphabet=m.load(path)
            self.assertEqual(alphabet,'abcd'); np.testing.assert_array_equal(m.forward(x)[0],restored.forward(x)[0])
    def test_compare_report_invariants(self):
        report=compare()
        for name,value in report.items():
            if name.startswith('max_error'): self.assertLess(value,1e-9,name)
        self.assertLess(report['sparse_score_pairs'],report['dense_causal_score_pairs'])

if __name__=='__main__': unittest.main()
