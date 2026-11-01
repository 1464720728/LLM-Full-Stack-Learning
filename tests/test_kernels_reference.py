import unittest
import numpy as np
from labs.kernels import reference as r
from labs.kernels.run import experiment

class KernelReferenceTests(unittest.TestCase):
    def setUp(self):self.rng=np.random.default_rng(9)
    def test_offsets_and_views(self):
        self.assertEqual(r.c_strides((2,3,4)),(12,4,1));self.assertEqual(r.offset((2,1),(4,3),(1,4)),6)
        with self.assertRaises(IndexError):r.offset((4,1),(4,3),(1,4))
    def test_grid_stride_covers_once(self):
        for n in [0,1,19,129]:
            flat=sum(r.launch_indices(n,8,3),[]);self.assertEqual(sorted(flat),list(range(n)))
    def test_coalescing_model(self):
        self.assertEqual(r.sectors(np.arange(32)*4),4)
        self.assertEqual(r.sectors(4+np.arange(32)*4),5)
        self.assertEqual(r.sectors(np.arange(32)*32),32)
    def test_bank_broadcast_and_padding(self):
        self.assertEqual(r.bank_conflicts([0]*32),1);self.assertEqual(r.bank_conflicts(np.arange(32)*32),32)
        self.assertEqual(r.bank_conflicts(np.arange(32)*33),1)
    def test_tree_sum_odd(self):
        for n in [0,1,3,19]:self.assertEqual(r.tree_sum(np.arange(n)),sum(range(n)))
    def test_softmax_stability(self):
        y=r.stable_softmax([1000,1001,1002]);self.assertAlmostEqual(y.sum(),1)
        np.testing.assert_allclose(y,r.stable_softmax([0,1,2]))
    def test_softmax_mask_empty_row(self):
        y=r.stable_softmax([[1,2],[3,4]],[[1,0],[0,0]])
        np.testing.assert_array_equal(y,[[1,0],[0,0]])
    def test_softmax_gradient(self):
        x=self.rng.normal(size=(2,5));g=self.rng.normal(size=x.shape)
        numeric=r.finite_difference(lambda z:float((r.stable_softmax(z)*g).sum()),x)
        np.testing.assert_allclose(r.softmax_backward(r.stable_softmax(x),g),numeric,atol=1e-9)
    def test_rmsnorm_gradients(self):
        x=self.rng.normal(size=(2,3));w=self.rng.normal(size=3);g=self.rng.normal(size=x.shape)
        dx,dw=r.rmsnorm_backward(x,w,g)
        np.testing.assert_allclose(dx,r.finite_difference(lambda z:float((r.rmsnorm(z,w)*g).sum()),x),atol=1e-9)
        np.testing.assert_allclose(dw,r.finite_difference(lambda z:float((r.rmsnorm(x,z)*g).sum()),w),atol=1e-9)
    def test_rmsnorm_zero(self):np.testing.assert_array_equal(r.rmsnorm(np.zeros((2,3)),np.ones(3)),np.zeros((2,3)))
    def test_gemm_nonmultiples(self):
        for m,k,n in [(1,1,1),(3,7,5),(17,3,9)]:
            a=self.rng.normal(size=(m,k));b=self.rng.normal(size=(k,n))
            for tile in [1,4,16]:np.testing.assert_allclose(r.tiled_matmul(a,b,tile),a@b,atol=1e-12)
    def test_online_attention_equivalence(self):
        for n,d,dv in [(1,1,1),(7,3,5),(17,8,3)]:
            q=self.rng.normal(size=(n,d));k=self.rng.normal(size=(n,d));v=self.rng.normal(size=(n,dv))
            for block in [1,3,8,32]:
                for causal in [False,True]:np.testing.assert_allclose(r.online_attention(q,k,v,block,causal),r.dense_attention(q,k,v,causal),atol=1e-12)
    def test_attention_causal_noninterference(self):
        q=self.rng.normal(size=(7,3));k=q.copy();v=self.rng.normal(size=(7,2));a=r.online_attention(q,k,v,2,True)
        k[6]+=100;v[6]+=100;b=r.online_attention(q,k,v,2,True)
        np.testing.assert_allclose(a[:6],b[:6])
    def test_online_attention_numerical_range(self):
        q=np.array([[1000.],[-1000.]]);k=np.array([[1000.],[-1000.]]);v=np.array([[2.],[3.]])
        np.testing.assert_allclose(r.online_attention(q,k,v,1),[[2.],[3.]])
    def test_attention_rectangular(self):
        q=self.rng.normal(size=(3,4));k=self.rng.normal(size=(7,4));v=self.rng.normal(size=(7,2))
        np.testing.assert_allclose(r.online_attention(q,k,v,3),r.dense_attention(q,k,v),atol=1e-12)
        with self.assertRaises(ValueError):r.online_attention(q,k,v,3,True)
    def test_quantize_zero_and_bound(self):
        x=np.array([[0,0,0],[-1,.13,1.]])
        q,s=r.quantize_int8_rows(x)
        self.assertTrue(np.all(np.abs(q*s-x)<=s/2+1e-12));self.assertEqual(s[0,0],1)
    def test_int4_all_values_odd_empty(self):
        for x in [np.arange(-8,8),np.array([-8,0,7]),np.array([],dtype=int)]:
            p,n=r.pack_int4(x);np.testing.assert_array_equal(r.unpack_int4(p,n),x)
        with self.assertRaises(ValueError):r.pack_int4(np.array([8]))
    def test_fp4_separate_from_int4(self):
        self.assertEqual(r.decode_e2m1([7])[0],6);self.assertEqual(r.decode_e2m1([15])[0],-6)
        self.assertTrue(np.signbit(r.decode_e2m1([8])[0]))
    def test_int8_accumulation_not_int8(self):
        a=np.ones((1,16));b=np.ones((16,1));np.testing.assert_allclose(r.int8_matmul(a,b),[[16]])
    def test_moe_duplicate_routes_scatter_add(self):
        x=np.array([[1.,2.],[3.,4.]])
        w=np.array([np.eye(2),2*np.eye(2)]);ids=np.array([[0,1],[1,1]]);g=np.array([[.25,.75],[.4,.6]])
        y,c=r.grouped_experts(x,ids,w,g)
        np.testing.assert_allclose(y,[[1.75,3.5],[6,8]]);np.testing.assert_array_equal(c,[1,3])
    def test_pipeline_fill_drain(self):
        self.assertEqual(r.pipeline_time(3,4,8),28);self.assertEqual(r.pipeline_time(0,4,8),0)
    def test_ring_one_rank(self):self.assertEqual(r.ring_allreduce_seconds(1,100,1e-6,1e9),0)
    def test_roofline_units(self):
        a=r.Roofline(2e9,1e9,1e12,1e11);self.assertEqual(a.lower_bound_s,.01);self.assertEqual(a.bottleneck,'memory')
    def test_all_labs_execute(self):
        for i in range(1,29):self.assertIsInstance(experiment(i),dict)

if __name__=='__main__':unittest.main()
