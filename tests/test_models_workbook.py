import unittest
import numpy as np
from labs.models.trace_decoder import trace,fixture
from labs.models.moe_decoder import TinyMoEDecoder
from labs.models.core import rms_norm

class WorkbookTests(unittest.TestCase):
    def test_trace_matches_reference_in_both_modes(self):
        for active in [False,True]:
            stages=trace(active); self.assertEqual(stages['25_logits'].shape,(2,5)); self.assertEqual(stages['14_probability'][0,0,1],0)
    def test_identity_fixture_head_symmetry(self):
        stages=trace(); np.testing.assert_allclose(stages['15_head_outputs'][0],stages['15_head_outputs'][1]); self.assertAlmostEqual(stages['25_logits'][1,4],0)
    def test_active_ffn_changes_function(self):
        self.assertGreater(float(abs(trace(False)['25_logits']-trace(True)['25_logits']).max()),1e-5)
    def test_two_step_fixture_equals_trace(self):
        model=fixture(True); _,s=model.step(0); last,_=model.step(1,s); np.testing.assert_allclose(last,trace(True)['25_logits'][1],atol=1e-12)
    def test_moe_full_step(self):
        m=TinyMoEDecoder(); ids=[1,1,2,4,7]; state=None; actual=[]
        for token in ids:
            z,state=m.step(token,state); actual.append(z)
        np.testing.assert_allclose(m.forward(ids),actual,atol=1e-10,rtol=1e-10)
    def test_moe_prefix_invariance(self):
        m=TinyMoEDecoder(); np.testing.assert_allclose(m.forward([1,2,3])[:2],m.forward([1,2,8])[:2])
    def test_moe_audit_conservation(self):
        m=TinyMoEDecoder(); x=rms_norm(m.embedding[[1,2,3]]); _,a=m.feed_forward(x,0)
        self.assertEqual(int(a['counts'].sum()),3*m.top_k); np.testing.assert_allclose(a['weights'].sum(1),1); self.assertFalse(a['dropped'].any())

if __name__=='__main__': unittest.main()
