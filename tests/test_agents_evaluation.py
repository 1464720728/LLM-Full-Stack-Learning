import unittest
from labs.agents.evaluation import *

class EvaluationTests(unittest.TestCase):
    def test_pass_at_k_boundaries(self):
        self.assertEqual(pass_at_k(4,0,2),0)
        self.assertEqual(pass_at_k(4,4,2),1)
        self.assertEqual(pass_at_k(4,1,2),.5)
        self.assertEqual(pass_at_k(4,2,1),.5)
    def test_wilson_nonzero_uncertainty(self):
        low,high=wilson_interval(10,10)
        self.assertLess(low,1);self.assertAlmostEqual(high,1)
    def test_invalid_statistics(self):
        with self.assertRaises(ValueError):wilson_interval(0,0)
        with self.assertRaises(ValueError):pass_at_k(3,4,1)
    def test_paired_identical(self):
        r=paired_bootstrap([0,1,1],[0,1,1],repeats=50)
        self.assertEqual(r['mean_difference'],0);self.assertEqual(r['percentile_95'],[0,0])
    def test_false_claim_fails(self):
        self.assertFalse(run_case(Case('x',2,3,'false_claim'))['success'])
    def test_correct_text_invalid_tool_fails(self):
        self.assertFalse(run_case(Case('x',2,3,'bad_argument'))['success'])
    def test_normal_success(self):
        self.assertTrue(run_case(Case('x',2,3))['success'])
if __name__=='__main__':unittest.main()
