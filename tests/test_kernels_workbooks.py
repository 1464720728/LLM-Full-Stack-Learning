import unittest
from labs.kernels.workbook import softmax_trace,gemm_trace,attention_trace
class WorkbookTests(unittest.TestCase):
    def test_softmax_padding_fault(self):
        x=softmax_trace();self.assertTrue(x['fault_detected']);self.assertEqual(x['padding_probability'],0);self.assertAlmostEqual(sum(x['probabilities']),1)
    def test_gemm_expected(self):
        x=gemm_trace();self.assertEqual(x['reference'],[[4,5],[10,11]]);self.assertTrue(x['fault_detected']);self.assertEqual(x['trace'][-1]['valid_k'],1)
    def test_attention_exact_toy(self):
        x=attention_trace();self.assertAlmostEqual(x['correct'][0][0],170/7);self.assertTrue(x['fault_detected'])
if __name__=='__main__':unittest.main()
