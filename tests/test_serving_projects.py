import unittest
from labs.serving.replay import compare, DEFAULT
from labs.serving.mock_server import validate_payload
from labs.serving.experiments import LABS

class ProjectTests(unittest.TestCase):
    def test_replay_outputs_equal(self):
        report=compare(DEFAULT,budgets=(2,8),policies=('fifo','round_robin'))
        self.assertTrue(report['output_equivalence']);self.assertEqual(len(report['runs']),4)
    def test_validation_limits(self):
        for data in [{},[],{'prompt_token_ids':[]},{'prompt_token_ids':[1],'max_tokens':65},{'prompt_token_ids':[True]}]:
            with self.assertRaises((ValueError,TypeError)):validate_payload(data)
        self.assertEqual(validate_payload({'prompt_token_ids':[1,2],'max_tokens':3}).max_new_tokens,3)
    def test_all_offline_labs(self):
        for name,fn in LABS.items():
            with self.subTest(name=name):self.assertIsInstance(fn(),dict)

if __name__=='__main__':unittest.main()
