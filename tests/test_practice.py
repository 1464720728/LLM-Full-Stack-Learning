import unittest
from practice import solutions
from practice.grade import grade, CHECKS

class PracticeReferenceTest(unittest.TestCase):
    def test_reference_satisfies_all_student_contracts(self):
        for task in CHECKS:
            with self.subTest(task=task):
                result=grade(solutions,[task])[0]
                self.assertTrue(result['passed'],result)
    def test_unfinished_starter_has_actionable_feedback(self):
        from practice import starter
        result=grade(starter,['F-P1'])[0]
        self.assertFalse(result['passed'])
        self.assertIn('NotImplementedError',result['error'])

if __name__=='__main__':unittest.main()
