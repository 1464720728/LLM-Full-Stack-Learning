import json
from pathlib import Path
import sys
import tempfile
import unittest
from tools.run_experiment import execute


class ExperimentRunnerTest(unittest.TestCase):
    def test_records_exit_and_hides_gpu_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"run"
            result = execute([sys.executable,"-c","import os; print(repr(os.environ['CUDA_VISIBLE_DEVICES']))"],out,10)
            self.assertEqual(result["exit_code"],0)
            self.assertEqual((out/"stdout.txt").read_text().strip(),"''")
            self.assertEqual(json.loads((out/"result.json").read_text())["visible_gpu"],None)

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            result = execute([sys.executable,"-c","import time; time.sleep(10)"],Path(d)/"run",.1)
            self.assertEqual(result["exit_code"],124)
            self.assertTrue(result["timed_out"])

    def test_failure_code_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            result = execute([sys.executable,"-c","raise SystemExit(7)"],Path(d)/"run",10)
            self.assertEqual(result["exit_code"],7)


if __name__ == "__main__":
    unittest.main()
