import unittest
import numpy as np
from labs.models.core import TinyDecoder, DecoderConfig
from projects.incremental_runtime import OnlineDecoder, IncrementalRunner, demo


class IncrementalIntegrationTests(unittest.TestCase):
    def test_online_replacement_matches_dense_full_and_cached(self):
        for kv_heads in (1, 2, 4):
            config = DecoderConfig(kv_heads=kv_heads, seed=17)
            reference, online = TinyDecoder(config), OnlineDecoder(config)
            tokens = [1, 5, 3, 7, 2, 9, 4]
            expected = reference.forward(tokens)
            np.testing.assert_allclose(online.forward(tokens), expected, rtol=1e-11, atol=1e-11)
            cache, values = None, []
            for token in tokens:
                logits, cache = online.step(token, cache)
                values.append(logits)
            np.testing.assert_allclose(values, expected, rtol=1e-11, atol=1e-11)

    def test_interleaving_preserves_outputs_and_releases_cache(self):
        for online in (False, True):
            for policy in ("fifo", "round_robin"):
                result = demo(online, policy)
                self.assertTrue(result["matches_full_context_reference"])
                self.assertEqual(result["initial"]["A"]["cache_lengths"], [2, 2])
                self.assertEqual(result["initial"]["A"]["cache_bytes"], 512)
                for record in result["final"].values():
                    self.assertEqual(record["status"], "finished")
                    self.assertEqual(record["cache_bytes"], 0)

    def test_cancelled_state_does_not_leak_into_other_or_new_request(self):
        runner = IncrementalRunner()
        runner.start("cancel", [1, 2], 4)
        runner.start("keep", [2, 3], 2)
        runner.advance("cancel")
        self.assertTrue(runner.cancel("cancel"))
        self.assertFalse(runner.cancel("cancel"))
        with self.assertRaises(ValueError):
            runner.advance("cancel")
        self.assertEqual(runner.snapshot()["cancel"]["cache_bytes"], 0)
        runner.run()
        runner.start("new", [1, 2], 1)
        runner.run()
        expected = int(np.argmax(runner.model.forward([1, 2])[-1]))
        self.assertEqual(runner.requests["new"].output, [expected])
        self.assertEqual(len(runner.requests["cancel"].output), 1)

    def test_round_robin_retains_order_when_a_request_finishes(self):
        runner = IncrementalRunner()
        for rid, count in (("A", 1), ("B", 2), ("C", 2)):
            runner.start(rid, [1, 2], count)
        trace = runner.run("round_robin", max_steps=5)
        self.assertEqual([row["request"] for row in trace], ["A", "B", "C", "B", "C"])
        self.assertTrue(all(r.status == "finished" for r in runner.requests.values()))

    def test_rejected_start_and_zero_length_generation_have_no_cache(self):
        runner = IncrementalRunner()
        for prompt in ([], [True], [-1], [32]):
            with self.assertRaises(ValueError):
                runner.start("bad", prompt)
        self.assertEqual(runner.requests, {})
        runner.start("zero", [1], 0)
        self.assertEqual(runner.run(max_steps=0), [])
        self.assertEqual(runner.snapshot()["zero"]["cache_bytes"], 0)
        with self.assertRaises(ValueError):
            runner.start("zero", [2])


if __name__ == "__main__":
    unittest.main()
