import json
from pathlib import Path
import tempfile
import unittest
from projects.evidence_agent import seed, make_harness, finalize, demo
from labs.agents.contracts import ToolCall, Turn
from labs.agents.runtime import ScriptedModel
from labs.agents.store import Store
from labs.knowledge.store import CorpusStore, Context


class EvidenceProjectTest(unittest.TestCase):
    def setUp(self):
        self.corpus = CorpusStore()
        self.ledger = Store()
        self.ctx = Context("demo", "reader")
        seed(self.corpus)

    def tearDown(self):
        self.corpus.close()
        self.ledger.close()

    def test_end_to_end(self):
        result = demo()
        self.assertTrue(result["result"]["accepted"])
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["tool_calls"], 1)
        text = json.dumps(result)
        self.assertNotIn("owner-only", text)
        self.assertNotIn("isolated tenant fixture", text)

    def test_revoked_evidence_between_tool_and_delivery(self):
        h = make_harness(self.corpus, self.ctx, self.ledger)
        s = h.new("revoked", "KV cache")
        h.step(s)
        h.step(s)
        self.corpus.delete("demo", "cache")
        h.run(s)
        result = finalize(s, self.corpus, self.ctx)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["claims"], [])

    def test_unknown_identity_parameter_is_rejected(self):
        m = ScriptedModel([Turn(calls=[ToolCall("x", "search_evidence", '{"query":"KV","actor":"owner"}')]), Turn(text='{"claims":[],"abstain":true}')])
        h = make_harness(self.corpus, self.ctx, self.ledger, m)
        s = h.new("injection", "query")
        h.run(s)
        tool = next(x for x in s.messages if x["role"] == "tool")
        self.assertEqual(json.loads(tool["content"])["error"]["code"], "invalid_arguments")

    def test_persisted_run_resumes_without_reissuing_completed_tool(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.sqlite3"
            with Store(path) as first:
                h = make_harness(self.corpus, self.ctx, first)
                state = h.new("resume", "KV cache")
                h.step(state)
                h.step(state)
            with Store(path) as second:
                h = make_harness(self.corpus, self.ctx, second)
                state = h.resume("resume")
                h.run(state)
                self.assertTrue(finalize(state, self.corpus, self.ctx)["accepted"])
                self.assertEqual(state.tool_calls, 1)

    def test_unsupported_synthesized_statement_rejected(self):
        h = make_harness(self.corpus, self.ctx, self.ledger)
        s = h.new("tamper", "KV cache")
        h.run(s)
        answer = json.loads(s.final)
        answer["claims"][0]["text"] = "Unsupported invented statement."
        s.final = json.dumps(answer)
        self.assertFalse(finalize(s, self.corpus, self.ctx)["accepted"])


class ModelRuntimeProjectTest(unittest.TestCase):
    def test_scheduler_policies_preserve_greedy_results(self):
        from projects.model_runtime import run
        for policy in ("fifo", "decode_first", "round_robin"):
            with self.subTest(policy=policy):
                result = run(policy)
                self.assertTrue(result["matches_direct"])
                self.assertLess(result["independent_attention_max_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
