import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from projects.evidence_agent import seed, make_harness, finalize, demo
from labs.agents.contracts import ToolCall, Turn
from labs.agents.runtime import ScriptedModel, Harness
from labs.agents.store import Store
from labs.knowledge.store import CorpusStore, Context


class ObservingModel:
    def __init__(self):
        self.calls = []

    def complete(self, messages, tools, index):
        self.calls.append(deepcopy(messages))
        return Turn(text='{"claims":[],"abstain":true}')


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

    def searched_state(self, run_id, context=None):
        harness = make_harness(self.corpus, context or self.ctx, self.ledger)
        state = harness.new(run_id, "KV cache")
        harness.step(state)
        harness.step(state)
        return harness, state

    def test_run_binding_is_persisted_without_an_extra_checkpoint(self):
        context = Context("demo", "reader", ("writers", "readers", "writers"))
        harness = make_harness(self.corpus, context, self.ledger)
        state = harness.new("bound", "KV cache")
        self.assertEqual(state.version, 1)
        self.assertEqual(state.events[0]["data"]["evidence_context"], {
            "tenant": "demo", "actor": "reader", "groups": ["readers", "writers"]})
        equivalent = make_harness(self.corpus,
            Context("demo", "reader", ("readers", "writers")), self.ledger)
        self.assertEqual(equivalent.resume("bound").version, 1)

    def test_mutating_callers_group_list_cannot_expand_search_scope(self):
        caller_groups = []
        context = Context("demo", "reader", caller_groups)
        self.corpus.ingest("demo", "group-private", "Group private", "KV cache group-only fixture",
                           allowed=("group:admins",))
        harness = make_harness(self.corpus, context, self.ledger)
        state = harness.new("mutable-groups", "KV cache")
        caller_groups.append("admins")
        harness.run(state)
        self.assertEqual(state.events[0]["data"]["evidence_context"]["groups"], [])
        self.assertNotIn("group-only", json.dumps(state.messages))
        self.assertTrue(finalize(state, self.corpus, self.ctx)["accepted"])

    def test_identity_rebinding_is_rejected_before_returning_a_resume(self):
        _, state = self.searched_state("owner-run", Context("demo", "owner"))
        self.assertIn("owner-only", json.dumps(state.messages))
        for context in (self.ctx, Context("another-tenant", "owner"),
                        Context("demo", "owner", ("new-group",))):
            with self.subTest(context=context):
                observer = ObservingModel()
                other = make_harness(self.corpus, context, self.ledger, observer)
                with self.assertRaises(PermissionError):
                    other.resume("owner-run")
                with self.assertRaises(PermissionError):
                    other.run(state)
                with self.assertRaises(PermissionError):
                    other.step(state)
                with self.assertRaises(PermissionError):
                    other.cancel(state)
                self.assertEqual(observer.calls, [])
        self.assertEqual(self.ledger.load("owner-run")["version"], state.version)

    def test_old_unbound_run_is_rejected_instead_of_assuming_current_user(self):
        legacy = Harness(ObservingModel(), [], store=self.ledger)
        state = legacy.new("legacy", "old goal")
        current = make_harness(self.corpus, self.ctx, self.ledger)
        with self.assertRaises(PermissionError):
            current.resume("legacy")
        with self.assertRaises(PermissionError):
            current.run(state, max_transitions=0)

    def test_revocation_blocks_next_model_call_and_preserves_controlled_audit(self):
        _, state = self.searched_state("acl-revoked")
        old = next(c for c in self.corpus.visible(self.ctx) if c.doc_id == "cache")
        self.corpus.ingest("demo", "cache", old.title, old.text,
                           allowed=("user:owner",))
        observer = ObservingModel()
        restored_harness = make_harness(self.corpus, self.ctx, self.ledger, observer)
        restored = restored_harness.resume(state.run_id)
        before_calls = restored.model_calls
        restored_harness.run(restored)
        self.assertEqual(restored.status, "error")
        self.assertEqual(restored.model_calls, before_calls)
        self.assertEqual(observer.calls, [])
        self.assertIn(old.text, json.loads(restored.messages[-1]["content"])["value"][0]["text"])
        self.assertEqual(restored.events[-1]["data"]["reason"],
                         "historical evidence invalid before model call")
        self.assertFalse(finalize(restored, self.corpus, self.ctx)["accepted"])

    def test_update_delete_and_recreate_block_before_model_even_with_same_actor(self):
        for change in ("update", "delete", "purge_recreate"):
            with self.subTest(change=change):
                with CorpusStore() as corpus, Store() as ledger:
                    seed(corpus)
                    h = make_harness(corpus, self.ctx, ledger)
                    state = h.new(change, "KV cache")
                    h.step(state)
                    h.step(state)
                    old = next(c for c in corpus.visible(self.ctx) if c.doc_id == "cache")
                    if change == "delete":
                        corpus.delete("demo", "cache")
                    elif change == "update":
                        corpus.ingest("demo", "cache", old.title, old.text + " New version.", allowed=("*",))
                    else:
                        corpus.purge("demo", "cache")
                        corpus.ingest("demo", "cache", old.title, old.text, allowed=("*",))
                    observer = ObservingModel()
                    restored_h = make_harness(corpus, self.ctx, ledger, observer)
                    restored = restored_h.resume(change)
                    restored_h.run(restored)
                    self.assertEqual(restored.status, "error")
                    self.assertEqual(restored.model_calls, 1)
                    self.assertEqual(observer.calls, [])

    def test_every_historical_result_is_checked_not_only_the_latest(self):
        self.corpus.ingest("demo", "fresh", "Fresh", "freshonly unrelated evidence", allowed=("*",))
        model = ScriptedModel([
            Turn(calls=[ToolCall("old-search", "search_evidence", '{"query":"KV"}')]),
            Turn(calls=[ToolCall("new-search", "search_evidence", '{"query":"freshonly"}')]),
        ])
        h = make_harness(self.corpus, self.ctx, self.ledger, model)
        state = h.new("two-history", "Compare evidence")
        for _ in range(4):
            h.step(state)
        observations = [m for m in state.messages if m["role"] == "tool"]
        self.assertEqual(len(observations), 2)
        self.assertEqual(json.loads(observations[-1]["content"])["value"][0]["doc_id"], "fresh")
        self.corpus.delete("demo", "cache")
        observer = ObservingModel()
        newer = make_harness(self.corpus, self.ctx, self.ledger, observer)
        newer.run(state)
        self.assertEqual(state.status, "error")
        self.assertEqual(state.model_calls, 2)
        self.assertEqual(observer.calls, [])

    def test_every_entry_inside_one_search_result_is_revalidated(self):
        owner = Context("demo", "owner")
        _, state = self.searched_state("multiple-entries", owner)
        entries = json.loads(state.messages[-1]["content"])["value"]
        self.assertEqual({entry["doc_id"] for entry in entries}, {"cache", "private"})
        self.corpus.delete("demo", "private")
        observer = ObservingModel()
        guarded = make_harness(self.corpus, owner, self.ledger, observer)
        guarded.run(state)
        self.assertEqual(state.status, "error")
        self.assertEqual(observer.calls, [])

    def test_empty_successful_search_can_still_produce_legal_abstention(self):
        h = make_harness(self.corpus, self.ctx, self.ledger)
        state = h.new("empty-search", "unfindabletermxyz")
        h.run(state)
        self.assertEqual(state.status, "finished")
        result = finalize(state, self.corpus, self.ctx)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["abstain"])

    def test_historical_record_fields_must_match_current_visible_chunk(self):
        for field, value in (("doc_id", "another"), ("revision", True),
                             ("revision", 999), ("text", "forged content"),
                             ("chunk_id", "absent")):
            with self.subTest(field=field, value=value):
                h, state = self.searched_state(f"tampered-{field}-{value}")
                payload = json.loads(state.messages[-1]["content"])
                payload["value"][0][field] = value
                state.messages[-1]["content"] = json.dumps(payload)
                observer = ObservingModel()
                guarded = make_harness(self.corpus, self.ctx, self.ledger, observer)
                guarded.run(state)
                self.assertEqual(state.status, "error")
                self.assertEqual(state.model_calls, 1)
                self.assertEqual(observer.calls, [])

    def test_completed_ledger_pending_snapshot_is_replayed_then_checked(self):
        h = make_harness(self.corpus, self.ctx, self.ledger)
        state = h.new("between-commits", "KV cache")
        h.step(state)
        # Deliberately stop after ledger completion, before transcript save.
        output = h._execute(state, ToolCall(**state.pending[0]))
        self.assertTrue(json.loads(output)["ok"])
        self.assertEqual(self.ledger.load(state.run_id)["tool_calls"], 0)
        self.corpus.delete("demo", "cache")
        observer = ObservingModel()
        recovered = make_harness(self.corpus, self.ctx, self.ledger, observer)
        restored = recovered.resume(state.run_id)
        with patch("projects.evidence_agent.hybrid", side_effect=AssertionError("must use ledger")):
            recovered.run(restored)
        self.assertEqual(restored.status, "error")
        self.assertEqual(restored.tool_calls, 1)
        self.assertEqual(restored.model_calls, 1)
        self.assertEqual(observer.calls, [])
        self.assertEqual(self.ledger.db.execute(
            "SELECT status FROM tool_ledger WHERE run_id=?", (state.run_id,)).fetchall(), [("done",)])

    def test_final_delivery_rechecks_current_evidence_and_bound_identity(self):
        h = make_harness(self.corpus, self.ctx, self.ledger)
        state = h.new("delivery-check", "KV cache")
        h.run(state)
        self.assertTrue(finalize(state, self.corpus, self.ctx)["accepted"])
        mismatch = finalize(state, self.corpus, Context("demo", "owner"))
        self.assertEqual(mismatch["reason"], "run_context_mismatch")
        self.corpus.delete("demo", "cache")
        rejected = finalize(state, self.corpus, self.ctx)
        self.assertEqual(state.status, "finished")
        self.assertEqual(rejected["reason"], "source_missing_stale_or_forbidden")


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
