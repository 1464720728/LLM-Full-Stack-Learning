import unittest
from projects import evidence_workshop as p1


class EvidenceWalkthroughTests(unittest.TestCase):
    def test_visible_scope_and_rankings_use_authorized_corpus(self):
        rows = p1.scope()
        self.assertEqual([c["doc_id"] for c in rows[0]["visible"]], ["cache"])
        self.assertEqual([c["doc_id"] for c in rows[1]["visible"]], ["cache", "private"])
        self.assertEqual({c["tenant"] for c in rows[2]["visible"]}, {"another-tenant"})
        ranking = p1.rankings()
        self.assertEqual(ranking["visible_documents"], ["cache", "schedule"])
        self.assertEqual(ranking["rrf"][0]["doc_id"], "cache")

    def test_trace_links_calls_and_observations_then_reloads_both_databases(self):
        trace = p1.trace()
        self.assertEqual([row["status"] for row in trace["states"]], ["ready", "tools", "ready", "finished"])
        self.assertEqual(trace["states"][2]["tool_calls"], 1)
        messages = trace["messages"]
        self.assertEqual(messages[1]["tool_calls"][0]["id"], messages[2]["tool_call_id"])
        recovery = p1.resume()
        self.assertEqual(recovery["before_close"], recovery["after_reload"])
        self.assertEqual(recovery["after_finish"]["tool_calls"], 1)
        self.assertEqual(recovery["ledger_rows"], [("search-1", "done")])
        self.assertTrue(recovery["result"]["accepted"])

    def test_stale_history_stops_before_model_and_final_claims_are_checked(self):
        for mutation in ("delete", "update", "purge_recreate"):
            with self.subTest(mutation=mutation):
                result = p1.change_after_retrieval(mutation)
                self.assertEqual(result["run_status"], "error")
                self.assertFalse(result["result"]["accepted"])
                self.assertEqual(result["result"]["claims"], [])
                self.assertEqual(result["result"]["reason"], "run_not_finished")
                if mutation != "delete":
                    self.assertGreater(result["current_revisions"][0], result["old_revision"])
        forged = p1.forged_identity()
        self.assertEqual(forged["tool_outcome"]["error"]["code"], "invalid_arguments")
        self.assertTrue(forged["result"]["abstain"])
        altered = p1.forged_claim()
        self.assertTrue(altered["before"]["accepted"])
        self.assertFalse(altered["after"]["accepted"])


if __name__ == "__main__":
    unittest.main()
