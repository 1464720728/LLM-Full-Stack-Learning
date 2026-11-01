"""Observable stages for P1. Offline fixtures only; no model/network calls."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from labs.agents.contracts import ToolCall, Turn
from labs.agents.runtime import ScriptedModel
from labs.agents.store import Store
from labs.knowledge.retrieval import BM25, TfidfVectors, rrf
from labs.knowledge.store import Context, CorpusStore
from projects.evidence_agent import finalize, make_harness, seed


def state_row(state):
    return {"status": state.status, "version": state.version,
            "model_calls": state.model_calls, "tool_calls": state.tool_calls,
            "roles": [m["role"] for m in state.messages],
            "pending_ids": [call["id"] for call in state.pending],
            "events": [event["kind"] for event in state.events]}


def scope():
    with CorpusStore() as corpus:
        seed(corpus)
        contexts = [Context("demo", "reader"), Context("demo", "owner"),
                    Context("another-tenant", "reader")]
        return [{"tenant": ctx.tenant, "actor": ctx.actor,
                 "principals": ctx.principals,
                 "visible": [{"tenant": chunk.tenant, "doc_id": chunk.doc_id,
                              "revision": chunk.revision}
                             for chunk in corpus.visible(ctx)]}
                for ctx in contexts]


def rankings():
    with CorpusStore() as corpus:
        seed(corpus)
        corpus.ingest("demo", "schedule", "Scheduler note",
                      "A scheduler selects requests. A cache can retain useful state.", allowed=("*",))
        chunks = corpus.visible(Context("demo", "reader"))
        lexical, vector = BM25(chunks), TfidfVectors(chunks)
        query = "KV cache"
        bm, tf = lexical.search(query, 5), vector.search(query, 5)
        def rows(hits):
            return [{"doc_id": hit.chunk.doc_id, "score": hit.score} for hit in hits]
        return {"query": query, "visible_documents": [chunk.doc_id for chunk in chunks],
                "bm25_terms": [{"doc_id": chunk.doc_id,
                                "contributions": lexical.contributions(query, i)}
                               for i, chunk in enumerate(chunks)],
                "bm25": rows(bm), "tfidf_cosine": rows(tf), "rrf": rows(rrf([bm, tf], 5))}


def trace():
    with CorpusStore() as corpus, Store() as ledger:
        seed(corpus)
        context = Context("demo", "reader")
        harness = make_harness(corpus, context, ledger)
        state = harness.new("trace", "KV cache")
        rows = [state_row(state)]
        for _ in range(3):
            harness.step(state)
            rows.append(state_row(state))
        return {"states": rows, "messages": deepcopy(state.messages),
                "result": finalize(state, corpus, context)}


def change_after_retrieval(kind):
    if kind not in {"delete", "update", "purge_recreate"}:
        raise ValueError("unknown mutation scenario")
    with CorpusStore() as corpus, Store() as ledger:
        seed(corpus)
        context = Context("demo", "reader")
        harness = make_harness(corpus, context, ledger)
        state = harness.new("change", "KV cache")
        harness.step(state)
        harness.step(state)
        evidence = json.loads(state.messages[-1]["content"])["value"]
        old_revision = evidence[0]["revision"]
        if kind == "delete":
            corpus.delete("demo", "cache", expected_revision=old_revision)
        elif kind == "update":
            corpus.ingest("demo", "cache", "Revised cache note",
                          "KV cache contains past projected keys and values.", allowed=("*",))
        else:
            corpus.purge("demo", "cache")
            corpus.ingest("demo", "cache", "KV cache primer", evidence[0]["text"], allowed=("*",))
        harness.run(state)
        return {"mutation": kind, "old_revision": old_revision,
                "current_revisions": [c.revision for c in corpus.visible(context) if c.doc_id == "cache"],
                "run_status": state.status, "result": finalize(state, corpus, context)}


def forged_identity():
    model = ScriptedModel([
        Turn(calls=[ToolCall("forged", "search_evidence", '{"query":"KV","actor":"owner"}')]),
        Turn(text='{"claims":[],"abstain":true}'),
    ])
    with CorpusStore() as corpus, Store() as ledger:
        seed(corpus)
        context = Context("demo", "reader")
        harness = make_harness(corpus, context, ledger, model)
        state = harness.new("forged", "KV cache")
        harness.run(state)
        outcome = next(m for m in state.messages if m["role"] == "tool")
        return {"tool_outcome": json.loads(outcome["content"]),
                "result": finalize(state, corpus, context)}


def forged_claim():
    with CorpusStore() as corpus, Store() as ledger:
        seed(corpus)
        context = Context("demo", "reader")
        harness = make_harness(corpus, context, ledger)
        state = harness.new("forged-claim", "KV cache")
        harness.run(state)
        good = finalize(state, corpus, context)
        answer = json.loads(state.final)
        answer["claims"][0]["text"] = "Every KV cache can be shared across all users and all models."
        state.final = json.dumps(answer)
        return {"before": good, "after": finalize(state, corpus, context)}


def resume():
    with tempfile.TemporaryDirectory() as directory:
        corpus_path = Path(directory) / "corpus.sqlite3"
        ledger_path = Path(directory) / "ledger.sqlite3"
        context = Context("demo", "reader")
        with CorpusStore(corpus_path) as corpus, Store(ledger_path) as ledger:
            seed(corpus)
            harness = make_harness(corpus, context, ledger)
            state = harness.new("resume", "KV cache")
            harness.step(state)
            harness.step(state)
            before = state_row(state)
        # Both connections close; reconstruct every service object from disk.
        with CorpusStore(corpus_path) as corpus, Store(ledger_path) as ledger:
            harness = make_harness(corpus, context, ledger)
            restored = harness.resume("resume")
            loaded = state_row(restored)
            harness.run(restored)
            ledger_rows = ledger.db.execute(
                "SELECT call_id,status FROM tool_ledger WHERE run_id=?", ("resume",)).fetchall()
            return {"before_close": before, "after_reload": loaded,
                    "after_finish": state_row(restored), "ledger_rows": ledger_rows,
                    "result": finalize(restored, corpus, context)}


def authorization_boundary():
    """A rejected final answer cannot undo a model-input disclosure."""
    class Observer:
        def __init__(self):
            self.calls = 0
        def complete(self, messages, tools, index):
            self.calls += 1
            return Turn(text='{"claims":[],"abstain":true}')

    with CorpusStore() as corpus, Store() as ledger:
        seed(corpus)
        owner = make_harness(corpus, Context("demo", "owner"), ledger)
        state = owner.new("bound-owner", "KV cache")
        owner.step(state)
        owner.step(state)
        assert "owner-only" in state.messages[-1]["content"]
        observer = Observer()
        reader = make_harness(corpus, Context("demo", "reader"), ledger, observer)
        try:
            reader.resume("bound-owner")
        except PermissionError:
            identity_rejected = True
        else:
            raise AssertionError("a run was rebound to another principal")
        assert observer.calls == 0

    with CorpusStore() as corpus, Store() as ledger:
        seed(corpus)
        context = Context("demo", "reader")
        first = make_harness(corpus, context, ledger)
        state = first.new("revoked-reader", "KV cache")
        first.step(state)
        first.step(state)
        prior_calls = state.model_calls
        corpus.ingest("demo", "cache", "Revoked cache note",
                      "KV cache stores past keys and values.", allowed=("user:owner",))
        observer = Observer()
        resumed = make_harness(corpus, context, ledger, observer)
        state = resumed.resume("revoked-reader")
        resumed.run(state)
        assert state.status == "error"
        assert state.model_calls == prior_calls
        assert observer.calls == 0
        return {"identity_rebinding_rejected": identity_rejected,
                "revoked_history_status": state.status,
                "next_model_calls": observer.calls,
                "reserved_model_calls": state.model_calls,
                "result": finalize(state, corpus, context)}


SCENARIOS = {"scope": scope, "rankings": rankings, "trace": trace,
             "delete": lambda: change_after_retrieval("delete"),
             "update": lambda: change_after_retrieval("update"),
             "purge-recreate": lambda: change_after_retrieval("purge_recreate"),
             "forged-identity": forged_identity, "forged-claim": forged_claim, "resume": resume,
             "auth-boundary": authorization_boundary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=[*SCENARIOS, "all"])
    args = parser.parse_args()
    result = {name: function() for name, function in SCENARIOS.items()} if args.scenario == "all" else SCENARIOS[args.scenario]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
