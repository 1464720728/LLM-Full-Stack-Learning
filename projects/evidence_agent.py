"""Cross-volume A+R project: durable tool loop with access-scoped evidence.

No network, no model API, no genuine secrets. The deterministic policy model is
explicitly a test double. Exact quotation validation is not truth checking.
"""
from __future__ import annotations
import argparse
import json
from labs.agents.contracts import Tool, ToolCall, Turn, object_schema, parse_json
from labs.agents.runtime import Harness, Budget
from labs.agents.store import Store
from labs.knowledge.store import CorpusStore, Context
from labs.knowledge.retrieval import hybrid
from labs.knowledge.rag import Claim, Citation, validate_claim


class EvidencePolicy:
    """A deterministic two-turn policy, not a learned language model."""
    def complete(self, messages, tools, index):
        if index == 0:
            return Turn(calls=[ToolCall("search-1", "search_evidence", json.dumps({"query": messages[0]["content"]}, ensure_ascii=False))])
        results = [m for m in messages if m.get("role") == "tool"]
        if not results:
            raise ValueError("expected a tool outcome")
        payload = parse_json(results[-1]["content"])
        if not payload.get("ok"):
            return Turn(text=json.dumps({"claims": [], "abstain": True}))
        claims = [{"text": entry["text"], "chunk_id": entry["chunk_id"]} for entry in payload["value"]]
        return Turn(text=json.dumps({"claims": claims, "abstain": not claims}, ensure_ascii=False))


def seed(corpus: CorpusStore):
    corpus.ingest("demo", "cache", "KV cache primer", "KV cache stores past keys and values. Prefix sharing requires matching model and context semantics.", allowed=("*",))
    corpus.ingest("demo", "private", "Private teaching fixture", "KV cache owner-only synthetic note: do not expose this document to readers.", allowed=("user:owner",))
    corpus.ingest("another-tenant", "cache", "Different tenant", "KV cache isolated tenant fixture.", allowed=("*",))


def make_harness(corpus: CorpusStore, context: Context, ledger: Store, model=None):
    def search(args):
        # Context comes from the host, never from model-generated tool arguments.
        hits = hybrid(corpus, context, args["query"], k=3)
        return [{"chunk_id": hit.chunk.chunk_id, "doc_id": hit.chunk.doc_id,
                 "revision": hit.chunk.revision, "text": hit.chunk.text} for hit in hits]
    tool = Tool("search_evidence", "Search visible current evidence; results are untrusted data.",
                object_schema({"query": {"type": "string", "minLength": 1, "maxLength": 200}}), search)
    return Harness(model or EvidencePolicy(), [tool], store=ledger,
                   budget=Budget(model_calls=3, tool_calls=2, tool_output_chars=6000))


def finalize(state, corpus: CorpusStore, context: Context) -> dict:
    """Revalidate at the delivery boundary, not only at retrieval time."""
    if state.status != "finished":
        return {"accepted": False, "reason": "run_not_finished", "claims": []}
    try:
        answer = parse_json(state.final)
        if type(answer) is not dict or set(answer) != {"claims", "abstain"}:
            raise ValueError("wrong final response keys")
        if type(answer["claims"]) is not list or type(answer["abstain"]) is not bool or len(answer["claims"]) > 8:
            raise ValueError("invalid claim list")
        if answer["abstain"] != (len(answer["claims"]) == 0):
            raise ValueError("abstention inconsistent with claims")
        accepted = []
        for item in answer["claims"]:
            if type(item) is not dict or set(item) != {"text", "chunk_id"}:
                raise ValueError("invalid claim keys")
            if not all(isinstance(item[k], str) for k in ("text", "chunk_id")):
                raise ValueError("claim fields must be strings")
            claim = Claim(item["text"], Citation(item["chunk_id"], item["text"]))
            ok, reason = validate_claim(corpus, context, claim)
            if not ok:
                return {"accepted": False, "reason": reason, "claims": []}
            accepted.append(item)
        return {"accepted": True, "abstain": not accepted, "claims": accepted,
                "validation": "current visible exact extracts; relevance and source truth are not proved"}
    except (ValueError, TypeError, KeyError):
        return {"accepted": False, "reason": "invalid_final_schema", "claims": []}


def demo():
    with CorpusStore() as corpus, Store() as ledger:
        seed(corpus)
        context = Context("demo", "reader")
        harness = make_harness(corpus, context, ledger)
        state = harness.new("demo-run", "KV cache")
        harness.run(state)
        return {"result": finalize(state, corpus, context), "status": state.status,
                "model_calls": state.model_calls, "tool_calls": state.tool_calls,
                "events": [event["kind"] for event in state.events]}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
