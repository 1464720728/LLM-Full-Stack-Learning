"""Strict extractive RAG: no external LLM, no unsupported synthesized claims."""
from dataclasses import dataclass
import json
from .store import CorpusStore,Context
from .retrieval import Hit,hybrid


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class Claim:
    text: str
    citation: Citation


def pack_context(hits: list[Hit],max_chars=1200):
    """JSON data payload measured in characters. Not a model-token budget."""
    if max_chars<2:
        raise ValueError("at least two characters required for []")
    entries=[]
    for hit in hits:
        c=hit.chunk
        item={"source_id":c.chunk_id,"doc_id":c.doc_id,"revision":c.revision,"text":c.text}
        candidate=json.dumps(entries+[item],ensure_ascii=False,separators=(",",":"))
        if len(candidate)<=max_chars:
            entries.append(item)
    return json.dumps(entries,ensure_ascii=False,separators=(",",":"))


def validate_claim(store: CorpusStore,ctx: Context,claim: Claim):
    c=store.get_visible_chunk(ctx,claim.citation.chunk_id)
    if c is None:
        return False,"source_missing_stale_or_forbidden"
    if not claim.citation.quote.strip() or claim.citation.quote not in c.text:
        return False,"quote_not_in_source"
    if claim.text != claim.citation.quote:
        return False,"nonextractive_claim_requires_semantic_review"
    return True,"verified_exact_extract"


def extractive_answer(store,ctx,query,k=3,max_chars=1200):
    packed=json.loads(pack_context(hybrid(store,ctx,query,k),max_chars))
    return [Claim(item['text'],Citation(item['source_id'],item['text'])) for item in packed]


def answer_report(store,ctx,query,k=3,max_chars=1200):
    claims=extractive_answer(store,ctx,query,k,max_chars)
    verified=[]
    for claim in claims:
        ok,reason=validate_claim(store,ctx,claim)
        if ok:
            c=store.get_visible_chunk(ctx,claim.citation.chunk_id)
            verified.append({"text":claim.text,"doc_id":c.doc_id,"revision":c.revision,
                             "source":c.source,"chunk_id":c.chunk_id,"validation":reason})
    return {"query":query,"mode":"strict_extractive","abstain":not verified,"claims":verified,
            "limitation":"Exact-source provenance is verified; relevance and source truth still require judgment."}


def bounded_research(store,ctx,queries,max_queries=3,k=3):
    """Explicit query plan demo; no model-generated actions or hidden unbounded loop."""
    if max_queries<=0:
        raise ValueError("positive query budget required")
    evidence={};events=[];seen=set()
    for q in queries:
        q=q.strip()
        if not q or q in seen:
            continue
        if len(seen)>=max_queries:
            events.append({"event":"stop","reason":"query_budget"});break
        seen.add(q)
        hits=hybrid(store,ctx,q,k)
        for hit in hits:
            evidence[hit.chunk.chunk_id]=hit
        events.append({"event":"retrieve","query":q,"new_total":len(evidence)})
    return list(evidence.values()),events
