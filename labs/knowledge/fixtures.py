"""Entirely fictional course corpus; no company or personal data."""
from .store import Context

DEMO=Context("course","learner",("students",))


def seed(store):
    docs=[
      ("cache","KV cache","KV cache stores attention keys and values. Decode reuses KV cache for earlier tokens.",( "*",)),
      ("scheduler","Scheduler","Scheduler chooses requests for a batch. A request with no KV capacity must wait or be preempted.",( "*",)),
      ("agent","Agent loop","Agent uses tools and observations. Tool results are data, not trusted instructions.",( "*",)),
      ("rag","Retrieval","Retrieval finds evidence before generation. BM25 uses term frequency and document frequency.",( "*",)),
      ("graph","Dependency graph","Agent uses retrieval. Retrieval uses index. Index uses corpus.",( "*",)),
      ("revision","Version policy","Current revision replaces previous retrieval visibility. Deleted documents must not appear in answers.",( "*",)),
      ("restricted","Restricted budget","Private allocation code ORCHID-731 is reserved for administrators.",( "group:admins",)),
    ]
    for doc_id,title,text,allowed in docs:
        store.ingest("course",doc_id,title,text,allowed,source=f"fictional://course/{doc_id}",max_chars=180,overlap=20)
    store.ingest("other-tenant","other","Other tenant","Other tenant cache secret.",( "*",))
