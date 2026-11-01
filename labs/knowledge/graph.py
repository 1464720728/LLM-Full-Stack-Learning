"""Evidence-backed directed graph, not Microsoft's full GraphRAG algorithm."""
from collections import deque
from dataclasses import dataclass
from .store import CorpusStore, Context
from .text import digest

PREDICATES={"depends_on","uses","implemented_by","part_of","supersedes","supports"}


@dataclass(frozen=True)
class Edge:
    edge_id: str
    tenant: str
    subject: str
    predicate: str
    object: str
    chunk_id: str
    quote: str


class EvidenceGraph:
    def __init__(self,store: CorpusStore):
        self.store=store

    def add(self,ctx: Context,subject,predicate,obj,chunk_id,quote):
        if not subject.strip() or not obj.strip() or predicate not in PREDICATES:
            raise ValueError("nonempty entity IDs and allowed predicate required")
        chunk=self.store.get_visible_chunk(ctx,chunk_id)
        if chunk is None or not quote.strip() or quote not in chunk.text:
            raise ValueError("source must be current, visible, and contain the quote")
        # Substring validation proves provenance, NOT semantic truth of the relation.
        key=digest(ctx.tenant,subject,predicate,obj,chunk_id,quote)
        with self.store.conn:
            self.store.conn.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?,?,?,?,?)",(key,ctx.tenant,subject,predicate,obj,chunk_id,quote))
        return key

    def visible_edges(self,ctx):
        allowed={c.chunk_id for c in self.store.visible(ctx)}
        rows=self.store.conn.execute("SELECT * FROM edges WHERE tenant=? ORDER BY edge_id",(ctx.tenant,)).fetchall()
        return [Edge(**dict(r)) for r in rows if r['chunk_id'] in allowed]

    def path(self,ctx,start,target,max_depth=3,max_expansions=100,predicates=None):
        if max_depth<0 or max_expansions<=0:
            raise ValueError("invalid traversal budget")
        if start==target:
            return []
        adjacency={}
        for e in self.visible_edges(ctx):
            if predicates is None or e.predicate in predicates:
                adjacency.setdefault(e.subject,[]).append(e)
        queue=deque([(start,[])])
        visited={start};expansions=0
        while queue and expansions<max_expansions:
            node,path=queue.popleft()
            expansions+=1
            if len(path)>=max_depth:
                continue
            for e in adjacency.get(node,[]):
                candidate=path+[e]
                if e.object==target:
                    return candidate
                if e.object not in visited:
                    visited.add(e.object);queue.append((e.object,candidate))
        return None
