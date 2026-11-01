"""Inspectable lexical retrieval. TF-IDF vectors here are NOT learned semantic embeddings."""
from collections import Counter
from dataclasses import dataclass
import math
from .store import Chunk, CorpusStore, Context
from .text import tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def _rank(chunks, scores, k):
    if k < 0:
        raise ValueError("k must be nonnegative")
    pairs = [Hit(c,float(s)) for c,s in zip(chunks,scores) if s > 0 and math.isfinite(s)]
    return sorted(pairs,key=lambda h:(-h.score,h.chunk.chunk_id))[:k]


class BM25:
    def __init__(self, chunks: list[Chunk], k1=1.2, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 > 0 and 0 <= b <= 1")
        self.chunks, self.k1, self.b = list(chunks), k1, b
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.lengths)/len(chunks) if chunks else 0.0
        self.df = Counter(t for counts in self.tf for t in counts)

    def idf(self, token):
        n, df = len(self.chunks), self.df.get(token,0)
        return math.log1p((n-df+0.5)/(df+0.5)) if n else 0.0

    def contributions(self, query: str, index: int):
        out = {}
        for term in sorted(set(tokenize(query))):
            f = self.tf[index].get(term,0)
            if not f or not self.avgdl:
                continue
            norm = self.k1*(1-self.b+self.b*self.lengths[index]/self.avgdl)
            out[term] = self.idf(term)*f*(self.k1+1)/(f+norm)
        return out

    def search(self, query: str, k=5):
        return _rank(self.chunks,[sum(self.contributions(query,i).values()) for i in range(len(self.chunks))],k)


def cosine(a: dict[str,float], b: dict[str,float]) -> float:
    den = math.sqrt(sum(x*x for x in a.values())*sum(x*x for x in b.values()))
    return sum(v*b.get(t,0) for t,v in a.items())/den if den else 0.0


class TfidfVectors:
    """Exact sparse-vector baseline; a teaching stand-in for the vector interface only."""
    def __init__(self,chunks):
        self.chunks = list(chunks)
        self.df = Counter(t for c in chunks for t in set(tokenize(c.text)))
        self.vectors = [self.encode(c.text) for c in chunks]

    def encode(self,text):
        tf = Counter(tokenize(text))
        return {t:(1+math.log(f))*(math.log((1+len(self.chunks))/(1+self.df[t]))+1)
                for t,f in tf.items() if t in self.df}

    def search(self,query,k=5):
        q = self.encode(query)
        return _rank(self.chunks,[cosine(q,v) for v in self.vectors],k)


def rrf(rankings: list[list[Hit]], k=5, rank_constant=60):
    if rank_constant < 0 or k < 0:
        raise ValueError("nonnegative constants required")
    totals, chunks = Counter(), {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking,1):
            key = hit.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            chunks[key] = hit.chunk
            totals[key] += 1/(rank_constant+rank)
    return sorted([Hit(chunks[i],s) for i,s in totals.items()],key=lambda h:(-h.score,h.chunk.chunk_id))[:k]


def hybrid(store: CorpusStore,ctx: Context,query: str,k=5):
    # Visibility before statistics, vectors, scores, or context assembly.
    chunks = store.visible(ctx)
    return rrf([BM25(chunks).search(query,max(k*3,10)),TfidfVectors(chunks).search(query,max(k*3,10))],k)


def lexical_rerank(query: str,hits: list[Hit],k=5):
    """Transparent phrase/coverage heuristic, explicitly NOT a neural cross-encoder."""
    terms = set(tokenize(query))
    def score(h):
        coverage = len(terms & set(tokenize(h.chunk.text)))/max(1,len(terms))
        phrase = float(bool(query.strip()) and query.lower() in h.chunk.text.lower())
        return coverage + 0.25*phrase
    return _rank([h.chunk for h in hits],[score(h) for h in hits],k)


def mmr(query, hits, k=3, weight=0.7):
    if not 0 <= weight <= 1 or k < 0:
        raise ValueError("invalid MMR parameters")
    index = TfidfVectors([h.chunk for h in hits])
    q, chosen, remaining = index.encode(query), [], list(range(len(hits)))
    while remaining and len(chosen) < k:
        def objective(i):
            redundancy = max((cosine(index.vectors[i],index.vectors[j]) for j in chosen),default=0)
            return weight*cosine(q,index.vectors[i])-(1-weight)*redundancy
        best = min(remaining,key=lambda i:(-objective(i),hits[i].chunk.chunk_id))
        chosen.append(best)
        remaining.remove(best)
    return [hits[i] for i in chosen]
