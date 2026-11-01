"""Small explicit ranking metrics; duplicate hits never earn extra relevance."""
import math


def unique(ids):
    return list(dict.fromkeys(ids))


def metrics(ranked_ids: list[str], relevance: dict[str,int], k: int):
    if k <= 0 or any(v < 0 for v in relevance.values()):
        raise ValueError("positive k and nonnegative relevance required")
    ranked = unique(ranked_ids)[:k]
    relevant = {d for d,g in relevance.items() if g > 0}
    binary = [int(d in relevant) for d in ranked]
    rr = next((1/(i+1) for i,x in enumerate(binary) if x),0.0)
    dcg = sum((2**relevance.get(d,0)-1)/math.log2(i+2) for i,d in enumerate(ranked))
    ideal = sorted(relevance.values(),reverse=True)[:k]
    idcg = sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
    return {"precision":sum(binary)/k,"recall":sum(binary)/len(relevant) if relevant else 0.0,
            "mrr":rr,"ndcg":dcg/idcg if idcg else 0.0}


def paired_delta(before: list[float],after: list[float]):
    if len(before) != len(after) or not before:
        raise ValueError("nonempty paired lists required")
    changes = [b-a for a,b in zip(before,after)]
    return {"mean_delta":sum(changes)/len(changes),"wins":sum(x>0 for x in changes),
            "losses":sum(x<0 for x in changes),"ties":sum(x==0 for x in changes)}
