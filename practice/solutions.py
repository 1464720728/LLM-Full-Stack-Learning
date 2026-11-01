"""Readable reference solutions. Every contract is independently graded."""
import json
import math
import numpy as np


def parse_jsonl(lines):
    records, errors = [], []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError("object required")
            records.append(value)
        except ValueError:
            errors.append(number)
    return records, errors


def merge_counts(left, right):
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0) + value
    return result


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("nonempty values and quantile in [0,1] required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def validate_add_args(value):
    if type(value) is not dict or set(value) != {"a", "b"}:
        raise ValueError("exact keys a,b required")
    if any(type(value[key]) is not int for key in ("a", "b")):
        raise ValueError("integers required, booleans rejected")
    return value["a"] + value["b"]


def merge_tool_fragments(fragments):
    calls = {}
    for fragment in fragments:
        index = fragment["index"]
        if type(index) is not int or index < 0:
            raise ValueError("nonnegative index required")
        current = calls.setdefault(index, {"id": None, "name": None, "arguments": ""})
        for key in ("id", "name"):
            if key in fragment:
                if current[key] is not None and current[key] != fragment[key]:
                    raise ValueError("conflicting identity")
                current[key] = fragment[key]
        current["arguments"] += fragment.get("arguments", "")
    result = []
    for index in sorted(calls):
        call = calls[index]
        if not call["id"] or not call["name"]:
            raise ValueError("incomplete identity")
        try:
            args = json.loads(call["arguments"])
        except ValueError as exc:
            raise ValueError("incomplete arguments") from exc
        if type(args) is not dict:
            raise ValueError("arguments must be object")
        result.append({"id": call["id"], "name": call["name"], "arguments": args})
    return result


def transition(state, event):
    table = {("ready", "tool_call"): "waiting_tool", ("ready", "answer"): "finished",
             ("waiting_tool", "tool_result"): "ready", ("ready", "cancel"): "cancelled",
             ("waiting_tool", "cancel"): "cancelled"}
    try:
        return table[(state, event)]
    except KeyError as exc:
        raise ValueError("illegal transition") from exc


def bm25_term(tf, df, n, length, average_length, k1=1.2, b=0.75):
    if min(tf, df, length) < 0 or n <= 0 or df > n or average_length <= 0 or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("invalid BM25 inputs")
    if tf == 0:
        return 0.0
    idf = math.log(1+(n-df+0.5)/(df+0.5))
    norm = k1*(1-b+b*length/average_length)
    return idf*tf*(k1+1)/(tf+norm)


def reciprocal_rank_fusion(rankings, k, constant=60):
    if k < 0 or constant < 0:
        raise ValueError("nonnegative parameters required")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, identity in enumerate(ranking, 1):
            if identity in seen:
                continue
            seen.add(identity)
            scores[identity] = scores.get(identity, 0) + 1/(constant+rank)
    return [(key, scores[key]) for key in sorted(scores, key=lambda key: (-scores[key], key))[:k]]


def visible_records(records, tenant, actor):
    allowed = {"*", "user:"+actor}
    return [row for row in records if row["tenant"] == tenant and not row["deleted"] and set(row["allowed"]) & allowed]


def masked_softmax(x, mask):
    x = np.asarray(x, dtype=np.float64)
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), x.shape)
    masked = np.where(mask, x, -np.inf)
    maximum = np.max(masked, axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0)
    exp = np.exp(masked-maximum)
    total = exp.sum(axis=-1, keepdims=True)
    return np.divide(exp, total, out=np.zeros_like(exp), where=total>0)


def causal_attention(q, k, v):
    q, k, v = [np.asarray(x, dtype=float) for x in (q,k,v)]
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0] or q.shape[1] == 0:
        raise ValueError("equal-length 2D self-attention required")
    mask = np.arange(k.shape[0])[None,:] <= np.arange(q.shape[0])[:,None]
    weights = masked_softmax(q @ k.T / np.sqrt(q.shape[1]), mask)
    return weights @ v


def linear_step(s, z, q, k, v):
    # This task receives already-positive feature vectors phi(q), phi(k).
    s, z, q, k, v = [np.asarray(x, dtype=float) for x in (s,z,q,k,v)]
    new_s = s + np.outer(k,v)
    new_z = z + k
    denominator = q @ new_z
    if denominator <= 0:
        raise ValueError("positive denominator required")
    return (q @ new_s)/denominator, new_s, new_z


def page_count(tokens, block_size):
    if type(tokens) is not int or type(block_size) is not int or tokens < 0 or block_size <= 0:
        raise ValueError("nonnegative token count and positive block size")
    return (tokens+block_size-1)//block_size


def page_refcounts(tables, num_pages):
    if num_pages < 0:
        raise ValueError("nonnegative page count")
    refs = [0]*num_pages
    for table in tables.values():
        for page in table:
            if type(page) is not int or not 0 <= page < num_pages:
                raise ValueError("invalid physical page")
            refs[page] += 1
    return refs


def allocate_token_budget(waiting, budget):
    if type(budget) is not int or budget < 0:
        raise ValueError("nonnegative budget required")
    if any(type(tokens) is not int or tokens < 0 for _,tokens in waiting):
        raise ValueError("invalid remaining tokens")
    result = []
    for identity, remaining in waiting:
        take = min(remaining,budget)
        if take:
            result.append((identity,take))
        budget -= take
        if budget == 0:
            break
    return result


def strided_offset(index, shape, strides, base=0):
    if not len(index) == len(shape) == len(strides):
        raise ValueError("rank mismatch")
    if any(type(i) is not int or not 0 <= i < n for i,n in zip(index,shape)):
        raise ValueError("index out of range")
    return base + sum(i*s for i,s in zip(index,strides))


def tiled_matmul(a,b,block):
    a,b = np.asarray(a),np.asarray(b)
    if type(block) is not int or block <= 0 or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("valid matrices and block required")
    out = np.zeros((a.shape[0],b.shape[1]),dtype=np.result_type(a,b,np.float64))
    for row in range(0,a.shape[0],block):
        for column in range(0,b.shape[1],block):
            for inner in range(0,a.shape[1],block):
                out[row:row+block,column:column+block] += a[row:row+block,inner:inner+block] @ b[inner:inner+block,column:column+block]
    return out


def online_attention(q,k,v,block):
    q,k,v = [np.asarray(x,dtype=float) for x in (q,k,v)]
    if type(block) is not int or block <= 0 or q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0] or k.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError("invalid attention input")
    out = np.zeros((q.shape[0],v.shape[1]))
    for row,query in enumerate(q):
        maximum,normalizer = -np.inf,0.0
        numerator = np.zeros(v.shape[1])
        for start in range(0,k.shape[0],block):
            scores = k[start:start+block] @ query / np.sqrt(q.shape[1])
            new_maximum = max(maximum,float(scores.max()))
            old_scale = np.exp(maximum-new_maximum)
            weights = np.exp(scores-new_maximum)
            numerator = old_scale*numerator + weights @ v[start:start+block]
            normalizer = old_scale*normalizer + weights.sum()
            maximum = new_maximum
        out[row] = numerator/normalizer
    return out
