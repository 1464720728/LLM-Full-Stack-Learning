"""Fill in one function at a time. See practice/README.md for exact contracts.

These intentional NotImplementedError stubs are exercises, not runtime modules.
Reference answers are in practice/solutions.py; do not import them in your work.
"""

def parse_jsonl(lines):
    raise NotImplementedError("F-P1")

def merge_counts(left, right):
    raise NotImplementedError("F-P2")

def percentile(values, q):
    raise NotImplementedError("F-P3")

def validate_add_args(value):
    raise NotImplementedError("A-P1")

def merge_tool_fragments(fragments):
    raise NotImplementedError("A-P2")

def transition(state, event):
    raise NotImplementedError("A-P3")

def bm25_term(tf, df, n, length, average_length, k1=1.2, b=0.75):
    raise NotImplementedError("R-P1")

def reciprocal_rank_fusion(rankings, k, constant=60):
    raise NotImplementedError("R-P2")

def visible_records(records, tenant, actor):
    raise NotImplementedError("R-P3")

def masked_softmax(x, mask):
    raise NotImplementedError("M-P1")

def causal_attention(q, k, v):
    raise NotImplementedError("M-P2")

def linear_step(s, z, q, k, v):
    raise NotImplementedError("M-P3")

def page_count(tokens, block_size):
    raise NotImplementedError("S-P1")

def page_refcounts(tables, num_pages):
    raise NotImplementedError("S-P2")

def allocate_token_budget(waiting, budget):
    raise NotImplementedError("S-P3")

def strided_offset(index, shape, strides, base=0):
    raise NotImplementedError("K-P1")

def tiled_matmul(a, b, block):
    raise NotImplementedError("K-P2")

def online_attention(q, k, v, block):
    raise NotImplementedError("K-P3")
