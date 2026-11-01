"""Optional CPU FAISS Flat-vs-HNSW experiment; no downloads or GPU use."""
import argparse
import json
import time


def run(seed=17,n=2000,dim=32,queries=40,k=10):
    try:
        import numpy as np
        import faiss
    except ImportError as exc:
        return {'status':'SKIP','reason':'Optional NumPy/FAISS dependency missing; no install or download attempted.'}
    if not 1<=k<=n or min(dim,queries)<=0:raise ValueError('invalid experiment size')
    if n>20000 or dim>2048 or queries>1000:raise ValueError('course CPU budget exceeded')
    faiss.omp_set_num_threads(1)
    rng=np.random.default_rng(seed)
    xb=rng.normal(size=(n,dim)).astype('float32')
    xq=rng.normal(size=(queries,dim)).astype('float32')
    faiss.normalize_L2(xb);faiss.normalize_L2(xq)
    exact=faiss.IndexFlatIP(dim);exact.add(xb)
    _,ground=exact.search(xq,k)
    ann=faiss.IndexHNSWFlat(dim,16,faiss.METRIC_INNER_PRODUCT)
    ann.hnsw.efConstruction=80
    build_start=time.perf_counter();ann.add(xb);build_seconds=time.perf_counter()-build_start
    measurements=[]
    for ef in [8,32,128]:
        ann.hnsw.efSearch=ef
        ann.search(xq[:1],k)  # Explicit warm-up; do not include it in measured run.
        start=time.perf_counter();_,indices=ann.search(xq,k);elapsed=time.perf_counter()-start
        recall=sum(len(set(a)&set(b))/k for a,b in zip(ground,indices))/queries
        measurements.append({'efSearch':ef,'ann_neighbor_recall_at_k':recall,'batch_search_seconds':elapsed})
    return {'status':'OK','faiss_version':getattr(faiss,'__version__','unknown'),'seed':seed,
            'n':n,'dimension':dim,'queries':queries,'k':k,'threads':1,'metric':'cosine via normalized inner product',
            'build_seconds':build_seconds,'measurements':measurements,
            'limitation':'Synthetic vectors measure ANN agreement with exact neighbors, not semantic task relevance.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=2000)
    args=parser.parse_args(argv)
    print(json.dumps(run(n=args.n),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
