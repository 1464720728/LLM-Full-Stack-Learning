"""Kernel correctness portfolio: emits reproducible JSON, never GPU claims."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
from . import reference as r


def correctness_report():
    rng=np.random.default_rng(11)
    records=[]
    for n,d,dv in [(1,1,1),(3,5,2),(17,7,4),(31,16,8)]:
        q=rng.normal(size=(n,d));k=rng.normal(size=(n,d));v=rng.normal(size=(n,dv))
        for causal in [False,True]:
            ref=r.dense_attention(q,k,v,causal)
            for block in [1,3,8,64]:
                got=r.online_attention(q,k,v,block,causal)
                error=float(np.max(np.abs(got-ref)))
                if not np.allclose(got,ref,rtol=1e-12,atol=1e-12):raise AssertionError((n,d,causal,block,error))
                records.append({'n':n,'d':d,'dv':dv,'causal':causal,'block':block,'max_abs_error':error})
    return {'environment':{'python':platform.python_version(),'numpy':np.__version__,'backend':'CPU NumPy'},'status':'passed','attention_cases':len(records),'worst_error':max(x['max_abs_error'] for x in records),'cases':records,'gpu_validation':'not performed by this command','timing':'no performance claims'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    text=json.dumps(correctness_report(),ensure_ascii=False,indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text+'\n',encoding='utf-8')
    print(text)
if __name__=='__main__':main()
