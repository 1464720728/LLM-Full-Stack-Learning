"""Trace and fault-injection companion to the three detailed workbooks."""
import argparse
import json
import numpy as np
from . import reference as r


def softmax_trace():
    x=np.array([1000.,1001.,1002.]);block=4
    padded=np.pad(x,(0,block-x.size),constant_values=-np.inf)
    maximum=float(np.max(padded));shift=padded-maximum;ex=np.exp(shift);den=float(ex.sum());p=ex/den
    wrong_pad=np.pad(np.array([-3.,-2.,-1.]),(0,1),constant_values=0.)
    wrong=r.stable_softmax(wrong_pad)[:3];correct=r.stable_softmax([-3,-2,-1])
    return {'logical_n':3,'block':block,'padded':['-inf' if not np.isfinite(v) else float(v) for v in padded],'maximum':maximum,'shifted':['-inf' if not np.isfinite(v) else float(v) for v in shift],'exponentials':ex.tolist(),'denominator':den,'probabilities':p[:3].tolist(),'padding_probability':float(p[-1]),'wrong_zero_padding_sum':float(wrong.sum()),'correct_sum':float(correct.sum()),'fault_detected':not np.allclose(wrong,correct)}


def gemm_trace():
    a=np.array([[1.,2.,3.],[4.,5.,6.]]);b=np.array([[1.,0.],[0.,1.],[1.,1.]])
    trace=[];accum=np.zeros((2,2))
    for start in [0,2]:
        ta=np.zeros((2,2));tb=np.zeros((2,2));valid=min(2,3-start)
        ta[:,:valid]=a[:,start:start+valid];tb[:valid]=b[start:start+valid]
        contribution=ta@tb;accum+=contribution
        trace.append({'k_start':start,'valid_k':valid,'tile_a':ta.tolist(),'tile_b':tb.tolist(),'contribution':contribution.tolist(),'accumulator':accum.copy().tolist()})
    # A deliberately wrong view of B: reinterpret as column-major without transpose.
    wrong=a@b.ravel().reshape(2,3).T
    return {'a':a.tolist(),'b':b.tolist(),'trace':trace,'reference':(a@b).tolist(),'wrong_b_stride_output':wrong.tolist(),'fault_detected':not np.allclose(wrong,a@b)}


def attention_trace():
    q=np.ones((1,1));k=np.log(np.array([[1.],[2.],[4.]]));v=np.array([[10.],[20.],[30.]])
    out,trace=r.online_attention(q,k,v,1,return_trace=True)
    # Deliberately omit old numerator rescaling, while preserving denominator.
    wrong_u=0.;den=0.;maximum=-np.inf;wrong=[]
    for score,value in zip(k[:,0],v[:,0]):
        new=max(maximum,score);alpha=np.exp(maximum-new);p=np.exp(score-new)
        den=den*alpha+p;wrong_u=wrong_u+p*value;maximum=new
        wrong.append(float(wrong_u/den))
    clean=[{key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in t.items()} for t in trace]
    return {'trace':clean,'correct':out.tolist(),'hand_answer':170/7,'wrong_outputs_without_rescaling':wrong,'fault_detected':not np.allclose(out,wrong[-1])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('topic',choices=['softmax','gemm','attention','all']);args=p.parse_args()
    mapping={'softmax':softmax_trace,'gemm':gemm_trace,'attention':attention_trace}
    for name,fn in mapping.items():
        if args.topic in [name,'all']:print(json.dumps({'topic':name,'result':fn()},ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
