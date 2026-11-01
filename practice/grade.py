"""Grade student functions without importing the reference implementation."""
import argparse
import importlib
import json
import math
import numpy as np


def require(value, message):
    if not value: raise AssertionError(message)


def raises(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('expected ValueError')


def f1(m):
    result=m.parse_jsonl(['{"x":1}','bad','  ','[1]','{"x":2}'])
    require(result==([{'x':1},{'x':2}],[2,4]),'preserve records and 1-based error line numbers')
    require(m.parse_jsonl([])==([],[]),'empty file')


def f2(m):
    a,b={'x':2},{'x':3,'y':1}
    require(m.merge_counts(a,b)=={'x':5,'y':1},'sum counts per key')
    require(a=={'x':2} and b=={'x':3,'y':1},'must not mutate inputs')


def f3(m):
    require(m.percentile([11,0,-1,4],.5)==2,'linear interpolation')
    require(m.percentile([2],0)==2 and m.percentile([2],1)==2,'singleton endpoints')
    raises(lambda:m.percentile([],0));raises(lambda:m.percentile([1],1.1))


def a1(m):
    require(m.validate_add_args({'a':2,'b':-3})==-1,'add validated integers')
    for value in ({'a':True,'b':1},{'a':1},{'a':1,'b':2,'extra':3},[1,2]):
        raises(lambda value=value:m.validate_add_args(value))


def a2(m):
    fragments=[{'index':1,'id':'b','name':'second','arguments':'{"x":'},
               {'index':0,'id':'a','name':'first','arguments':'{"y":1}'},
               {'index':1,'arguments':'2}'}]
    require(m.merge_tool_fragments(fragments)==[{'id':'a','name':'first','arguments':{'y':1}},{'id':'b','name':'second','arguments':{'x':2}}],'group by call index, preserve identity, parse only at completion')
    raises(lambda:m.merge_tool_fragments([{'index':0,'id':'a','name':'x','arguments':'{'}]))
    raises(lambda:m.merge_tool_fragments([{'index':0,'id':'a'},{'index':0,'id':'b'}]))


def a3(m):
    state='ready'
    for event, expected in [('tool_call','waiting_tool'),('tool_result','ready'),('answer','finished')]:
        state=m.transition(state,event);require(state==expected,'incorrect state transition')
    raises(lambda:m.transition('finished','tool_call'))
    require(m.transition('waiting_tool','cancel')=='cancelled','cancellation while waiting')


def r1(m):
    require(abs(m.bm25_term(1,1,3,10,10)-.9808292530117263)<1e-12,'hand-computed single term score')
    require(m.bm25_term(0,1,3,10,10)==0,'absent term contributes zero')
    require(m.bm25_term(2,1,3,10,10)>m.bm25_term(1,1,3,10,10),'frequency effect')
    require(m.bm25_term(1,1,3,20,10)<m.bm25_term(1,1,3,10,10),'length normalization')
    raises(lambda:m.bm25_term(1,4,3,10,10))


def r2(m):
    result=m.reciprocal_rank_fusion([['a','b'],['b','c']],3,60)
    require([x[0] for x in result]==['b','a','c'],'fusion rank')
    require(abs(result[0][1]-(1/62+1/61))<1e-12,'sum reciprocal positions')
    once=m.reciprocal_rank_fusion([['a','a']],5,0)
    require(once==[('a',1.0)],'duplicates cannot count twice within a ranking')


def r3(m):
    rows=[{'id':1,'tenant':'x','allowed':['*'],'deleted':False},
          {'id':2,'tenant':'x','allowed':['user:a'],'deleted':False},
          {'id':3,'tenant':'y','allowed':['*'],'deleted':False},
          {'id':4,'tenant':'x','allowed':['*'],'deleted':True},
          {'id':5,'tenant':'x','allowed':[],'deleted':False}]
    require([x['id'] for x in m.visible_records(rows,'x','b')]==[1],'filter tenant, deleted and ACL before retrieval')
    require([x['id'] for x in m.visible_records(rows,'x','a')]==[1,2],'explicit user grant')


def m1(m):
    x=np.array([[1000.,1000.,-1000.],[3.,4.,5.]])
    out=m.masked_softmax(x,[[True,True,False],[False,False,False]])
    np.testing.assert_allclose(out,[[.5,.5,0],[0,0,0]],atol=1e-12)
    require(np.isfinite(out).all(),'fully masked row must not be NaN')


def m2(m):
    q=k=np.ones((3,2));v=np.array([[1.,0.],[3.,2.],[8.,4.]])
    out=m.causal_attention(q,k,v)
    np.testing.assert_allclose(out,[[1,0],[2,1],[4,2]])
    changed=v.copy();changed[-1]=1000
    np.testing.assert_allclose(m.causal_attention(q,k,changed)[:2],out[:2])


def m3(m):
    s=np.zeros((2,2));z=np.zeros(2); q=np.array([1.,2.]);k=np.array([2.,1.]);v=np.array([3.,4.])
    out,ns,nz=m.linear_step(s,z,q,k,v)
    np.testing.assert_allclose(out,v)
    np.testing.assert_array_equal(s,np.zeros_like(s));np.testing.assert_array_equal(z,np.zeros_like(z))
    k2=np.array([1.,3.]);v2=np.array([0.,2.])
    out2,_,_=m.linear_step(ns,nz,q,k2,v2)
    expected=((q@k)*v+(q@k2)*v2)/(q@k+q@k2)
    np.testing.assert_allclose(out2,expected)


def s1(m):
    require([m.page_count(x,4) for x in (0,1,4,5)]==[0,1,1,2],'ceiling pages, zero has no page')
    raises(lambda:m.page_count(1,0));raises(lambda:m.page_count(-1,4))


def s2(m):
    require(m.page_refcounts({'a':[0,1],'b':[0,2]},4)==[2,1,1,0],'shared physical pages count owners')
    require(m.page_refcounts({'b':[0,2]},4)==[1,0,1,0],'release one owner')
    raises(lambda:m.page_refcounts({'a':[4]},4))


def s3(m):
    waiting=[('a',3),('b',5),('c',0)]
    require(m.allocate_token_budget(waiting,5)==[('a',3),('b',2)],'partial allocation with global budget')
    require(m.allocate_token_budget(waiting,0)==[],'zero budget')
    require(waiting==[('a',3),('b',5),('c',0)],'no input mutation')
    raises(lambda:m.allocate_token_budget([('x',-1)],2))


def k1(m):
    require(m.strided_offset((1,2),(2,3),(3,1))==5,'row-major offset')
    require(m.strided_offset((2,1),(3,2),(1,3))==5,'transposed view')
    require(m.strided_offset((1,),(3,),(-1,),base=2)==1,'negative stride with explicit base')
    raises(lambda:m.strided_offset((2,0),(2,3),(3,1)))


def k2(m):
    rng=np.random.default_rng(19)
    a,b=rng.normal(size=(5,7)),rng.normal(size=(7,3))
    for block in (1,2,4,16):
        np.testing.assert_allclose(m.tiled_matmul(a,b,block),a@b,rtol=1e-12,atol=1e-12)
    raises(lambda:m.tiled_matmul(a,b,0))


def k3(m):
    rng=np.random.default_rng(23)
    q,k,v=rng.normal(size=(5,3)),rng.normal(size=(7,3)),rng.normal(size=(7,4))
    for scale in (1.,30.):
        scores=(q*scale)@k.T/np.sqrt(3)
        exp=np.exp(scores-scores.max(-1,keepdims=True));expected=(exp/exp.sum(-1,keepdims=True))@v
        for block in (1,3,8):
            np.testing.assert_allclose(m.online_attention(q*scale,k,v,block),expected,rtol=1e-11,atol=1e-11)
    raises(lambda:m.online_attention(q,k,v,0))


CHECKS={'F-P1':f1,'F-P2':f2,'F-P3':f3,'A-P1':a1,'A-P2':a2,'A-P3':a3,
        'R-P1':r1,'R-P2':r2,'R-P3':r3,'M-P1':m1,'M-P2':m2,'M-P3':m3,
        'S-P1':s1,'S-P2':s2,'S-P3':s3,'K-P1':k1,'K-P2':k2,'K-P3':k3}


def grade(module, selected=None):
    results=[]
    for task,check in CHECKS.items():
        if selected and task not in selected:continue
        try:
            check(module)
            results.append({'task':task,'passed':True})
        except Exception as exc:
            results.append({'task':task,'passed':False,'error':f'{type(exc).__name__}: {exc}'})
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--module',default='practice.starter')
    p.add_argument('--task',action='append',choices=list(CHECKS))
    args=p.parse_args()
    module=importlib.import_module(args.module)
    results=grade(module,args.task)
    print(json.dumps(results,ensure_ascii=False,indent=2))
    passed=sum(x['passed'] for x in results)
    print(f'{passed}/{len(results)} tasks passed; passing tests is not a substitute for explaining invariants.')
    return 0 if passed==len(results) else 1

if __name__=='__main__':raise SystemExit(main())
