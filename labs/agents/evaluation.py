"""Small offline evaluation tools with explicit statistical units."""
from dataclasses import dataclass,asdict
import argparse
import json
import math
import random
from .contracts import ToolCall,Turn
from .runtime import Harness,ScriptedModel,Budget
from .tools import arithmetic_tool


def wilson_interval(successes,n,z=1.959963984540054):
    if type(successes) is not int or type(n) is not int or n<=0 or not 0<=successes<=n:
        raise ValueError('require integer 0 <= successes <= n, n > 0')
    p=successes/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.0,mid-half),min(1.0,mid+half)

def pass_at_k(n,c,k):
    if not all(type(x) is int for x in (n,c,k)) or not 0<=c<=n or not 1<=k<=n:
        raise ValueError('require integers 0 <= c <= n and 1 <= k <= n')
    return 1.0 if n-c<k else 1-math.comb(n-c,k)/math.comb(n,k)

def paired_bootstrap(before,after,repeats=2000,seed=0):
    if len(before)!=len(after) or not before or repeats<1:raise ValueError('nonempty paired arrays and positive repeats required')
    if any(x not in (0,1,False,True) for x in list(before)+list(after)):raise ValueError('binary outcomes required')
    diffs=[int(b)-int(a) for a,b in zip(before,after)];rng=random.Random(seed);n=len(diffs)
    samples=sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(repeats))
    return {'mean_difference':sum(diffs)/n,'percentile_95':[samples[int(.025*(repeats-1))],samples[int(.975*(repeats-1))]],'unit':'paired task','repeats':repeats,'seed':seed}

@dataclass
class Case:
    id:str
    a:int
    b:int
    behavior:str='normal'

def run_case(case):
    arg={'a':case.a,'b':case.b}
    if case.behavior=='bad_argument':arg['a']=True
    turns=[Turn(calls=[ToolCall('c','add',json.dumps(arg))]),Turn(str(case.a+case.b))]
    if case.behavior=='false_claim':turns=[Turn('999')]
    budget=Budget(tool_calls=0) if case.behavior=='tool_budget' else Budget()
    h=Harness(ScriptedModel(turns),[arithmetic_tool()],budget=budget)
    s=h.run(h.new(case.id,f'Use add to calculate {case.a}+{case.b}'))
    observations=[json.loads(m['content']) for m in s.messages if m.get('role')=='tool']
    success=(s.status=='finished' and s.final==str(case.a+case.b) and len(observations)==1
             and observations[0].get('ok') is True and observations[0].get('value')==case.a+case.b)
    return {'case':asdict(case),'success':success,'status':s.status,'model_calls':s.model_calls,'tool_calls':s.tool_calls,
            'classification': 'verified' if success else ('invalid_tool_input' if case.behavior=='bad_argument' else 'budget' if s.status=='limit' else 'unverified_claim')}

def main():
    cases=[Case('normal',17,25),Case('negative',-3,5),Case('bad-args',2,3,'bad_argument'),Case('claim',2,3,'false_claim'),Case('budget',2,3,'tool_budget')]
    rows=[run_case(c) for c in cases]
    print(json.dumps({'purpose':'deterministic harness regression, not model capability benchmark','results':rows,
                     'expected_pass_pattern':[True,True,False,False,False],
                     'illustrative_wilson_16_of_20':wilson_interval(16,20),
                     'illustrative_pass_at_2_n4_c1':pass_at_k(4,1,2)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
