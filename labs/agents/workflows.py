"""A serial explicit-state graph with approval binding; async fanout separately."""
from dataclasses import dataclass,field
from copy import deepcopy
import asyncio
import hashlib
import json
from .contracts import ContractError

@dataclass
class GraphState:
    node: str
    data: dict=field(default_factory=dict)
    history: list=field(default_factory=list)
    status: str='running'

def action_digest(action):
    return hashlib.sha256(json.dumps(action,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

class Workflow:
    def __init__(self,nodes,edges,end='END'):
        self.nodes=nodes;self.edges=edges;self.end=end
        if set(edges)-set(nodes):raise ContractError('edge source missing node')
    def step(self,state):
        if state.status!='running':return state
        if state.node==self.end:state.status='finished';return state
        if state.node not in self.nodes:raise ContractError('missing node')
        before=state.node
        patch=self.nodes[before](deepcopy(state.data))
        if not isinstance(patch,dict):raise ContractError('node must return a mapping')
        if patch.get('_interrupt'):
            state.data.update({k:v for k,v in patch.items() if k!='_interrupt'})
            state.status='waiting';return state
        state.data.update(patch)
        target=self.edges[before]
        state.node=target(state.data) if callable(target) else target
        state.history.append(before)
        if state.node==self.end:state.status='finished'
        return state
    def run(self,state,max_steps=20):
        if type(max_steps) is not int or max_steps<0:
            raise ValueError('max_steps must be a nonnegative integer')
        for _ in range(max_steps):
            if state.status!='running':return state
            self.step(state)
        if state.status!='running':return state
        raise ContractError('graph step budget exceeded')
    def approve(self,state,expected_digest,approved):
        if type(approved) is not bool:raise ContractError('approval must be a boolean')
        if state.status!='waiting':raise ContractError('not waiting')
        actual=action_digest(state.data['proposed_action'])
        if actual!=expected_digest:raise ContractError('action changed after review')
        state.data['approval']={'digest':actual,'approved':approved}
        state.status='running'
        return state

def approval_node(data):
    digest=action_digest(data['proposed_action'])
    decision=data.get('approval')
    if not decision or decision['digest']!=digest:return {'_interrupt':True,'review_digest':digest}
    return {'authorized':decision['approved']}

async def bounded_fanout(items,worker,limit=3):
    if type(limit) is not int or limit<1:raise ValueError('limit must be a positive integer')
    semaphore=asyncio.Semaphore(limit)
    async def one(item):
        async with semaphore:return await worker(item)
    # Explicit fail-fast cancellation and draining on exceptions.
    tasks=[asyncio.create_task(one(x)) for x in items]
    try:return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        raise

def merge_claims(worker_outputs):
    """Merge evidence-labelled claims, flag disagreement instead of majority truth."""
    merged={}
    for worker,claims in worker_outputs.items():
        for claim in claims:
            key=claim['subject']
            slot=merged.setdefault(key,{'values':{},'conflict':False})
            value=json.dumps(claim['value'],sort_keys=True,ensure_ascii=False)
            slot['values'].setdefault(value,[]).append({'worker':worker,'source':claim['source']})
            slot['conflict']=len(slot['values'])>1
    return merged
