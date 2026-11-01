"""Complete reference solutions for the deeper workbook assignments."""
from __future__ import annotations
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
import sqlite3
from .contracts import ContractError,Tool,validate
from .context import turn_groups
from .streaming import ToolDeltaAssembler
from .workflows import GraphState


def validate_state(state):
    if min(state.model_calls,state.tool_calls,state.output_chars,state.version)<0:
        raise ContractError('negative counter')
    if [e['seq'] for e in state.events]!=list(range(len(state.events))):
        raise ContractError('event sequence gap or duplicate')
    if state.status=='ready' and state.pending:raise ContractError('ready state has pending tools')
    if state.status=='tools' and not state.pending:raise ContractError('tools state has no pending tools')
    if state.status=='finished' and (state.pending or state.inflight_model or not state.final.strip()):
        raise ContractError('invalid finished state')
    # Full exchange validation is possible only when no unfinished tool group remains.
    if not state.pending:turn_groups(state.messages)
    ids=[c['id'] for c in state.pending]
    if len(ids)!=len(set(ids)):raise ContractError('duplicate pending IDs')
    return True


def output_checked(tool,output_schema):
    if tool.side_effect:raise ContractError('this wrapper is intentionally restricted to pure tools')
    def call(arguments):
        result=tool.function(arguments)
        try:validate(result,output_schema)
        except ContractError as exc:raise RuntimeError('tool output contract failed') from exc
        return result
    return Tool(tool.name,tool.description,tool.schema,call,False)


def pinned_context(history,goal,constraints,max_chars,output_reserve=0):
    """Goal/constraints remain user data, never promoted into system instructions."""
    groups=turn_groups(history)
    systems=[]
    while groups and groups[0][0].get('role')=='system':systems.extend(groups.pop(0))
    task={'role':'user','content':json.dumps({'current_task':goal,'constraints':constraints},ensure_ascii=False)}
    pinned=systems+[task]
    cost=lambda messages:len(json.dumps(messages,ensure_ascii=False,separators=(',',':')))
    limit=max_chars-output_reserve
    if cost(pinned)>limit:raise ContractError('pinned task itself exceeds context budget')
    selected=[]
    for group in reversed(groups):
        candidate=group+selected
        if cost(pinned+candidate)<=limit:selected=candidate
        elif not selected:raise ContractError('latest exchange cannot fit with task contract')
        else:break
    return deepcopy(pinned+selected)

class SequencedAssembler:
    """Reference solution for a CUSTOM sequence-numbered stream, not a provider field."""
    def __init__(self):self.seen={};self.last=-1;self.inner=ToolDeltaAssembler()
    def feed(self,seq,event):
        canonical=json.dumps(event,sort_keys=True,separators=(',',':'))
        if seq in self.seen:
            if self.seen[seq]!=canonical:raise ContractError('sequence replay changed content')
            return False
        if seq!=self.last+1:raise ContractError('sequence gap or out-of-order event')
        self.inner.feed(**event);self.seen[seq]=canonical;self.last=seq
        return True
    def finish(self):return self.inner.finish()

class AsyncBudget:
    def __init__(self,total):
        if type(total) is not int or total<0:raise ValueError('nonnegative integer total required')
        self.total=total;self.used=0;self.lock=asyncio.Lock()
    async def reserve(self,amount=1):
        if type(amount) is not int or amount<1:raise ValueError('positive integer reservation required')
        async with self.lock:
            if self.used+amount>self.total:return False
            self.used+=amount
            return True

async def resource_fanout(jobs,worker,limit=3):
    """jobs are dicts with id and resource_key. Same key serializes within one batch."""
    if type(limit) is not int or limit<1:raise ValueError('positive integer concurrency limit required')
    keys={j['resource_key']:asyncio.Lock() for j in jobs}
    semaphore=asyncio.Semaphore(limit)
    async def one(job):
        # Wait for a resource before spending a global slot.
        async with keys[job['resource_key']]:
            async with semaphore:return await worker(job)
    tasks=[asyncio.create_task(one(j)) for j in jobs]
    try:return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        raise

class GraphSnapshots:
    def __init__(self,path=':memory:'):
        self.db=sqlite3.connect(str(path))
        self.db.execute('CREATE TABLE IF NOT EXISTS graph_runs(id TEXT PRIMARY KEY,version INTEGER NOT NULL,payload TEXT NOT NULL)')
    def save(self,run_id,state,expected_version):
        payload=json.dumps({'schema_version':1,'state':asdict(state)},ensure_ascii=False)
        with self.db:
            if expected_version==0:
                try:self.db.execute('INSERT INTO graph_runs VALUES(?,?,?)',(run_id,1,payload))
                except sqlite3.IntegrityError as exc:raise ContractError('graph already exists') from exc
            else:
                c=self.db.execute('UPDATE graph_runs SET version=?,payload=? WHERE id=? AND version=?',(expected_version+1,payload,run_id,expected_version))
                if c.rowcount!=1:raise ContractError('stale graph version')
        return expected_version+1
    def load(self,run_id):
        row=self.db.execute('SELECT version,payload FROM graph_runs WHERE id=?',(run_id,)).fetchone()
        if row is None:raise KeyError(run_id)
        payload=json.loads(row[1])
        if payload['schema_version']!=1:raise ContractError('unsupported graph schema')
        return GraphState(**payload['state']),row[0]
    def close(self):self.db.close()

class Outbox:
    """Producer DB and separate consumer DB deliberately have separate transactions."""
    def __init__(self,producer_path=':memory:',consumer_path=':memory:'):
        self.producer=sqlite3.connect(str(producer_path));self.consumer=sqlite3.connect(str(consumer_path))
        self.producer.executescript('CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,title TEXT NOT NULL);CREATE TABLE IF NOT EXISTS outbox(event_id TEXT PRIMARY KEY,payload TEXT NOT NULL,sent INTEGER NOT NULL DEFAULT 0);')
        self.consumer.execute('CREATE TABLE IF NOT EXISTS inbox(event_id TEXT PRIMARY KEY,payload TEXT NOT NULL)');self.consumer.commit()
    def create(self,task_id,title):
        payload=json.dumps({'task_id':task_id,'title':title},sort_keys=True)
        with self.producer:
            # First DML obtains SQLite's writer lock; all checks stay in this transaction.
            self.producer.execute('INSERT OR IGNORE INTO tasks VALUES(?,?)',(task_id,title))
            row=self.producer.execute('SELECT title FROM tasks WHERE id=?',(task_id,)).fetchone()
            if row[0]!=title:raise ContractError('business key conflict')
            event_id='created:'+task_id
            self.producer.execute('INSERT OR IGNORE INTO outbox(event_id,payload) VALUES(?,?)',(event_id,payload))
            existing=self.producer.execute('SELECT payload FROM outbox WHERE event_id=?',(event_id,)).fetchone()
            if existing[0]!=payload:raise ContractError('event identity collision')
    def deliver_one(self,crash_after_delivery=False):
        row=self.producer.execute('SELECT event_id,payload FROM outbox WHERE sent=0 ORDER BY event_id LIMIT 1').fetchone()
        if not row:return False
        with self.consumer:
            self.consumer.execute('INSERT OR IGNORE INTO inbox VALUES(?,?)',row)
            earlier=self.consumer.execute('SELECT payload FROM inbox WHERE event_id=?',(row[0],)).fetchone()
            if earlier[0]!=row[1]:raise ContractError('event identity collision')
        if crash_after_delivery:raise RuntimeError('simulated crash after receiver commit, before sender mark')
        with self.producer:self.producer.execute('UPDATE outbox SET sent=1 WHERE event_id=?',(row[0],))
        return True
    def close(self):self.producer.close();self.consumer.close()


def main():
    from .workflows import Workflow,approval_node,action_digest
    async def budget_example():
        b=AsyncBudget(3);results=await asyncio.gather(*(b.reserve() for _ in range(7)));return {'accepted':sum(results),'used':b.used}
    action={'type':'create_ticket','title':'workbook example'}
    graph=Workflow({'review':approval_node},{'review':'END'});state=graph.run(GraphState('review',{'proposed_action':action}))
    db=GraphSnapshots();v=db.save('g',state,0);restored,v=db.load('g');graph.approve(restored,action_digest(action),True);graph.run(restored);db.save('g',restored,v);db.close()
    outbox=Outbox();outbox.create('task-1','only once')
    try:outbox.deliver_one(crash_after_delivery=True)
    except RuntimeError:pass
    outbox.deliver_one();count=outbox.consumer.execute('SELECT count(*) FROM inbox').fetchone()[0];outbox.close()
    print(json.dumps({'budget':asyncio.run(budget_example()),'restored_graph':restored.status,'outbox_inbox_rows':count},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
