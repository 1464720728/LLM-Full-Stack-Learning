"""Run `python -m labs.agents.experiments --list` or a numbered laboratory."""
from pathlib import Path
import argparse
import asyncio
import hashlib
import json
import random
import tempfile
from .contracts import *
from .runtime import *
from .store import *
from .tools import *
from .protocols import *
from .streaming import *
from .context import *
from .workflows import *
from .reliability import *
from .capstones import ticket_project,coding_project
from .integrations import ToolRPC,request_message

LABS={}
def lab(name):
    def register(fn):LABS[name]=fn;return fn
    return register

def arithmetic():
    h=Harness(ScriptedModel([Turn(calls=[ToolCall('c1','add','{"a":17,"b":25}')]),Turn('42')]),[arithmetic_tool()])
    return h,h.run(h.new('demo','Compute 17 + 25'))

@lab('01-loop')
def loop():
    _,s=arithmetic();return {'final':s.final,'model_calls':s.model_calls,'tool_calls':s.tool_calls,'events':[x['kind'] for x in s.events]}
@lab('02-contract')
def contract():
    result={}
    for name,args in [('valid',{'a':1,'b':2}),('boolean',{'a':True,'b':2}),('extra',{'a':1,'b':2,'shell':'oops'}),('missing',{'a':1})]:
        try:validate(args,arithmetic_tool().schema);result[name]='accepted'
        except ContractError as e:result[name]=str(e)
    return result
@lab('03-chat')
def chat():
    msg={'role':'assistant','content':None,'reasoning_content':'provider-specific teaching placeholder','tool_calls':[ToolCall('c1','add','{"a":1,"b":2}').wire()]}
    turn=from_chat({'choices':[{'message':msg,'finish_reason':'tool_calls'}]})
    return {'call':turn.calls[0].wire(),'extension_preserved':turn.message()['reasoning_content']==msg['reasoning_content']}
@lab('04-responses')
def responses():
    p={'status':'completed','output':[{'type':'reasoning','id':'opaque-1','summary':[]},{'type':'function_call','call_id':'c1','name':'add','arguments':'{"a":1,"b":2}'}]}
    return {'calls':len(from_responses(p).calls),'continuation':responses_continuation(p,{'c1':'3'})}
@lab('05-messages')
def messages_lab():
    p={'stop_reason':'tool_use','content':[{'type':'tool_use','id':'c1','name':'add','input':{'a':1,'b':2}}]}
    return messages_continuation(p,{'c1':'3'})
@lab('06-sse')
def sse():
    data='data: {"text":"鲸"}\r\n\r\ndata: [DONE]\n\n'.encode()
    return list(sse_events([data[i:i+1] for i in range(len(data))]))
@lab('07-deltas')
def deltas():
    a=ToolDeltaAssembler();a.feed(1,'b','add','{"a":');a.feed(0,'a','add','{"a":1,');a.feed(1,arguments='2,"b":3}');a.feed(0,arguments='"b":2}')
    return [x.wire() for x in a.finish()]
@lab('08-errors')
def errors():
    h=Harness(ScriptedModel([Turn(calls=[ToolCall('x','add','{"a":true,"b":2}')]),Turn('Correct the integer argument.')]),[arithmetic_tool()]);s=h.run(h.new('errors','test'))
    return {'status':s.status,'observation':json.loads(s.messages[2]['content'])}
@lab('09-budget')
def budget():
    h=Harness(ScriptedModel([Turn(calls=[ToolCall('x','add','{"a":1,"b":2}')])]),[arithmetic_tool()],budget=Budget(model_calls=1,tool_calls=0));s=h.run(h.new('budget','test'))
    return {'status':s.status,'tool_calls':s.tool_calls,'reason':s.events[-1]['data']}
@lab('10-retry')
def retry_lab():
    calls=[];delays=[]
    def op():
        calls.append(1)
        if len(calls)<3:raise TransientFailure()
        return 42
    return {'value':retry(op,sleep=delays.append,rng=random.Random(7)),'attempts':len(calls),'backoff_delays':delays}
@lab('11-context')
def context_lab():
    history=[{'role':'system','content':'Keep evidence.'},{'role':'user','content':'old '*100},{'role':'assistant','content':'ok'},{'role':'user','content':'new question'}]
    return select_context(history,200,output_reserve=20)
@lab('12-provenance')
def provenance():
    return evidence_summary([{'statement':'42','source':'tool:c1','verified':True,'confidence':'exact'},{'statement':'guess','source':'model','verified':False}])
@lab('13-paths')
def paths():
    with tempfile.TemporaryDirectory() as d:
        w=Workspace(d);(Path(d)/'note.txt').write_text('hello')
        result={'good':w.read('note.txt')['text']}
        try:w.read('../escape')
        except ContractError:result['escape']='rejected'
        return result
@lab('14-edit')
def edit():
    with tempfile.TemporaryDirectory() as d:
        (Path(d)/'a.py').write_text('answer = 41\n');w=Workspace(d);before=w.read('a.py');w.replace('a.py','41','42',before['sha256'])
        return w.read('a.py')
@lab('15-idempotency')
def idempotency():
    b=TicketBook();a=b.create('k','same task');c=b.create('k','same task');b.close();return [a,c]
@lab('16-checkpoint')
def checkpoint():
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'run.db';model=ScriptedModel([Turn(calls=[ToolCall('x','add','{"a":2,"b":3}')]),Turn('5')])
        with Store(path) as db:
            h=Harness(model,[arithmetic_tool()],db);s=h.new('r','test');h.step(s);saved=s.version
        with Store(path) as db:
            h=Harness(model,[arithmetic_tool()],db);s=h.run(h.resume('r'));return {'saved_version':saved,'final':s.final,'tool_calls':s.tool_calls}
@lab('17-conflict')
def conflict():
    with Store() as db:
        h=Harness(ScriptedModel([Turn('done')]),[],db);s=h.new('r','test');old=h.resume('r');h.step(s)
        try:h._save(old)
        except Conflict:return {'stale_save':'rejected'}
@lab('18-uncertain')
def uncertain():
    b=TicketBook();t=b.tool();call=ToolCall('c','create_ticket','{"request_id":"k","title":"t"}')
    with Store() as db:
        h=Harness(ScriptedModel([Turn(calls=[call])]),[t],db);s=h.new('r','test');h.step(s)
        sig=hashlib.sha256((call.name+'\0'+call.arguments).encode()).hexdigest();db.start('r','c',sig,True)
        h.step(s);b.close();return {'status':s.status,'reason':s.events[-1]['data']}
@lab('19-cancellation')
def cancellation():
    h=Harness(ScriptedModel([]),[]);s=h.new('r','test');h.cancel(s);h.run(s);return {'status':s.status,'model_calls':s.model_calls}
@lab('20-fanout')
def fanout():
    async def run():
        active=0;peak=0
        async def worker(n):
            nonlocal active,peak
            active+=1;peak=max(peak,active);await asyncio.sleep(0);active-=1;return n*n
        out=await bounded_fanout(range(6),worker,limit=2);return {'values':out,'peak':peak}
    return asyncio.run(run())
@lab('21-approval')
def approval():
    action={'name':'create_ticket','title':'example'};g=Workflow({'review':approval_node},{'review':'END'});s=g.run(GraphState('review',{'proposed_action':action}));initial=s.status;g.approve(s,action_digest(action),True);g.run(s);return {'before':initial,'after':s.status,'authorized':s.data['authorized']}
@lab('22-multi-agent')
def multi_agent():
    return merge_claims({'a':[{'subject':'version','value':'v1','source':'file:old'}],'b':[{'subject':'version','value':'v2','source':'file:new'}]})
@lab('23-metrics')
def metrics_lab():
    return metrics([{'success':True,'model_calls':2,'tool_calls':1},{'success':False,'model_calls':4,'tool_calls':3},{'success':False,'model_calls':1,'tool_calls':0}])
@lab('24-redaction')
def redaction():
    return redact({'authorization':'Bearer example','nested':{'api_key':'fake'},'text':'placeholder sk-abcdefgh123456789'})
@lab('25-ticket-project')
def ticket():
    with tempfile.TemporaryDirectory() as d:return ticket_project(d)
@lab('26-coding-project')
def coding():
    with tempfile.TemporaryDirectory() as d:return coding_project(d)
@lab('27-tool-rpc')
def tool_rpc():
    rpc=ToolRPC([arithmetic_tool()]);return [rpc.request(request_message(1,'tools/list')),rpc.request(request_message(2,'tools/call',{'name':'add','arguments':{'a':17,'b':25}}))]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('lab',nargs='?',choices=list(LABS)+['all']);p.add_argument('--list',action='store_true');a=p.parse_args()
    if a.list or not a.lab:print('\n'.join(LABS));return
    names=list(LABS) if a.lab=='all' else [a.lab]
    for name in names:print(json.dumps({'lab':name,'result':LABS[name]()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
