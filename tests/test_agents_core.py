import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from labs.agents.contracts import *
from labs.agents.runtime import *
from labs.agents.store import *
from labs.agents.protocols import *
from labs.agents.streaming import *
from labs.agents.context import *
from labs.agents.tools import *
from labs.agents.workflows import *
from labs.agents.reliability import *
from labs.agents.capstones import coding_project,ticket_project
from labs.agents.integrations import ToolRPC,request_message

class ContractsTests(unittest.TestCase):
    def test_boolean_not_integer(self):
        with self.assertRaises(ContractError):validate(True,{'type':'integer'})
    def test_unknown_schema_not_silently_ignored(self):
        with self.assertRaises(ContractError):validate('x',{'type':'string','pattern':'x'})
    def test_duplicate_json_keys(self):
        with self.assertRaises(ContractError):parse_json('{"a":1,"a":2}')
    def test_nan_rejected(self):
        with self.assertRaises(ContractError):parse_json('{"a":NaN}')
    def test_nested_paths(self):
        with self.assertRaisesRegex(ContractError,r'\$\.xs\[1\]'):
            validate({'xs':[1,'x']},object_schema({'xs':{'type':'array','items':{'type':'integer'}}}))
    def test_huge_integer_is_finite_number_without_float_conversion(self):
        value=parse_json("1" + "0" * 1000)
        validate(value,{"type":"number"})
        with self.assertRaises(ContractError):
            validate(value,{"type":"number","maximum":100})

    def test_extra_key_rejected(self):
        with self.assertRaises(ContractError):validate({'a':1,'b':2,'unexpected':0},arithmetic_tool().schema)

class RuntimeTests(unittest.TestCase):
    def make(self,budget=None,store=None):
        return Harness(ScriptedModel([Turn(calls=[ToolCall('x','add','{"a":17,"b":25}')]),Turn('42')]),[arithmetic_tool()],store,budget)
    def test_finished_on_last_transition_stays_finished(self):
        h=Harness(ScriptedModel([Turn('done')]),[])
        s=h.run(h.new('r','goal'),max_transitions=1)
        self.assertEqual((s.status,s.final),('finished','done'))
        self.assertEqual(h.resume('r').status,'finished')
        self.assertFalse(any(e['kind']=='run_stopped' for e in s.events))
    def test_error_on_last_transition_stays_error(self):
        h=Harness(ScriptedModel([RuntimeError('planned')]),[])
        s=h.run(h.new('r','goal'),max_transitions=1)
        self.assertEqual(s.status,'error');self.assertEqual(h.resume('r').status,'error')
        self.assertEqual(sum(e['kind']=='run_stopped' for e in s.events),1)
    def test_uncertain_on_last_transition_stays_uncertain(self):
        effects=[]
        def effect(args):effects.append('created');raise ContractError('after effect')
        tool=Tool('effect','demo',object_schema({}),effect,True)
        h=Harness(ScriptedModel([Turn(calls=[ToolCall('c','effect','{}')])]),[tool])
        s=h.new('r','goal');h.step(s);h.run(s,max_transitions=1)
        self.assertEqual(s.status,'uncertain');self.assertEqual(effects,['created'])
        self.assertEqual(h.resume('r').status,'uncertain')
        self.assertEqual(sum(e['kind']=='run_stopped' for e in s.events),1)
    def test_terminal_zero_transitions_is_unchanged(self):
        h=Harness(ScriptedModel([Turn('done')]),[]);s=h.run(h.new('r','goal'))
        before=deepcopy(s);h.run(s,max_transitions=0);self.assertEqual(s,before)
    def test_active_transition_exhaustion_is_limit(self):
        h=self.make();s=h.run(h.new('r','goal'),max_transitions=1)
        self.assertEqual(s.status,'limit');self.assertEqual(len(s.pending),1)
        self.assertEqual(s.events[-1]['data']['reason'],'transition guard')
    def test_invalid_transition_limit_does_not_mutate_state(self):
        h=self.make();s=h.new('r','goal');before=deepcopy(s)
        for limit in [-1,True,1.5,float('nan'),float('inf'),'1']:
            with self.subTest(limit=limit),self.assertRaises(ValueError):h.run(s,max_transitions=limit)
            self.assertEqual(s,before)
    def test_count_budgets_require_nonnegative_integers(self):
        for field in ['model_calls','tool_calls','tool_output_chars','max_argument_chars']:
            for value in [-1,True,1.5,float('nan'),float('inf'),'1']:
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):Budget(**{field:value})
            self.assertEqual(getattr(Budget(**{field:0}),field),0)
    def test_loop(self):
        h=self.make();s=h.run(h.new('r','add'));self.assertEqual((s.status,s.final,s.model_calls,s.tool_calls),('finished','42',2,1))
        self.assertEqual([e['seq'] for e in s.events],list(range(len(s.events))))
    def test_budget_prevents_tool_execution(self):
        calls=[];t=Tool('effect','demo',object_schema({}),lambda a:calls.append(1),True)
        h=Harness(ScriptedModel([Turn(calls=[ToolCall('x','effect','{}')])]),[t],budget=Budget(tool_calls=0))
        s=h.run(h.new('r','x'));self.assertEqual(s.status,'limit');self.assertEqual(calls,[])
    def test_model_error_charged(self):
        h=Harness(ScriptedModel([RuntimeError('secret')]),[]);s=h.run(h.new('r','x'));self.assertEqual(s.model_calls,1);self.assertEqual(s.status,'error');self.assertNotIn('secret',json.dumps(s.events))
    def test_invalid_tool_becomes_observation(self):
        h=Harness(ScriptedModel([Turn(calls=[ToolCall('x','add','{"a":true,"b":2}')]),Turn('retry needed')]),[arithmetic_tool()])
        s=h.run(h.new('r','x'));self.assertFalse(json.loads(s.messages[2]['content'])['ok'])
    def test_empty_final_rejected(self):
        h=Harness(ScriptedModel([Turn()]),[]);self.assertEqual(h.run(h.new('r','x')).status,'error')
    def test_duplicate_ids_rejected(self):
        h=Harness(ScriptedModel([Turn(calls=[ToolCall('x','add','{}'),ToolCall('x','add','{}')])]),[])
        self.assertEqual(h.run(h.new('r','x')).status,'error')
    def test_deadline_before_request(self):
        h=Harness(ScriptedModel([]),[],deadline=10,clock=lambda:10);s=h.run(h.new('r','x'));self.assertEqual(s.status,'limit');self.assertEqual(s.model_calls,0)
    def test_cancel_is_terminal(self):
        h=self.make();s=h.new('r','x');h.cancel(s);h.run(s);self.assertEqual(s.model_calls,0)
    def test_late_cancel_preserves_finished_outcome_and_checkpoint(self):
        h=Harness(ScriptedModel([Turn(text='done')]),[])
        s=h.run(h.new('late-cancel','x'));before=deepcopy(s)
        self.assertIs(h.cancel(s),s)
        self.assertEqual(s,before)
        self.assertEqual(h.resume('late-cancel'),before)
    def test_repeated_cancel_does_not_add_event_or_version(self):
        h=self.make();s=h.new('repeat-cancel','x');h.cancel(s);before=deepcopy(s)
        self.assertIs(h.cancel(s),s)
        self.assertEqual(s,before)
        self.assertEqual(h.resume('repeat-cancel'),before)
    def test_resume_pending_tool(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.db'
            with Store(p) as db:
                h=self.make(store=db);s=h.new('r','x');h.step(s)
            with Store(p) as db:
                h=self.make(store=db);s=h.run(h.resume('r'));self.assertEqual(s.final,'42');self.assertEqual(s.tool_calls,1)
    def test_model_crash_window_does_not_skip_response(self):
        class CrashingModel:
            def complete(self,*args):raise KeyboardInterrupt()
        with Store() as db:
            h=Harness(CrashingModel(),[],db);s=h.new('r','x')
            with self.assertRaises(KeyboardInterrupt):h.step(s)
            h2=Harness(ScriptedModel([Turn('wrongly skipped'),Turn('not allowed')]),[],db)
            resumed=h2.run(h2.resume('r'))
            self.assertEqual(resumed.status,'uncertain');self.assertEqual(resumed.model_calls,1)
    def test_stale_snapshot_rejected(self):
        h=self.make();s=h.new('r','x');old=h.resume('r');h.step(s)
        with self.assertRaises(Conflict):h._save(old)
    def test_side_effect_exception_is_uncertain(self):
        effects=[]
        def f(a):effects.append(1);raise RuntimeError('response lost')
        t=Tool('effect','demo',object_schema({}),f,True)
        h=Harness(ScriptedModel([Turn(calls=[ToolCall('x','effect','{}')])]),[t]);s=h.run(h.new('r','x'))
        self.assertEqual(s.status,'uncertain');self.assertEqual(effects,[1])
    def test_tool_contract_error_after_effect_is_uncertain(self):
        effects=[]
        def effect_then_contract_error(args):
            effects.append('created')
            raise ContractError('late tool failure')
        tool=Tool('effect','demo',object_schema({}),effect_then_contract_error,True)
        h=Harness(ScriptedModel([Turn(calls=[ToolCall('c','effect','{}')])]),[tool])
        s=h.run(h.new('r','demo'))
        self.assertEqual(s.status,'uncertain');self.assertEqual(effects,['created'])
        row=h.store.db.execute('SELECT status,output FROM tool_ledger WHERE run_id=? AND call_id=?',('r','c')).fetchone()
        self.assertEqual(row,('running',None))
        self.assertFalse(any(m.get('role')=='tool' for m in s.messages))
    def test_side_effect_invalid_input_never_invokes_tool(self):
        effects=[]
        tool=Tool('effect','demo',object_schema({'n':{'type':'integer'}}),lambda args:effects.append(args),True)
        h=Harness(ScriptedModel([Turn(calls=[ToolCall('c','effect','{"n":true}')]),Turn('correct input required')]),[tool])
        s=h.run(h.new('r','demo'))
        self.assertEqual(effects,[]);self.assertEqual(s.status,'finished')
        result=json.loads(next(m['content'] for m in s.messages if m.get('role')=='tool'))
        self.assertEqual(result['error']['code'],'invalid_arguments')
        self.assertEqual(h.store.db.execute('SELECT status FROM tool_ledger').fetchone()[0],'done')
    def test_pure_tool_contract_error_is_tool_failure(self):
        def internal_error(args):raise ContractError('internal tool contract failed')
        tool=Tool('pure','demo',object_schema({}),internal_error,False)
        h=Harness(ScriptedModel([Turn(calls=[ToolCall('c','pure','{}')]),Turn('tool failed')]),[tool])
        s=h.run(h.new('r','demo'))
        result=json.loads(next(m['content'] for m in s.messages if m.get('role')=='tool'))
        self.assertEqual(result['error'],{'code':'tool_failed','type':'ContractError'})
    def test_ledger_replays_completed_without_effect(self):
        count=[];t=Tool('effect','demo',object_schema({}),lambda a:count.append(1) or 'ok',True)
        h=Harness(ScriptedModel([]),[t]);s=h.new('r','x');call=ToolCall('x','effect','{}')
        self.assertEqual(h._execute(s,call),h._execute(s,call));self.assertEqual(count,[1])
    def test_reused_id_different_arguments(self):
        h=self.make();s=h.new('r','x');h._execute(s,ToolCall('same','add','{"a":1,"b":2}'))
        with self.assertRaises(Conflict):h._execute(s,ToolCall('same','add','{"a":3,"b":2}'))

class ProtocolTests(unittest.TestCase):
    def test_chat_preserves_extension(self):
        msg={'role':'assistant','content':'ok','reasoning_content':'opaque'}
        self.assertEqual(from_chat({'choices':[{'message':msg}]}).message(),msg)
    def test_responses_retains_reasoning(self):
        p={'output':[{'type':'reasoning','id':'r'},{'type':'function_call','call_id':'c','name':'add','arguments':'{}'}]}
        self.assertEqual(responses_continuation(p,{'c':'3'})[0],p['output'][0])
        with self.assertRaises(ContractError):responses_continuation(p,{})
    def test_messages_result_role(self):
        p={'content':[{'type':'tool_use','id':'c','name':'add','input':{}}]}
        self.assertEqual(messages_continuation(p,{'c':'3'})[1]['role'],'user')
    def test_incomplete_not_success(self):
        with self.assertRaises(ContractError):from_chat({'choices':[{'finish_reason':'length','message':{'content':'partial'}}]})
        with self.assertRaises(ContractError):from_responses({'status':'incomplete','output':[]})
    def test_rpc_invalid_shapes(self):
        rpc=ToolRPC([arithmetic_tool()])
        for message in [[],{'jsonrpc':'2.0','id':True},{'jsonrpc':'2.0','id':1,'params':[]},request_message(1,'tools/call',{'name':{}})]:
            self.assertIn('error',rpc.request(message))
    def test_rpc_current_metadata(self):
        rpc=ToolRPC([arithmetic_tool()])
        self.assertIn('error',rpc.request({'jsonrpc':'2.0','id':1,'method':'tools/list'}))
        response=rpc.request(request_message(1,'tools/call',{'name':'add','arguments':{'a':17,'b':25}}))
        self.assertEqual(response['result']['resultType'],'complete');self.assertEqual(response['result']['content'][0]['text'],'42')

class StreamContextTests(unittest.TestCase):
    def test_every_byte_boundary(self):
        raw='data: {"text":"鲸鱼"}\r\n\r\ndata: [DONE]\n\n'.encode()
        for i in range(len(raw)+1):self.assertEqual(list(sse_events([raw[:i],raw[i:]])),['{"text":"鲸鱼"}','[DONE]'])
    def test_many_small_events_in_one_large_chunk(self):
        raw=b'data: x\n\n'*50
        self.assertEqual(list(sse_events([raw],max_event_chars=16)),['x']*50)
    def test_truncated_sse(self):
        with self.assertRaises(ContractError):list(sse_events([b'data: abc\n']))
    def test_interleaved_deltas(self):
        a=ToolDeltaAssembler();a.feed(1,'b','add','{"a":');a.feed(0,'a','add','{}');a.feed(1,arguments='1}')
        self.assertEqual([c.id for c in a.finish()],['a','b'])
    def test_conflicting_delta(self):
        a=ToolDeltaAssembler();a.feed(0,'a','add','')
        with self.assertRaises(ContractError):a.feed(0,'b')
    def test_context_preserves_tool_pair(self):
        messages=[{'role':'user','content':'old '*100},{'role':'assistant','content':'','tool_calls':[ToolCall('c','add','{}').wire()]},{'role':'tool','tool_call_id':'c','content':'3'}]
        selected=select_context(messages,400)
        self.assertEqual([m['role'] for m in selected],['assistant','tool'])
    def test_orphan_context_rejected(self):
        with self.assertRaises(ContractError):select_context([{'role':'tool','tool_call_id':'c','content':'3'}],1000)

class WorkspaceAndGraphTests(unittest.TestCase):
    def test_path_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ContractError):Workspace(d).read('../outside')
    def test_stale_edit_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_text('before');w=Workspace(d);old=w.read('a');p.write_text('changed')
            with self.assertRaises(ContractError):w.replace('a','before','after',old['sha256'])
            self.assertEqual(p.read_text(),'changed')
    def test_approval_action_binding(self):
        g=Workflow({'review':approval_node},{'review':'END'});s=g.run(GraphState('review',{'proposed_action':{'x':1}}));digest=s.data['review_digest'];s.data['proposed_action']['x']=2
        with self.assertRaises(ContractError):g.approve(s,digest,True)
    def test_ticket_business_idempotence(self):
        b=TicketBook();self.assertTrue(b.create('k','x')['created']);self.assertFalse(b.create('k','x')['created'])
        with self.assertRaises(ContractError):b.create('k','y')
        self.assertEqual(b.db.execute('SELECT count(*) FROM tickets').fetchone()[0],1);b.close()
    def test_graph_cycle_bounded(self):
        g=Workflow({'n':lambda d:{}},{'n':'n'})
        with self.assertRaises(ContractError):g.run(GraphState('n'),max_steps=3)
    def test_graph_finished_on_last_step(self):
        graph=Workflow({'n':lambda data:{'answer':42}},{'n':'END'})
        state=graph.run(GraphState('n'),max_steps=1)
        self.assertEqual((state.status,state.data['answer']),('finished',42))
        before=deepcopy(state);graph.run(state,max_steps=0);self.assertEqual(state,before)
    def test_graph_waiting_on_last_step(self):
        graph=Workflow({'n':approval_node},{'n':'END'})
        state=graph.run(GraphState('n',{'proposed_action':{'x':1}}),max_steps=1)
        self.assertEqual(state.status,'waiting')
        self.assertNotIn('approval',state.data)
    def test_graph_limits_are_nonnegative_integers(self):
        graph=Workflow({'n':lambda data:{}},{'n':'END'});state=GraphState('n')
        for limit in [-1,True,1.5,float('nan'),float('inf'),'1']:
            with self.subTest(limit=limit),self.assertRaises(ValueError):graph.run(state,max_steps=limit)
        self.assertEqual(state,GraphState('n'))
        with self.assertRaises(ContractError):graph.run(state,max_steps=0)
    def test_approval_rejects_truthy_non_boolean(self):
        graph=Workflow({'n':approval_node},{'n':'END'})
        state=graph.run(GraphState('n',{'proposed_action':{'x':1}}));before=deepcopy(state)
        for value in ['false','true',1,0,None]:
            with self.subTest(value=value),self.assertRaises(ContractError):graph.approve(state,state.data['review_digest'],value)
            self.assertEqual(state,before)
        graph.approve(state,state.data['review_digest'],False);graph.run(state)
        self.assertFalse(state.data['authorized'])
    def test_fanout_rejects_noninteger_limits_before_workers(self):
        calls=[]
        async def worker(value):calls.append(value)
        async def run():
            for limit in [0,-1,True,1.5,float('nan'),float('inf'),'2']:
                with self.subTest(limit=limit),self.assertRaises(ValueError):await bounded_fanout([1,2],worker,limit)
        asyncio.run(run());self.assertEqual(calls,[])
    def test_fanout_order_and_limit(self):
        async def run():
            active=0;peak=0
            async def worker(x):
                nonlocal active,peak
                active+=1;peak=max(peak,active);await asyncio.sleep(0);active-=1;return x*x
            result=await bounded_fanout(range(10),worker,2)
            return result,peak
        result,peak=asyncio.run(run());self.assertEqual(result,[x*x for x in range(10)]);self.assertLessEqual(peak,2)
    def test_fanout_failure_drained(self):
        async def run():
            done=[]
            async def worker(x):
                try:
                    if x==0:raise ValueError('expected')
                    await asyncio.sleep(100)
                finally:done.append(x)
            with self.assertRaises(ValueError):await bounded_fanout([0,1],worker,2)
            return done
        self.assertIn(0,asyncio.run(run()))
    def test_projects(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';a.mkdir();b=Path(d)/'b';b.mkdir()
            self.assertEqual(ticket_project(a)['rows'],1);self.assertTrue(coding_project(b)['success'])

class ReliabilityTests(unittest.TestCase):
    def test_permanent_failure_not_retried(self):
        calls=[]
        def f():calls.append(1);raise PermanentFailure()
        with self.assertRaises(PermanentFailure):retry(f)
        self.assertEqual(len(calls),1)
    def test_redaction_nested(self):
        self.assertEqual(redact({'x':[{'api_key':'secret'}]})['x'][0]['api_key'],'[REDACTED]')
    def test_empty_metrics_not_zero_success(self):
        self.assertIsNone(metrics([])['success_rate'])

if __name__=='__main__':unittest.main()
