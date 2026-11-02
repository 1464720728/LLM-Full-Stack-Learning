"""Two complete deterministic integration projects, not claims of LLM competence."""
from pathlib import Path
import hashlib
import json
import tempfile
from .contracts import Tool,ToolCall,Turn,object_schema
from .runtime import Harness,ScriptedModel
from .store import Store
from .tools import Workspace,TicketBook,run_trusted_fixture_tests
from .workflows import Workflow,GraphState,approval_node,action_digest

def ticket_project(root):
    root=Path(root)
    book=TicketBook(root/'tickets.db')
    action={'request_id':'request-001','title':'Investigate retry storm'}
    executed=[]
    def execute(data):
        if not data['authorized']:return {'outcome':'declined'}
        result=book.create(**data['proposed_action'])
        executed.append(result)
        return {'outcome':result}
    graph=Workflow({'review':approval_node,'execute':execute},{'review':'execute','execute':'END'})
    state=graph.run(GraphState('review',{'proposed_action':action}))
    assert state.status=='waiting' and executed==[]
    # Simulates a human approving a concrete action digest. Not automatic production consent.
    graph.approve(state,action_digest(action),True)
    graph.run(state)
    replay=book.create(**action)
    count=book.db.execute('SELECT count(*) FROM tickets').fetchone()[0]
    book.close()
    return {'status':state.status,'rows':count,'created_first':executed[0]['created'],'created_replay':replay['created'],'history':state.history}

def coding_project(root):
    root=Path(root)
    buggy='def total(price, rate):\n    return price * (1 - rate)\n'
    (root/'price.py').write_text(buggy,encoding='utf-8',newline='\n')
    (root/'test_price.py').write_text('import unittest\nfrom price import total\n\nclass PriceTests(unittest.TestCase):\n    def test_tax(self):\n        self.assertAlmostEqual(total(100,0.2),120)\n    def test_zero(self):\n        self.assertEqual(total(100,0),100)\n',encoding='utf-8')
    before=run_trusted_fixture_tests(root)
    workspace=Workspace(root)
    sha=hashlib.sha256(buggy.encode()).hexdigest()
    turns=[Turn(calls=[ToolCall('read-1','read_file','{"path":"price.py"}')]),
           Turn(calls=[ToolCall('edit-1','replace_text',json.dumps({'path':'price.py','old':'1 - rate','new':'1 + rate','expected_sha256':sha}))]),
           Turn(calls=[ToolCall('test-1','test_fixture','{}')]),
           Turn('Changed subtraction to addition; independent tests confirm the tax formula.')]
    test_tool=Tool('test_fixture','Run only the trusted course fixture tests.',object_schema({}),lambda _:run_trusted_fixture_tests(root))
    with Store(root/'agent.db') as store:
        h=Harness(ScriptedModel(turns),workspace.tools()+[test_tool],store)
        state=h.run(h.new('coding-001','Fix tax addition; preserve the zero-rate case.'))
        # Completion is independently checked; the model's final claim is not the verifier.
        verification=run_trusted_fixture_tests(root)
        return {'status':state.status,'before_exit':before['exit_code'],'after_exit':verification['exit_code'],
                'tool_calls':state.tool_calls,'model_calls':state.model_calls,'final_code':(root/'price.py').read_text(),
                'success':state.status=='finished' and before['exit_code']!=0 and verification['exit_code']==0}

def main():
    with tempfile.TemporaryDirectory() as tmp:
        a=Path(tmp)/'tickets';a.mkdir()
        b=Path(tmp)/'coding';b.mkdir()
        print(json.dumps({'ticket':ticket_project(a),'coding':coding_project(b)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
