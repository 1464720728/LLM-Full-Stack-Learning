"""Optional, offline real-SDK labs. No model API, tracing export, or HTTP listener.
Verified: langgraph 1.2.19, openai-agents 0.24.0, mcp 2.3.0.
Install only in an isolated environment; see A15. Core labs do not import this.
"""
import argparse
import asyncio
import json
from importlib.metadata import version


def langgraph_case():
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import interrupt, Command

    class ReviewState(TypedDict):
        title: str
        approved: bool
        result: str

    entries = []
    def review(state):
        entries.append(state['title'])  # Deliberate evidence of node re-entry.
        decision = interrupt({'title': state['title'], 'question': 'Approve?'})
        if type(decision) is not bool:
            raise ValueError('exact boolean approval required')
        return {'approved': decision}

    def finish(state):
        return {'result': 'approved' if state['approved'] else 'rejected'}

    builder = StateGraph(ReviewState)
    builder.add_node('review', review)
    builder.add_node('finish', finish)
    builder.add_edge(START, 'review')
    builder.add_edge('review', 'finish')
    builder.add_edge('finish', END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'course-review-1'}}
    paused = graph.invoke({'title': 'Read-only audit', 'approved': False,
                           'result': ''}, config)
    assert paused['__interrupt__']
    resumed = graph.invoke(Command(resume=True), config)
    assert resumed['result'] == 'approved'
    assert entries == ['Read-only audit', 'Read-only audit']
    assert graph.get_state(config).next == ()
    return {'version': version('langgraph'), 'node_entries': len(entries),
            'result': resumed['result']}


async def mcp_case():
    from mcp import Client
    from mcp.server import MCPServer
    from pydantic import StrictInt

    server = MCPServer('Course offline arithmetic')
    @server.tool()
    def exact_add(a: StrictInt, b: StrictInt) -> int:
        """Add two exact integers without coercing booleans or strings."""
        return a + b

    async with Client(server) as client:
        listed = await client.list_tools()
        # The installed SDK returns a ListToolsResult object.
        names = [tool.name for tool in listed.tools]
        assert 'exact_add' in names
        good = await client.call_tool('exact_add', {'a': 17, 'b': 25})
        assert good.structured_content == {'result': 42}
        bad = await client.call_tool('exact_add', {'a': True, 'b': 25})
        assert bad.is_error is True
    return {'version': version('mcp'), 'tools': names,
            'result': good.structured_content, 'strict_bool_rejected': bad.is_error}


async def agents_case():
    from agents import Agent, Runner, Model, ModelResponse, RunConfig, function_tool
    from agents.usage import Usage
    from agents.tracing import set_trace_provider
    from agents.tracing.provider import DefaultTraceProvider
    # A provider with no exporters avoids even constructing a network client.
    set_trace_provider(DefaultTraceProvider())
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
    from pydantic import StrictInt

    calls = []
    @function_tool
    def exact_add(a: StrictInt, b: StrictInt) -> int:
        """Add two exact integers."""
        calls.append((a, b))
        return a + b

    class ScriptedSDKModel(Model):
        def __init__(self):
            self.inputs = []
        async def get_response(self, system_instructions, input, model_settings,
                               tools, output_schema, handoffs, tracing, **kwargs):
            self.inputs.append(input)
            if len(self.inputs) == 1:
                output = [ResponseFunctionToolCall(type='function_call',
                    name='exact_add', arguments='{"a":17,"b":25}',
                    call_id='sdk-call-1', id='sdk-item-1', status='completed')]
            else:
                assert any(isinstance(x, dict) and x.get('type') == 'function_call_output'
                           and x.get('call_id') == 'sdk-call-1' for x in input)
                output = [ResponseOutputMessage(id='sdk-message-1', type='message',
                    role='assistant', status='completed', content=[ResponseOutputText(
                        type='output_text', text='42', annotations=[])])]
            return ModelResponse(output=output, usage=Usage(), response_id=None)
        async def stream_response(self, *args, **kwargs):
            raise NotImplementedError('offline fixture is non-streaming')
            yield  # Async-generator interface; deliberately unsupported.

    model = ScriptedSDKModel()
    agent = Agent(name='Course arithmetic', model=model, tools=[exact_add])
    result = await Runner.run(agent, 'Compute 17 + 25', max_turns=2,
                              run_config=RunConfig(tracing_disabled=True))
    assert result.final_output == '42'
    assert calls == [(17, 25)]
    assert len(model.inputs) == 2
    return {'version': version('openai-agents'), 'model_calls': len(model.inputs),
            'tool_calls': len(calls), 'final': result.final_output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['langgraph', 'mcp', 'agents', 'all'])
    part = parser.parse_args().part
    results = {}
    try:
        if part in ('langgraph', 'all'):
            results['langgraph'] = langgraph_case()
        if part in ('mcp', 'all'):
            results['mcp'] = asyncio.run(mcp_case())
        if part in ('agents', 'all'):
            results['agents'] = asyncio.run(agents_case())
    except ImportError as exc:
        raise SystemExit(f'Optional SDK import failed: {exc}. Use the isolated A15 environment.') from exc
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
