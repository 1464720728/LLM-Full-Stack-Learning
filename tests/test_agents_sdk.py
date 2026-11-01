"""Optional real-framework integration checks, with no model APIs or listeners."""
import importlib.util
import unittest
from labs.agents.sdk_bridge import langgraph_case, mcp_case, agents_case


class OptionalSDKTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('langgraph'), 'optional langgraph absent')
    def test_graph_interrupt_reenters_before_resuming(self):
        result = langgraph_case()
        self.assertEqual(result['node_entries'], 2)
        self.assertEqual(result['result'], 'approved')


class OptionalAsyncSDKTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(importlib.util.find_spec('mcp'), 'optional mcp absent')
    async def test_real_mcp_strict_arguments_and_structured_result(self):
        result = await mcp_case()
        self.assertEqual(result['result'], {'result': 42})
        self.assertTrue(result['strict_bool_rejected'])

    @unittest.skipUnless(importlib.util.find_spec('agents'), 'optional agents SDK absent')
    async def test_real_runner_calls_tool_and_continues(self):
        result = await agents_case()
        self.assertEqual(result['model_calls'], 2)
        self.assertEqual(result['tool_calls'], 1)
        self.assertEqual(result['final'], '42')


if __name__ == '__main__':
    unittest.main()
