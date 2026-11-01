"""Offline JSON-RPC tools method exercise. Not a complete MCP transport or SDK.
The real host owns version/capability metadata, authorization, transport, and
current specification validation. See chapter A15 before interoperability work.
"""
from .contracts import parse_json,validate,ContractError

class ToolRPC:
    def __init__(self,tools):self.tools={t.name:t for t in tools}
    def request(self,message):
        if type(message) is not dict:return {'jsonrpc':'2.0','id':None,'error':{'code':-32600,'message':'request must be an object'}}
        mid=message.get('id')
        def error(code,detail):return {'jsonrpc':'2.0','id':mid,'error':{'code':code,'message':detail}}
        if message.get('jsonrpc')!='2.0' or type(mid) not in (str,int):return error(-32600,'invalid request; notifications not supported by this exercise')
        params=message.get('params',{})
        if type(params) is not dict:return error(-32602,'params must be an object')
        meta=params.get('_meta',{})
        if type(meta) is not dict:return error(-32602,'metadata must be an object')
        if meta.get('io.modelcontextprotocol/protocolVersion')!='2026-07-28' or 'io.modelcontextprotocol/clientInfo' not in meta or 'io.modelcontextprotocol/clientCapabilities' not in meta:
            return error(-32602,'required current protocol metadata missing')
        method=message.get('method')
        if method=='tools/list':
            result={'resultType':'complete','tools':[{'name':t.name,'description':t.description,'inputSchema':t.schema} for t in self.tools.values()]}
        elif method=='tools/call':
            name=params.get('name')
            if type(name) is not str:return error(-32602,'tool name must be a string')
            tool=self.tools.get(name)
            if tool is None:return error(-32602,'unknown tool')
            if tool.side_effect:return error(-32602,'offline RPC exercise permits pure tools only')
            try:
                args=params.get('arguments',{})
                validate(args,tool.schema)
                value=tool.function(args)
                result={'resultType':'complete','content':[{'type':'text','text':str(value)}],'isError':False}
            except ContractError as exc:return error(-32602,str(exc))
            except Exception:
                result={'resultType':'complete','content':[{'type':'text','text':'tool execution failed'}],'isError':True}
        else:return error(-32601,'method not found')
        return {'jsonrpc':'2.0','id':mid,'result':result}

def request_message(mid,method,params=None):
    params=dict(params or {})
    params['_meta']={'io.modelcontextprotocol/protocolVersion':'2026-07-28','io.modelcontextprotocol/clientInfo':{'name':'course-offline','version':'1.0.0'},'io.modelcontextprotocol/clientCapabilities':{}}
    return {'jsonrpc':'2.0','id':mid,'method':method,'params':params}
