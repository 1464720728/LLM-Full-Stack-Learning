"""Explicitly opt-in Chat Completions tool loop using only the standard library.

Not run by tests. The model, endpoint and supported optional fields are supplied
by the learner. No automatic retry or cross-provider protocol conversion.
"""
import argparse
import json
import os
from urllib import request,parse,error
from .contracts import ContractError
from .protocols import from_chat
from .runtime import Harness,Budget
from .tools import arithmetic_tool

class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise ContractError('redirect refused: verify the explicit API endpoint')

class ChatModel:
    def __init__(self,endpoint,model,key,extra=None,timeout=30,allow_http_localhost=False):
        u=parse.urlparse(endpoint)
        local=u.hostname in {'localhost','127.0.0.1','::1'}
        if u.username or u.password or u.fragment or not u.hostname:raise ContractError('invalid endpoint URL')
        if u.scheme!='https' and not (allow_http_localhost and local and u.scheme=='http'):
            raise ContractError('HTTPS required except explicit localhost testing')
        self.endpoint=endpoint;self.model=model;self.key=key;self.extra=extra or {};self.timeout=timeout
        if type(self.extra) is not dict:raise ContractError('extra fields must be a JSON object')
        if {'messages','tools','model','stream'} & self.extra.keys():raise ContractError('extra fields cannot replace core request fields')
    def complete(self,messages,tools,index):
        body=dict(self.extra,model=self.model,messages=messages,tools=tools,stream=False)
        req=request.Request(self.endpoint,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},method='POST')
        with request.build_opener(NoRedirect()).open(req,timeout=self.timeout) as response:
            raw=response.read(2_000_001)
            if len(raw)>2_000_000:raise ContractError('response size limit')
        return from_chat(json.loads(raw))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-live',action='store_true',help='explicitly allow paid network requests')
    p.add_argument('--endpoint',required=True,help='complete /chat/completions HTTPS URL')
    p.add_argument('--model',required=True)
    p.add_argument('--key-env',default='COURSE_API_KEY')
    p.add_argument('--extra-json',default='{}',help='provider-specific fields confirmed from its docs')
    p.add_argument('--allow-http-localhost',action='store_true')
    args=p.parse_args()
    if not args.allow_live:p.error('add --allow-live after reviewing potential API cost')
    key=os.environ.get(args.key_env)
    if not key:p.error('set the named API key environment variable; do not put the key in the command')
    model=ChatModel(args.endpoint,args.model,key,json.loads(args.extra_json),allow_http_localhost=args.allow_http_localhost)
    h=Harness(model,[arithmetic_tool()],budget=Budget(model_calls=4,tool_calls=4,tool_output_chars=4000))
    state=h.run(h.new('live-arithmetic','Use the add tool to compute 17 + 25, then briefly explain the result.'))
    print(json.dumps({'status':state.status,'final':state.final,'model_calls':state.model_calls,'tool_calls':state.tool_calls},ensure_ascii=False,indent=2))
    if state.status!='finished':raise SystemExit(1)

if __name__=='__main__':main()
