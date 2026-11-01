"""UTF-8/SSE parsing and tool-call delta assembly, intentionally separate layers."""
from __future__ import annotations
import codecs
import json
from .contracts import ContractError, ToolCall, parse_json

def sse_events(chunks, max_event_chars=100000):
    """Yield SSE data strings. Handles UTF-8 splits, LF/CRLF and multiline data.
    Requires a terminating blank line; truncated final frames are errors.
    """
    decoder=codecs.getincrementaldecoder("utf-8")()
    buffer="";lines=[];size=0
    def process_line(line):
        nonlocal lines,size
        if line=="":
            result="\n".join(lines) if lines else None
            lines=[];size=0
            return result
        if line.startswith("data:"):
            value=line[5:]
            if value.startswith(" "):value=value[1:]
            lines.append(value);size+=len(value)
            if size>max_event_chars:raise ContractError("SSE event exceeds limit")
        return None
    for chunk in chunks:
        buffer+=decoder.decode(chunk)
        while "\n" in buffer:
            line,buffer=buffer.split("\n",1)
            if len(line)>max_event_chars:raise ContractError("SSE line exceeds limit")
            if line.endswith("\r"):line=line[:-1]
            event=process_line(line)
            if event is not None:yield event
        if len(buffer)>max_event_chars:raise ContractError("SSE line exceeds limit")
    buffer+=decoder.decode(b"",final=True)
    if buffer or lines:raise ContractError("truncated SSE frame")

class ToolDeltaAssembler:
    def __init__(self, max_argument_chars=10000,max_calls=16):
        self.parts={};self.limit=max_argument_chars;self.max_calls=max_calls
    def feed(self,index,call_id=None,name=None,arguments=""):
        if type(index) is not int or index<0:raise ContractError("invalid delta index")
        if index not in self.parts and len(self.parts)>=self.max_calls:raise ContractError("too many calls")
        p=self.parts.setdefault(index,{"id":None,"name":None,"arguments":""})
        for key,value in [("id",call_id),("name",name)]:
            if value is not None:
                if p[key] is not None and p[key]!=value:raise ContractError("conflicting delta metadata")
                p[key]=value
        p["arguments"]+=arguments
        if len(p["arguments"])>self.limit:raise ContractError("arguments exceed limit")
    def finish(self):
        result=[]
        for index in sorted(self.parts):
            p=self.parts[index]
            if not p["id"] or not p["name"]:raise ContractError("incomplete call metadata")
            parse_json(p["arguments"])
            result.append(ToolCall(**p))
        if len({x.id for x in result})!=len(result):raise ContractError("duplicate call ids")
        return result
