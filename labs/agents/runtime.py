"""A synchronous, checkpointable teaching harness with one transition per step."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import hashlib
import json
import time
from .contracts import Tool, ToolCall, Turn, ContractError, parse_json, validate
from .store import Store, UncertainExecution, Conflict

@dataclass
class Budget:
    model_calls: int = 8
    tool_calls: int = 16
    tool_output_chars: int = 20000
    max_argument_chars: int = 10000
    def __post_init__(self):
        values=(self.model_calls,self.tool_calls,self.tool_output_chars,self.max_argument_chars)
        if any(type(value) is not int or value<0 for value in values):
            raise ValueError("budgets must be nonnegative integers")

@dataclass
class State:
    run_id: str
    messages: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    events: list = field(default_factory=list)
    model_calls: int = 0
    tool_calls: int = 0
    output_chars: int = 0
    status: str = "ready"
    version: int = 0
    final: str = ""
    inflight_model: bool = False

class ScriptedModel:
    """Deterministic scripted responses; cursor is derived from persisted model calls."""
    def __init__(self, turns):
        self.turns=turns
    def complete(self, messages, tools, index):
        if index >= len(self.turns):
            raise RuntimeError("script exhausted")
        value=self.turns[index]
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)

class Harness:
    def __init__(self, model, tools, store=None, budget=None, deadline=None, clock=time.monotonic):
        self.model=model
        self.tools={tool.name:tool for tool in tools}
        if len(self.tools)!=len(tools):
            raise ValueError("duplicate tool names")
        self.store=store or Store()
        self.budget=budget or Budget()
        self.deadline=deadline
        self.clock=clock
    def new(self, run_id, goal):
        state=State(run_id=run_id,messages=[{"role":"user","content":goal}])
        self._event(state,"run_started",{})
        self._save(state)
        return state
    def resume(self, run_id):
        return State(**self.store.load(run_id))
    def _save(self,state):
        state.version=self.store.save(asdict(state),state.version)
    def _event(self,state,kind,data):
        state.events.append({"seq":len(state.events),"kind":kind,"data":data})
    def _stop(self,state,status,reason):
        state.status=status
        self._event(state,"run_stopped",{"reason":reason})
        self._save(state)
        return state
    def cancel(self,state):
        return self._stop(state,"cancelled","requested by caller")
    def step(self,state):
        if state.status in {"finished","cancelled","limit","uncertain","error"}:
            return state
        if state.inflight_model:
            return self._stop(state,"uncertain","model request outcome unknown after interruption; reconcile before continuing")
        if self.deadline is not None and self.clock() >= self.deadline:
            return self._stop(state,"limit","deadline reached")
        if state.pending:
            if state.tool_calls >= self.budget.tool_calls:
                return self._stop(state,"limit","tool call budget")
            call=ToolCall(**state.pending[0])
            try:
                output=self._execute(state,call)
            except (UncertainExecution,Conflict) as exc:
                return self._stop(state,"uncertain",str(exc))
            # Tool outcome exists in durable ledger before transcript checkpoint.
            if state.output_chars+len(output)>self.budget.tool_output_chars:
                return self._stop(state,"limit","tool output budget; output retained in ledger")
            state.pending.pop(0)
            state.tool_calls+=1
            state.output_chars+=len(output)
            state.messages.append({"role":"tool","tool_call_id":call.id,"content":output})
            self._event(state,"tool_completed",{"call_id":call.id,"name":call.name,"chars":len(output)})
            state.status="tools" if state.pending else "ready"
            self._save(state)
            return state
        if state.model_calls >= self.budget.model_calls:
            return self._stop(state,"limit","model call budget")
        # Requests count even if they fail. Persist reservation before external request.
        index=state.model_calls
        state.model_calls+=1
        state.inflight_model=True
        self._event(state,"model_requested",{"index":index})
        self._save(state)
        try:
            turn=self.model.complete(deepcopy(state.messages),[t.definition() for t in self.tools.values()],index)
            if not isinstance(turn,Turn):
                raise ContractError("model adapter must return Turn")
            ids=[c.id for c in turn.calls]
            previous={m.get("tool_call_id") for m in state.messages if m.get("role")=="tool"}
            if len(set(ids))!=len(ids) or set(ids)&previous:
                raise ContractError("duplicate tool call id in transcript")
            if not turn.calls and not turn.text.strip():
                raise ContractError("empty final response is not completion")
        except Exception as exc:
            state.inflight_model=False
            self._event(state,"model_error",{"type":type(exc).__name__})
            return self._stop(state,"error","model request failed; inspect events")
        state.inflight_model=False
        state.messages.append(turn.message())
        state.pending=[asdict(c) for c in turn.calls]
        self._event(state,"model_completed",{"index":index,"tool_count":len(turn.calls)})
        if turn.calls:
            state.status="tools"
        else:
            state.status="finished"
            state.final=turn.text
        self._save(state)
        return state
    def _execute(self,state,call):
        signature=hashlib.sha256((call.name+"\0"+call.arguments).encode()).hexdigest()
        tool=self.tools.get(call.name)
        side_effect=bool(tool and tool.side_effect)
        cached=self.store.start(state.run_id,call.id,signature,side_effect)
        if cached is not None:
            return cached
        try:
            if tool is None:
                raise ContractError("unknown tool")
            if len(call.arguments)>self.budget.max_argument_chars:
                raise ContractError("arguments too large")
            args=parse_json(call.arguments)
            validate(args,tool.schema)
        except ContractError as exc:
            # Only this pre-invocation scope can prove the tool never started.
            output=json.dumps({"ok":False,"error":{"code":"invalid_arguments","detail":str(exc)}},ensure_ascii=False)
        else:
            try:
                result=tool.function(args)
                output=json.dumps({"ok":True,"value":result},ensure_ascii=False,allow_nan=False)
            except Exception as exc:
                # Exception names cannot establish rollback after invocation.
                # This includes a tool's own ContractError and serialization errors.
                if side_effect:
                    raise UncertainExecution(f"tool raised {type(exc).__name__}; effect requires reconciliation") from exc
                output=json.dumps({"ok":False,"error":{"code":"tool_failed","type":type(exc).__name__}})
        self.store.complete(state.run_id,call.id,output)
        return output
    def run(self,state,max_transitions=100):
        if type(max_transitions) is not int or max_transitions<0:
            raise ValueError("max_transitions must be a nonnegative integer")
        for _ in range(max_transitions):
            if state.status not in {"ready","tools"}:
                return state
            self.step(state)
        # A final permitted transition may already have reached a terminal state.
        if state.status not in {"ready","tools"}:
            return state
        return self._stop(state,"limit","transition guard")
