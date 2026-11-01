"""Context planning using explicit character units, not pretending to count tokens."""
import json
from .contracts import ContractError

def turn_groups(messages):
    """Atomic assistant-tool groups. User/system messages each form one group."""
    groups=[];i=0
    while i<len(messages):
        message=messages[i]
        if message.get("role")=="tool":raise ContractError("orphan tool result")
        group=[message];i+=1
        calls=message.get("tool_calls",[])
        if calls:
            need={x["id"] for x in calls}
            if len(need)!=len(calls):raise ContractError("duplicate pending call")
            seen=set()
            while i<len(messages) and messages[i].get("role")=="tool":
                mid=messages[i].get("tool_call_id")
                if mid not in need or mid in seen:raise ContractError("bad tool association")
                seen.add(mid);group.append(messages[i]);i+=1
            if seen!=need:raise ContractError("incomplete tool exchange")
        groups.append(group)
    return groups

def select_context(messages, max_chars, output_reserve=0):
    """Keep leading system instructions and newest complete suffix; fail if essential
    content cannot fit. Production uses the provider tokenizer + template overhead.
    """
    available=max_chars-output_reserve
    if available<0:raise ContractError("output reserve exceeds context")
    groups=turn_groups(messages)
    pinned=[]
    while groups and groups[0][0].get("role")=="system":pinned.extend(groups.pop(0))
    cost=lambda x:len(json.dumps(x,ensure_ascii=False,separators=(",",":")))
    chosen=[]
    for group in reversed(groups):
        candidate=group+chosen
        if cost(pinned+candidate)<=available:chosen=candidate
        elif not chosen:raise ContractError("latest atomic exchange cannot fit")
        else:break
    if cost(pinned+chosen)>available:raise ContractError("system instructions cannot fit")
    return pinned+chosen

def evidence_summary(facts):
    """Deterministic extractive summary retaining provenance and uncertainty."""
    return [{"statement":f["statement"],"source":f["source"],"confidence":f.get("confidence","unknown")}
            for f in facts if f.get("verified") is True]
