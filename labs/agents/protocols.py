"""Protocol parsing, not a claim of universal cross-provider equivalence."""
from copy import deepcopy
import json
from .contracts import ContractError, ToolCall, Turn

def from_chat(payload):
    try:
        choice=payload["choices"][0]
        if choice.get("finish_reason") in {"length","content_filter"}:
            raise ContractError("chat response did not complete normally")
        msg=deepcopy(choice["message"])
        calls=[ToolCall(x["id"],x["function"]["name"],x["function"]["arguments"]) for x in msg.get("tool_calls",[])]
        return Turn(msg.get("content") or "",calls,msg)
    except (KeyError,IndexError,TypeError) as exc:
        raise ContractError("malformed Chat Completions response") from exc

def from_responses(payload):
    if payload.get("status") not in {None,"completed"}:
        raise ContractError("Responses output is incomplete")
    text=[]; calls=[]
    for item in payload.get("output",[]):
        if item.get("type")=="function_call":
            calls.append(ToolCall(item["call_id"],item["name"],item["arguments"]))
        elif item.get("type")=="message":
            for block in item.get("content",[]):
                if block.get("type")=="output_text": text.append(block["text"])
                elif block.get("type")=="refusal": raise ContractError("provider refusal")
        elif item.get("type")=="reasoning":
            continue # caller retains full payload separately for same-protocol replay
        else:
            raise ContractError("unsupported Responses item; do not silently discard")
    return Turn("\n".join(text),calls)

def responses_continuation(previous_payload, outputs):
    """Replay ALL output items (including opaque reasoning) plus function outputs.
    Built-in tools/custom tools are outside this adapter's supported subset.
    """
    expected={x["call_id"] for x in previous_payload.get("output",[]) if x.get("type")=="function_call"}
    if expected != set(outputs):
        raise ContractError("one output per function call is required")
    return deepcopy(previous_payload.get("output",[]))+[{"type":"function_call_output","call_id":k,"output":v} for k,v in outputs.items()]

def from_messages(payload):
    if payload.get("stop_reason") in {"max_tokens","refusal"}:
        raise ContractError("Messages output is incomplete or refused")
    text=[];calls=[]
    for block in payload.get("content",[]):
        typ=block.get("type")
        if typ=="text": text.append(block["text"])
        elif typ=="tool_use": calls.append(ToolCall(block["id"],block["name"],json.dumps(block["input"],ensure_ascii=False)))
        elif typ in {"thinking","redacted_thinking"}: continue
        else: raise ContractError("unsupported Messages block")
    return Turn("\n".join(text),calls)

def messages_continuation(previous_payload, outputs):
    expected={b["id"] for b in previous_payload.get("content",[]) if b.get("type")=="tool_use"}
    if expected != set(outputs): raise ContractError("missing or extra tool result")
    return [{"role":"assistant","content":deepcopy(previous_payload["content"])},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":k,"content":v} for k,v in outputs.items()]}]
