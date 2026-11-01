"""Contracts and a deliberately documented JSON Schema subset."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import math
from typing import Any, Callable

class ContractError(ValueError):
    pass


def parse_json(text: str) -> Any:
    """Reject duplicate keys and non-JSON NaN/Infinity instead of silently accepting."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(name):
        raise ContractError(f"non-JSON numeric constant: {name}")
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ContractError("invalid JSON arguments") from exc


def validate(value: Any, schema: dict, path: str = "$") -> None:
    """Subset: type, properties, required, additionalProperties=False, enum,
    items, min/maxItems, min/maxLength, minimum, maximum. Unsupported keywords
    are rejected so this cannot masquerade as a complete JSON Schema validator.
    """
    allowed = {"type", "properties", "required", "additionalProperties", "enum",
               "items", "minItems", "maxItems", "minLength", "maxLength",
               "minimum", "maximum", "description"}
    unknown = set(schema) - allowed
    if unknown:
        raise ContractError(f"unsupported schema keywords: {sorted(unknown)}")
    typ = schema.get("type")
    checks = {"object": lambda x: type(x) is dict, "array": lambda x: type(x) is list,
              "string": lambda x: type(x) is str, "integer": lambda x: type(x) is int,
              "number": lambda x: type(x) in (int, float) and math.isfinite(x),
              "boolean": lambda x: type(x) is bool, "null": lambda x: x is None}
    if typ not in checks:
        raise ContractError(f"{path}: unsupported or missing type")
    if not checks[typ](value):
        raise ContractError(f"{path}: expected {typ}")
    if "enum" in schema and not any(type(value) is type(x) and value == x for x in schema["enum"]):
        raise ContractError(f"{path}: not in enum")
    if typ == "object":
        props = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise ContractError(f"{path}: missing {sorted(missing)}")
        extra = set(value) - set(props)
        if extra and schema.get("additionalProperties") is False:
            raise ContractError(f"{path}: unexpected {sorted(extra)}")
        for key in value.keys() & props.keys():
            validate(value[key], props[key], f"{path}.{key}")
    if typ == "array":
        if "items" not in schema:
            raise ContractError("arrays require an items schema in this laboratory")
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{i}]")
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")):
            raise ContractError(f"{path}: array length out of range")
    if typ == "string" and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", float("inf")):
        raise ContractError(f"{path}: string length out of range")
    if typ in ("integer", "number") and not schema.get("minimum", -float("inf")) <= value <= schema.get("maximum", float("inf")):
        raise ContractError(f"{path}: numeric value out of range")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def __post_init__(self):
        if not self.id or not self.name:
            raise ContractError("tool call id and name must be nonempty")

    def wire(self):
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class Turn:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    # Preserves provider-specific assistant fields; not portable to other protocols.
    provider_message: dict | None = None

    def message(self):
        if self.provider_message is not None:
            return dict(self.provider_message)
        result = {"role": "assistant", "content": self.text}
        if self.calls:
            result["tool_calls"] = [call.wire() for call in self.calls]
        return result


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    function: Callable[[dict], Any]
    side_effect: bool = False
    def definition(self):
        return {"type": "function", "function": {"name": self.name,
                "description": self.description, "parameters": self.schema}}


def object_schema(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else list(required), "additionalProperties": False}
