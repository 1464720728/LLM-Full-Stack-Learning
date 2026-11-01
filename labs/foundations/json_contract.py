"""JSON syntax validity and business validity are different checks."""
import json
from dataclasses import dataclass

@dataclass(frozen=True)
class Request:
    request_id: str
    tokens: int

def parse_request(text: str) -> Request:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"request_id", "tokens"}:
        raise ValueError("object with exactly request_id and tokens required")
    if not isinstance(value["request_id"], str) or not value["request_id"].strip():
        raise ValueError("non-empty request_id required")
    if type(value["tokens"]) is not int or value["tokens"] < 0:
        raise ValueError("tokens must be a non-negative integer")
    return Request(value["request_id"], value["tokens"])

if __name__ == "__main__":
    print(parse_request('{"request_id":"r1","tokens":12}'))
