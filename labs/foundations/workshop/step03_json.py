"""Step 3: transform external JSON into validated immutable records."""
from dataclasses import dataclass
import json

@dataclass(frozen=True)
class TokenRecord:
    request_id: str
    tokens: int

def parse(line):
    row = json.loads(line)
    if not isinstance(row, dict) or set(row) != {'request_id', 'tokens'}:
        raise ValueError('exactly request_id and tokens are required')
    if not isinstance(row['request_id'], str) or not row['request_id'].strip():
        raise ValueError('non-empty request_id required')
    if type(row['tokens']) is not int or row['tokens'] < 0:
        raise ValueError('tokens must be a non-negative integer')
    return TokenRecord(row['request_id'], row['tokens'])

if __name__ == '__main__':
    record = parse('{"request_id":"a","tokens":3}')
    print(record)
    print('type of tokens:', type(record.tokens).__name__)
