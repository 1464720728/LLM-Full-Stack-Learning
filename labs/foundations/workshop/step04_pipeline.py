"""Step 4: stream input and classify bad records without external I/O."""
import json
from .step03_json import parse
from .step02_function import summarize_tokens

SAMPLE = ['{"request_id":"a","tokens":3}', '{"request_id":"b","tokens":5}',
          '{"request_id":"c","tokens":0}', 'broken']

def analyze(lines):
    values, errors, seen = [], [], set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = parse(line)
            if record.request_id in seen:
                raise ValueError('duplicate request_id')
        except ValueError as error:
            errors.append({'line': line_number, 'reason': str(error)})
            continue
        seen.add(record.request_id)
        values.append(record.tokens)
    result = summarize_tokens(values)
    result['invalid'] = len(errors)
    result['errors'] = errors
    return result

if __name__ == '__main__':
    print(json.dumps(analyze(SAMPLE), indent=2))
