"""Testable retry policy and log redaction; no sleeps in offline tests."""
import random
import re

class TransientFailure(Exception):pass
class PermanentFailure(Exception):pass

def retry(operation,max_attempts=3,base_delay=0.1,max_delay=2.0,sleep=lambda seconds:None,rng=None):
    """Retry only the explicit transient class. Caller owns idempotency and budget."""
    if max_attempts<1:raise ValueError('at least one attempt required')
    rng=rng or random.Random()
    for attempt in range(max_attempts):
        try:return operation()
        except TransientFailure:
            if attempt+1==max_attempts:raise
            cap=min(max_delay,base_delay*2**attempt)
            sleep(rng.uniform(0,cap))


def redact(record):
    """Structured redaction is preferred. Regex is only a defense in depth example."""
    private={'authorization','api_key','password','secret','token'}
    if isinstance(record,dict):return {k:'[REDACTED]' if k.lower() in private else redact(v) for k,v in record.items()}
    if isinstance(record,list):return [redact(x) for x in record]
    if isinstance(record,str):return re.sub(r'\bsk-[A-Za-z0-9_-]{8,}\b','[REDACTED]',record)
    return record


def metrics(outcomes):
    """Keep denominator semantics explicit; abstentions are not successes."""
    n=len(outcomes)
    if not n:return {'n':0,'success_rate':None,'mean_model_calls':None,'mean_tool_calls':None}
    return {'n':n,'success_rate':sum(x['success'] is True for x in outcomes)/n,
            'mean_model_calls':sum(x['model_calls'] for x in outcomes)/n,
            'mean_tool_calls':sum(x['tool_calls'] for x in outcomes)/n}
