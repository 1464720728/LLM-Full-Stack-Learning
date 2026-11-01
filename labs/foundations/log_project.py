"""Project A: validate JSONL records and produce a deterministic report.

Percentiles use nearest-rank, intentionally stated rather than delegated to a
library default. Exact percentiles retain valid records, so memory is O(n).
"""
import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from .file_io import atomic_json

@dataclass(frozen=True)
class Record:
    request_id: str
    latency_ms: float
    tokens: int
    status: str

class InvalidRecord(ValueError):
    pass

def parse_record(line: str) -> Record:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as e:
        raise InvalidRecord("invalid JSON") from e
    if not isinstance(row, dict):
        raise InvalidRecord("expected object")
    if set(row) != {"request_id", "latency_ms", "tokens", "status"}:
        raise InvalidRecord("unexpected or missing fields")
    if not isinstance(row["request_id"], str) or not row["request_id"].strip():
        raise InvalidRecord("empty request_id")
    latency = row["latency_ms"]
    if type(latency) not in (int, float) or latency < 0:
        raise InvalidRecord("latency_ms must be finite and non-negative")
    try:
        latency = float(latency)
    except OverflowError as e:
        raise InvalidRecord("latency_ms must be finite and non-negative") from e
    if not math.isfinite(latency):
        raise InvalidRecord("latency_ms must be finite and non-negative")
    if type(row["tokens"]) is not int or row["tokens"] < 0:
        raise InvalidRecord("tokens must be a non-negative integer")
    if row["status"] not in ("ok", "error"):
        raise InvalidRecord("status must be ok or error")
    return Record(row["request_id"], float(latency), row["tokens"], row["status"])

def nearest_rank(values: list[float], percentile: float):
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]

def analyze(lines: Iterable[str], strict=False) -> dict:
    records, errors, seen = [], [], set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = parse_record(line)
            if record.request_id in seen:
                raise InvalidRecord("duplicate request_id")
            seen.add(record.request_id)
            records.append(record)
        except InvalidRecord as e:
            if strict:
                raise InvalidRecord(f"line {line_number}: {e}") from e
            errors.append({"line": line_number, "reason": str(e)})
    latencies = [r.latency_ms for r in records]
    statuses = Counter(r.status for r in records)
    return {"schema_version": 1, "valid": len(records), "invalid": len(errors),
            "tokens": sum(r.tokens for r in records),
            "status_counts": {key: statuses[key] for key in ("ok", "error")},
            "latency_ms": {"mean": sum(latencies) / len(latencies) if latencies else None,
                           "p50_nearest_rank": nearest_rank(latencies, .5),
                           "p95_nearest_rank": nearest_rank(latencies, .95)},
            "errors": errors}

def sample_lines():
    rows = [dict(request_id="a", latency_ms=10, tokens=3, status="ok"),
            dict(request_id="b", latency_ms=30, tokens=5, status="ok"),
            dict(request_id="c", latency_ms=20, tokens=0, status="error")]
    return [json.dumps(r) for r in rows] + ["broken"]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.input:
            with args.input.open(encoding="utf-8") as f:
                report = analyze(f, args.strict)
        else:
            report = analyze(sample_lines(), args.strict)
        if args.output:
            atomic_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, UnicodeError, InvalidRecord) as e:
        parser.exit(2, f"error: {e}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
