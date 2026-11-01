"""Bounded concurrent HTTP lesson; dry-run by default, loopback only.

The thread pool limits in-flight calls, not total wall time. Use an outer
process timeout for a hard experiment deadline. Streaming chunks are not tokens.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import time
import urllib.parse
from .http_benchmark import load_prompt, measure


def run_bounded(call, count=4, concurrency=2):
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("count must be 1..8")
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("concurrency must be 1..4")
    submitted = time.perf_counter()

    def task(index):
        started = time.perf_counter()
        try:
            payload = call(index)
            result = {"index": index, "ok": True, "payload": payload}
        except Exception as exc:
            result = {"index": index, "ok": False, "error_type": type(exc).__name__}
        result["client_queue_ms"] = (started-submitted)*1000
        result["call_wall_ms"] = (time.perf_counter()-started)*1000
        return result

    rows = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(task, index) for index in range(count)]
        for future in as_completed(futures):
            rows.append(future.result())
    return {"time_kind": "client_wall_clock", "concurrency_limit": concurrency,
            "attempted": count, "successful": sum(row["ok"] for row in rows),
            "wall_ms": (time.perf_counter()-submitted)*1000,
            "rows": sorted(rows,key=lambda row:row["index"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url",default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model",required=True)
    parser.add_argument("--requests",type=int,default=4)
    parser.add_argument("--concurrency",type=int,default=2)
    parser.add_argument("--max-tokens",type=int,default=16)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--prompt-file",help="local UTF-8 public fixture, at most 65536 bytes")
    args = parser.parse_args()
    parsed = urllib.parse.urlparse(args.base_url)
    if (parsed.scheme not in {"http","https"}
            or parsed.hostname not in {"localhost","127.0.0.1","::1"}
            or parsed.username or parsed.password):
        parser.error("this lesson accepts loopback HTTP(S) only, without embedded credentials")
    if not 1 <= args.requests <= 8 or not 1 <= args.concurrency <= 4 or not 1 <= args.max_tokens <= 64:
        parser.error("bounds: requests 1..8, concurrency 1..4, max-tokens 1..64")
    try:
        prompt = load_prompt(args.prompt_file)
    except (OSError,ValueError) as exc:
        parser.error(str(exc))
    if not args.execute:
        print(json.dumps({"dry_run":True,"requests":args.requests,
                          "concurrency_limit":args.concurrency,
                          "max_output_tokens_total":args.requests*args.max_tokens,
                          "prompt_utf8_bytes":len(prompt.encode("utf-8"))}))
        return
    report = run_bounded(
        lambda index: measure(args.base_url,args.model,requests=1,max_tokens=args.max_tokens,prompt=prompt)[0],
        args.requests,args.concurrency)
    print(json.dumps(report,indent=2))
    if report["successful"] != report["attempted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
