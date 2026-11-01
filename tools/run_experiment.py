#!/usr/bin/env python3
"""Run one bounded experiment and save local evidence, not a security sandbox."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def execute(command, directory: Path, timeout: float, gpu: str | None = None):
    if not command or not 0 < timeout <= 3600:
        raise ValueError("nonempty command and timeout in (0,3600] required")
    directory.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu if gpu is not None else ""
    kwargs = {"start_new_session": True} if os.name == "posix" else {}
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.monotonic()
    timed_out = False
    with (directory / "stdout.txt").open("wb") as out, (directory / "stderr.txt").open("wb") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, env=environment, shell=False, **kwargs)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
            code = 124
    report = {"started_at_utc": started, "command": command, "exit_code": code,
              "timeout_seconds": timeout, "timed_out": timed_out,
              "wall_seconds": time.monotonic()-t0, "visible_gpu": gpu,
              "scope": "process timeout only; no network isolation or VRAM limit"}
    (directory / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--gpu", help="explicit assigned device ID(s); omit to hide CUDA devices")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command and command[0] in ("python", "python3"):
        command[0] = sys.executable
    root = Path(__file__).resolve().parents[1]
    directory = root / ".course" / "runs" / uuid.uuid4().hex[:12]
    report = execute(command, directory, args.seconds, args.gpu)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Local evidence:", directory)
    return report["exit_code"]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
