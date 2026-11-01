#!/usr/bin/env python3
"""Offline course navigator. Uses only the Python standard library."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TRACKS = {"F": "foundations", "A": "agents", "R": "knowledge", "M": "models", "S": "serving", "K": "kernels"}
STATES = ("reading", "practiced", "mastered")


def catalog(root: Path = ROOT) -> tuple[list[dict], list[dict]]:
    tracks, chapters = [], []
    cross_path = root / "docs" / "PREREQUISITES.json"
    cross = json.loads(cross_path.read_text(encoding="utf-8")) if cross_path.exists() else {}
    for path in sorted((root / "books").glob("*/manifest.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_manifest"] = str(path.relative_to(root))
        tracks.append(data)
        for source in data.get("chapters", []):
            chapter = dict(source)
            chapter["track_id"] = data["track_id"]
            chapter.setdefault("prerequisites", [])
            chapter["prerequisites"] = list(dict.fromkeys(chapter["prerequisites"] + cross.get(chapter.get("id"), [])))
            chapter.setdefault("labs", [])
            chapters.append(chapter)
    return tracks, chapters


def safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes repository: {value}")
    return path


def validate(root: Path = ROOT) -> list[str]:
    errors = []
    try:
        tracks, chapters = catalog(root)
    except (ValueError, OSError, KeyError) as exc:
        return [f"cannot read manifests: {exc}"]
    if not tracks:
        return ["no book manifests found"]
    ids = [c.get("id") for c in chapters]
    if len(set(ids)) != len(ids):
        errors.append("duplicate chapter IDs")
    if {t.get("track_id") for t in tracks} != set(TRACKS):
        errors.append("expected six track IDs F, A, R, M, S, K")
    known = set(ids)
    cross_path = root / "docs" / "PREREQUISITES.json"
    if cross_path.exists():
        cross = json.loads(cross_path.read_text(encoding="utf-8"))
        for target in set(cross) - known:
            errors.append(f"cross-volume prerequisite target missing: {target}")
    labs = set()
    for c in chapters:
        ident = c.get("id", "?")
        try:
            path = safe_path(root, c["path"])
            if not path.is_file():
                errors.append(f"{ident}: missing chapter {c['path']}")
        except (KeyError, ValueError) as exc:
            errors.append(f"{ident}: {exc}")
        for dependency in c.get("prerequisites", []):
            if dependency not in known:
                errors.append(f"{ident}: unknown prerequisite {dependency}")
        for lab in c.get("labs", []):
            if not lab.get("id") or lab["id"] in labs:
                errors.append(f"{ident}: missing/duplicate lab ID {lab.get('id')}")
            labs.add(lab.get("id"))
            if not lab.get("command"):
                errors.append(f"{ident}: lab without command {lab.get('id')}")
            if not isinstance(lab.get("network", False), bool):
                errors.append(f"{ident}: network must be boolean")
    graph = {c["id"]: c.get("prerequisites", []) for c in chapters}
    visiting, visited = set(), set()

    def visit(node):
        if node in visiting:
            errors.append(f"prerequisite cycle at {node}")
            return
        if node in visited or node not in graph:
            return
        visiting.add(node)
        for child in graph[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
    return errors


def read_progress(root: Path = ROOT) -> dict:
    path = root / ".course" / "progress.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema_version": 1, "chapters": {}}


def record(chapter_id: str, status: str, evidence: str, root: Path = ROOT) -> None:
    _, chapters = catalog(root)
    if chapter_id not in {c["id"] for c in chapters}:
        raise ValueError(f"unknown chapter: {chapter_id}")
    if status not in STATES:
        raise ValueError(f"unknown status: {status}")
    if not evidence.strip():
        raise ValueError("evidence must not be empty")
    data = read_progress(root)
    data["chapters"][chapter_id] = {
        "status": status, "evidence": evidence,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    directory = root / ".course"
    directory.mkdir(exist_ok=True)
    fd, name = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(name, directory / "progress.json")
    finally:
        if os.path.exists(name):
            os.unlink(name)


def eligible(chapters: list[dict], progress: dict) -> list[dict]:
    mastered = {key for key, value in progress.get("chapters", {}).items() if value.get("status") == "mastered"}
    return [c for c in chapters if c["id"] not in mastered and set(c.get("prerequisites", [])).issubset(mastered)]


def doctor() -> dict:
    result = {
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(),
        "optional_commands": {name: shutil.which(name) for name in ("git", "docker", "nvcc", "nvidia-smi")},
        "modules": {name: importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "triton")},
        "gpu_probe": "not performed; this command does not import torch or initialize CUDA",
    }
    try:
        from importlib.metadata import version, PackageNotFoundError
        result["installed_versions"] = {}
        for name in ("numpy", "torch", "triton"):
            try:
                result["installed_versions"][name] = version(name)
            except PackageNotFoundError:
                pass
    except ImportError:
        pass
    return result


def prepare_command(lab: dict, allow_gpu: bool, allow_network: bool) -> list[str]:
    if lab.get("network", False) and not allow_network:
        raise ValueError("this lab uses network/API; read the chapter, then opt in with --allow-network")
    if lab.get("hardware", "cpu").lower() not in ("cpu", "none") and not allow_gpu:
        raise ValueError("this lab requires GPU; reserve the device, then opt in with --allow-gpu")
    command = shlex.split(lab["command"])
    if not command:
        raise ValueError("empty command")
    if command[0] in ("python", "python3"):
        command[0] = sys.executable
    if any(token in ("&&", "||", "|", ";", ">", "<") for token in command):
        raise ValueError("shell command chains are not supported; follow the chapter manually")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 全栈学习工程：离线导航与学习记录")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("map", help="列出所有分卷与章节")
    sub.add_parser("doctor", help="检查 Python 和可选工具，不初始化 GPU")
    sub.add_parser("check", help="校验课程路径与先修关系")
    nx = sub.add_parser("next", help="显示先修已满足的章节")
    nx.add_argument("--track", choices=list(TRACKS))
    ls = sub.add_parser("lesson", help="显示章节入口和实验")
    ls.add_argument("id")
    sr = sub.add_parser("search", help="检索教材正文")
    sr.add_argument("query")
    sr.add_argument("--limit", type=int, default=20)
    pr = sub.add_parser("record", help="记录学习证据")
    pr.add_argument("id")
    pr.add_argument("--status", choices=STATES, required=True)
    pr.add_argument("--evidence", required=True)
    vr = sub.add_parser("verify", help="运行离线测试")
    vr.add_argument("--track", choices=list(TRACKS))
    rn = sub.add_parser("run", help="运行一个实验，默认禁止 GPU／网络实验")
    rn.add_argument("id")
    rn.add_argument("--allow-gpu", action="store_true")
    rn.add_argument("--allow-network", action="store_true")
    rn.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    if args.action == "doctor":
        print(json.dumps(doctor(), ensure_ascii=False, indent=2))
        return 0
    if args.action == "check":
        errors = validate()
        for error in errors:
            print("ERROR:", error)
        if not errors:
            tracks, chapters = catalog()
            print(f"PASS: {len(tracks)} tracks, {len(chapters)} chapters, {sum(len(c['labs']) for c in chapters)} registered labs")
        return int(bool(errors))
    if args.action == "verify":
        pattern = f"test_{TRACKS[args.track]}*.py" if args.track else "test_*.py"
        return subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"], cwd=ROOT).returncode
    tracks, chapters = catalog()
    if args.action == "map":
        for track in tracks:
            print(f"\n{track['track_id']}  {track['title']}")
            for chapter in chapters:
                if chapter["track_id"] == track["track_id"]:
                    print(f"  {chapter['id']:5} {chapter['title']}")
        return 0
    if args.action == "next":
        choices = eligible(chapters, read_progress())
        if args.track:
            choices = [c for c in choices if c["track_id"] == args.track]
        for chapter in choices[:12]:
            print(chapter["id"], chapter["title"], "->", chapter["path"])
        if not choices:
            print("没有新的可学章节；用 map 检查先修关系和本地学习记录。")
        print("mastered 需要自己的学习证据；测试通过不会自动将章节标为掌握。")
        return 0
    if args.action == "record":
        record(args.id, args.status, args.evidence)
        print("Recorded locally:", args.id, args.status)
        return 0
    if args.action == "lesson":
        chapter = next((c for c in chapters if c["id"] == args.id), None)
        if chapter is None:
            raise ValueError(f"unknown chapter: {args.id}")
        print(chapter["id"], chapter["title"])
        print("Read:", chapter["path"])
        print("Prerequisites:", ", ".join(chapter["prerequisites"]) or "none")
        for lab in chapter["labs"]:
            print(f"  {lab['id']}: {lab.get('title', '')}\n    {lab['command']} [{lab.get('hardware', 'cpu')}, network={lab.get('network', False)}]")
        return 0
    if args.action == "search":
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        matches = 0
        query = args.query.casefold()
        for path in sorted((ROOT / "books").rglob("*.md")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if query in line.casefold():
                    print(f"{path.relative_to(ROOT)}:{number}: {line[:220]}")
                    matches += 1
                    if matches >= args.limit:
                        return 0
        print(f"{matches} matches")
        return 0
    if args.action == "run":
        if not 0 < args.timeout <= 3600:
            raise ValueError("timeout must be in 1..3600 seconds")
        lab = next((lab for c in chapters for lab in c["labs"] if lab["id"] == args.id), None)
        if lab is None:
            raise ValueError(f"unknown lab: {args.id}")
        command = prepare_command(lab, args.allow_gpu, args.allow_network)
        print("Running:", " ".join(command), flush=True)
        environment = dict(os.environ)
        if not args.allow_gpu:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(command, cwd=ROOT, env=environment, timeout=args.timeout, shell=False).returncode
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
