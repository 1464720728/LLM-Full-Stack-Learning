#!/usr/bin/env python3
"""Measure chapter expansion; this is a length check, never a quality certificate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TRACKS = {"F": "00-foundations", "A": "01-agent-engineering",
          "R": "02-retrieval-memory-graphs", "M": "03-model-architectures",
          "S": "04-inference-systems", "K": "05-kernels-hardware"}


def prose_without_fences(text: str) -> str:
    """Handle backtick/tilde fences and longer fences containing shorter ones."""
    kept = []
    opened = None
    for line in text.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if opened:
            if (match and match[1][0] == opened[0] and len(match[1]) >= opened[1]
                    and not match[2].strip()):
                opened = None
            continue
        if match:
            opened = (match[1][0], len(match[1]))
        else:
            kept.append(line)
    return "\n".join(kept)


def cjk_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]", text))


def measure(root: Path = ROOT, track: str | None = None) -> dict:
    config = json.loads((root / "docs/DEPTH_TARGETS.json").read_text(encoding="utf-8"))
    previous = json.loads((root / "docs/depth_baseline_v1.json").read_text(encoding="utf-8"))
    baseline = {entry["path"]: entry["prose_chinese_characters"] for entry in previous["chapters"]}
    rows, paragraphs = [], defaultdict(list)
    excerpts = {}
    for entry in config["entries"]:
        relative = entry["path"]
        if track and not relative.startswith(f"books/{TRACKS[track]}/"):
            continue
        path = root / relative
        exists = path.is_file()
        text = path.read_text(encoding="utf-8") if exists else ""
        prose = prose_without_fences(text)
        count = cjk_count(prose)
        rows.append({"path": relative, "exists": exists,
                     "baseline_prose_cjk": baseline.get(relative), "prose_cjk": count,
                     "minimum_prose_cjk": entry["minimum_prose_cjk"],
                     "length_met": exists and count >= entry["minimum_prose_cjk"]})
        for paragraph in re.split(r"\n\s*\n", prose):
            normalized = re.sub(r"\s+", "", paragraph)
            if cjk_count(normalized) < 120:
                continue
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            paragraphs[digest].append(relative)
            excerpts[digest] = normalized[:160]
    duplicates = [{"paths": sorted(set(paths)), "excerpt": excerpts[key]}
                  for key, paths in paragraphs.items() if len(set(paths)) > 1]
    return {"version": config["version"], "baseline_commit": previous["baseline_commit"],
            "scope": track or "all", "chapters": rows,
            "total_prose_cjk": sum(row["prose_cjk"] for row in rows),
            "below_minimum": sum(not row["length_met"] for row in rows),
            "duplicate_paragraphs_for_review": duplicates,
            "limitation": "Length and exact repeated paragraphs are review signals, not evidence of pedagogical correctness or completeness."}


def write_report(data: dict, root: Path = ROOT) -> None:
    (root / "docs/depth_report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 逐章深写对照", "", "由 `python tools/check_depth.py --write-report` 生成。统计去掉 fenced code block 后的中文汉字，含标题、表格和习题答案。篇幅是审稿线索，不是质量证明；教学标准见 [DEPTH_STANDARD](DEPTH_STANDARD.md)。", "",
             f"初版快照：`{data['baseline_commit']}`。目标版本：{data['version']}。", "",
             f"统计 {len(data['chapters'])} 篇；当前合计 {data['total_prose_cjk']:,} 汉字；未达到本轮最低篇幅 {data['below_minimum']} 篇。", "",
             "| 章节 | 初版 | 当前 | 本轮下限 | 篇幅检查 |", "|---|---:|---:|---:|---|"]
    for row in data["chapters"]:
        previous = row["baseline_prose_cjk"]
        lines.append(f"| [{row['path']}](../{row['path']}) | {previous if previous is not None else '未纳入初版主章统计'} | {row['prose_cjk']} | {row['minimum_prose_cjk']} | {'达到' if row['length_met'] else '未达到'} |")
    lines += ["", "## 重复段落审稿线索", "", "仅列跨文档完全相同且至少含 120 个汉字的段落。共同定义可能需要重复，须结合上下文判断。近似重复和语义空泛仍需人工检查。", ""]
    for index, duplicate in enumerate(data["duplicate_paragraphs_for_review"], 1):
        lines += [f"{index}. " + "；".join(duplicate["paths"]), "", "> " + duplicate["excerpt"], ""]
    if not data["duplicate_paragraphs_for_review"]:
        lines += ["未检出满足此规则的跨文档完全重复段落。", ""]
    (root / "docs/DEPTH_REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=TRACKS)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    if args.track and args.write_report:
        parser.error("write the release report for all tracks, not a partial track")
    data = measure(track=args.track)
    if args.write_report:
        write_report(data)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"chapters={len(data['chapters'])} prose_cjk={data['total_prose_cjk']} below_minimum={data['below_minimum']} repeated_paragraphs={len(data['duplicate_paragraphs_for_review'])}")
        for row in data["chapters"]:
            if not row["length_met"]:
                print(f"SHORT {row['prose_cjk']:>6}/{row['minimum_prose_cjk']} {row['path']}")
    return int(args.check and data["below_minimum"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
