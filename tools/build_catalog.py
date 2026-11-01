#!/usr/bin/env python3
"""Generate human-readable navigation from track manifests; no dependencies."""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.course import ROOT, catalog, validate


def build(root: Path = ROOT) -> dict:
    errors = validate(root)
    if errors:
        raise ValueError("\n".join(errors))
    tracks, chapters = catalog(root)
    toc = ["# 完整课程索引", "", "由各卷 manifest.json 生成。每课的详细练习、答案和验收在正文中。先修是最低建议，不等于必须先读完所有上游章节。", ""]
    labs = ["# 可执行实验索引", "", "所有命令在仓库根目录运行。CPU 实验可以先学；GPU、联网和真实 API 实验须按章节准备环境。目录中的实验数量不包括正文中手算、代码修改、故障诊断与开源扩展题。", ""]
    for track in tracks:
        tid = track["track_id"]
        toc += [f"## {tid} {track['title']}", "", "| 章 | 标题 | 先修 | 注册实验数 |", "|---|---|---|---|"]
        labs += [f"## {tid} {track['title']}", ""]
        for c in [x for x in chapters if x["track_id"] == tid]:
            toc.append(f"| {c['id']} | [{c['title']}](../{c['path']}) | {', '.join(c.get('prerequisites', [])) or '无'} | {len(c['labs'])} |")
            for lab in c["labs"]:
                labs += [f"### {lab['id']} {lab.get('title', '')}", "", f"教材：[{c['id']} {c['title']}](../{c['path']}) · 环境：{lab.get('hardware', 'cpu')} · 联网：{lab.get('network', False)}", "", "```bash", lab["command"], "```", ""]
        toc.append("")
    text_files = list((root / "books").rglob("*.md"))
    chinese = sum(len(re.findall(r"[\u3400-\u9fff]", p.read_text(encoding="utf-8"))) for p in text_files)
    code_files = [p for p in (root / "labs").rglob("*") if p.suffix in (".py", ".cu", ".cpp", ".h", ".sh")]
    stats = {"tracks": len(tracks), "chapters": len(chapters), "registered_labs": sum(len(c["labs"]) for c in chapters), "book_markdown_files": len(text_files), "book_chinese_characters": chinese, "lab_code_files": len(code_files), "lab_code_lines": sum(len(p.read_text(encoding="utf-8").splitlines()) for p in code_files)}
    (root / "docs" / "CATALOG.md").write_text("\n".join(toc).rstrip() + "\n", encoding="utf-8")
    (root / "docs" / "LABS.md").write_text("\n".join(labs).rstrip() + "\n", encoding="utf-8")
    (root / "docs" / "course_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
