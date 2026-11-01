#!/usr/bin/env python3
"""Check local Markdown link targets and Python syntax without executing labs."""
from __future__ import annotations
import ast
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_depth import prose_without_fences


def check(root=ROOT):
    errors=[]
    for path in root.rglob("*.md"):
        if any(part in {".git", ".venv", ".course", ".practice"} for part in path.relative_to(root).parts):
            continue
        text=path.read_text(encoding="utf-8")
        text=prose_without_fences(text)
        # Inline code such as checks[typ](value) is Python, not a Markdown link.
        text=re.sub(r"(?<!`)(`+)(?!`)(.*?)\1(?!`)", "", text, flags=re.S)
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)",text):
            value=match.group(1).strip("<>")
            parsed=urlsplit(value)
            if parsed.scheme or not parsed.path:
                continue
            target=path.parent/unquote(parsed.path)
            if not target.exists():
                errors.append(f"{path.relative_to(root)}: missing link {value}")
    for path in root.rglob("*.py"):
        if any(part in {".git", ".venv", ".course", ".practice"} for part in path.relative_to(root).parts):
            continue
        try:ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
        except (SyntaxError,UnicodeError) as exc:errors.append(str(exc))
    return errors


if __name__=="__main__":
    errors=check()
    print("\n".join(errors) if errors else "PASS: local Markdown targets and Python syntax")
    raise SystemExit(bool(errors))
