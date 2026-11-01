#!/usr/bin/env python3
"""Normalize prose math delimiters for GitHub without editing code examples."""
from __future__ import annotations

import argparse
from pathlib import Path
import re


def inline_segments(line):
    """Yield (is_code, text), respecting equal-length backtick delimiters."""
    cursor = 0
    while cursor < len(line):
        opening = re.search(r"`+", line[cursor:])
        if not opening:
            yield False, line[cursor:]
            return
        start = cursor + opening.start()
        end = cursor + opening.end()
        width = end - start
        closing = next((m for m in re.finditer(r"`+", line[end:])
                        if len(m[0]) == width), None)
        if closing is None:
            yield False, line[cursor:]
            return
        stop = end + closing.end()
        yield False, line[cursor:start]
        yield True, line[start:stop]
        cursor = stop


def normalize(text):
    output, fence = [], None
    for line in text.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*?)(?:\n)?$", line)
        if fence:
            output.append(line)
            if (match and match[1][0] == fence[0] and len(match[1]) >= fence[1]
                    and not match[2].strip()):
                fence = None
            continue
        if match:
            fence = (match[1][0], len(match[1]))
            output.append(line)
            continue
        if line.strip() in (r"\[", r"\]"):
            output.append(line.replace(line.strip(), "$$"))
            continue
        fragments = []
        for is_code, segment in inline_segments(line):
            fragments.append(segment if is_code else
                             re.sub(r"\\\((.*?)\\\)", lambda m: "$" + m[1] + "$", segment))
        output.append("".join(fragments))
    return "".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    changed = 0
    for path in args.paths:
        before = path.read_text(encoding="utf-8")
        after = normalize(before)
        if before != after:
            changed += 1
            print(path)
            if args.write:
                path.write_text(after, encoding="utf-8")
    print(f"files_requiring_normalization={changed}; written={args.write}")


if __name__ == "__main__":
    main()
