"""Text contracts: offsets are Python Unicode character offsets, never token IDs."""
from dataclasses import dataclass
import hashlib
import json
import re

TOKENIZER_VERSION = "latin-word-cjk-char-v1"
_TOKEN = re.compile(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Deliberately simple lexical tokenizer, not an LLM tokenizer."""
    return [m.group(0).lower() for m in _TOKEN.finditer(text)]


def digest(*parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


def chunk_text(text: str, max_chars: int = 240, overlap: int = 32) -> list[Span]:
    """Exact slices; prefer newline/sentence endings in the latter half of a window."""
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("require max_chars > 0 and 0 <= overlap < max_chars")
    if not text:
        return []
    spans, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            lower = start + max(max_chars // 2, overlap + 1)
            candidates = [text.rfind(mark, lower, end) for mark in ("\n", "。", ". ", "; ")]
            boundary = max(candidates)
            if boundary >= lower:
                end = boundary + 1
        spans.append(Span(start, end, text[start:end]))
        if end == len(text):
            break
        start = end - overlap
    return spans
