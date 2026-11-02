"""Files live in a temporary directory; names are not file contents."""
from pathlib import Path
from tempfile import TemporaryDirectory

def contained_path(root: Path, relative: str) -> Path:
    """Resolve a relative path, refusing absolute paths and traversal.

    This validates a path at one instant; it is not a race-proof OS sandbox.
    """
    root = root.resolve()
    part = Path(relative)
    if part.is_absolute():
        raise ValueError("relative path required")
    candidate = (root / part).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("path escapes root")
    return candidate

def demo():
    with TemporaryDirectory() as d:
        root = Path(d).resolve()
        target = contained_path(root, "notes/lesson.txt")
        target.parent.mkdir(parents=True)
        target.write_text("hello\n学习\n", encoding="utf-8", newline="\n")
        return {"relative": str(target.relative_to(root)),
                "characters": len(target.read_text(encoding="utf-8")),
                "bytes": len(target.read_bytes()),
                "lines": target.read_text(encoding="utf-8").splitlines()}

if __name__ == "__main__":
    print(demo())
