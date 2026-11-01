"""Atomic replacement prevents partial JSON files under normal failure.

Same-filesystem replace is atomic for observers. This does not promise complete
power-loss durability on every filesystem; directory fsync is not done here.
"""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

if __name__ == "__main__":
    with TemporaryDirectory() as d:
        target = Path(d) / "progress.json"
        atomic_json(target, {"lesson": "files", "done": True})
        print(target.read_text(encoding="utf-8").strip())
