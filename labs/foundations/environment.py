"""Observe the active interpreter without dumping environment secrets."""
import importlib.util
import json
import platform
import sys
from pathlib import Path

def report():
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.system(), "cwd": str(Path.cwd()),
            "virtual_env": sys.prefix != sys.base_prefix,
            "numpy_available": importlib.util.find_spec("numpy") is not None}

if __name__ == "__main__":
    print(json.dumps(report(), ensure_ascii=False, indent=2))
