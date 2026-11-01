"""Execute a fixed executable with separate argv; no shell interpolation."""
import subprocess
import sys

def run_child(text: str) -> dict:
    result = subprocess.run([sys.executable, "-c", "import sys; print(sys.argv[1]); print('diagnostic', file=sys.stderr)", text],
                            text=True, capture_output=True, timeout=3, check=True)
    return {"stdout": result.stdout.rstrip("\n"), "stderr": result.stderr.rstrip("\n"), "code": result.returncode}

if __name__ == "__main__":
    print(run_child("hello; this is one argument"))
