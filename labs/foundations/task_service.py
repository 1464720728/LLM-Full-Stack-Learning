"""Project B: HTTP/JSON/SQLite tasks with validation and idempotency.

Default runs an in-memory demonstration without opening a port. `--serve` starts
an explicitly local, serial educational server. It is not a production server.
"""
import argparse
import json
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

MAX_BODY = 8192

class Conflict(ValueError):
    pass

class TaskStore:
    def __init__(self, database=":memory:"):
        self.db = sqlite3.connect(database)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            done INTEGER NOT NULL DEFAULT 0 CHECK(done IN (0,1)),
            idempotency_key TEXT UNIQUE NOT NULL)""")
        self.db.commit()

    def close(self):
        self.db.close()

    @staticmethod
    def as_task(row):
        return {"id": row["id"], "title": row["title"], "done": bool(row["done"])}

    def add(self, title, key):
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 200:
            raise ValueError("title must contain 1..200 characters")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key):
            raise ValueError("idempotency_key must be 1..80 letters, digits, _ or -")
        title = title.strip()
        with self.db:
            previous = self.db.execute("SELECT * FROM tasks WHERE idempotency_key=?", (key,)).fetchone()
            if previous:
                if previous["title"] != title:
                    raise Conflict("idempotency key reused with a different title")
                return self.as_task(previous), False
            cursor = self.db.execute("INSERT INTO tasks(title,idempotency_key) VALUES (?,?)", (title,key))
            row = self.db.execute("SELECT * FROM tasks WHERE id=?", (cursor.lastrowid,)).fetchone()
            return self.as_task(row), True

    def list(self):
        return [self.as_task(r) for r in self.db.execute("SELECT * FROM tasks ORDER BY id")]

    def complete(self, task_id):
        if type(task_id) is not int or task_id <= 0:
            raise ValueError("positive integer task id required")
        with self.db:
            cursor = self.db.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,))
            if cursor.rowcount == 0:
                raise KeyError(task_id)
            row = self.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            return self.as_task(row)

def dispatch(store, method, path, body=b""):
    """Pure transport boundary: no sockets are required to test API behavior."""
    if len(body) > MAX_BODY:
        return 413, {"error": "request body too large"}
    parsed = urlsplit(path)
    if parsed.query:
        return 400, {"error": "query parameters are unsupported"}
    route = parsed.path
    try:
        if method == "GET" and route == "/health":
            return 200, {"status": "ok"}
        if method == "GET" and route == "/tasks":
            return 200, {"tasks": store.list()}
        if method == "POST" and route == "/tasks":
            data = json.loads(body.decode("utf-8"))
            if not isinstance(data, dict) or set(data) != {"title", "idempotency_key"}:
                raise ValueError("title and idempotency_key required")
            task, created = store.add(data["title"], data["idempotency_key"])
            return (201 if created else 200), task
        match = re.fullmatch(r"/tasks/([1-9][0-9]*)/complete", route)
        if method == "POST" and match:
            if body:
                raise ValueError("completion request must have an empty body")
            return 200, store.complete(int(match.group(1)))
        return 404, {"error": "route not found"}
    except Conflict as e:
        return 409, {"error": str(e)}
    except KeyError:
        return 404, {"error": "task not found"}
    except (ValueError, UnicodeError) as e:
        return 400, {"error": str(e)}

def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            if self.headers.get("Transfer-Encoding"):
                self.send_json(400, {"error": "transfer encoding unsupported"})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) > 1:
                self.send_json(400, {"error": "duplicate Content-Length"})
                return
            try:
                length = int(lengths[0]) if lengths else 0
                if length < 0:
                    raise ValueError
            except ValueError:
                self.send_json(400, {"error": "invalid Content-Length"})
                return
            if length > MAX_BODY:
                self.send_json(413, {"error": "request body too large"})
                return
            if length and self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "application/json required"})
                return
            self.connection.settimeout(5)
            try:
                data = self.rfile.read(length)
            except TimeoutError:
                self.send_json(408, {"error": "request body timeout"})
                return
            if len(data) != length:
                self.send_json(400, {"error": "incomplete request body"})
                return
            status, payload = dispatch(store, self.command, self.path, data)
            self.send_json(status, payload)

        def send_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        do_GET = handle_request
        do_POST = handle_request

        def log_message(self, format, *args):
            # Do not log bodies or credentials. Base implementation is omitted.
            pass
    return Handler

def demo():
    store = TaskStore()
    try:
        body = json.dumps({"title": "完成基础课程", "idempotency_key": "demo-1"}).encode()
        return [dispatch(store, "POST", "/tasks", body),
                dispatch(store, "POST", "/tasks", body),
                dispatch(store, "POST", "/tasks/1/complete"),
                dispatch(store, "GET", "/tasks")]
    finally:
        store.close()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--database", default="tasks.sqlite3")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not args.serve:
        print(json.dumps(demo(), ensure_ascii=False, indent=2))
        return 0
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    store = TaskStore(args.database)
    try:
        with HTTPServer(("127.0.0.1", args.port), make_handler(store)) as server:
            print(f"Local educational server: http://127.0.0.1:{args.port}; Ctrl+C stops it", flush=True)
            try:
                server.serve_forever(poll_interval=.2)
            except KeyboardInterrupt:
                pass
    finally:
        store.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
