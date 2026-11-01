"""SQLite snapshots, optimistic versions and a tool execution ledger.

The ledger detects ambiguous side effects. It does NOT promise exactly-once
execution across an arbitrary external API and a local SQLite transaction.
"""
from __future__ import annotations
import json
import sqlite3

class Conflict(RuntimeError):
    pass
class UncertainExecution(RuntimeError):
    pass

class Store:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tool_ledger(
          run_id TEXT NOT NULL, call_id TEXT NOT NULL, signature TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('running','done')), output TEXT,
          PRIMARY KEY(run_id,call_id));
        """)
    def close(self):
        self.db.close()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    def save(self, state: dict, expected: int) -> int:
        new = expected + 1
        payload = dict(state, version=new)
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self.db:
            if expected == 0:
                try:
                    self.db.execute("INSERT INTO runs VALUES(?,?,?)", (state["run_id"],new,encoded))
                except sqlite3.IntegrityError as exc:
                    raise Conflict("run already exists") from exc
            else:
                cur = self.db.execute("UPDATE runs SET version=?,payload=? WHERE id=? AND version=?",
                                      (new,encoded,state["run_id"],expected))
                if cur.rowcount != 1:
                    raise Conflict("stale checkpoint version")
        return new
    def load(self, run_id):
        row = self.db.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row[0])
    def lookup(self, run_id, call_id, signature):
        row = self.db.execute("SELECT signature,status,output FROM tool_ledger WHERE run_id=? AND call_id=?", (run_id,call_id)).fetchone()
        if row and row[0] != signature:
            raise Conflict("call id reused with different arguments")
        return None if row is None else {"status":row[1],"output":row[2]}
    def start(self, run_id, call_id, signature, side_effect):
        # Single run owner; CAS on snapshots prevents blind overwrite, not concurrent tool dispatch.
        with self.db:
            row = self.lookup(run_id,call_id,signature)
            if row:
                if row["status"] == "done":
                    return row["output"]
                if side_effect:
                    raise UncertainExecution("side effect may have completed; reconcile before replay")
                return None  # Safe replay allowed only for declared pure operations.
            self.db.execute("INSERT INTO tool_ledger VALUES(?,?,?,'running',NULL)", (run_id,call_id,signature))
        return None
    def complete(self, run_id, call_id, output):
        with self.db:
            cur=self.db.execute("UPDATE tool_ledger SET status='done',output=? WHERE run_id=? AND call_id=?", (output,run_id,call_id))
            if cur.rowcount != 1:
                raise Conflict("missing execution claim")
