"""Versioned memory ledger: source evidence, namespace, expiration, correction and purge.

This is an application-owned local store. Its caller must authenticate tenant/owner.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
import time
from .store import ConflictError
from .text import digest

KINDS = {"semantic", "episodic", "procedural"}


@dataclass(frozen=True)
class Memory:
    tenant: str
    owner: str
    key: str
    revision: int
    kind: str
    value: object
    evidence: str
    confidence: float
    expires_at: float | None


class MemoryStore:
    def __init__(self,path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS memory_revision_counters(
          identity_hash TEXT PRIMARY KEY, last_revision INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS memory_versions(
          tenant TEXT NOT NULL, owner TEXT NOT NULL, key TEXT NOT NULL, revision INTEGER NOT NULL,
          kind TEXT NOT NULL, value_json TEXT NOT NULL, evidence TEXT NOT NULL,
          confidence REAL NOT NULL, expires_at REAL, state TEXT NOT NULL,
          PRIMARY KEY(tenant,owner,key,revision));
        CREATE INDEX IF NOT EXISTS memory_active ON memory_versions(tenant,owner,state);
        ''')
        with self.conn:
            rows = self.conn.execute("SELECT tenant,owner,key,MAX(revision) AS revision FROM memory_versions GROUP BY tenant,owner,key").fetchall()
            for row in rows:
                self.conn.execute('''INSERT INTO memory_revision_counters VALUES(?,?)
                  ON CONFLICT(identity_hash) DO UPDATE SET last_revision=MAX(last_revision,excluded.last_revision)''',
                  (digest(row['tenant'],row['owner'],row['key']),row['revision']))

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()

    def remember(self,tenant,owner,key,value,evidence,kind="semantic",confidence=1.0,
                 expires_at=None,expected_revision=None):
        if not all(isinstance(x,str) and x.strip() for x in (tenant,owner,key,evidence)):
            raise ValueError("nonempty tenant, owner, key and evidence required")
        if kind not in KINDS or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid kind/confidence")
        if expires_at is not None and not math.isfinite(expires_at):
            raise ValueError("finite expiration required")
        serialized = json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            current = self.conn.execute("SELECT MAX(revision) FROM memory_versions WHERE tenant=? AND owner=? AND key=?",(tenant,owner,key)).fetchone()[0] or 0
            if expected_revision is not None and expected_revision != current:
                raise ConflictError(f"memory revision changed: {current}")
            identity = digest(tenant,owner,key)
            counter = self.conn.execute("SELECT last_revision FROM memory_revision_counters WHERE identity_hash=?",(identity,)).fetchone()
            rev = max(current,counter[0] if counter else 0)+1
            self.conn.execute('''INSERT INTO memory_revision_counters VALUES(?,?)
              ON CONFLICT(identity_hash) DO UPDATE SET last_revision=excluded.last_revision''',(identity,rev))
            self.conn.execute("UPDATE memory_versions SET state='superseded' WHERE tenant=? AND owner=? AND key=? AND state='active'",(tenant,owner,key))
            self.conn.execute("INSERT INTO memory_versions VALUES(?,?,?,?,?,?,?,?,?,'active')",(tenant,owner,key,rev,kind,serialized,evidence,confidence,expires_at))
            self.conn.commit()
            return rev
        except BaseException:
            self.conn.rollback()
            raise

    def recall(self,tenant,owner,now=None,kind=None):
        now = time.time() if now is None else now
        if kind is not None and kind not in KINDS:
            raise ValueError("invalid memory kind")
        rows = self.conn.execute('''SELECT * FROM memory_versions WHERE tenant=? AND owner=?
          AND state='active' AND (expires_at IS NULL OR expires_at>?)
          ORDER BY key''',(tenant,owner,now)).fetchall()
        return [Memory(r['tenant'],r['owner'],r['key'],r['revision'],r['kind'],json.loads(r['value_json']),r['evidence'],r['confidence'],r['expires_at'])
                for r in rows if kind is None or r['kind']==kind]

    def retract(self,tenant,owner,key):
        with self.conn:
            self.conn.execute("UPDATE memory_versions SET state='retracted' WHERE tenant=? AND owner=? AND key=? AND state='active'",(tenant,owner,key))

    def forget(self,tenant,owner,key):
        """Delete content history, retaining a hashed revision counter against ABA.

        The stable hash is pseudonymous metadata; external logs/backups and full
        metadata erasure need separate retention/identity-rotation policies.
        """
        with self.conn:
            self.conn.execute("DELETE FROM memory_versions WHERE tenant=? AND owner=? AND key=?",(tenant,owner,key))

    def history(self,tenant,owner,key):
        # Administrative inspection, not automatically included in model context.
        return [dict(r) for r in self.conn.execute("SELECT * FROM memory_versions WHERE tenant=? AND owner=? AND key=? ORDER BY revision",(tenant,owner,key))]


def select_memory(memories,query_terms: set[str],budget_chars=400):
    """Simple deterministic key/value lexical policy, not a learned importance estimator."""
    if budget_chars < 0:
        raise ValueError("nonnegative budget")
    from .text import tokenize
    scored=[]
    for m in memories:
        text=f"{m.key}={json.dumps(m.value,ensure_ascii=False)}"
        overlap=len(query_terms & set(tokenize(text)))
        if overlap:
            scored.append((overlap*m.confidence,m.key,text))
    selected,used=[],0
    for _,_,text in sorted(scored,key=lambda x:(-x[0],x[1])):
        if used+len(text)<=budget_chars:
            selected.append(text);used+=len(text)
    return selected
