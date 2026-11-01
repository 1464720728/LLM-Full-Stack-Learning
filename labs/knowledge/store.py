"""SQLite versioned corpus with tenant/ACL filters before retrieval.

Ingest/delete/purge are trusted administrative APIs, not remotely authenticated APIs.
A service must authenticate Context and authorize these mutations separately.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
from .text import chunk_text, digest, TOKENIZER_VERSION


@dataclass(frozen=True)
class Context:
    tenant: str
    actor: str
    groups: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.tenant or not self.actor or self.actor == "*":
            raise ValueError("a real tenant and actor are required")
        if "*" in self.groups:
            raise ValueError("wildcard is a grant, not an actor/group")

    @property
    def principals(self):
        return tuple(sorted({"*", "user:" + self.actor, *("group:" + g for g in self.groups)}))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    tenant: str
    doc_id: str
    revision: int
    title: str
    source: str
    start: int
    end: int
    text: str


class ConflictError(ValueError):
    pass


class CorpusStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS document_revision_counters(
          identity_hash TEXT PRIMARY KEY, last_revision INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS documents(
          tenant TEXT NOT NULL, doc_id TEXT NOT NULL, current_revision INTEGER NOT NULL,
          deleted INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(tenant,doc_id));
        CREATE TABLE IF NOT EXISTS revisions(
          tenant TEXT NOT NULL, doc_id TEXT NOT NULL, revision INTEGER NOT NULL,
          title TEXT NOT NULL, source TEXT NOT NULL, body TEXT NOT NULL,
          fingerprint TEXT NOT NULL, tokenizer TEXT NOT NULL,
          PRIMARY KEY(tenant,doc_id,revision),
          FOREIGN KEY(tenant,doc_id) REFERENCES documents ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS chunks(
          chunk_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, doc_id TEXT NOT NULL,
          revision INTEGER NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
          text TEXT NOT NULL,
          FOREIGN KEY(tenant,doc_id,revision) REFERENCES revisions ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS grants(
          tenant TEXT NOT NULL, doc_id TEXT NOT NULL, revision INTEGER NOT NULL,
          principal TEXT NOT NULL, PRIMARY KEY(tenant,doc_id,revision,principal),
          FOREIGN KEY(tenant,doc_id,revision) REFERENCES revisions ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS edges(
          edge_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, subject TEXT NOT NULL,
          predicate TEXT NOT NULL, object TEXT NOT NULL, chunk_id TEXT NOT NULL,
          quote TEXT NOT NULL, FOREIGN KEY(chunk_id) REFERENCES chunks ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS chunks_document ON chunks(tenant,doc_id,revision);
        CREATE INDEX IF NOT EXISTS edges_tenant ON edges(tenant,subject);
        ''')
        # Migrate pre-counter databases before they can purge a document. The
        # tombstone retains no body/title/source/grants, but its stable hash is
        # still pseudonymous metadata, not a promise of complete erasure.
        with self.conn:
            for row in self.conn.execute("SELECT tenant,doc_id,current_revision FROM documents").fetchall():
                self.conn.execute('''INSERT INTO document_revision_counters VALUES(?,?)
                  ON CONFLICT(identity_hash) DO UPDATE SET last_revision=MAX(last_revision,excluded.last_revision)''',
                  (digest(row['tenant'],row['doc_id']),row['current_revision']))

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def ingest(self, tenant: str, doc_id: str, title: str, body: str,
               allowed: tuple[str, ...] = (), source: str = "fixture",
               max_chars: int = 240, overlap: int = 32,
               expected_revision: int | None = None) -> int:
        """Atomic replace. Empty allowed list means deny all; '*' means tenant-public."""
        if not tenant or not doc_id or not title or not source:
            raise ValueError("tenant, doc_id, title, source are required")
        if any(not (p == "*" or (p.startswith(("user:", "group:")) and len(p.split(':',1)[1]) > 0)) for p in allowed):
            raise ValueError("grants use '*', 'user:NAME', or 'group:NAME'")
        spans = chunk_text(body, max_chars, overlap)
        grants = tuple(sorted(set(allowed)))
        fp = digest(title, body, source, grants, max_chars, overlap, TOKENIZER_VERSION)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            old = self.conn.execute("SELECT * FROM documents WHERE tenant=? AND doc_id=?", (tenant, doc_id)).fetchone()
            current = old["current_revision"] if old else 0
            if expected_revision is not None and current != expected_revision:
                raise ConflictError(f"expected revision {expected_revision}; current {current}")
            if old and not old["deleted"]:
                previous = self.conn.execute("SELECT fingerprint FROM revisions WHERE tenant=? AND doc_id=? AND revision=?", (tenant, doc_id, current)).fetchone()
                if previous[0] == fp:
                    self.conn.commit()
                    return current
            identity = digest(tenant,doc_id)
            counter = self.conn.execute("SELECT last_revision FROM document_revision_counters WHERE identity_hash=?", (identity,)).fetchone()
            rev = max(current,counter[0] if counter else 0) + 1
            self.conn.execute('''INSERT INTO document_revision_counters VALUES(?,?)
              ON CONFLICT(identity_hash) DO UPDATE SET last_revision=excluded.last_revision''', (identity,rev))
            if old:
                self.conn.execute("UPDATE documents SET current_revision=?,deleted=0 WHERE tenant=? AND doc_id=?", (rev, tenant, doc_id))
            else:
                self.conn.execute("INSERT INTO documents VALUES(?,?,?,0)", (tenant, doc_id, rev))
            self.conn.execute("INSERT INTO revisions VALUES(?,?,?,?,?,?,?,?)", (tenant, doc_id, rev, title, source, body, fp, TOKENIZER_VERSION))
            self.conn.executemany("INSERT INTO grants VALUES(?,?,?,?)", [(tenant,doc_id,rev,g) for g in grants])
            self.conn.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?)", [
                (digest(tenant,doc_id,rev,s.start,s.end,s.text),tenant,doc_id,rev,s.start,s.end,s.text) for s in spans])
            self.conn.commit()
            return rev
        except BaseException:
            self.conn.rollback()
            raise

    def visible(self, ctx: Context) -> list[Chunk]:
        placeholders = ",".join("?" for _ in ctx.principals)
        query = f'''SELECT c.*,r.title,r.source FROM chunks c
          JOIN documents d ON d.tenant=c.tenant AND d.doc_id=c.doc_id
          JOIN revisions r ON r.tenant=c.tenant AND r.doc_id=c.doc_id AND r.revision=c.revision
          WHERE c.tenant=? AND d.deleted=0 AND c.revision=d.current_revision
          AND EXISTS(SELECT 1 FROM grants g WHERE g.tenant=c.tenant AND g.doc_id=c.doc_id
            AND g.revision=c.revision AND g.principal IN ({placeholders}))
          ORDER BY c.doc_id,c.start,c.chunk_id'''
        rows = self.conn.execute(query, (ctx.tenant, *ctx.principals)).fetchall()
        return [Chunk(**{k: r[k] for k in Chunk.__dataclass_fields__}) for r in rows]

    def get_visible_chunk(self, ctx: Context, chunk_id: str) -> Chunk | None:
        # Tiny-corpus baseline: same visibility contract as retrieval. Production use indexed join.
        return next((c for c in self.visible(ctx) if c.chunk_id == chunk_id), None)

    def delete(self, tenant: str, doc_id: str, expected_revision: int | None = None):
        with self.conn:
            # Put the expected revision in the write itself: a separate SELECT
            # followed by an unconditional UPDATE has a cross-connection race.
            query = "UPDATE documents SET deleted=1 WHERE tenant=? AND doc_id=?"
            params = (tenant,doc_id)
            if expected_revision is not None:
                query += " AND current_revision=?"
                params += (expected_revision,)
            changed = self.conn.execute(query,params).rowcount
            if changed:
                return True
            exists = self.conn.execute("SELECT 1 FROM documents WHERE tenant=? AND doc_id=?", (tenant,doc_id)).fetchone()
            if exists is not None:
                raise ConflictError("stale delete")
            return False

    def purge(self, tenant: str, doc_id: str):
        """Purge content/descendants, retaining a hashed revision counter against ABA.

        This is not forensic disk/backup erasure or deletion of all metadata.
        """
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE tenant=? AND doc_id=?", (tenant,doc_id))

    def counts(self):
        return {t:self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("documents","revisions","chunks","grants","edges")}
