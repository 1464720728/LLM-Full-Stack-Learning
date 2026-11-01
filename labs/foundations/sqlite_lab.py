"""Transactions commit together, or roll back together."""
import sqlite3

def transaction_demo():
    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE TABLE counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL CHECK(value>=0))")
        with db:
            db.execute("INSERT INTO counters VALUES (?, ?)", ("tokens", 10))
        try:
            with db:
                db.execute("UPDATE counters SET value=20 WHERE name='tokens'")
                db.execute("INSERT INTO counters VALUES (?, ?)", ("invalid", -1))
        except sqlite3.IntegrityError:
            pass
        return db.execute("SELECT name,value FROM counters ORDER BY name").fetchall()
    finally:
        db.close()

if __name__ == "__main__":
    print(transaction_demo())
