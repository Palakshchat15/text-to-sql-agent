"""SQLite request tracing (data/traces.db). One row per answered question."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT,              -- app | eval | smoke
    question TEXT,
    config TEXT,              -- ablation name
    model TEXT,
    retrieved TEXT,           -- JSON [[table, score], ...]
    status TEXT,              -- ok | refused | error | blocked
    n_attempts INTEGER,
    guard_blocks INTEGER,
    attempts TEXT,            -- JSON list of attempts (sql, error, rows, timings, tokens)
    final_sql TEXT,
    n_rows INTEGER,
    t_retrieve REAL, t_llm REAL, t_exec REAL, t_total REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    summary TEXT,
    error TEXT
);
"""


class Tracer:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def log(self, **kw) -> int:
        row = {"ts": time.time(), **kw}
        for k in ("retrieved", "attempts"):
            if k in row and not isinstance(row[k], str):
                row[k] = json.dumps(row[k], default=str)
        cols = ", ".join(row)
        q = f"INSERT INTO traces ({cols}) VALUES ({', '.join('?' * len(row))})"
        with self._conn() as c:
            return int(c.execute(q, list(row.values())).lastrowid)

    def read(self, limit: int | None = None):
        import pandas as pd

        q = "SELECT * FROM traces ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        with self._conn() as c:
            return pd.read_sql_query(q, c)
