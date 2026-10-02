"""Read-only DuckDB execution with a timeout and a maximum result size."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb


@dataclass
class ExecResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    elapsed_s: float = 0.0
    timed_out: bool = False


class Executor:
    """One read-only connection per Executor. External file access is disabled, so even a query
    that slipped past the guardrails cannot read files or write anything."""

    def __init__(self, db_path: Path, timeout_s: float = 30, max_rows: int = 1000,
                 memory_limit: str = "1GB", threads: int = 4):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"warehouse not found: {self.db_path} (run scripts/export_warehouses.py)")
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.con = duckdb.connect(str(self.db_path), read_only=True, config={
            "enable_external_access": False,
            "memory_limit": memory_limit,
            "threads": threads,
        })
        try:
            self.con.execute("SET lock_configuration = true")
        except duckdb.InvalidInputException:
            pass  # another connection in this process (same database instance) already locked it
        self._lock = threading.Lock()

    def run(self, sql: str) -> ExecResult:
        with self._lock:
            timer = threading.Timer(self.timeout_s, self.con.interrupt)
            t0 = time.perf_counter()
            timer.start()
            try:
                cur = self.con.execute(sql)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchmany(self.max_rows + 1)
                trunc = len(rows) > self.max_rows
                return ExecResult(cols, [tuple(r) for r in rows[: self.max_rows]], trunc, None,
                                  time.perf_counter() - t0)
            except duckdb.InterruptException:
                return ExecResult(error=f"query timed out after {self.timeout_s:.0f}s",
                                  elapsed_s=time.perf_counter() - t0, timed_out=True)
            except duckdb.Error as e:
                msg = str(e).strip().splitlines()
                return ExecResult(error=" ".join(msg[:3])[:500], elapsed_s=time.perf_counter() - t0)
            finally:
                timer.cancel()

    def close(self) -> None:
        self.con.close()
