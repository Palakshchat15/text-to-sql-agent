"""Copy the 4 data science warehouses (Postgres in Docker volumes) into data/warehouse.duckdb.

For ONE project at a time: `docker compose up -d postgres` (only that service), wait until it is
ready, export every table in its `warehouse` schema (except ML artefacts listed in config.yaml) to a
temporary CSV, then `docker compose stop`. Credentials never leave the container: psql runs inside
it via `sh -c` and uses the container's own POSTGRES_USER / POSTGRES_DB variables. Nothing here
reads the projects' .env files.

Then the CSVs are loaded into DuckDB (schema per project) with types mapped from Postgres
information_schema, the temporary CSVs are deleted and data/MANIFEST.csv is written.

Usage:  .venv\\Scripts\\python scripts\\export_warehouses.py [--only flight,fraud]
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config, path  # noqa: E402

PSQL = 'psql -X -q -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -f -'

# Postgres -> DuckDB types
TYPE_MAP = {
    "integer": "INTEGER", "smallint": "SMALLINT", "bigint": "BIGINT",
    "numeric": "DOUBLE", "double precision": "DOUBLE", "real": "DOUBLE",
    "text": "VARCHAR", "character varying": "VARCHAR", "character": "VARCHAR",
    "boolean": "BOOLEAN", "date": "DATE",
    "timestamp without time zone": "TIMESTAMP", "timestamp with time zone": "TIMESTAMPTZ",
    "time without time zone": "TIME", "interval": "INTERVAL",
}


def compose(folder: Path, *args: str, stdin: str | None = None, stdout=None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", *args], cwd=folder, input=stdin.encode() if stdin else None,
                          stdout=stdout or subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def psql(folder: Path, sql: str, stdout=None) -> str:
    r = compose(folder, "exec", "-T", "postgres", "sh", "-c", PSQL, stdin=sql, stdout=stdout)
    if r.returncode != 0:
        raise RuntimeError(f"psql failed in {folder.name}: {r.stderr.decode(errors='replace')[:500]}")
    return r.stdout.decode("utf-8") if stdout is None else ""


def wait_ready(folder: Path, timeout_s: int = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = compose(folder, "exec", "-T", "postgres", "sh", "-c",
                    'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"')
        if r.returncode == 0:
            return
        time.sleep(2)
    raise TimeoutError(f"postgres in {folder.name} not ready after {timeout_s}s")


def export_project(name: str, folder: Path, src_schema: str, exclude: list[str], tmp: Path) -> list[dict]:
    print(f"== {name}: starting postgres in {folder}", flush=True)
    r = compose(folder, "up", "-d", "postgres")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace"))
    tables = []
    try:
        wait_ready(folder)
        cols_csv = psql(folder, (
            "\\copy (SELECT table_name, column_name, data_type, ordinal_position "
            f"FROM information_schema.columns WHERE table_schema = '{src_schema}' "
            "ORDER BY table_name, ordinal_position) TO STDOUT WITH CSV HEADER\n"))
        cols: dict[str, list[tuple[str, str]]] = {}
        for row in csv.DictReader(io.StringIO(cols_csv)):
            cols.setdefault(row["table_name"], []).append((row["column_name"], row["data_type"]))
        for t, c in sorted(cols.items()):
            if t in exclude:
                print(f"   skip {t} (excluded in config)", flush=True)
                continue
            out = tmp / f"{name}__{t}.csv"
            t0 = time.time()
            with open(out, "wb") as f:
                psql(folder, f"\\copy (SELECT * FROM {src_schema}.\"{t}\") TO STDOUT WITH CSV HEADER\n", stdout=f)
            print(f"   {t}: {out.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.1f}s", flush=True)
            tables.append({"schema": name, "table": t, "csv": out, "columns": c})
    finally:
        compose(folder, "stop")
        print(f"   stopped {name}", flush=True)
    return tables


def load_duckdb(tables: list[dict], db: Path) -> list[dict]:
    import duckdb

    con = duckdb.connect(str(db))
    rows = []
    for t in tables:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {t['schema']}")
        unknown = [dt for _, dt in t["columns"] if dt not in TYPE_MAP]
        if unknown:
            print(f"   WARNING {t['table']}: unmapped types {unknown} -> VARCHAR")
        types = {c: TYPE_MAP.get(dt, "VARCHAR") for c, dt in t["columns"]}
        col_spec = "{" + ", ".join(f"'{c}': '{ty}'" for c, ty in types.items()) + "}"
        fq = f'{t["schema"]}."{t["table"]}"'
        con.execute(f"DROP TABLE IF EXISTS {fq}")
        con.execute(f"CREATE TABLE {fq} AS SELECT * FROM read_csv('{t['csv'].as_posix()}', header=true, "
                    f"columns={col_spec}, nullstr='', quote='\"', escape='\"')")
        n = con.execute(f"SELECT count(*) FROM {fq}").fetchone()[0]
        rows.append({"schema": t["schema"], "table": t["table"], "rows": n, "columns": len(types),
                     "source": f"postgres warehouse.{t['table']}",
                     "exported_at": datetime.now().isoformat(timespec="seconds")})
        print(f"   loaded {fq}: {n:,} rows", flush=True)
    con.execute("CHECKPOINT")
    con.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    cfg = load_config()["sources"]
    root = (ROOT / cfg["projects_root"]).resolve()
    names = [n for n in cfg["projects"] if not a.only or n in a.only.split(",")]
    db = path("warehouse")
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="t2sql_export_"))
    manifest = path("manifest")
    old = []
    if manifest.exists() and a.only:
        with open(manifest, encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r["schema"] not in names]
    try:
        rows = []
        for n in names:
            tables = export_project(n, root / cfg["projects"][n]["folder"], cfg["source_schema"],
                                    cfg["exclude"].get(n, []), tmp)
            rows += load_duckdb(tables, db)
            for t in tables:            # free disk as we go
                t["csv"].unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["schema", "table", "rows", "columns", "source", "exported_at"])
        w.writeheader()
        w.writerows(old + rows)
    print(f"wrote {manifest} ({len(old) + len(rows)} tables)")


if __name__ == "__main__":
    main()
