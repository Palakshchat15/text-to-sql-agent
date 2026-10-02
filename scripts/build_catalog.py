"""Build the schema catalogue (data/catalog.json) used for schema retrieval and the Schema explorer.

Sources, per table in data/warehouse.duckdb:
  - table / column descriptions and accepted_values tests from each project's dbt models/*/schema.yml
    (read-only; the DS projects are not modified),
  - hand-written notes in data/column_notes.yaml (dbt has few column descriptions),
  - column types and up to N distinct sample values from DuckDB, plus the row count.

Usage:  .venv\\Scripts\\python scripts\\build_catalog.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config, path  # noqa: E402


def dbt_docs(dbt_dir: Path) -> dict[str, dict]:
    """model name -> {description, columns: {name: text}} from every schema.yml under models/."""
    out: dict[str, dict] = {}
    for f in sorted((dbt_dir / "models").rglob("schema.yml")):
        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        for m in doc.get("models", []) or []:
            cols = {}
            for c in m.get("columns", []) or []:
                parts = []
                if c.get("description"):
                    parts.append(c["description"])
                for t in c.get("tests", []) or []:
                    if isinstance(t, dict) and "accepted_values" in t:
                        vals = t["accepted_values"].get("values", [])
                        parts.append("Allowed values: " + ", ".join(map(str, vals)) + ".")
                    elif t == "unique":
                        parts.append("Unique key.")
                if parts:
                    cols[c["name"]] = " ".join(parts)
            out[m["name"]] = {"description": m.get("description", ""), "columns": cols}
    return out


def main() -> None:
    cfg = load_config()
    src = cfg["sources"]
    root = (ROOT / src["projects_root"]).resolve()
    notes = yaml.safe_load((ROOT / "data" / "column_notes.yaml").read_text(encoding="utf-8"))
    n_samples = cfg["retrieval"]["sample_values"]
    con = duckdb.connect(str(path("warehouse")), read_only=True)
    tables = []
    for schema, p in src["projects"].items():
        docs = dbt_docs(root / p["folder"] / p["dbt"])
        sn = notes.get(schema, {})
        names = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = ? ORDER BY 1", [schema]).fetchall()]
        for t in names:
            d = docs.get(t, {"description": "", "columns": {}})
            tn = sn.get(t, {}) or {}
            desc = d["description"] or tn.get("description", "")
            fq = f"{schema}.{t}"
            rows = con.execute(f"SELECT count(*) FROM {fq}").fetchone()[0]
            cols = []
            for name, typ in con.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position", [schema, t]).fetchall():
                text = " ".join(x for x in [(tn.get("columns") or {}).get(name, ""), d["columns"].get(name, "")] if x)
                samples = [r[0] for r in con.execute(
                    f'SELECT DISTINCT "{name}" FROM {fq} WHERE "{name}" IS NOT NULL LIMIT {n_samples}').fetchall()]
                samples = [s if isinstance(s, (int, float, bool, str)) else str(s) for s in samples]
                samples = [round(s, 4) if isinstance(s, float) else s for s in samples]
                cols.append({"name": name, "type": typ, "description": text, "samples": samples})
            tables.append({"schema": schema, "table": t, "fq": fq, "project": sn.get("_project", ""),
                           "description": desc, "rows": rows, "columns": cols,
                           "dbt_documented": t in docs})
    con.close()
    out = path("catalog")
    out.write_text(json.dumps(tables, indent=1, default=str), encoding="utf-8")
    n_desc = sum(1 for t in tables for c in t["columns"] if c["description"])
    n_cols = sum(len(t["columns"]) for t in tables)
    print(f"wrote {out}: {len(tables)} tables, {n_cols} columns, {n_desc} with descriptions")


if __name__ == "__main__":
    main()
