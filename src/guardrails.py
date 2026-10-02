"""SQL guardrails: allow exactly one read-only SELECT / WITH query over the warehouse schemas.

Checks (with sqlglot, DuckDB dialect):
  - the text parses, and is exactly one statement;
  - the root is a query (SELECT, WITH ... SELECT, UNION / INTERSECT / EXCEPT);
  - no DDL / DML / PRAGMA / ATTACH / COPY / SET / INSTALL / LOAD ... anywhere in the tree;
  - no file- or environment-reading functions (read_csv, read_parquet, glob, getenv, ...);
  - every table is a CTE name or a schema-qualified warehouse table (flight/fraud/retail/sales);
  - all tables come from ONE schema (the four projects are unrelated, so cross-schema joins are blocked);
  - a LIMIT is added when the outer query has none.
The executor also opens DuckDB read-only with external access disabled (defence in depth).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

ALLOWED_SCHEMAS = frozenset({"flight", "fraud", "retail", "sales"})

# Expression class names that must never appear (checked by name so sqlglot versions don't matter).
FORBIDDEN_NODES = frozenset({
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterTable", "TruncateTable",
    "Command", "Pragma", "Attach", "Detach", "Copy", "Set", "SetItem", "Use", "Transaction",
    "Commit", "Rollback", "LoadData", "Install", "Load", "Export", "Grant", "Revoke", "Cache",
    "Uncache", "Refresh", "Describe", "Show", "Summarize", "Analyze", "Into",
})

FORBIDDEN_FUNCS = frozenset({
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "parquet_metadata", "parquet_schema",
    "read_json", "read_json_auto", "read_json_objects", "read_ndjson", "read_ndjson_auto",
    "read_text", "read_blob", "glob", "sniff_csv", "getenv", "query", "query_table", "sqlite_scan",
    "postgres_scan", "mysql_scan", "iceberg_scan", "delta_scan", "read_xlsx", "duckdb_secrets",
    "duckdb_settings", "duckdb_extensions", "pragma_database_list", "current_setting",
})

QUERY_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)


@dataclass
class GuardResult:
    ok: bool
    sql: str = ""
    reason: str = ""
    tables: list[str] = field(default_factory=list)
    limit_added: bool = False


def _func_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    if isinstance(node, exp.Func):
        return node.sql_name().lower()
    return ""


def check_sql(sql: str, add_limit: int | None = 1000,
              allowed_tables: set[str] | None = None) -> GuardResult:
    """Validate `sql`; return GuardResult with the (possibly LIMIT-ed) SQL to run, or the reason."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        return GuardResult(False, reason="empty SQL")
    try:
        stmts = [s for s in sqlglot.parse(text, read="duckdb") if s is not None]
    except sqlglot.errors.SqlglotError as e:  # ParseError and TokenError (e.g. unterminated quote)
        return GuardResult(False, reason=f"SQL parse error: {str(e).splitlines()[0][:200]}")
    if len(stmts) != 1:
        return GuardResult(False, reason=f"exactly one statement allowed, got {len(stmts)}")
    tree = stmts[0]
    if not isinstance(tree, QUERY_ROOTS):
        return GuardResult(False, reason=f"only SELECT / WITH queries are allowed (got {tree.key.upper()})")
    for node in tree.walk():
        name = type(node).__name__
        if name in FORBIDDEN_NODES:
            return GuardResult(False, reason=f"forbidden statement or clause: {name.upper()}")
        fn = _func_name(node)
        if fn in FORBIDDEN_FUNCS or fn.startswith(("read_", "pragma_")):
            return GuardResult(False, reason=f"forbidden function: {fn}")
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = []
    for t in tree.find_all(exp.Table):
        name = (t.name or "").lower()
        db = (t.db or "").lower()
        if isinstance(t.this, exp.Literal) or not name:
            return GuardResult(False, reason="reading files or literals as tables is not allowed")
        if t.args.get("catalog"):
            return GuardResult(False, reason=f"catalog-qualified table not allowed: {t.sql()}")
        if not db:
            if name in cte_names:
                continue
            return GuardResult(False, reason=(
                f"table '{name}' must be qualified with its schema (flight, fraud, retail or sales), "
                f"e.g. flight.fact_flights"))
        if db not in ALLOWED_SCHEMAS:
            return GuardResult(False, reason=f"schema '{db}' is not allowed; use flight, fraud, retail or sales")
        fq = f"{db}.{name}"
        if allowed_tables is not None and fq not in allowed_tables:
            return GuardResult(False, reason=f"unknown table {fq}")
        tables.append(fq)
    schemas = {t.split(".")[0] for t in tables}
    if len(schemas) > 1:
        return GuardResult(False, reason=(
            f"the query combines unrelated schemas ({', '.join(sorted(schemas))}); each project's data stands "
            f"alone, so use tables from one schema only"))
    limit_added = False
    if add_limit and not tree.args.get("limit"):
        tree = tree.limit(add_limit)
        limit_added = True
    return GuardResult(True, sql=tree.sql(dialect="duckdb"), tables=sorted(set(tables)),
                       limit_added=limit_added)


def has_order_by(sql: str) -> bool:
    """True if the outermost query has an ORDER BY."""
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except sqlglot.errors.SqlglotError:
        return False
    return tree.args.get("order") is not None
