"""Bucket the failures of every scored run (results/scored/{split}__*.jsonl). No LLM calls.

Buckets (rule-based, first match wins; heuristic, so spot-check the examples):
  refusal_false      refused an answerable question
  refusal_missed     answered an unanswerable question
  hallucinated_column  final query failed because a column does not exist
  wrong_table          final query used a table that does not exist / is not allowed, or different tables than gold
  sql_error_other      any other execution / guardrail / parse failure
  wrong_join           same tables as gold but a different number of joins or join keys
  wrong_aggregation    different aggregate functions or GROUP BY columns than gold
  date_logic           gold uses date functions/literals and the predicted date logic differs
  other_wrong_result   ran fine, result differs (filters, values, rounding/units, extra/missing rows)

Usage:  .venv\\Scripts\\python scripts\\error_analysis.py [--split test]
Writes: results/error_analysis_{split}.json  ({config: {bucket: {count, examples}}})
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import path  # noqa: E402

AGGS = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
DATE_FUNCS = {"year", "month", "dayofweek", "date_trunc", "extract", "strftime", "date_part", "dayname",
              "monthname", "week", "quarter"}


def parse(sql: str | None):
    if not sql:
        return None
    try:
        return sqlglot.parse_one(sql, read="duckdb")
    except Exception:  # noqa: BLE001
        return None


def features(tree) -> dict:
    tables = {f"{t.db}.{t.name}".lower() for t in tree.find_all(exp.Table) if t.db}
    joins = list(tree.find_all(exp.Join))
    join_cols = set()
    for j in joins:
        for c in j.find_all(exp.Column):
            join_cols.add(c.name.lower())
        for u in j.args.get("using") or []:
            join_cols.add(u.name.lower())
    aggs = sorted(type(a).__name__ for a in tree.find_all(*AGGS))
    groups = set()
    for g in tree.find_all(exp.Group):
        for c in g.find_all(exp.Column):
            groups.add(c.name.lower())
    fn_names = set()
    for f in tree.find_all(exp.Func):
        fn_names.add(f.sql_name().lower())
    for f in tree.find_all(exp.Anonymous):
        fn_names.add(str(f.this).lower())
    has_date = bool(fn_names & DATE_FUNCS) or bool(re.search(r"\d{4}-\d{2}-\d{2}", tree.sql())) or \
        any(isinstance(c, exp.Cast) and c.to.is_type("date") for c in tree.find_all(exp.Cast))
    dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", tree.sql())) | (fn_names & DATE_FUNCS)
    return {"tables": tables, "n_joins": len(joins), "join_cols": join_cols, "aggs": aggs, "groups": groups,
            "has_date": has_date, "dates": dates}


def bucket(s: dict) -> str:
    if s.get("false_refusal"):
        return "refusal_false"
    if s.get("missed_refusal"):
        return "refusal_missed"
    err = (s.get("error") or "").lower()
    if s["status"] in ("error", "blocked"):
        if "column" in err and ("not found" in err or "does not have a column" in err or "referenced column" in err):
            return "hallucinated_column"
        if "table" in err and ("does not exist" in err or "unknown table" in err) or "schema '" in err:
            return "wrong_table"
        return "sql_error_other"
    g, p = parse(s.get("gold_sql")), parse(s.get("pred_sql"))
    if g is None or p is None:
        return "other_wrong_result"
    fg, fp = features(g), features(p)
    if fg["tables"] != fp["tables"]:
        return "wrong_table"
    if fg["n_joins"] != fp["n_joins"] or (fg["n_joins"] and fg["join_cols"] != fp["join_cols"]):
        return "wrong_join"
    if fg["aggs"] != fp["aggs"] or fg["groups"] != fp["groups"]:
        return "wrong_aggregation"
    if fg["has_date"] and fg["dates"] != fp["dates"]:
        return "date_logic"
    return "other_wrong_result"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    files = sorted((path("results") / "scored").glob(f"{a.split}__*.jsonl"))
    if not files:
        sys.exit("no scored runs; run scripts/score.py first")
    out = {}
    for f in files:
        config = f.stem.split("__", 1)[1]
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        b = defaultdict(list)
        for s in rows:
            if not s["correct"]:
                b[bucket(s)].append(s)
        out[config] = {
            "n": len(rows), "n_wrong": sum(not s["correct"] for s in rows),
            "buckets": {k: {"count": len(v), "examples": [
                {"id": s["id"], "question": s["question"], "gold_sql": s["gold_sql"], "pred_sql": s["pred_sql"],
                 "status": s["status"], "reason": s["reason"], "error": s.get("error") or s.get("refusal_reason")}
                for s in v[: a.examples]], "ids": [s["id"] for s in v]}
                for k, v in sorted(b.items(), key=lambda kv: -len(kv[1]))},
        }
        print(config, {k: v["count"] for k, v in out[config]["buckets"].items()})
    p = path("results") / f"error_analysis_{a.split}.json"
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
