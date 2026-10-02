"""Choose the number of retrieved tables (retrieval.top_tables) on DEV only. No LLM calls.

Metric: table recall = share of answerable questions whose gold-SQL tables are all retrieved,
and the mean prompt-schema size (chars / 4 as a rough token estimate).

Usage:  .venv\\Scripts\\python scripts\\retrieval_dev.py [--split dev]
Writes: results/retrieval_<split>.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import path, questions_for  # noqa: E402
from src.guardrails import check_sql  # noqa: E402
from src.retrieval import SchemaRetriever, render_schema  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    r = SchemaRetriever()
    gold = [q for q in questions_for(a.split) if q["answerable"]]
    rows = []
    for k in range(2, 11):
        hit, toks = 0, 0.0
        for q in gold:
            got = [t for t, _ in r.retrieve(q["question"], k)]
            hit += set(check_sql(q["gold_sql"]).tables) <= set(got)
            toks += len(render_schema([r.by_fq[t] for t in got])) / 4
        rows.append({"k": k, "table_recall": round(hit / len(gold), 3), "n": len(gold),
                     "mean_schema_tokens_est": int(toks / len(gold))})
        print(rows[-1])
    full_docs = len(render_schema(r.catalog)) / 4
    full_bare = len(render_schema(r.catalog, docs=False)) / 4
    print(f"full schema with docs ~{int(full_docs)} tokens; bare names+types ~{int(full_bare)} tokens")
    out = path("results") / f"retrieval_{a.split}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
