"""Verify the hand-written golden set and write data/golden.jsonl + data/splits.json.

Every answerable question's gold SQL is executed (read-only) against data/warehouse.duckdb and must
pass the agent's own guardrails and return at least one row with at least one non-NULL value.
The split is stratified by difficulty with a fixed seed (config data.split_seed): dev 30 / test 70.

Usage:  .venv\\Scripts\\python scripts\\build_golden.py
Writes: data/golden.jsonl, data/splits.json, results/golden_verification.csv
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config, path  # noqa: E402
from src.executor import Executor  # noqa: E402
from src.guardrails import check_sql  # noqa: E402

LEVELS = ["single_table", "aggregation", "join", "window", "date", "multi_step", "unanswerable"]


def stratified_split(items: list[dict], dev_size: int, seed: int) -> dict[str, list[str]]:
    by = defaultdict(list)
    for q in items:
        by[q["difficulty"]].append(q["id"])
    n = len(items)
    # largest-remainder allocation of dev slots per level; remainder ties broken by the seeded RNG
    rng = random.Random(seed)
    quotas = {k: dev_size * len(v) / n for k, v in by.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    order = sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), rng.random()))
    for k in order[: dev_size - sum(alloc.values())]:
        alloc[k] += 1
    dev, test = [], []
    for k in sorted(by):
        ids = sorted(by[k])
        rng.shuffle(ids)
        dev += ids[: alloc[k]]
        test += ids[alloc[k]:]
    return {"dev": sorted(dev), "test": sorted(test)}


def main() -> None:
    cfg = load_config()
    items = yaml.safe_load((ROOT / "data" / "golden_questions.yaml").read_text(encoding="utf-8"))
    ids = [q["id"] for q in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    ex = Executor(path("warehouse"), timeout_s=120, max_rows=100_000)
    report, out, bad = [], [], []
    for q in items:
        assert q["difficulty"] in LEVELS, q
        rec = {"id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
               "gold_sql": q.get("sql"), "answerable": q.get("sql") is not None,
               "order_matters": bool(q.get("order_matters", False)),
               "schema": None, "gold_rows": None, "gold_columns": None}
        if rec["answerable"]:
            g = check_sql(rec["gold_sql"], add_limit=None)
            if not g.ok:
                bad.append((q["id"], f"guardrail: {g.reason}"))
                continue
            res = ex.run(g.sql)
            if res.error:
                bad.append((q["id"], res.error))
                continue
            nonnull = any(v is not None for row in res.rows for v in row)
            if not res.rows or not nonnull:
                bad.append((q["id"], "empty or all-NULL result"))
                continue
            rec["schema"] = sorted({t.split(".")[0] for t in g.tables})
            rec["gold_rows"] = len(res.rows)
            rec["gold_columns"] = res.columns
            preview = json.dumps([list(r) for r in res.rows[:3]], default=str)[:160]
        else:
            preview = "(unanswerable: correct behaviour is a refusal)"
        out.append(rec)
        report.append({"id": q["id"], "difficulty": q["difficulty"], "rows": rec["gold_rows"],
                       "columns": "|".join(rec["gold_columns"] or []), "preview": preview})
    if bad:
        for b in bad:
            print("REJECTED", *b)
        sys.exit(1)
    with open(path("golden"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    splits = stratified_split(out, cfg["data"]["dev_size"], cfg["data"]["split_seed"])
    path("splits").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    res_dir = path("results")
    res_dir.mkdir(exist_ok=True)
    with open(res_dir / "golden_verification.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    lvl = {q["id"]: q["difficulty"] for q in out}
    print(f"verified {sum(r['answerable'] for r in out)} gold queries + "
          f"{sum(not r['answerable'] for r in out)} unanswerable -> {path('golden')}")
    for s in ("dev", "test"):
        print(s, len(splits[s]), dict(sorted(Counter(lvl[i] for i in splits[s]).items())))


if __name__ == "__main__":
    main()
