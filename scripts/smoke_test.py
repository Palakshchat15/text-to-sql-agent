"""End-to-end smoke test on a few DEV questions (proves the pipeline works; not an evaluation).

Usage:  .venv\\Scripts\\python scripts\\smoke_test.py [--ids st13,jn13,un09] [--model qwen2.5-coder:7b]
Writes: results/smoke_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_agent import jsonable  # noqa: E402
from src.config import load_config, path, questions_for  # noqa: E402
from src.evaluation import score_item  # noqa: E402
from src.executor import Executor  # noqa: E402
from src.guardrails import check_sql  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--model", default=load_config()["llm"]["model"])
    a = ap.parse_args()
    dev = questions_for("dev")
    if a.ids:
        qs = [q for q in dev if q["id"] in a.ids.split(",")]
    else:   # one simple, one join, one unanswerable question from dev
        pick = {}
        for q in dev:
            pick.setdefault(q["difficulty"], q)
        qs = [pick[k] for k in ("single_table", "join", "unanswerable") if k in pick]
    from src.agent import Agent

    agent = Agent("full_agent", model=a.model, trace_source="smoke")
    gold_ex = Executor(path("warehouse"), timeout_s=120, max_rows=100_000)
    out = []
    for q in qs:
        r = agent.ask(q["question"])
        g = dict(q)
        if q["answerable"]:
            gr = gold_ex.run(check_sql(q["gold_sql"], add_limit=None).sql)
            g["gold_result"] = {"columns": gr.columns, "rows": gr.rows}
        s = score_item(g, {"status": r.status, "columns": r.columns, "rows": r.rows, "error": r.error})
        rec = {"id": q["id"], "difficulty": q["difficulty"], "question": q["question"], "model": a.model,
               "status": r.status, "sql": r.sql, "refusal_reason": r.refusal_reason, "error": r.error,
               "rows_preview": [[jsonable(v) for v in row] for row in r.rows[:5]], "n_rows": len(r.rows),
               "summary": r.summary, "chart": r.chart, "n_attempts": len(r.attempts),
               "retrieved": [t for t, _ in r.retrieved], "prompt_tokens": r.prompt_tokens,
               "completion_tokens": r.completion_tokens, "t_llm": round(r.t_llm, 1),
               "t_total": round(r.t_total, 1), "correct": s["correct"], "reason": s["reason"]}
        out.append(rec)
        print(json.dumps(rec, default=str), flush=True)
    p = path("results") / "smoke_test.json"
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"{sum(r['correct'] for r in out)}/{len(out)} correct -> {p}")


if __name__ == "__main__":
    main()
