"""Run one agent configuration (ablation) over a split, caching one JSONL record per question.

Re-running resumes: questions already in the cache are skipped. Holds the shared
AI_Engineer_Portfolio/.llm_lock while running (waits if another project holds it).

Usage:
  .venv\\Scripts\\python scripts\\eval_agent.py --split test --config full_agent
  .venv\\Scripts\\python scripts\\eval_agent.py --split dev --config zero_shot_full --ids st01,ag02
  configs: zero_shot_full | retrieval_docs | full_agent | full_agent_3b | oracle_tables_upper
Cache:  results/eval/{split}__{config}.jsonl   (scored by scripts/score.py)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config, path, questions_for  # noqa: E402
from src.guardrails import check_sql  # noqa: E402
from src.llm import check_ollama  # noqa: E402
from src.llm_lock import llm_lock  # noqa: E402


def jsonable(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, dt.datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, (dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    if isinstance(v, float) and v != v:
        return None
    return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)


def cache_path(split: str, config: str) -> Path:
    return path("results") / "eval" / f"{split}__{config}.jsonl"


def load_cache(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                out[r["id"]] = r
            except json.JSONDecodeError:
                pass  # a line cut off by an interruption; that question is re-run
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="override the ablation's model")
    ap.add_argument("--ids", default="", help="comma-separated question ids (sanity runs)")
    a = ap.parse_args()
    cfg = load_config()
    if a.config not in cfg["ablations"]:
        sys.exit(f"unknown config {a.config}; choose from {list(cfg['ablations'])}")
    model = a.model or cfg["ablations"][a.config]["model"]
    qs = questions_for(a.split)
    if a.ids:
        want = set(a.ids.split(","))
        qs = [q for q in qs if q["id"] in want]
    out = cache_path(a.split, a.config if not a.model else f"{a.config}@{model.replace(':', '-')}")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_cache(out)
    todo = [q for q in qs if q["id"] not in done]
    print(f"{a.split}/{a.config} model={model}: {len(qs)} questions, {len(done)} cached, {len(todo)} to run",
          flush=True)
    if not todo:
        return
    check_ollama(model)
    from src.agent import Agent

    with llm_lock(f"eval_agent {a.split} {a.config} {model}"):
        agent = Agent(a.config, model=model, trace_source="eval")
        t_start = time.time()
        for i, q in enumerate(todo, 1):
            oracle = None
            if cfg["ablations"][a.config]["retrieval"] == "oracle" and q["answerable"]:
                oracle = check_sql(q["gold_sql"]).tables
            r = agent.ask(q["question"], oracle_tables=oracle)
            rec = {
                "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
                "split": a.split, "config": a.config, "model": model, "status": r.status,
                "sql": r.sql, "columns": r.columns, "rows": [[jsonable(v) for v in row] for row in r.rows],
                "truncated": r.truncated, "refusal_reason": r.refusal_reason, "error": r.error,
                "retrieved": [t for t, _ in r.retrieved], "n_attempts": len(r.attempts),
                "guard_blocks": r.guard_blocks,
                "attempts": [{k: v for k, v in att.items() if k != "reply"} | {"reply": (att["reply"] or "")[:1500]}
                             for att in r.attempts],
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                "t_retrieve": round(r.t_retrieve, 4), "t_llm": round(r.t_llm, 3), "t_exec": round(r.t_exec, 4),
                "t_total": round(r.t_total, 3), "trace_id": r.trace_id,
                "ts": dt.datetime.now().isoformat(timespec="seconds"),
            }
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            el = time.time() - t_start
            print(f"[{i}/{len(todo)}] {q['id']} {r.status:8s} attempts={len(r.attempts)} "
                  f"{r.t_total:6.1f}s  (elapsed {el / 60:.1f} min, eta {el / i * (len(todo) - i) / 60:.1f} min)",
                  flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
