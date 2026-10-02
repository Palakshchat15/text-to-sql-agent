"""Score every cached evaluation run of a split (results/eval/{split}__*.jsonl). No LLM calls.

Gold results are recomputed by executing the gold SQL (read-only). Outputs:
  results/scored/{split}__{config}.jsonl   per-question score + reason
  results/summary_{split}.csv              one row per config (accuracy, refusals, attempts, latency, tokens)
  results/by_difficulty_{split}.csv        accuracy per config x difficulty

Usage:  .venv\\Scripts\\python scripts\\score.py [--split test]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config, path, questions_for  # noqa: E402
from src.evaluation import score_item  # noqa: E402
from src.executor import Executor  # noqa: E402
from src.guardrails import check_sql  # noqa: E402

LEVELS = ["single_table", "aggregation", "join", "window", "date", "multi_step", "unanswerable"]


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def gold_results(qs: list[dict]) -> dict[str, dict]:
    ex = Executor(path("warehouse"), timeout_s=120, max_rows=100_000)
    out = {}
    for q in qs:
        if q["answerable"]:
            r = ex.run(check_sql(q["gold_sql"], add_limit=None).sql)
            if r.error:
                raise RuntimeError(f"gold SQL failed for {q['id']}: {r.error}")
            out[q["id"]] = {"columns": r.columns, "rows": r.rows}
    ex.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    ev = load_config()["eval"]
    qs = {q["id"]: q for q in questions_for(a.split)}
    files = sorted((path("results") / "eval").glob(f"{a.split}__*.jsonl"))
    if not files:
        sys.exit(f"no cached runs for split {a.split} in results/eval/ (run scripts/eval_agent.py first)")
    gold = gold_results(list(qs.values()))
    sdir = path("results") / "scored"
    sdir.mkdir(parents=True, exist_ok=True)
    summary, by_diff = [], []
    for f in files:
        config = f.stem.split("__", 1)[1]
        recs = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    recs[r["id"]] = r
                except json.JSONDecodeError:
                    pass
        scored = []
        for qid, r in recs.items():
            if qid not in qs:
                continue
            g = dict(qs[qid])
            if g["answerable"]:
                g["gold_result"] = gold[qid]
            s = score_item(g, r, rel_tol=ev["float_rel_tol"], abs_tol=ev["float_abs_tol"])
            scored.append({"id": qid, "difficulty": g["difficulty"], "answerable": g["answerable"],
                           "question": g["question"], "gold_sql": g["gold_sql"], "pred_sql": r.get("sql"),
                           "status": r["status"], "error": r.get("error"),
                           "refusal_reason": r.get("refusal_reason"), "n_attempts": r["n_attempts"],
                           "guard_blocks": r["guard_blocks"], "t_total": r["t_total"],
                           "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                           "model": r["model"], **s})
        with open(sdir / f"{a.split}__{config}.jsonl", "w", encoding="utf-8") as fo:
            for s in scored:
                fo.write(json.dumps(s, default=str) + "\n")
        ans = [s for s in scored if s["answerable"]]
        una = [s for s in scored if not s["answerable"]]
        lat = [s["t_total"] for s in scored]
        row = {
            "config": config, "model": scored[0]["model"] if scored else "",
            "n_done": len(scored), "n_split": len(qs), "complete": len(scored) == len(qs),
            "overall_acc": round(sum(s["correct"] for s in scored) / len(scored), 3) if scored else None,
            "exec_acc_answerable": round(sum(s["correct"] for s in ans) / len(ans), 3) if ans else None,
            "refusal_acc_unanswerable": round(sum(s["correct"] for s in una) / len(una), 3) if una else None,
            "false_refusals": sum(s["false_refusal"] for s in scored),
            "no_result": sum(s["status"] in ("error", "blocked") for s in ans),
            "avg_attempts": round(statistics.mean(s["n_attempts"] for s in scored), 2) if scored else None,
            "guard_blocks": sum(s["guard_blocks"] for s in scored),
            "items_with_guard_block": sum(s["guard_blocks"] > 0 for s in scored),
            "latency_p50_s": round(pct(lat, 0.5), 1) if lat else None,
            "latency_p95_s": round(pct(lat, 0.95), 1) if lat else None,
            "mean_prompt_tokens": round(statistics.mean(s["prompt_tokens"] for s in scored)) if scored else None,
            "mean_completion_tokens": round(statistics.mean(s["completion_tokens"] for s in scored)) if scored else None,
        }
        summary.append(row)
        for lv in LEVELS:
            xs = [s for s in scored if s["difficulty"] == lv]
            by_diff.append({"config": config, "difficulty": lv, "n": len(xs),
                            "accuracy": round(sum(s["correct"] for s in xs) / len(xs), 3) if xs else None})
        print(json.dumps(row))
    for name, rows in ((f"summary_{a.split}.csv", summary), (f"by_difficulty_{a.split}.csv", by_diff)):
        with open(path("results") / name, "w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("wrote", path("results") / f"summary_{a.split}.csv")


if __name__ == "__main__":
    main()
