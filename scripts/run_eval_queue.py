"""Run the test-set ablations strictly one after another (never two LLM jobs at once), scoring
after each job and running the error analysis at the end. Every job resumes from its per-question
JSONL cache, so re-running this script after an interruption continues where it stopped.

Usage:  .venv\\Scripts\\python scripts\\run_eval_queue.py [--split test] [--jobs zero_shot_full,full_agent]
Logs:   results/logs/eval_{split}__{config}.log, results/logs/score_{split}.log
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# Fastest first, so the most finishes if the run is interrupted. oracle_tables_upper is optional:
# add it with --jobs ...,oracle_tables_upper
DEFAULT_JOBS = ["zero_shot_full", "retrieval_docs", "full_agent_3b", "full_agent"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=",".join(DEFAULT_JOBS))
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    logs = ROOT / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from src.llm_lock import llm_lock

    # Hold the shared lock for the whole queue (the child jobs see our own lock and proceed).
    with llm_lock(f"run_eval_queue {a.split} {a.jobs}"):
        for job in a.jobs.split(","):
            log = logs / f"eval_{a.split}__{job}.log"
            print(f"== {job} -> {log.name}", flush=True)
            with open(log, "a", encoding="utf-8") as f:
                rc = subprocess.call([PY, "-u", str(ROOT / "scripts" / "eval_agent.py"), "--split", a.split,
                                      "--config", job], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
            print(f"   exit {rc}", flush=True)
            if rc != 0:
                sys.exit(rc)
            with open(logs / f"score_{a.split}.log", "a", encoding="utf-8") as f:
                rc = subprocess.call([PY, "-u", str(ROOT / "scripts" / "score.py"), "--split", a.split],
                                     stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
            print(f"   scoring exit {rc}", flush=True)
    rc = subprocess.call([PY, "-u", str(ROOT / "scripts" / "error_analysis.py"), "--split", a.split], cwd=ROOT)
    print(f"error analysis exit {rc}", flush=True)


if __name__ == "__main__":
    main()
