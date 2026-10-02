# PROGRESS (resume file)

All commands run from this folder in PowerShell, using the project venv: `.venv\Scripts\python ...`
Build laptop = CPU only. Batch evaluations run later on the GPU machine.

## Status
- [x] 0. venv (Python 3.13) + pinned requirements.txt. Ollama models: qwen2.5-coder:7b and :3b (pulled 2026-10-01).
- [x] 1. Warehouse export DONE 2026-10-01: `.venv\Scripts\python scripts\export_warehouses.py`
      -> data/warehouse.duckdb (22 tables, 4 schemas), data/MANIFEST.csv. Row counts match Postgres exactly.
      Left out (ML artefacts, not analytics tables): flight.mart_flight_features, flight.flight_cutoff_features,
      flight.model_test_predictions, fraud.mart_transaction_features. All 4 stacks stopped afterwards.
- [x] 2. Schema catalogue: `.venv\Scripts\python scripts\build_catalog.py` -> data/catalog.json (22 tables, 227 cols; dbt schema.yml + data/column_notes.yaml + types + 3 samples)
- [x] 3. Golden set (builder-written): data/golden_questions.yaml -> `.venv\Scripts\python scripts\build_golden.py`
      -> data/golden.jsonl, data/splits.json, results/golden_verification.csv. 90 gold queries all executed OK
      (15 each: single_table, aggregation, join, window, date, multi_step) + 10 unanswerable.
      Split seed 42: dev 30 / test 70, stratified. Ties at top-N cut-offs checked by hand (none at the cut).
- [x] 4. Agent code: src/retrieval.py (BM25 tables, k=7 chosen on dev: `scripts\retrieval_dev.py` ->
      results/retrieval_dev.csv), prompts.py, llm.py (Ollama 127.0.0.1:11434), guardrails.py (sqlglot),
      executor.py (read-only DuckDB, external access off, timeout, max rows), agent.py (3 retries, refusal),
      answer.py (summary from real values, chart choice), tracing.py (data/traces.db), llm_lock.py.
- [x] 5. Comparator (src/evaluation.py) + tests: `.venv\Scripts\python -m pytest -q` -> 105 passed (final).
      Self-test found and fixed: 0.1% float tolerance accepted COUNT(*)+1 on large counts -> whole-number
      counts are now compared exactly.
- [x] 6. Eval tooling written, NOT run in batch: scripts/eval_agent.py (JSONL cache per question, resumes,
      takes ../.llm_lock), run_eval_queue.py, score.py, error_analysis.py, run_queue_task.cmd.
      Verified score.py + error_analysis.py on a synthetic (gold-SQL-based, perturbed) cache, then deleted it.
      Optional upper bound ablation: oracle_tables_upper (gold tables given).
- [x] 7. Smoke test: `.venv\Scripts\python scripts\smoke_test.py` (dev st01, jn03, un01; 7B on CPU).
      Run 1 (prompt v1, results/smoke_test_v1_prompt.json): st01 correct (1 attempt, 203 s), jn03 correct
      (2 attempts, 394 s), un01 WRONG: instead of refusing "average ticket price on Delta" the 7B joined
      sales.fact_daily_sales to flight.dim_carrier and used avg_unit_price (4 attempts, 608 s, all failed).
      Dev fix (prompt v2): rule "schemas are unrelated, never join across them; don't substitute a different
      measure" + guardrail blocking cross-schema queries (+1 test). Run 2 (prompt v2) found two bugs, fixed:
      sqlglot TokenError (unterminated quote) crashed the guardrail -> now caught as a parse error; the 7B
      wrapped "CANNOT_ANSWER: ..." in a ```sql fence -> parse_reply now treats it as a refusal (+3 tests).
      FINAL smoke (results/smoke_test.json, log results/smoke_test.log): 3/3 correct.
      st01 ok 1 attempt 346 s (3.1k prompt tok); jn03 ok 2 attempts 439 s (7.5k prompt tok total);
      un01 refused on attempt 2 after the cross-schema guardrail block, 27 s (prompt cache warm).
      CPU latencies are not representative of the GPU machine.
- [x] 8. Streamlit app: `.venv\Scripts\streamlit run streamlit_app.py` (port 8611 via .streamlit/config.toml).
      Pages: Ask (root URL), Schema explorer (/schema), Evaluation (/evaluation, shows "not run yet"),
      Monitoring (/monitoring). Screenshots via headless Edge + DevTools protocol (scratch scripts; plain
      --screenshot only captures the skeleton). Real question asked in the app with the 7B:
      "What is the fraud rate for each channel?" -> correct SQL, 4 rows, bar chart, 246 s on CPU.
      Fixed: "Page not found" for /ask (default page now at root), sidebar forced expanded, toolbar minimal,
      thousands separators in the schema table. Second app question "Which 5 origin airports have the most
      total weather delay minutes?" -> DFW 180,315 ... ATL 67,790 (= gold ag04), 266 s.
      Known cosmetic issue, headless only: in some CDP screenshots the sidebar is captured mid-animation
      (partly cut off); the Evaluation screenshot shows it fully expanded. Not seen as a real app bug.
      App RAM (Streamlit process, excluding Ollama): 65 MB working set at start; after a question
      188 MB working set / 505 MB private (peak WS 191 MB). Server stopped afterwards.
- [x] 9. README draft: results marked "not run yet", exact evaluation commands, design decisions,
      limitations, 8 GB notes.

## COMPLETE (2026-10-02, GPU laptop: RTX 3050 Laptop 4 GB VRAM, 8 GB RAM, Python 3.14.0)
- [x] 10. venv recreated on Python 3.14.0 (all pins installed); pytest 105 passed.
- [x] 11. Test queue run via Task Scheduler (02:46-03:49 IST) + oracle_tables_upper (04:44-05:02). All 5 configs
      70/70. Answerable exec. accuracy: zero_shot_full 0.651, retrieval_docs 0.746, full_agent 0.778,
      full_agent_3b 0.619, oracle_tables_upper 0.857. Refusal: 6/7, 4/7, 4/7, 5/7, 5/7.
- [x] 12. README filled (results, by difficulty, error buckets with real examples, latency labelled as GPU
      laptop, speed check: qwen2.5-coder:7b ~550 tok/s prompt, 5.3-6.7 tok/s generation, 55% CPU / 45% GPU).
- [x] 13. App re-checked with screenshots (results/screenshots/ask, schema, evaluation, monitoring).
      Fixed on Monitoring: refused traces showed "nan" as SQL; times were UTC, now local.
      App RAM: 125 MB working set / 518 MB private.

## (History) Instructions that were used for the GPU-machine run
From this folder, PowerShell:
```powershell
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:3b
.venv\Scripts\python -m pip install -r requirements.txt      # only if the venv was not copied / is broken
.venv\Scripts\python -m pytest -q                            # expect 105 passed
scripts\run_queue_task.cmd                                    # one click; = .venv\Scripts\python scripts\run_eval_queue.py --split test
# optional upper bound afterwards:
.venv\Scripts\python scripts\run_eval_queue.py --split test --jobs oracle_tables_upper
```
- Queue order: zero_shot_full, retrieval_docs, full_agent_3b, full_agent (70 test questions each), scoring
  after each job, error analysis at the end. Logs: results/logs/. Resumes from results/eval/test__<config>.jsonl.
- Stale lock after a crash: if ..\.llm_lock exists and its PID is not running, delete it.
- Then: copy results/summary_test.csv, by_difficulty_test.csv and error_analysis_test.json numbers into the
  README "Results" and "Error analysis" sections (replace every "not run yet"), and re-screenshot the
  Evaluation page.
- Note: the venv was created on the build laptop with host Python 3.13; if it does not work on the GPU
  machine, recreate it: `& "C:/Program Files/Python313/python.exe" -m venv .venv` then pip install -r requirements.txt.
