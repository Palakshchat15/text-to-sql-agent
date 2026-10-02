# Build spec: Text-to-SQL analytics agent over the data science warehouses

Follow `../CONVENTIONS.md` for all shared rules.

## Goal
An agent that answers plain-English analytics questions about the owner's four data science projects' data. It writes SQL, runs it read-only, fixes its own errors, and shows a table plus a chart. It is evaluated by **execution accuracy** on a golden question set.

## Data: the warehouse copy
- The DS projects' Postgres warehouses live in Docker volumes on this laptop:

  | Project folder | Postgres port |
  |---|---|
  | `Airline_Flight_Delay/flight_delay_project` | 5436 |
  | `Bank_Fraud_Detection/fraud_detection_project` | 5435 |
  | `Retail_Ecommerce_Analytics/retail_analytics_project` | 5434 |
  | `Sales_Demand_Forecasting/sales_forecast_project` | 5433 |

- **Export one stack at a time:**
  1. `docker compose up -d postgres` in that folder (start **only** the postgres service; don't run DAGs).
  2. Export the **mart/dimension/fact tables** (the `warehouse` schema and similar; skip raw and staging) to parquet.
  3. `docker compose stop`.
- **Never modify their data, files or containers.**
- Read DB credentials from each project's `.env` inside your script. **Never print or store them**, and never copy the `.env` values into this project.
- **Load everything into one DuckDB file**, `data/warehouse.duckdb`, with a schema per project: `flight`, `fraud`, `retail`, `sales`.
  - Huge fact tables (e.g. 2.2M flights) are fine in DuckDB.
  - If a table is only needed for the dashboards, you may leave it out. Document the choice.
- **Schema documentation:** build a catalogue of table and column descriptions from each project's dbt `schema.yml` files, plus column types and a few sample values. It's used to retrieve the relevant schema for each question.
- **If a stack's data volume is empty** (tables missing), don't run pipelines to rebuild it. Skip that project and document the gap, unless it's quick and safe.

## Golden set
- Write about **90 questions** with **gold SQL**, verified by executing every one. Mix the difficulty levels:
  - single table and filter;
  - aggregation and group-by;
  - joins;
  - window functions and top-N per group;
  - date logic;
  - multi-step questions.
- Add about **10 unanswerable questions** (data not in the warehouse). The correct behaviour is a clear refusal.
- **Split dev 30 / test 70** (stratified by difficulty, fixed seed).
- State in the README that the builder wrote the questions.

## Agent
- **Schema retrieval:** choose the relevant tables and columns from the catalogue (BM25 or embeddings), so the prompt stays small.
- **SQL generation:** Ollama, `qwen2.5-coder:7b` (main) and `qwen2.5-coder:3b`; check the tags with `ollama list`. `temperature=0`.
- **Execution guardrails:**
  - parse with `sqlglot` and allow only SELECT/WITH (reject DDL/DML/PRAGMA/ATTACH/COPY etc.);
  - open DuckDB **read-only**;
  - add a LIMIT if it's missing;
  - timeout;
  - maximum result size.
- **Self-correction loop:** on an error or an empty or suspicious result, feed the error back and retry, at most 3 times. Log each attempt.
- **Answer:** a result table, a short natural-language summary built from the actual result values (not invented), and an automatic chart choice (bar, line or table).
- **Refusal:** if the question can't be answered from the schema, say so.

## Evaluation (test)
- **Execution accuracy:** compare result sets, order-insensitive unless the question asks for an order, with float tolerance and column-order independence. Test the comparator well.
- **Ablations:**

  | Configuration | What changes |
  |---|---|
  | Zero-shot, full schema dump | no retrieval, no retries |
  | Schema retrieval + docs | no retries |
  | Full agent | with retries |
  | Full agent, 3B model | the small-model option |

- **Also report:** accuracy by difficulty; refusal accuracy on unanswerable questions; average attempts; guardrail blocks; latency p50/p95; tokens.
- **Error analysis:** wrong table or column, wrong join, wrong aggregation, date logic, hallucinated column, refusal errors, with examples.

## Streamlit app
1. **Ask:** question, generated SQL (collapsible), attempts log, result table, chart and summary. Include example questions.
2. **Schema explorer:** tables and descriptions per project.
3. **Evaluation:** ablation tables and charts, and a per-question browser.
4. **Monitoring:** traces.
