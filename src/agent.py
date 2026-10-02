"""The Text-to-SQL agent: retrieve schema -> prompt -> generate SQL -> guardrails -> execute ->
self-correct (up to N retries) -> summary + chart, or a refusal. Every request is traced to SQLite."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from . import llm
from .answer import choose_chart, summarize
from .config import load_config, path
from .executor import Executor
from .guardrails import check_sql
from .prompts import build_messages, parse_reply, retry_message
from .retrieval import SchemaRetriever, render_schema
from .tracing import Tracer


@dataclass
class AgentResult:
    question: str
    config: str
    model: str
    status: str = "error"                 # ok | refused | error | blocked
    sql: str | None = None                # final SQL that was run
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    refusal_reason: str | None = None
    error: str | None = None
    summary: str = ""
    chart: dict = field(default_factory=lambda: {"type": "table"})
    retrieved: list[tuple[str, float]] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    guard_blocks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    t_retrieve: float = 0.0
    t_llm: float = 0.0
    t_exec: float = 0.0
    t_total: float = 0.0
    trace_id: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rows"] = [list(r) for r in self.rows]
        return d


class Agent:
    def __init__(self, config_name: str = "full_agent", model: str | None = None,
                 trace_source: str | None = "app", retriever: SchemaRetriever | None = None,
                 executor: Executor | None = None):
        cfg = load_config()
        self.cfg = cfg
        ab = cfg["ablations"][config_name]
        self.config_name = config_name
        self.use_retrieval = ab["retrieval"]
        self.use_docs = ab["docs"]
        self.max_retries = ab["max_retries"]
        # The app passes the model chosen in the sidebar (default llm.model); evals use the ablation's model.
        self.model = model or ab.get("model") or cfg["llm"]["model"]
        a = cfg["agent"]
        self.retriever = retriever or SchemaRetriever()
        self.executor = executor or Executor(path("warehouse"), timeout_s=a["timeout_s"], max_rows=a["max_rows"])
        self.default_limit = a["default_limit"]
        self.retry_on_empty = a["retry_on_empty"]
        self.top_k = cfg["retrieval"]["top_tables"]
        self.tracer = Tracer(path("traces_db")) if trace_source else None
        self.trace_source = trace_source

    # ------------------------------------------------------------------------------------------
    def schema_for(self, question: str, oracle_tables: list[str] | None = None
                   ) -> tuple[str, list[tuple[str, float]]]:
        if self.use_retrieval == "oracle" and oracle_tables:
            # Upper-bound ablation: the gold query's tables (eval only; unanswerable -> normal retrieval).
            hits = [(t, 1.0) for t in oracle_tables]
            tables = [self.retriever.by_fq[t] for t in oracle_tables]
        elif self.use_retrieval:
            hits = self.retriever.retrieve(question, self.top_k)
            tables = [self.retriever.by_fq[t] for t, _ in hits]
        else:
            hits = [(t, 0.0) for t in self.retriever.all_tables()]
            tables = self.retriever.catalog
        return render_schema(tables, docs=self.use_docs), hits

    def ask(self, question: str, oracle_tables: list[str] | None = None) -> AgentResult:
        t0 = time.perf_counter()
        res = AgentResult(question=question, config=self.config_name, model=self.model)
        try:
            self._run(question, res, oracle_tables)
        except llm.OllamaUnavailable as e:
            res.status, res.error = "error", str(e)
            raise
        finally:
            res.t_total = time.perf_counter() - t0
            if self.tracer:
                res.trace_id = self._trace(res)
        return res

    def _run(self, question: str, res: AgentResult, oracle_tables: list[str] | None = None) -> None:
        t = time.perf_counter()
        schema_text, res.retrieved = self.schema_for(question, oracle_tables)
        res.t_retrieve = time.perf_counter() - t
        messages = build_messages(question, schema_text)
        allowed = self.retriever.table_names()
        last_problem = None
        for n in range(1 + self.max_retries):
            att = {"n": n + 1, "reply": None, "sql": None, "sql_run": None, "guard_reason": None,
                   "error": None, "n_rows": None, "t_llm": 0.0, "t_exec": 0.0,
                   "prompt_tokens": 0, "completion_tokens": 0}
            res.attempts.append(att)
            out = llm.chat(messages, self.model)
            att.update(reply=out["content"][:4000], t_llm=round(out["wall_s"], 3),
                       prompt_tokens=out["prompt_tokens"], completion_tokens=out["completion_tokens"])
            res.t_llm += out["wall_s"]
            res.prompt_tokens += out["prompt_tokens"]
            res.completion_tokens += out["completion_tokens"]
            messages.append({"role": "assistant", "content": out["content"]})
            sql, refusal = parse_reply(out["content"])
            if refusal is not None:
                res.status, res.refusal_reason = "refused", refusal
                res.summary = f"I can't answer this from the warehouse: {refusal}"
                return
            if not sql:
                last_problem = "Your reply did not contain a SQL query."
                att["error"] = last_problem
                res.status, res.error = "error", last_problem
                messages.append(retry_message(last_problem))
                continue
            att["sql"] = sql
            g = check_sql(sql, add_limit=self.default_limit, allowed_tables=allowed)
            if not g.ok:
                res.guard_blocks += 1
                att["guard_reason"] = g.reason
                last_problem = f"The query was rejected by the SQL guardrails: {g.reason}"
                res.status, res.error, res.sql = "blocked", g.reason, sql
                messages.append(retry_message(last_problem))
                continue
            att["sql_run"] = g.sql
            ex = self.executor.run(g.sql)
            att["t_exec"] = round(ex.elapsed_s, 4)
            res.t_exec += ex.elapsed_s
            res.sql = g.sql
            if ex.error:
                att["error"] = ex.error
                res.status, res.error = "error", ex.error
                messages.append(retry_message(f"The query failed with this DuckDB error: {ex.error}"))
                continue
            att["n_rows"] = len(ex.rows)
            res.status, res.error = "ok", None
            res.columns, res.rows, res.truncated = ex.columns, ex.rows, ex.truncated
            suspicious = self._suspicious(ex.rows)
            if suspicious and n < self.max_retries:
                att["error"] = f"suspicious result: {suspicious}"
                messages.append(retry_message(
                    f"The query ran but {suspicious}. Check the filters, values and joins against the "
                    f"schema examples (e.g. exact spelling and case of text values, rates as fractions)."))
                continue
            break
        if res.status == "ok":
            res.summary = summarize(res.columns, res.rows, res.truncated)
            res.chart = choose_chart(res.columns, res.rows)

    def _suspicious(self, rows: list[tuple]) -> str | None:
        if not self.retry_on_empty:
            return None
        if not rows:
            return "returned no rows"
        if all(v is None for r in rows for v in r):
            return "returned only NULL values"
        return None

    def _trace(self, r: AgentResult) -> int:
        return self.tracer.log(
            source=self.trace_source, question=r.question, config=r.config, model=r.model,
            retrieved=r.retrieved, status=r.status, n_attempts=len(r.attempts), guard_blocks=r.guard_blocks,
            attempts=r.attempts, final_sql=r.sql, n_rows=len(r.rows),
            t_retrieve=round(r.t_retrieve, 4), t_llm=round(r.t_llm, 3), t_exec=round(r.t_exec, 4),
            t_total=round(r.t_total, 3), prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
            summary=r.summary, error=r.error or r.refusal_reason)
