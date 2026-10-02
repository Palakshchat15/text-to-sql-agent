"""Retrieval, prompt parsing, answer building, executor safety and the golden set (no LLM calls)."""
import datetime as dt

import pytest

from src.answer import choose_chart, summarize
from src.config import load_golden, load_splits, path
from src.evaluation import compare_results
from src.executor import Executor
from src.guardrails import check_sql
from src.prompts import REFUSAL_TOKEN, parse_reply
from src.retrieval import SchemaRetriever, render_schema

HAVE_DB = path("warehouse").exists() and path("catalog").exists()
needs_db = pytest.mark.skipif(not HAVE_DB, reason="warehouse.duckdb / catalog.json not built")


@pytest.fixture(scope="module")
def retriever():
    return SchemaRetriever()


@pytest.fixture(scope="module")
def ex():
    e = Executor(path("warehouse"), timeout_s=10, max_rows=100)
    yield e
    e.close()


# ---- retrieval ------------------------------------------------------------------------------
@needs_db
@pytest.mark.parametrize("q,table", [
    ("What is the full name of the airline with carrier code B6?", "flight.dim_carrier"),
    ("How many card transactions were fraudulent?", "fraud.fact_transactions"),
    ("How many retail customers are in the Champions segment?", "retail.mart_customer_rfm"),
    ("What is the seasonality profile of the Toys category?", "sales.dim_product_category"),
    ("Prophet forecast yhat for Toys in February 2026", "sales.demand_forecast"),
])
def test_retrieval_finds_table(retriever, q, table):
    assert table in [t for t, _ in retriever.retrieve(q, 5)]


@needs_db
def test_retrieval_k_and_scores(retriever):
    hits = retriever.retrieve("flights cancelled by weather", 3)
    assert len(hits) == 3 and hits[0][1] >= hits[-1][1]


@needs_db
def test_retrieval_empty_question(retriever):
    assert len(retriever.retrieve("???", 4)) == 4


@needs_db
def test_render_schema_docs_vs_bare(retriever):
    t = [retriever.by_fq["flight.dim_carrier"]]
    with_docs, bare = render_schema(t), render_schema(t, docs=False)
    assert "CREATE TABLE flight.dim_carrier" in bare and "--" not in bare
    assert "Examples:" in with_docs and len(with_docs) > len(bare)


@needs_db
def test_catalog_covers_warehouse(retriever, ex):
    r = ex.run("SELECT table_schema || '.' || table_name FROM information_schema.tables")
    assert {x[0] for x in r.rows} == retriever.table_names()


# ---- prompt parsing -------------------------------------------------------------------------
def test_parse_fenced_sql():
    assert parse_reply("```sql\nSELECT 1;\n```") == ("SELECT 1", None)


def test_parse_bare_sql():
    assert parse_reply("Here you go: SELECT 2") == ("SELECT 2", None)


def test_parse_refusal():
    sql, reason = parse_reply(f"{REFUSAL_TOKEN}: no ticket prices in the data")
    assert sql is None and "ticket" in reason


def test_parse_nothing():
    assert parse_reply("I am not sure.") == (None, None)


# ---- answer ---------------------------------------------------------------------------------
def test_summary_scalar_uses_value():
    assert "1,234" in summarize(["n"], [(1234,)])


def test_summary_top_bottom_from_values():
    s = summarize(["carrier", "flights"], [("AA", 10), ("DL", 30), ("UA", 20)])
    assert "DL (30)" in s and "AA (10)" in s and "3 rows" in s


def test_summary_empty():
    assert "no rows" in summarize(["a"], [])


def test_chart_choice():
    assert choose_chart(["d", "n"], [(dt.date(2024, 1, i), i) for i in range(1, 5)])["type"] == "line"
    assert choose_chart(["flight_month", "n"], [(1, 5), (2, 6)])["type"] == "line"
    assert choose_chart(["carrier", "n"], [("AA", 1), ("DL", 2)])["type"] == "bar"
    assert choose_chart(["n"], [(1,)])["type"] == "table"
    assert choose_chart(["a", "b"], [("x", "y"), ("z", "w")])["type"] == "table"


# ---- executor safety ------------------------------------------------------------------------
@needs_db
def test_executor_is_read_only(ex):
    assert ex.run("CREATE TABLE flight.x AS SELECT 1").error


@needs_db
def test_executor_blocks_file_access(ex):
    assert ex.run("SELECT * FROM read_csv('config.yaml')").error


@needs_db
def test_executor_max_rows(ex):
    r = ex.run("SELECT * FROM flight.dim_airport")
    assert len(r.rows) == 100 and r.truncated


@needs_db
def test_executor_timeout():
    e = Executor(path("warehouse"), timeout_s=0.5, max_rows=10)
    r = e.run("SELECT count(*) FROM range(1000000000) a, range(100) b WHERE a.range * b.range = -1")
    e.close()
    assert r.timed_out and "timed out" in r.error


# ---- golden set -----------------------------------------------------------------------------
@needs_db
def test_golden_split_and_size():
    g = load_golden()
    s = load_splits()
    assert len(g) == 100 and sum(not q["answerable"] for q in g) == 10
    assert len(s["dev"]) == 30 and len(s["test"]) == 70
    assert not set(s["dev"]) & set(s["test"])
    assert set(s["dev"]) | set(s["test"]) == {q["id"] for q in g}


@needs_db
def test_every_gold_query_runs_and_matches_itself(ex):
    big = Executor(path("warehouse"), timeout_s=120, max_rows=100_000)
    for q in load_golden():
        if not q["answerable"]:
            continue
        r = big.run(check_sql(q["gold_sql"], add_limit=None).sql)
        assert r.error is None and r.rows, q["id"]
        rev = list(reversed(r.rows)) if not q["order_matters"] else r.rows
        assert compare_results(r.columns, r.rows, r.columns[::-1], [t[::-1] for t in rev],
                               order_matters=q["order_matters"])[0], q["id"]
    big.close()


def test_parse_refusal_inside_sql_fence():
    # seen in the smoke test: the 7B wrapped its refusal in a sql code block
    sql, reason = parse_reply(f"```sql\n{REFUSAL_TOKEN}: no ticket prices in the warehouse.\n```")
    assert sql is None and reason.startswith("no ticket prices")


def test_parse_sql_mentioning_token_in_comment_still_sql():
    sql, _ = parse_reply(f"```sql\nSELECT 1 -- not {REFUSAL_TOKEN}\n```")
    assert sql.startswith("SELECT 1")
