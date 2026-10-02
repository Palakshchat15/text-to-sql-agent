import pytest

from src.guardrails import check_sql, has_order_by

OK = [
    "SELECT count(*) FROM flight.fact_flights",
    "select carrier_name from flight.dim_carrier where carrier_code = 'B6';",
    "WITH t AS (SELECT carrier_code, count(*) n FROM flight.fact_flights GROUP BY 1) SELECT * FROM t ORDER BY n DESC",
    "SELECT a.city_name FROM flight.fact_flights f JOIN flight.dim_airport a ON a.airport_code = f.origin_airport_code",
    "SELECT channel FROM fraud.fact_transactions UNION SELECT 'x'",
    "SELECT * FROM (SELECT customer_id, row_number() OVER (ORDER BY monetary DESC) rn FROM retail.mart_customer_rfm) WHERE rn <= 3",
    "SELECT sum(revenue) FILTER (WHERE year(event_date) = 2025) FROM sales.fact_daily_sales",
    "SELECT 1",
]

BLOCKED = [
    ("DROP TABLE flight.fact_flights", "only SELECT"),
    ("DELETE FROM flight.fact_flights", "only SELECT"),
    ("UPDATE flight.dim_carrier SET carrier_name = 'x'", "only SELECT"),
    ("INSERT INTO flight.dim_carrier VALUES ('x', 'y', 1)", "only SELECT"),
    ("CREATE TABLE x AS SELECT 1", "only SELECT"),
    ("ALTER TABLE flight.dim_carrier ADD COLUMN x INT", "only SELECT"),
    ("PRAGMA database_list", ""),
    ("ATTACH 'other.db' AS o", ""),
    ("COPY flight.dim_carrier TO 'out.csv'", ""),
    ("SET memory_limit = '10GB'", ""),
    ("INSTALL httpfs", ""),
    ("LOAD httpfs", ""),
    ("SELECT 1; DROP TABLE flight.dim_carrier", "exactly one statement"),
    ("SELECT 1; SELECT 2", "exactly one statement"),
    ("SELECT * FROM read_csv('C:/secret.csv')", "forbidden function"),
    ("SELECT * FROM read_parquet('x.parquet')", "forbidden function"),
    ("SELECT * FROM 'data.csv'", ""),
    ("SELECT getenv('HOME')", "forbidden function"),
    ("SELECT * FROM glob('*')", "forbidden function"),
    ("SELECT * FROM information_schema.tables", "not allowed"),
    ("SELECT * FROM main.secrets", "not allowed"),
    ("SELECT * FROM fact_flights", "must be qualified"),
    ("SELECT * INTO newtab FROM flight.dim_carrier", ""),
    ("", "empty"),
    ("SELEC count(*) FROM", ""),
]


@pytest.mark.parametrize("sql", OK)
def test_allowed(sql):
    r = check_sql(sql)
    assert r.ok, r.reason


@pytest.mark.parametrize("sql,why", BLOCKED)
def test_blocked(sql, why):
    r = check_sql(sql)
    assert not r.ok
    if why:
        assert why in r.reason


def test_limit_added_when_missing():
    r = check_sql("SELECT * FROM flight.dim_carrier", add_limit=1000)
    assert r.ok and r.limit_added and "LIMIT 1000" in r.sql


def test_existing_limit_kept():
    r = check_sql("SELECT * FROM flight.dim_carrier LIMIT 5", add_limit=1000)
    assert r.ok and not r.limit_added and "LIMIT 5" in r.sql and "1000" not in r.sql


def test_no_limit_when_disabled():
    r = check_sql("SELECT * FROM flight.dim_carrier", add_limit=None)
    assert r.ok and "LIMIT" not in r.sql


def test_limit_on_union():
    r = check_sql("SELECT 1 AS a UNION ALL SELECT 2", add_limit=10)
    assert r.ok and "LIMIT 10" in r.sql


def test_tables_reported_and_ctes_excluded():
    r = check_sql("WITH t AS (SELECT * FROM fraud.fact_transactions) SELECT * FROM t JOIN fraud.dim_account USING (account_id)")
    assert r.ok and r.tables == ["fraud.dim_account", "fraud.fact_transactions"]


def test_allowed_tables_whitelist():
    r = check_sql("SELECT * FROM flight.not_a_table", allowed_tables={"flight.fact_flights"})
    assert not r.ok and "unknown table" in r.reason


def test_cte_named_like_forbidden_still_parses():
    assert check_sql("WITH read_data AS (SELECT 1 AS x) SELECT x FROM read_data").ok


def test_has_order_by():
    assert has_order_by("SELECT a FROM flight.dim_carrier ORDER BY a")
    assert not has_order_by("SELECT a FROM (SELECT a FROM flight.dim_carrier ORDER BY a)")


def test_cross_schema_join_blocked():
    r = check_sql("SELECT * FROM sales.fact_daily_sales s JOIN flight.dim_carrier c ON s.category_id = 1")
    assert not r.ok and "unrelated schemas" in r.reason


def test_unterminated_string_is_rejected_not_raised():
    r = check_sql("SELECT * FROM flight.dim_carrier WHERE carrier_name = 'Delta")
    assert not r.ok and "parse error" in r.reason
