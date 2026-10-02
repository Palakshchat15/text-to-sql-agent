"""Prompt building and parsing of the model's reply (SQL or refusal)."""
from __future__ import annotations

import re

REFUSAL_TOKEN = "CANNOT_ANSWER"

SYSTEM = f"""You are a careful analytics engineer who writes DuckDB SQL.
You answer questions about four data warehouses (schemas flight, fraud, retail, sales) using ONLY the tables
and columns given below.

Rules:
1. Write exactly one read-only query (SELECT or WITH ... SELECT). Never modify data.
2. Always qualify tables with their schema, e.g. flight.fact_flights.
3. Use only columns that appear in the schema. Do not invent tables or columns.
4. Rates and shares stored in the tables are fractions between 0 and 1 unless stated otherwise.
5. Return only the columns needed to answer, with readable aliases. Add ORDER BY only when the question
   asks for an order or a top-N.
6. The four schemas are unrelated projects: never join or combine tables from different schemas.
7. If the question cannot be answered from these tables (the data does not exist, e.g. no column holds the
   asked-for measure), reply with exactly one line: {REFUSAL_TOKEN}: <short reason>. Do not guess or
   substitute a different measure.
8. Otherwise reply with the SQL in a single ```sql code block and nothing else."""


def build_messages(question: str, schema_text: str) -> list[dict]:
    user = f"Schema:\n{schema_text}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def retry_message(problem: str) -> dict:
    return {"role": "user", "content": (
        f"{problem}\nFix the query and reply with the corrected SQL in a ```sql code block, "
        f"or with {REFUSAL_TOKEN}: <reason> if the question cannot be answered from the schema.")}


_FENCE = re.compile(r"```(?:sql|duckdb)?\s*(.*?)```", re.S | re.I)
_BARE = re.compile(r"(?is)\b(WITH|SELECT)\b.*")


def parse_reply(text: str) -> tuple[str | None, str | None]:
    """Return (sql, refusal_reason). Exactly one of them is set (sql may be None if unparseable)."""
    text = (text or "").strip()
    m = _FENCE.search(text)
    fenced = m.group(1).strip() if m else ""
    # A refusal wins unless a real query was given (the 7B sometimes wraps CANNOT_ANSWER in a sql fence).
    if fenced and not (REFUSAL_TOKEN in text.upper() and not re.match(r"(?is)\s*(WITH|SELECT)\b", fenced)):
        return fenced.rstrip(";").strip(), None
    if REFUSAL_TOKEN in text.upper():
        reason = text[text.upper().index(REFUSAL_TOKEN) + len(REFUSAL_TOKEN):].lstrip(" :").strip()
        return None, reason.splitlines()[0] if reason else "the data needed is not in the warehouse"
    m = _BARE.search(text)
    if m:
        return m.group(0).strip().rstrip(";").strip(), None
    return None, None
