"""Answer presentation: a summary built only from the actual result values, and a chart choice.

No LLM is used here, so the summary cannot contain invented numbers: every number in it is copied
from the result table.
"""
from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal

TIME_HINTS = ("date", "month", "day", "week", "year", "hour", "ds", "time", "period")


def fmt(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if v == int(v) and abs(v) < 1e15:
            return f"{int(v):,}"
        if abs(v) >= 100:
            return f"{v:,.2f}"
        if abs(v) >= 1:
            return f"{v:,.3f}".rstrip("0").rstrip(".")
        return f"{v:.4g}"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat(sep=" ") if isinstance(v, dt.datetime) else v.isoformat()
    return str(v)


def _is_num(v) -> bool:
    return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)


def column_kinds(columns: list[str], rows: list[tuple]) -> list[str]:
    """'num' | 'time' | 'cat' per column, from the values (and the column name for integer periods)."""
    kinds = []
    for j, c in enumerate(columns):
        vals = [r[j] for r in rows if r[j] is not None]
        if vals and all(isinstance(v, (dt.date, dt.datetime)) for v in vals):
            kinds.append("time")
        elif vals and all(_is_num(v) for v in vals):
            low = c.lower()
            is_period = any(h == low or low.endswith("_" + h) or low.startswith(h) for h in TIME_HINTS)
            kinds.append("time" if is_period and all(float(v) == int(v) for v in vals) else "num")
        else:
            kinds.append("cat")
    return kinds


def choose_chart(columns: list[str], rows: list[tuple]) -> dict:
    """Return {'type': 'line'|'bar'|'table', 'x': col, 'y': [cols]}."""
    if len(rows) < 2 or len(columns) < 2:
        return {"type": "table"}
    kinds = column_kinds(columns, rows)
    nums = [c for c, k in zip(columns, kinds) if k == "num"]
    if not nums:
        return {"type": "table"}
    times = [c for c, k in zip(columns, kinds) if k == "time"]
    cats = [c for c, k in zip(columns, kinds) if k == "cat"]
    if times:
        return {"type": "line", "x": times[0], "y": nums[:3]}
    if len(cats) == 1 and len(rows) <= 40:
        return {"type": "bar", "x": cats[0], "y": nums[:1]}
    return {"type": "table"}


def summarize(columns: list[str], rows: list[tuple], truncated: bool = False) -> str:
    """Short plain-English summary using only values present in the result."""
    if not rows:
        return "The query ran but returned no rows."
    if len(rows) == 1:
        if len(columns) == 1:
            return f"Answer: **{fmt(rows[0][0])}** ({columns[0]})."
        return "Result: " + "; ".join(f"{c} = **{fmt(v)}**" for c, v in zip(columns, rows[0])) + "."
    n = f"{len(rows):,}{'+ (truncated)' if truncated else ''}"
    kinds = column_kinds(columns, rows)
    nums = [j for j, k in enumerate(kinds) if k == "num"]
    labels = [j for j, k in enumerate(kinds) if k != "num"]
    parts = [f"{n} rows returned."]
    if nums and labels:
        y, x = nums[0], labels[0]
        valid = [r for r in rows if r[y] is not None]
        if valid:
            hi = max(valid, key=lambda r: float(r[y]))
            lo = min(valid, key=lambda r: float(r[y]))
            parts.append(f"Highest {columns[y]}: {fmt(hi[x])} ({fmt(hi[y])}); "
                         f"lowest: {fmt(lo[x])} ({fmt(lo[y])}).")
    elif nums:
        y = nums[0]
        vals = [float(r[y]) for r in rows if r[y] is not None]
        if vals:
            parts.append(f"{columns[y]} ranges from {fmt(min(vals))} to {fmt(max(vals))}.")
    else:
        shown = ", ".join(fmt(r[0]) for r in rows[:5])
        parts.append(f"First values of {columns[0]}: {shown}{' ...' if len(rows) > 5 else ''}.")
    return " ".join(parts)
