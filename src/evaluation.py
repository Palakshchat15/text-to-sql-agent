"""Execution-accuracy comparator.

A predicted result matches the gold result when:
  - both have the same number of rows;
  - every gold column can be matched to a different predicted column (column order does not matter;
    extra predicted columns are allowed by default, since a helpful answer may add context such as a
    name next to a code);
  - on the matched columns, the rows are equal as a multiset (order-insensitive), or as a sequence
    when the question requires an order (order_matters);
  - numbers are compared with a relative + absolute tolerance; ints, floats, Decimals and booleans
    are all numbers (whole-number counts must match exactly); a midnight timestamp equals the same date; NULL equals NULL.
"""
from __future__ import annotations

import datetime as dt
import itertools
import math
import re
from decimal import Decimal

MAX_ASSIGNMENTS = 20000
_MIDNIGHT = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]00:00:00$")


def norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() and v.as_tuple().exponent >= 0 else float(v)
    if isinstance(v, float):
        return None if math.isnan(v) else v
    if isinstance(v, dt.datetime):
        if v.time() == dt.time(0) and v.tzinfo is None:
            return v.date().isoformat()
        return v.isoformat(sep=" ")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, (dt.time, dt.timedelta)):
        return str(v)
    if isinstance(v, str):
        v = v.strip()
        m = _MIDNIGHT.match(v)       # JSON-cached timestamps come back as text
        return m.group(1) if m else v
    return str(v)


def values_equal(a, b, rel_tol: float, abs_tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    na, nb = isinstance(a, (int, float)), isinstance(b, (int, float))
    if na and nb:
        # Counts are exact: when either side is an integer and both are whole numbers, require equality
        # (a 0.1% tolerance would accept COUNT(*) + 1 on large counts).
        if (isinstance(a, int) or isinstance(b, int)) and float(a).is_integer() and float(b).is_integer():
            return int(a) == int(b)
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b


def _key(v):
    if v is None:
        return (0, 0.0, "")
    if isinstance(v, (int, float)):
        return (1, float(v), "")
    return (2, 0.0, v)


def _row_key(r):
    return tuple(_key(v) for v in r)


def rows_equal(a: tuple, b: tuple, rel_tol: float, abs_tol: float) -> bool:
    return len(a) == len(b) and all(values_equal(x, y, rel_tol, abs_tol) for x, y in zip(a, b))


def multiset_equal(xs: list[tuple], ys: list[tuple], rel_tol: float, abs_tol: float) -> bool:
    if len(xs) != len(ys):
        return False
    sx, sy = sorted(xs, key=_row_key), sorted(ys, key=_row_key)
    if all(rows_equal(a, b, rel_tol, abs_tol) for a, b in zip(sx, sy)):
        return True
    # Sorting can misalign rows whose floats differ within tolerance; fall back to greedy matching.
    if len(xs) > 3000:
        return False
    used = [False] * len(sy)
    for a in sx:
        for i, b in enumerate(sy):
            if not used[i] and rows_equal(a, b, rel_tol, abs_tol):
                used[i] = True
                break
        else:
            return False
    return True


def compare_results(gold_cols: list[str], gold_rows: list, pred_cols: list[str], pred_rows: list,
                    order_matters: bool = False, rel_tol: float = 1e-3, abs_tol: float = 1e-6,
                    allow_extra_columns: bool = True) -> tuple[bool, str]:
    """Return (match, reason)."""
    g = [tuple(norm(v) for v in r) for r in gold_rows]
    p = [tuple(norm(v) for v in r) for r in pred_rows]
    ng, np_ = len(gold_cols) if gold_cols else (len(g[0]) if g else 0), \
        len(pred_cols) if pred_cols else (len(p[0]) if p else 0)
    if len(g) != len(p):
        return False, f"row count differs (gold {len(g)}, predicted {len(p)})"
    if not g:
        return True, "both empty"
    if np_ < ng:
        return False, f"too few columns (gold {ng}, predicted {np_})"
    if np_ > ng and not allow_extra_columns:
        return False, f"extra columns (gold {ng}, predicted {np_})"
    gcols = [[r[j] for r in g] for j in range(ng)]
    pcols = [[r[k] for r in p] for k in range(np_)]

    def col_match(a, b):
        if order_matters:
            return all(values_equal(x, y, rel_tol, abs_tol) for x, y in zip(a, b))
        return multiset_equal([(x,) for x in a], [(y,) for y in b], rel_tol, abs_tol)

    cands = []
    for j in range(ng):
        c = [k for k in range(np_) if col_match(gcols[j], pcols[k])]
        if not c:
            return False, f"no predicted column matches gold column {j + 1} ({gold_cols[j] if gold_cols else j})"
        cands.append(c)
    tried = 0
    for assign in itertools.product(*cands):
        if len(set(assign)) < ng:
            continue
        tried += 1
        if tried > MAX_ASSIGNMENTS:
            break
        proj = [tuple(r[k] for k in assign) for r in p]
        if order_matters:
            if all(rows_equal(a, b, rel_tol, abs_tol) for a, b in zip(g, proj)):
                return True, "match (ordered)"
        elif multiset_equal(g, proj, rel_tol, abs_tol):
            return True, "match"
    if order_matters:
        return False, "values match column-wise but not row-wise in the required order"
    return False, "columns match individually but rows differ"


def score_item(gold: dict, pred: dict, rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> dict:
    """Score one cached prediction against its golden item.

    gold: golden.jsonl record plus 'gold_result' {'columns', 'rows'} for answerable questions.
    pred: eval cache record (status, columns, rows, ...).
    """
    status = pred.get("status")
    if not gold["answerable"]:
        ok = status == "refused"
        return {"correct": ok, "reason": "correct refusal" if ok else f"should refuse, got {status}",
                "false_refusal": False, "missed_refusal": not ok}
    if status == "refused":
        return {"correct": False, "reason": "refused an answerable question", "false_refusal": True,
                "missed_refusal": False}
    if status != "ok":
        return {"correct": False, "reason": f"no result ({status}: {str(pred.get('error'))[:120]})",
                "false_refusal": False, "missed_refusal": False}
    gr = gold["gold_result"]
    ok, reason = compare_results(gr["columns"], gr["rows"], pred.get("columns") or [], pred.get("rows") or [],
                                 order_matters=gold.get("order_matters", False), rel_tol=rel_tol, abs_tol=abs_tol)
    return {"correct": ok, "reason": reason, "false_refusal": False, "missed_refusal": False}
