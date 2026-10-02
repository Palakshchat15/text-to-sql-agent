import datetime as dt
from decimal import Decimal

import pytest

from src.evaluation import compare_results, norm, score_item, values_equal


def cmp(gc, gr, pc, pr, **kw):
    return compare_results(gc, gr, pc, pr, **kw)[0]


# ---- exact and order handling ---------------------------------------------------------------
def test_identical():
    assert cmp(["a"], [(1,), (2,)], ["a"], [(1,), (2,)])


def test_row_order_ignored_by_default():
    assert cmp(["c", "n"], [("x", 1), ("y", 2)], ["c", "n"], [("y", 2), ("x", 1)])


def test_row_order_enforced_when_required():
    assert not cmp(["c", "n"], [("x", 1), ("y", 2)], ["c", "n"], [("y", 2), ("x", 1)], order_matters=True)
    assert cmp(["c", "n"], [("x", 1), ("y", 2)], ["c", "n"], [("x", 1), ("y", 2)], order_matters=True)


def test_column_order_ignored():
    assert cmp(["c", "n"], [("x", 1), ("y", 2)], ["n", "c"], [(1, "x"), (2, "y")])


def test_column_order_ignored_with_row_order_required():
    assert cmp(["c", "n"], [("x", 1), ("y", 2)], ["n", "c"], [(1, "x"), (2, "y")], order_matters=True)


def test_column_names_do_not_matter():
    assert cmp(["count_star()"], [(5,)], ["n_flights"], [(5,)])


# ---- row / column counts ----------------------------------------------------------------------
def test_row_count_mismatch():
    ok, reason = compare_results(["a"], [(1,), (2,)], ["a"], [(1,)])
    assert not ok and "row count" in reason


def test_duplicates_count_as_rows():
    assert not cmp(["a"], [(1,), (1,), (2,)], ["a"], [(1,), (2,), (2,)])


def test_too_few_columns():
    ok, reason = compare_results(["a", "b"], [(1, 2)], ["a"], [(1,)])
    assert not ok and "too few" in reason


def test_extra_columns_allowed_by_default():
    assert cmp(["name"], [("Delta",), ("United",)], ["code", "name"], [("DL", "Delta"), ("UA", "United")])


def test_extra_columns_rejected_when_strict():
    assert not cmp(["name"], [("Delta",)], ["code", "name"], [("DL", "Delta")], allow_extra_columns=False)


def test_both_empty():
    assert cmp(["a"], [], ["a"], [])


def test_rows_must_be_consistent_across_columns():
    # each column matches as a multiset, but the row pairing is wrong
    ok, reason = compare_results(["c", "n"], [("x", 1), ("y", 2)], ["c", "n"], [("x", 2), ("y", 1)])
    assert not ok and "rows differ" in reason


def test_ambiguous_identical_columns_backtracking():
    # two gold columns with identical values; predicted has three candidates
    g = [(1, 1, "a"), (2, 2, "b")]
    p = [("a", 1, 1), ("b", 2, 2)]
    assert cmp(["x", "y", "z"], g, ["z", "x", "y"], p)


# ---- numeric tolerance and types --------------------------------------------------------------
def test_float_tolerance_relative():
    assert cmp(["v"], [(12.3456,)], ["v"], [(12.35,)])          # ROUND(x, 2)
    assert not cmp(["v"], [(12.3456,)], ["v"], [(12.5,)])


def test_float_tolerance_small_values():
    assert cmp(["r"], [(0.123456,)], ["r"], [(0.1235,)])
    assert not cmp(["r"], [(0.0096,)], ["r"], [(0.0100,)])


def test_fraction_vs_percentage_is_not_equal():
    assert not cmp(["r"], [(0.25,)], ["r"], [(25.0,)])


def test_int_float_decimal_equal():
    assert cmp(["n"], [(3,)], ["n"], [(3.0,)])
    assert cmp(["n"], [(Decimal("2.50"),)], ["n"], [(2.5,)])


def test_bool_vs_int():
    assert cmp(["f", "n"], [(True, 5), (False, 7)], ["f", "n"], [(1, 5), (0, 7)])


def test_null_handling():
    assert cmp(["a"], [(None,), (1,)], ["a"], [(1,), (None,)])
    assert not cmp(["a"], [(None,)], ["a"], [(0,)])


def test_nan_is_null():
    assert norm(float("nan")) is None


def test_dates_and_midnight_timestamps():
    assert cmp(["d"], [(dt.date(2024, 1, 15),)], ["d"], [(dt.datetime(2024, 1, 15),)])
    assert cmp(["d"], [(dt.date(2024, 1, 15),)], ["d"], [("2024-01-15 00:00:00",)])   # JSON-cached
    assert cmp(["d"], [(dt.date(2024, 1, 15),)], ["d"], [("2024-01-15",)])
    assert not cmp(["d"], [(dt.date(2024, 1, 15),)], ["d"], [(dt.date(2024, 1, 16),)])


def test_non_midnight_timestamp_string_roundtrip():
    ts = dt.datetime(2010, 12, 1, 8, 34)
    assert cmp(["t"], [(ts,)], ["t"], [(str(ts),)])


def test_strings_are_case_sensitive_but_trimmed():
    assert cmp(["s"], [("Toys",)], ["s"], [(" Toys ",)])
    assert not cmp(["s"], [("Toys",)], ["s"], [("toys",)])


def test_tolerance_sort_misalignment_falls_back_to_greedy():
    g = [(1.0000, "a"), (1.0005, "b")]
    p = [(1.0006, "b"), (0.9999, "a")]          # sorting by float swaps a/b relative to gold
    assert cmp(["v", "s"], g, ["v", "s"], p)


def test_values_equal_mixed_types():
    assert not values_equal("1", 1.0, 1e-3, 1e-6)


@pytest.mark.parametrize("n", [1, 50, 500])
def test_large_unordered(n):
    g = [(i, f"k{i}", i * 0.5) for i in range(n)]
    p = list(reversed(g))
    assert cmp(["a", "b", "c"], g, ["c", "a", "b"], [(r[2], r[0], r[1]) for r in p])


# ---- item scoring ---------------------------------------------------------------------------
GOLD = {"answerable": True, "order_matters": False, "gold_result": {"columns": ["n"], "rows": [[5]]}}
UNANS = {"answerable": False}


def test_score_correct_answer():
    assert score_item(GOLD, {"status": "ok", "columns": ["x"], "rows": [[5]]})["correct"]


def test_score_false_refusal():
    s = score_item(GOLD, {"status": "refused"})
    assert not s["correct"] and s["false_refusal"]


def test_score_error_is_wrong():
    assert not score_item(GOLD, {"status": "error", "error": "boom"})["correct"]


def test_score_unanswerable_refused_is_correct():
    assert score_item(UNANS, {"status": "refused"})["correct"]


def test_score_unanswerable_answered_is_wrong():
    s = score_item(UNANS, {"status": "ok", "columns": ["x"], "rows": [[1]]})
    assert not s["correct"] and s["missed_refusal"]


def test_counts_are_exact_even_when_large():
    assert not cmp(["n"], [(30001,)], ["n"], [(30002,)])
    assert not cmp(["n"], [(30001,)], ["n"], [(30002.0,)])
    assert cmp(["n"], [(30001,)], ["n"], [(30001.0,)])


def test_int_vs_rounded_float_uses_tolerance():
    assert cmp(["v"], [(12345,)], ["v"], [(12345.4,)])


def test_decimal_integral_is_int():
    assert cmp(["n"], [(Decimal("7"),)], ["n"], [(7,)])
    assert not cmp(["n"], [(Decimal("7000"),)], ["n"], [(7001,)])
