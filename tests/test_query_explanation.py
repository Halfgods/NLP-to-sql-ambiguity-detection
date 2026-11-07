"""Stress checks for the deterministic SQL explanation shown with results."""

import importlib
import sys
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[1] / "UI" / "server"
sys.path.insert(0, str(SERVER_DIR))
_stdout = sys.stdout
try:
    model_worker = importlib.import_module("model_worker")
finally:
    sys.stdout = _stdout


def test_explanation_covers_projection_filter_group_sort_and_limit():
    explanation = model_worker._explain_read_query(
        "SELECT department, AVG(marks) AS mean_marks FROM students "
        "WHERE active = 1 GROUP BY department ORDER BY mean_marks DESC LIMIT 3")

    assert explanation is not None
    assert "returns" in explanation
    assert "`students`" in explanation
    assert "filters rows with" in explanation
    assert "groups by `department`" in explanation
    assert "sorts by" in explanation
    assert "limits the output to 3 rows" in explanation


def test_explanation_handles_join_and_cte_without_claiming_wrong_source():
    explanation = model_worker._explain_read_query(
        "WITH totals AS (SELECT student_id, SUM(marks) AS total "
        "FROM grades GROUP BY student_id) "
        "SELECT s.name, t.total FROM students s JOIN totals t "
        "ON s.student_id = t.student_id")

    assert explanation is not None
    assert "`students`" in explanation
    assert "`totals`" in explanation
    assert "returns" in explanation


def test_explanation_refuses_invalid_or_multiple_statements():
    assert model_worker._explain_read_query("not sql") is None
    assert model_worker._explain_read_query(
        "SELECT 1; SELECT 2") is None
