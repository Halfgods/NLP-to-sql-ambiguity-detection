"""Tests for pipeline's sqlglot-backed SQL validation (P1-2).

Covers the contract of validate_read_only (single read-only SELECT/WITH
statement, write/DDL/multi-statement rejection, LIMIT injection) and
schema_mismatch_reason (hallucinated tables/columns, alias/CTE resolution,
derived tables), plus the documented degraded mode when sqlglot is absent.
Databases are tiny throwaway SQLite files in pytest's tmp_path.
"""

import sqlite3

import pytest

import pipeline


@pytest.fixture
def stu_db(tmp_path):
    db = tmp_path / "students.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE students ("
                "student_id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
                "department TEXT NOT NULL, average_marks REAL NOT NULL, "
                "pending_fees REAL NOT NULL)")
    con.commit()
    con.close()
    return db


# --------------------------------------------------------------------------
# validate_read_only
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT name FROM students WHERE pending_fees > 0",
    "WITH t AS (SELECT name FROM students) SELECT * FROM t",
    "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt "
    "WHERE x < 5) SELECT SUM(x) FROM cnt",
    'SELECT "weird ""col""" FROM students  -- a real comment, not code',
    "select 1",
])
def test_validate_accepts_read_queries(sql):
    out, why = pipeline.validate_read_only(sql)
    assert out is not None, why
    assert why == ""
    assert "LIMIT 100;" in out   # row cap injected for every no-limit query


def test_validate_keeps_existing_limit_and_trailing_semicolon():
    out, why = pipeline.validate_read_only(
        "SELECT name FROM students ORDER BY average_marks DESC LIMIT 5;")
    assert why == ""
    assert out.count("LIMIT") == 1          # no double cap
    assert out.rstrip().endswith(";")

@pytest.mark.parametrize("sql,why_has", [
    ("SELECT 1; DROP TABLE students", "multiple statements"),
    ("DROP TABLE students", "read-only mode"),
    ("CREATE TABLE t (a INTEGER)", "read-only mode"),
    ("INSERT INTO students VALUES (1, 'x', 'y', 1.0, 0)", "read-only mode"),
    ("UPDATE students SET name='x'", "read-only mode"),
    ("DELETE FROM students", "read-only mode"),
    ("PRAGMA data_version", "read-only mode"),
    ("ATTACH DATABASE 'x.db' AS x", "read-only mode"),
])
def test_validate_rejects_writes_and_multi(sql, why_has):
    out, why = pipeline.validate_read_only(sql)
    assert out is None
    assert why_has in why.lower()


def test_validate_rejects_nested_write_ddl():
    # data-modifying CTE: whether the sqlite dialect refuses to parse it or
    # parses it, the outcome must be rejection
    sql = ("WITH t AS (INSERT INTO students VALUES (1,'x','y',1.0,0) "
           "RETURNING *) SELECT * FROM t")
    out, _ = pipeline.validate_read_only(sql)
    assert out is None


def test_validate_rejects_empty():
    assert pipeline.validate_read_only("")[0] is None
    assert pipeline.validate_read_only("   ")[0] is None


def test_validate_no_sqlglot_degraded_mode(monkeypatch):
    monkeypatch.setattr(pipeline, "_HAS_SQLGLOT", False)
    ok, why = pipeline.validate_read_only("SELECT 1")
    assert ok is not None and why == ""
    bad, why = pipeline.validate_read_only("DROP TABLE students")
    assert bad is None                       # rejected via the regex gate
    assert "select" in why.lower() or "write/DDL" in why
    # a SELECT-looking write still trips the keyword check in degraded mode
    bad2, why2 = pipeline.validate_read_only(
        "SELECT * FROM students; DELETE FROM students")
    assert bad2 is None and "write/DDL keyword" in why2


# --------------------------------------------------------------------------
# schema_mismatch_reason
# --------------------------------------------------------------------------

def test_mismatch_flags_hallucinated_table(stu_db):
    why = pipeline.schema_mismatch_reason(
        "SELECT name FROM employees", stu_db)
    assert "employees" in why and "does not exist" in why


def test_mismatch_flags_hallucinated_column(stu_db):
    why = pipeline.schema_mismatch_reason(
        "SELECT ghost_col FROM students", stu_db)
    # unqualified columns are ambiguous by design — only qualified refs and
    # tables are provable; so this must NOT crash and must not invent a reason
    # (documented), but a hallucinated QUALIFIED ref must be caught.
    assert why == ""
    why = pipeline.schema_mismatch_reason(
        "SELECT s.ghost_col FROM students s", stu_db)
    assert "ghost_col" in why and "does not exist" in why


def test_mismatch_allows_aliases_and_ctes(stu_db):
    ok = ("SELECT s.name, AVG(s.average_marks) AS m FROM students s "
          "GROUP BY s.name ORDER BY m DESC LIMIT 3")
    assert pipeline.schema_mismatch_reason(ok, stu_db) == ""
    cte = ("WITH flagged AS (SELECT * FROM students WHERE pending_fees > 0) "
           "SELECT f.name FROM flagged f")
    assert pipeline.schema_mismatch_reason(cte, stu_db) == ""


def test_mismatch_flags_unknown_qualifier(stu_db):
    why = pipeline.schema_mismatch_reason(
        "SELECT c.name FROM students", stu_db)
    assert "c.name" in why


def test_mismatch_handles_derived_tables(stu_db):
    good = ("SELECT d.name FROM (SELECT name FROM students "
            "WHERE pending_fees > 0) d")
    assert pipeline.schema_mismatch_reason(good, stu_db) == ""
    bad = ("SELECT d.ghosty FROM (SELECT name FROM students "
           "WHERE pending_fees > 0) d")
    why = pipeline.schema_mismatch_reason(bad, stu_db)
    assert "ghosty" in why


def test_mismatch_star_subquery_is_opaque(stu_db):
    sql = ("SELECT d.anything FROM (SELECT * FROM students) d")
    assert pipeline.schema_mismatch_reason(sql, stu_db) == ""


def test_mismatch_case_insensitive_identifiers(stu_db):
    why = pipeline.schema_mismatch_reason(
        "SELECT s.PENDING_FEES FROM STUDENTS s", stu_db)
    assert why == ""
    why = pipeline.schema_mismatch_reason(
        "SELECT s.PENDING_FEESX FROM STUDENTS s", stu_db)
    assert "PENDING_FEESX" in why


def test_mismatch_empty_db_returns_empty(tmp_path):
    db = tmp_path / "empty.db"
    con = sqlite3.connect(db)
    con.commit()
    con.close()
    assert pipeline.schema_mismatch_reason(
        "SELECT 1", db) == ""


def test_mismatch_quoted_identifiers(tmp_path):
    db = tmp_path / "weird.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "weird table" ("col ""a""" TEXT)')
    con.commit()
    con.close()
    assert pipeline.schema_mismatch_reason(
        'SELECT "col ""a""" FROM "weird table"', db) == ""
