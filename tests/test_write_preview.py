"""Tests for the P1-3 write-preview/transaction-safety additions.

Contract under test (pipeline.py):
* `_apply_write_transaction(preview=True)` returns (affected, info) with a
  column-level before/after diff, duplicate-row and net-zero-change flags,
  and leaves the database byte-identical (rollback);
* the authorizer refuses writes to any table other than the validated target
  (regression) while allowing the preview temp snapshot;
* `propose_write` cannot be tested here (needs the served model) — its
  plumbing (fingerprint, preview passthrough) is exercised through
  confirm_write + the low-level transaction instead;
* `confirm_write(expect_fingerprint=...)` refuses a stale fingerprint
  (TOCTOU guard) and applies a fresh one.
"""

import json
import sqlite3
import time

import pytest

import pipeline
from pipeline import InputError, _apply_write_transaction, _db_fingerprint, \
    confirm_write


@pytest.fixture
def prod_db(tmp_path):
    db = tmp_path / "products.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE products ("
                "product_id INTEGER PRIMARY KEY, product_name TEXT, "
                "price REAL, note TEXT)")
    con.executemany("INSERT INTO products VALUES (?,?,?,?)", [
        (1, "Cable", 10.0, None),
        (2, "Keyboard", 25.0, None),
        (3, "Mouse", 8.0, None),
    ])
    con.commit()
    con.close()
    return db


def _price_of(db, pid=1):
    con = sqlite3.connect(db)
    v = con.execute("SELECT price FROM products WHERE product_id=?",
                    (pid,)).fetchone()[0]
    con.close()
    return v


def test_update_preview_diff_before_after(prod_db):
    affected, info = _apply_write_transaction(
        "UPDATE products SET price = 99.0 WHERE product_id = 1",
        prod_db, preview=True)
    assert affected == 1
    assert info["columns"] == ["product_id", "product_name", "price", "note"]
    assert info["before"] == [[1, "Cable", 10.0, None]]
    assert info["after"] == [[1, "Cable", 99.0, None]]
    assert not info["net_zero_change"] and not info["duplicate_existing"]
    # rollback discipline: the database is unchanged
    assert _price_of(prod_db) == 10.0


def test_update_preview_null_and_zero_change(prod_db):
    affected, info = _apply_write_transaction(
        "UPDATE products SET note = 'ui' WHERE product_id = 1",
        prod_db, preview=True)
    assert info["before"] == [[1, "Cable", 10.0, None]]
    assert info["after"] == [[1, "Cable", 10.0, "ui"]]

    affected, info = _apply_write_transaction(
        "UPDATE products SET price = 10.0 WHERE product_id = 1",
        prod_db, preview=True)
    assert affected == 1                      # SQLite counts the matched row
    assert info["before"] == [] and info["after"] == []
    assert info["net_zero_change"] is True


def test_insert_preview_and_duplicate_detection(prod_db):
    # duplicate = identical on NON-key columns (re-inserting the same entity
    # under a new/an explicit id is the classic user mistake)
    affected, info = _apply_write_transaction(
        "INSERT INTO products VALUES (4, 'Cable', 10.0, NULL)",
        prod_db, preview=True)
    assert affected == 1
    assert info["before"] == []
    assert info["after"] == [[4, "Cable", 10.0, None]]
    assert info["duplicate_existing"] is True

    affected2, info2 = _apply_write_transaction(
        "INSERT INTO products VALUES (5, 'Chair', 40.0, 'new')",
        prod_db, preview=True)
    assert info2["duplicate_existing"] is False
    con = sqlite3.connect(prod_db)              # nothing was applied
    assert con.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 3
    con.close()


def test_multi_row_update_and_rows_omitted(prod_db):
    con = sqlite3.connect(prod_db)
    for i in range(30):
        con.execute("INSERT INTO products VALUES (?, ?, ?, NULL)", (100 + i, f"b{i}", i))
    con.commit()
    con.close()
    affected, info = _apply_write_transaction(
        "UPDATE products SET note = 'x' WHERE note IS NULL",
        prod_db, preview=True)
    assert affected >= 30
    assert info["rows_omitted"] is True
    assert len(info["after"]) == 20             # hard cap
    assert _price_of(prod_db) == 10.0           # rollback left db untouched


def test_authorizer_blocks_foreign_table(prod_db):
    # write-amplification isolation: a trigger that fans out to another table
    # must be denied by the authorizer (writes may only reach `target`)
    con = sqlite3.connect(prod_db)
    con.execute("CREATE TABLE audit (a INTEGER)")
    con.execute("CREATE TRIGGER fanout AFTER INSERT ON products BEGIN "
                "UPDATE audit SET a = 1; END")
    con.commit()
    con.close()
    with pytest.raises(sqlite3.Error) as e:
        _apply_write_transaction("INSERT INTO products VALUES (9, 'z', 1, NULL)",
                                 prod_db, preview=True)
    assert "not authorized" in str(e.value).lower()
    # nothing applied: products unchanged, audit unchanged
    con = sqlite3.connect(prod_db)
    assert con.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 3
    assert con.execute("SELECT COUNT(*) FROM audit WHERE a = 1").fetchone()[0] == 0
    con.close()


def test_fingerprint_changes_after_commit(prod_db):
    fp_before = _db_fingerprint(prod_db)
    confirm_write("UPDATE products SET price = 77.0 WHERE product_id = 1",
                  "update the price for product id 1", db_path=prod_db)
    assert _price_of(prod_db) == 77.0
    assert _db_fingerprint(prod_db) != fp_before


def test_confirm_refuses_stale_fingerprint(prod_db):
    stale = _db_fingerprint(prod_db)
    # another process writes underneath the preview
    con = sqlite3.connect(prod_db, timeout=5)
    con.execute("UPDATE products SET price = price + 1")
    con.commit()
    con.close()
    fp_mid = _db_fingerprint(prod_db)
    if fp_mid == stale:                        # mtime resolution guard
        con = sqlite3.connect(prod_db)
        con.execute("INSERT INTO products VALUES (7, 'Pad', 5.0, None)")
        con.commit()
        con.close()
        time.sleep(0.2)
    with pytest.raises(InputError) as e:
        confirm_write("UPDATE products SET price = 99.0 WHERE product_id = 1",
                      "update the price for product id 1", db_path=prod_db,
                      expect_fingerprint=stale)
    assert "fresh preview" in str(e.value)
    assert _price_of(prod_db) != 99.0          # nothing was applied


def test_confirm_allows_fresh_fingerprint(prod_db):
    fp = _db_fingerprint(prod_db)
    confirm_write("UPDATE products SET price = 77.0 WHERE product_id = 1",
                  "update the price for product id 1", db_path=prod_db,
                  expect_fingerprint=fp)
    assert _price_of(prod_db) == 77.0


def test_confirm_without_fingerprint_still_works(prod_db):
    confirm_write("UPDATE products SET price = 66.0 WHERE product_id = 1",
                  "update the price for product id 1", db_path=prod_db)
    assert _price_of(prod_db) == 66.0


def test_preview_info_is_json_serializable(prod_db):
    _, info = _apply_write_transaction(
        "UPDATE products SET price = 12.0 WHERE product_id = 1",
        prod_db, preview=True)
    json.dumps(info)     # must never carry non-JSON types to the worker


@pytest.fixture()
def no_pk_db(tmp_path):
    """An imported CSV-style table: integer *_id column, no PK/autoincrement."""
    import sqlite3
    db = tmp_path / "no_pk.db"
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE "orders" ("order_id" INTEGER, "customer" TEXT)')
    conn.execute("INSERT INTO orders VALUES (1005, 'Pat')")
    conn.commit()
    conn.close()
    return str(db)


def test_insert_id_warnings_audits_imported_no_pk_tables(no_pk_db):
    # id omitted entirely → its column silently holds NULL
    (warn,) = pipeline.insert_id_warnings(
        "INSERT INTO orders (customer) VALUES ('Eve');", no_pk_db)
    assert "order_id" in warn and "NULL" in warn

    # id provided as an explicit literal NULL
    (warn,) = pipeline.insert_id_warnings(
        "INSERT INTO orders (order_id, customer) VALUES (NULL, 'Eve');",
        no_pk_db)
    assert "NULL" in warn

    # a fresh id clears the audit; a guessed duplicate is caught, and a
    # comma inside a quoted value must not shift the column mapping
    assert pipeline.insert_id_warnings(
        "INSERT INTO orders (order_id, customer) VALUES (1007, 'Eve');",
        no_pk_db) == []
    (warn,) = pipeline.insert_id_warnings(
        "INSERT INTO orders (order_id, customer) "
        "VALUES (1005, 'Doe, John');", no_pk_db)
    assert "1005 already exists" in warn

    # not an INSERT at all → additive audit returns nothing
    assert pipeline.insert_id_warnings(
        "UPDATE orders SET customer = 'X' WHERE order_id = 1", no_pk_db) == []


def test_explicit_write_gate_catches_texting_shorthand():
    from pipeline import explicitly_requests_write
    # historical live-bug: "can u add …" passed the gate, became a READ and
    # the model answered with the silent fallback SELECT 1 WHERE 0.
    assert explicitly_requests_write("can u add a new order and a new item with id")
    assert explicitly_requests_write("can you add a new order")
    assert explicitly_requests_write("add a new row to orders")
    # ability questions and "set up" phrasing must stay reads
    assert not explicitly_requests_write("can u set up a new workshop")
    assert not explicitly_requests_write("can u see the orders table?")
