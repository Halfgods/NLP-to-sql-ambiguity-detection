"""Tests for the db_introspect schema cache (P1: retrieval + caching).

Contract under test:
*   same file signature -> same dict object identity is NEVER exposed
    (extract_schema returns a fresh deep copy each call);
*   writing to the database (mtime/size change) invalidates the cache;
*   deleting the file bypasses the cache (raises sqlite3-ish error, not a
    cached lookup);
*   cache_clear() drops all entries; cache_stats() reports the version.
"""

import sqlite3
import time

import pytest

import db_introspect as di


@pytest.fixture
def sample_db(tmp_path):
    db = tmp_path / "cache_probe.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE items (item_id INTEGER PRIMARY KEY, "
                "item_name TEXT NOT NULL, price REAL)")
    con.execute("INSERT INTO items VALUES (1, 'Alpha', 10.5)")
    con.commit()
    con.close()
    return db


def test_cache_returns_equal_but_not_identical(sample_db):
    di.cache_clear()
    a = di.extract_schema(sample_db)
    b = di.extract_schema(sample_db)
    assert a == b
    assert a is not b
    assert a["tables"]["items"] is not b["tables"]["items"]


def test_cache_isolation_from_caller_mutations(sample_db):
    di.cache_clear()
    a = di.extract_schema(sample_db)
    a["tables"]["items"]["columns"].clear()   # caller mutates its copy
    b = di.extract_schema(sample_db)
    assert b["tables"]["items"]["columns"], "mutation leaked into the cache"


def test_write_invalidates_cache(sample_db):
    di.cache_clear()
    before = di.extract_schema(sample_db)
    assert before["tables"]["items"]["sample_values"]["price"] == ["10.5"]

    con = sqlite3.connect(sample_db)
    con.execute("INSERT INTO items VALUES (2, 'Beta', 99.0)")
    con.commit()
    con.close()

    after = di.extract_schema(sample_db)
    assert after["tables"]["items"]["sample_values"]["price"] == ["10.5", "99.0"]


def test_missing_file_bypasses_cache(tmp_path):
    di.cache_clear()
    ghost = tmp_path / "gone.db"
    with pytest.raises((sqlite3.OperationalError, FileNotFoundError)):
        di.extract_schema(ghost)


def test_cache_clear_and_stats():
    di.cache_clear()
    assert di.cache_stats()["entries"] == 0
    assert di.cache_stats()["version"] == di._CACHE_VERSION
