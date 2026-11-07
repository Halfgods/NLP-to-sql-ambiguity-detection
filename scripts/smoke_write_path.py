"""Exercise the fine-tuned write path against a disposable CSV-backed DB."""

import csv
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_runtime
from UI.server import model_worker
from pipeline import validate_write_sql


MODEL = ROOT / "models" / "qwen25-coder-1.5b-sql-write-q4_k_m.gguf"


def main():
    with tempfile.TemporaryDirectory(prefix="clarion-write-smoke-") as temp:
        temp = Path(temp)
        csv_path = temp / "uploaded_people.csv"
        csv_path.write_text("id,name,city\n1,Ada Lovelace,London\n", encoding="utf-8")
        db_path = temp / "uploaded-workspace.db"

        # Mirror the CSV-to-SQLite table shape used by workspace.importTable.
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE uploaded_people (id INTEGER PRIMARY KEY, name TEXT, city TEXT)")
        conn.executemany("INSERT INTO uploaded_people VALUES (?, ?, ?)",
                         [(int(r["id"]), r["name"], r["city"]) for r in rows])
        conn.execute("CREATE TABLE courses (course_name TEXT, credits INTEGER, difficulty_score REAL)")
        conn.execute("INSERT INTO courses VALUES ('Algorithms', 4, 8.5)")
        conn.commit()
        conn.close()

        assert not model_worker.explicitly_requests_write("How many people live in London?")
        assert model_worker.explicitly_requests_write(
            "Insert a new row into uploaded_people with id 2, name Grace Hopper, city New York.")
        rejected, why = validate_write_sql(
            "UPDATE courses SET credits = 1 WHERE course_name = 'Algorithms'",
            "change the credits from 4 to 1 and 8.5 to 10", db_path)
        assert rejected is None and "grounded" in why.lower(), why

        model_runtime.get_runtime(MODEL)
        try:
            first = model_worker._query({
                "message": "Insert a new row into uploaded_people with id 2, name Grace Hopper, city New York and then show the list of people.",
                "dbPath": str(db_path),
            })
            assert first["status"] == "confirm_write", first
            assert first["affected_rows"] == 1
            assert first["sql"].strip().upper().startswith("INSERT INTO")
            conn = sqlite3.connect(db_path)
            before_insert = conn.execute("SELECT COUNT(*) FROM uploaded_people").fetchone()[0]
            conn.close()
            assert before_insert == 1, "preview unexpectedly changed the database"
            blocked = model_worker._query({
                "message": "show me the people", "dbPath": str(db_path)})
            assert blocked["status"] == "write_pending", blocked
            applied = model_worker._confirm_write({"token": first["write_token"]})
            assert len(applied["results"]["rows"]) == 2, applied

            conn = sqlite3.connect(db_path)
            inserted = conn.execute(
                "SELECT name, city FROM uploaded_people WHERE id = 2").fetchone()
            conn.close()
            assert inserted == ("Grace Hopper", "New York"), (first["sql"], inserted)

            second = model_worker._query({
                "message": "Update uploaded_people row with id 2 and set city to Cambridge.",
                "dbPath": str(db_path),
            })
            assert second["status"] == "confirm_write", second
            conn = sqlite3.connect(db_path)
            before_update = conn.execute(
                "SELECT city FROM uploaded_people WHERE id = 2").fetchone()[0]
            conn.close()
            assert before_update == "New York", "UPDATE preview unexpectedly committed"
            cancelled = model_worker._cancel_write({"token": second["write_token"]})
            assert cancelled["status"] == "write_cancelled", cancelled
            conn = sqlite3.connect(db_path)
            after_cancel = conn.execute(
                "SELECT city FROM uploaded_people WHERE id = 2").fetchone()[0]
            conn.close()
            assert after_cancel == "New York", "cancel unexpectedly changed the database"

            second = model_worker._query({
                "message": "Update uploaded_people row with id 2 and set city to Cambridge.",
                "dbPath": str(db_path),
            })
            assert second["status"] == "confirm_write", second
            model_worker._confirm_write({"token": second["write_token"]})

            conn = sqlite3.connect(db_path)
            updated = conn.execute(
                "SELECT city FROM uploaded_people WHERE id = 2").fetchone()[0]
            conn.close()
            assert updated == "Cambridge", (second["sql"], updated)
            print("PASS: ungrounded UPDATE rejected; new queries blocked while a write is pending; composite INSERT+SELECT, explicit UPDATE, and cancel all behaved correctly on the disposable SQLite DB.")
            print("INSERT SQL:", first["sql"])
            print("UPDATE SQL:", second["sql"])
        finally:
            model_runtime.shutdown_runtime()


if __name__ == "__main__":
    main()
