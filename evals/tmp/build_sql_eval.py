"""Build evals/heldout/sql_eval.jsonl — execution-verified gold pairs for the
students.db domain (plus a JOIN slice against evals/tmp/join_demo.db).

Every gold SQL in this file is EXECUTED here against its target DB (read-only
URI connections).  Any error aborts the build, so a committed file means every
pair ran clean at authoring time.  expected_result_row_count comes from that
execution, not from guesses.

Run:  .venv/bin/python evals/tmp/build_sql_eval.py
"""

import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "evals" / "heldout" / "sql_eval.jsonl"

DBS = {
    "students": f"file:{ROOT / 'students.db'}?mode=ro",
    "join_demo": f"file:{HERE / 'join_demo.db'}?mode=ro",
}

# (question, gold_sql, category, db, rationale)
PAIRS = [
    # ---------- filters
    ("List the names of all students who are in the Electronics Engineering department.",
     "SELECT name FROM students WHERE department = 'Electronics Engineering';",
     "filter", "students", "equality on literal department"),
    ("Give the names of students whose average marks are above 90.",
     "SELECT name FROM students WHERE average_marks > 90;",
     "filter", "students", "strict > on REAL"),
    ("Show the names of students whose pending fees are between 1000 and 5000.",
     "SELECT name FROM students WHERE pending_fees BETWEEN 1000 AND 5000;",
     "filter", "students", "BETWEEN inclusive; all debtors, zero-fee rows excluded by range"),
    ("Which student names have attendance percentage less than 80?",
     "SELECT name FROM students WHERE attendance_percentage < 80;",
     "filter", "students", "strict < on REAL"),
    ("List the names of students in Computer Engineering or Electronics Engineering whose average marks are at least 90.",
     "SELECT name FROM students WHERE department IN ('Computer Engineering', 'Electronics Engineering') AND average_marks >= 90;",
     "filter", "students", "IN-list + >= boundary (Tanvi at exactly 90 must appear)"),
    ("Give the names of students who owe fees and whose attendance is below 90.",
     "SELECT name FROM students WHERE pending_fees > 0 AND attendance_percentage < 90;",
     "filter", "students", "conjunction: presence of debt + threshold"),
    ("Which students' names start with the letter A?",
     "SELECT name FROM students WHERE name LIKE 'A%';",
     "filter", "students", "LIKE prefix"),
    # ---------- aggregation
    ("What is the overall average attendance of the students?",
     "SELECT AVG(attendance_percentage) FROM students;",
     "aggregate", "students", "scalar AVG"),
    ("What is the total pending fees across all students?",
     "SELECT SUM(pending_fees) FROM students;",
     "aggregate", "students", "scalar SUM"),
    ("How many students have pending fees greater than zero?",
     "SELECT COUNT(*) FROM students WHERE pending_fees > 0;",
     "aggregate", "students", "COUNT with filter"),
    ("What is the highest average marks recorded?",
     "SELECT MAX(average_marks) FROM students;",
     "aggregate", "students", "scalar MAX"),
    ("What is the difference between the highest and lowest attendance percentages?",
     "SELECT MAX(attendance_percentage) - MIN(attendance_percentage) FROM students;",
     "aggregate", "students", "MAX-MIN in one scalar (naive keyword rules stumble here)"),
    ("How many students are named Meera?",
     "SELECT COUNT(*) FROM students WHERE name LIKE '%Meera%';",
     "aggregate", "students", "substring match count"),
    # ---------- GROUP BY / HAVING
    ("What is the average assignment score for each department?",
     "SELECT department, AVG(assignment_score) FROM students GROUP BY department;",
     "group_having", "students", "simple GROUP BY"),
    ("Which departments have a total pending fees above 10000?",
     "SELECT department FROM students GROUP BY department HAVING SUM(pending_fees) > 10000;",
     "group_having", "students", "HAVING SUM (CE 11500, IT 14000, EE 9200 fails)"),
    ("Which departments have an average attendance below 87?",
     "SELECT department FROM students GROUP BY department HAVING AVG(attendance_percentage) < 87;",
     "group_having", "students", "HAVING AVG (CE 85.2, IT 86.2 pass; EE 89.5 fails)"),
    ("For each department, how many students owe fees?",
     "SELECT department, COUNT(*) FROM students WHERE pending_fees > 0 GROUP BY department;",
     "group_having", "students", "WHERE before GROUP BY (filtered count)"),
    ("Which departments have more than 4 students?",
     "SELECT department FROM students GROUP BY department HAVING COUNT(*) > 4;",
     "group_having", "students", "HAVING COUNT"),
    ("Give the names and departments of students whose pending fees are above the average pending fees of their department.",
     "SELECT name, department FROM students s WHERE pending_fees > (SELECT AVG(pending_fees) FROM students w WHERE w.department = s.department);",
     "group_having", "students", "correlated subquery per-department average"),
    # ---------- ORDER BY / LIMIT
    ("Give the names of the three students with the highest attendance.",
     "SELECT name FROM students ORDER BY attendance_percentage DESC, name LIMIT 3;",
     "order_limit", "students", "ORDER+LIMIT with deterministic tiebreak (97/97 tie)"),
    ("Which student owes the most in pending fees? Give the name.",
     "SELECT name FROM students ORDER BY pending_fees DESC LIMIT 1;",
     "order_limit", "students", "top-1"),
    ("Give the names of the top 2 students by assignment score.",
     "SELECT name FROM students ORDER BY assignment_score DESC, name LIMIT 2;",
     "order_limit", "students", "top-2"),
    ("Give the names of the three students with the lowest average marks.",
     "SELECT name FROM students ORDER BY average_marks ASC, name LIMIT 3;",
     "order_limit", "students", "ASC ORDER+LIMIT"),
    ("Give the name of the student who has the fifth highest attendance percentage.",
     "SELECT name FROM students ORDER BY attendance_percentage DESC, name LIMIT 1 OFFSET 4;",
     "order_limit", "students", "LIMIT/OFFSET (naive keyword rules fail OFFSET)"),
    # ---------- JOINs (run against evals/tmp/join_demo.db)
    ("List each course name together with the number of students enrolled in it, including courses with no enrollments.",
     "SELECT c.course_name, COUNT(e.enrollment_id) FROM courses c LEFT JOIN enrollments e ON e.course_id = c.course_id GROUP BY c.course_id, c.course_name ORDER BY c.course_name;",
     "join", "join_demo", "LEFT JOIN + COUNT includes the zero-enrollment course"),
    ("Which course names had no enrolled students?",
     "SELECT course_name FROM courses WHERE course_id NOT IN (SELECT course_id FROM enrollments);",
     "join", "join_demo", "anti-join via NOT IN"),
    ("What are the names of students enrolled in more than one course?",
     "SELECT s.name FROM students s JOIN enrollments e ON e.student_id = s.student_id GROUP BY s.student_id, s.name HAVING COUNT(*) > 1;",
     "join", "join_demo", "JOIN + GROUP BY + HAVING"),
    ("How many students are enrolled in the Data Structures course?",
     "SELECT COUNT(*) FROM enrollments e JOIN courses c ON c.course_id = e.course_id WHERE c.course_name = 'Data Structures';",
     "join", "join_demo", "two-table join + COUNT"),
    ("What is the average enrollment marks of each course, including courses with no enrolled students?",
     "SELECT c.course_name, AVG(e.marks) FROM courses c LEFT JOIN enrollments e ON e.course_id = c.course_id GROUP BY c.course_id, c.course_name ORDER BY c.course_name;",
     "join", "join_demo", "LEFT JOIN AVG (NULL row must appear)"),
    ("List the student names with their course names for every enrollment.",
     "SELECT s.name, c.course_name FROM enrollments e JOIN students s ON s.student_id = e.student_id JOIN courses c ON c.course_id = e.course_id;",
     "join", "join_demo", "3-way join projection"),
]


def main():
    rows, problems = [], []
    for question, gold, cat, db, why in PAIRS:
        uri = DBS[db]
        con = sqlite3.connect(uri, uri=True)
        try:
            cur = con.execute(gold)
            n = len(cur.fetchall())
        except Exception as exc:
            problems.append(f"{question} -> {exc}")
            continue
        finally:
            con.close()
        rows.append({
            "question": question,
            "gold_sql": gold,
            "category": cat,
            "db": db,
            "expected_result_row_count": n,
        })
        print(f"[OK {db:9s} {cat:12s} rows={n:3d}] {question[:60]}")

    if problems:
        print("\nBUILD FAILED — gold pairs that did not run clean:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"\nwrote {OUT} ({len(rows)} pairs); categories:",
          dict(Counter(r["category"] for r in rows)),
          "dbs:", dict(Counter(r["db"] for r in rows)))


if __name__ == "__main__":
    main()
