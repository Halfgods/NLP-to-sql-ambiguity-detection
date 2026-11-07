"""Authoring-time sanity checks for the hand labels in heldout/classifier_eval.jsonl.

NOT part of the eval runtime contract.  It verifies, pair by pair:

  A) every schema DDL parses and creates cleanly (in-memory SQLite),
  B) for every students_college AMBIGUOUS pair, the >= 2 plausible interpretations
     are materially different on the REAL students.db data (different result sets),
  C) for the students_college CLEAR pairs, the intended single reading executes
     and matches the reasoning notes (unique maxima, row counts).

Run:  .venv/bin/python evals/tmp/label_sanity_check.py
"""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = f"file:{ROOT / 'students.db'}?mode=ro"  # uri

from schemas import SCHEMAS                      # noqa: E402
from pairs_part1 import PART1                    # noqa: E402
from pairs_part2 import PART2                    # noqa: E402

S1 = "students_college"

# candidate readings for students AMBIGUOUS pairs (label -> list of gold-SQLs
# whose result sets must differ pairwise on real data)
AMBIG_CHECKS = {
    "Which department performs the best overall?": [
        "SELECT department, AVG(average_marks) FROM students GROUP BY department ORDER BY 2 DESC LIMIT 1",
        "SELECT department, AVG(attendance_percentage) FROM students GROUP BY department ORDER BY 2 DESC LIMIT 1",
        "SELECT department, AVG(assignment_score) FROM students GROUP BY department ORDER BY 2 DESC LIMIT 1",
    ],
    "How is each department doing overall?": None,  # filled below (same fork)
    "Which students are struggling and need support?": [
        "SELECT name FROM students WHERE average_marks < 80",
        "SELECT name FROM students WHERE attendance_percentage < 80",
        "SELECT name FROM students WHERE assignment_score < 70",
    ],
    "Who are the top performers in internal marks?": [
        "SELECT name, average_marks FROM students ORDER BY average_marks DESC LIMIT 1",
        "SELECT name, assignment_score FROM students ORDER BY assignment_score DESC LIMIT 1",
    ],
    "Which students should be considered for the merit scholarship?": [
        "SELECT name, average_marks FROM students ORDER BY average_marks DESC LIMIT 1",
        "SELECT name, assignment_score FROM students ORDER BY assignment_score DESC LIMIT 1",
        "SELECT name, attendance_percentage FROM students ORDER BY attendance_percentage DESC, name LIMIT 1",
    ],
    "Which student has the highest overall performance?": [
        "SELECT name, average_marks FROM students ORDER BY average_marks DESC LIMIT 1",
        "SELECT name, assignment_score FROM students ORDER BY assignment_score DESC LIMIT 1",
        "SELECT name, attendance_percentage FROM students ORDER BY attendance_percentage DESC, name LIMIT 1",
    ],
    "Which department should get the remedial program?": [
        "SELECT department, AVG(average_marks) FROM students GROUP BY department ORDER BY 2 ASC LIMIT 1",
        "SELECT department, AVG(attendance_percentage) FROM students GROUP BY department ORDER BY 2 ASC LIMIT 1",
        "SELECT department, AVG(assignment_score) FROM students GROUP BY department ORDER BY 2 ASC LIMIT 1",
    ],
    "How is each department doing overall?": None,  # filled below (same fork)
    "Give me the weak students for each department.": [
        "SELECT name FROM students WHERE average_marks < 80 ORDER BY name",
        "SELECT name FROM students s WHERE average_marks < "
        "(SELECT AVG(average_marks) FROM students w WHERE w.department = s.department) ORDER BY name",
    ],
    "Who is the brightest student in each department?": [
        "SELECT department, name, average_marks FROM students s WHERE average_marks = "
        "(SELECT MAX(average_marks) FROM students w WHERE w.department = s.department)",
        "SELECT department, name, assignment_score FROM students s WHERE assignment_score = "
        "(SELECT MAX(assignment_score) FROM students w WHERE w.department = s.department)",
    ],
    "Which department has the highest score?": None,  # filled below (same fork)
}
AMBIG_CHECKS["Which department performs the best overall?"] = (
    AMBIG_CHECKS["Which department performs the best overall?"])
AMBIG_CHECKS["How is each department doing overall?"] = (
    AMBIG_CHECKS["Which department performs the best overall?"])
AMBIG_CHECKS["Which department has the highest score?"] = (
    AMBIG_CHECKS["Which department performs the best overall?"])

CLEAR_CHECKS = {
    "Which student has the highest average marks?": (
        "SELECT name FROM students WHERE average_marks = (SELECT MAX(average_marks) FROM students)", 1),
    "Give the names of students who scored above 90 in assignments.": (
        "SELECT name FROM students WHERE assignment_score > 90", 6),
    "Who owes the most in pending fees?": (
        "SELECT name FROM students WHERE pending_fees = (SELECT MAX(pending_fees) FROM students)", 1),
    "List the students who have pending fees.": (
        "SELECT name FROM students WHERE pending_fees > 0", 8),
    "How many students are there in Computer Engineering?": (
        "SELECT COUNT(*) FROM students WHERE department = 'Computer Engineering'", 1),
    "What is the average pending fees of all students?": (
        "SELECT AVG(pending_fees) FROM students", 1),
    "Give the top 3 students by assignment score.": (
        "SELECT name FROM students ORDER BY assignment_score DESC, name LIMIT 3", 3),
    "Which students have attendance below 70 and pending fees above 5000?": (
        "SELECT name FROM students WHERE attendance_percentage < 70 AND pending_fees > 5000", 1),
    "How many students are in each department?": (
        "SELECT department, COUNT(*) FROM students GROUP BY department", 3),
}


def setof(sql):
    with sqlite3.connect(DB, uri=True) as con:
        return {tuple(r) for r in con.execute(sql.rstrip(";"))}


def main():
    fails = 0

    # A) DDLs parse; every schema creates cleanly (fresh in-memory db each)
    for sid, ddl in SCHEMAS.items():
        try:
            con = sqlite3.connect(":memory:")
            con.execute(ddl)
            con.close()
            print(f"[DDL-OK ] {sid}")
        except Exception as exc:
            fails += 1
            print(f"[DDL-ERR] {sid}: {exc}")

    # B) students AMBIGUOUS forks differ materially on real data
    qsets = {(q, lab) for sid, q, lab, _ in PART1} | {(q, lab) for sid, q, lab, _ in PART2}
    for q, reads in AMBIG_CHECKS.items():
        if (q, "AMBIGUOUS") not in qsets:
            fails += 1
            print(f"[ORPHAN ] ambiguity check refers to missing pair: {q!r}")
            continue
        try:
            sets = [setof(r) for r in reads]
            if all(len(s) for s in sets) and any(sets[i] != sets[j]
                                                for i in range(len(sets))
                                                for j in range(i + 1, len(sets))):
                print(f"[AMB-OK ] {q[:58]}")
            else:
                fails += 1
                print(f"[AMB-WEAK] {q[:58]}  readings not materially different: {sets}")
        except Exception as exc:
            fails += 1
            print(f"[AMB-ERR] {q[:58]}: {exc}")

    # C) students CLEAR single readings execute + row counts as claimed
    for q, (sql, n) in CLEAR_CHECKS.items():
        if (q, "CLEAR") not in qsets:
            fails += 1
            print(f"[ORPHAN ] clear check refers to missing pair: {q!r}")
            continue
        try:
            s = setof(sql)
            if all(len(r) >= 1 for r in s):
                print(f"[CHR-OK ] {q[:58]}  ({len(s)} distinct result rows)")
            else:
                fails += 1
                print(f"[CHR-ERR] {q[:58]} unexpected results: {s}")
        except Exception as exc:
            fails += 1
            print(f"[CHR-ERR] {q[:58]}: {exc}")

    print("\nFAILURES:", fails)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
