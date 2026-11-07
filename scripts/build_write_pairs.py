"""Build explicit, schema-grounded INSERT/UPDATE continuation examples.

The examples teach syntax and phrasing only. They are never executed against
the demo database. Runtime authorization and confirmation remain app-side.
"""

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DBS = [ROOT / "data" / "people.db", ROOT / "data" / "academics.db"]
OUT = ROOT / "data" / "write_pairs.jsonl"


def schemas():
    blocks = []
    for db_path in DBS:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            for (table,) in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"):
                safe = table.replace('"', '""')
                cols = conn.execute(f'PRAGMA table_info("{safe}")').fetchall()
                decl = []
                for col in cols:
                    bits = [f'"{col[1]}" {col[2] or "TEXT"}']
                    if col[5]:
                        bits.append("PRIMARY KEY")
                    elif col[3]:
                        bits.append("NOT NULL")
                    decl.append("  " + " ".join(bits))
                blocks.append(f'CREATE TABLE "{table}" (\n' + ",\n".join(decl) + "\n);")
        finally:
            conn.close()
    return "\n\n".join(blocks)


def main():
    schema = schemas()
    records = []

    def add(mode, sql, prompts):
        for question in prompts:
            variants = [question,
                        f"Please explicitly carry this out: {question}",
                        f"Perform this requested database change: {question}",
                        f"Write SQL to make this change: {question}"]
            for variant in variants:
                records.append({"mode": mode, "schema": schema,
                                "question": variant, "sql": sql,
                                "source": "clarion_explicit_write"})

    add("WRITE", "INSERT INTO students (student_id, name, department, year_of_study) VALUES (9001, 'Aarav Mehta', 'Computer Engineering', 2);", [
        "Explicitly insert a new student named Aarav Mehta in Computer Engineering, year 2, with student ID 9001.",
        "Add a student row: ID 9001, name Aarav Mehta, department Computer Engineering, year of study 2.",
        "Please insert Aarav Mehta into students with student_id 9001, department Computer Engineering, and year_of_study 2.",
    ])
    add("WRITE", "UPDATE students SET department = 'Information Technology' WHERE student_id = 12;", [
        "Explicitly update student 12's department to Information Technology.",
        "Change the department for the student whose ID is 12 to Information Technology.",
        "Set students.department to Information Technology for student_id 12.",
    ])
    add("WRITE", "UPDATE students SET year_of_study = 3 WHERE student_id = 21;", [
        "Update student 21 so their year of study is 3.",
        "Change year_of_study to 3 for the row with student_id 21.",
        "Set the study year to 3 for student ID 21.",
    ])
    add("WRITE", "INSERT INTO departments (department, head_name, total_seats) VALUES ('Data Science', 'Nisha Rao', 120);", [
        "Insert a department named Data Science, headed by Nisha Rao, with 120 total seats.",
        "Add a new departments row for Data Science with head_name Nisha Rao and total_seats 120.",
        "Please create a department record: Data Science, head Nisha Rao, 120 seats.",
    ])
    add("WRITE", "UPDATE departments SET total_seats = 150 WHERE department = 'Computer Engineering';", [
        "Update Computer Engineering to have 150 total seats.",
        "Set total_seats to 150 for the Computer Engineering department.",
        "Change the seat count for Computer Engineering to 150.",
    ])
    add("WRITE", "INSERT INTO courses (course_id, course_name, credits, difficulty_score) VALUES (9001, 'Applied NLP', 4, 3.5);", [
        "Insert a course called Applied NLP with course ID 9001, 4 credits, and difficulty score 3.5.",
        "Add a courses row: course_id 9001, course_name Applied NLP, credits 4, difficulty_score 3.5.",
        "Create a course record for Applied NLP, ID 9001, worth 4 credits, with difficulty 3.5.",
    ])
    add("WRITE", "UPDATE courses SET difficulty_score = 2.5 WHERE course_id = 4;", [
        "Update course 4's difficulty score to 2.5.",
        "Set difficulty_score to 2.5 for course_id 4.",
        "Change the difficulty of the course with ID 4 to 2.5.",
    ])
    add("WRITE", "INSERT INTO enrollments (enrollment_id, student_id, course_id, grade, fees_due) VALUES (9001, 12, 4, 88.5, 250);", [
        "Insert enrollment 9001 for student 12 in course 4 with grade 88.5 and fees due 250.",
        "Add an enrollment row with enrollment_id 9001, student_id 12, course_id 4, grade 88.5, fees_due 250.",
        "Create an enrollment for student 12 and course 4: ID 9001, grade 88.5, fees due 250.",
    ])
    add("WRITE", "UPDATE enrollments SET fees_due = 0 WHERE enrollment_id = 15;", [
        "Update enrollment 15 so its fees_due is zero.",
        "Set fees_due to 0 for the enrollment with ID 15.",
        "Change the outstanding fees on enrollment_id 15 to 0.",
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8")
    print(f"wrote {len(records)} explicit WRITE examples to {OUT}")


if __name__ == "__main__":
    main()
