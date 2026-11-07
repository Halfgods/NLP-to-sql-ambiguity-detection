"""Build evals/tmp/join_demo.db — a deterministic, upsert-free, 2-table join
sandbox for the JOIN slice of heldout/sql_eval.jsonl.

The real students.db has ONE table (students), so honest JOIN questions need a
dedicated sandbox rather than pretending courses lives in students.db.  This DB
contains the REAL students rows (copied verbatim) plus courses and enrollments
defined here.  Re-running this script recreates the file from scratch.

Run:  .venv/bin/python evals/tmp/make_join_demo_db.py
"""

import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE / "join_demo.db"

DDL = """
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    average_marks REAL NOT NULL,
    attendance_percentage REAL NOT NULL,
    assignment_score REAL NOT NULL,
    pending_fees REAL NOT NULL
);

CREATE TABLE courses (
    course_id INTEGER PRIMARY KEY,
    course_name TEXT NOT NULL,
    department TEXT NOT NULL,
    credits INTEGER NOT NULL,
    difficulty_score REAL NOT NULL
);

CREATE TABLE enrollments (
    enrollment_id INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(student_id),
    course_id INTEGER NOT NULL REFERENCES courses(course_id),
    marks REAL NOT NULL
);
"""

STUDENTS = [
    (1, 'Aarav Shah', 'Computer Engineering', 91.5, 88.0, 94.0, 0.0),
    (2, 'Meera Patil', 'Information Technology', 87.0, 96.0, 89.0, 2500.0),
    (3, 'Rohan Desai', 'Computer Engineering', 94.0, 91.0, 85.0, 0.0),
    (4, 'Ananya Rao', 'Electronics Engineering', 89.5, 84.0, 98.0, 5000.0),
    (5, 'Kabir Mehta', 'Information Technology', 82.0, 93.0, 92.0, 1500.0),
    (6, 'Isha Kulkarni', 'Computer Engineering', 96.0, 97.0, 95.0, 0.0),
    (7, 'Vihaan Joshi', 'Electronics Engineering', 85.5, 90.0, 87.0, 3000.0),
    (8, 'Sara Fernandes', 'Information Technology', 92.0, 86.0, 96.0, 0.0),
    (9, 'Arjun Nair', 'Computer Engineering', 78.0, 72.0, 68.0, 7500.0),
    (10, 'Diya Kapoor', 'Electronics Engineering', 93.5, 89.0, 91.0, 0.0),
    (11, 'Vivaan Sharma', 'Information Technology', 68.5, 64.0, 59.0, 10000.0),
    (12, 'Tanvi Iyer', 'Electronics Engineering', 90.0, 95.0, 88.0, 1200.0),
    (13, 'Aditya Bose', 'Computer Engineering', 84.0, 78.0, 80.0, 4000.0),
    (14, 'Nisha Verma', 'Information Technology', 95.0, 92.0, 97.0, 0.0),
]

COURSES = [
    (1, 'Data Structures', 'Computer Engineering', 4, 3.2),
    (2, 'Microprocessors', 'Computer Engineering', 3, 3.8),
    (3, 'Web Technologies', 'Information Technology', 2, 2.1),
    (4, 'Image Processing', 'Electronics Engineering', 3, 3.5),
    (5, 'Machine Learning', 'Computer Engineering', 3, 3.0),   # zero enrollments
]

ENROLLMENTS = [  # (enrollment_id, student_id, course_id, marks)
    (1, 1, 1, 88.0),   # Aarav       - Data Structures
    (2, 3, 1, 81.5),   # Rohan       - Data Structures
    (3, 6, 1, 93.0),   # Isha        - Data Structures
    (4, 6, 5, 91.5),   # Isha        - Machine Learning
    (5, 9, 1, 61.0),   # Arjun       - Data Structures
    (6, 2, 3, 84.0),   # Meera       - Web Technologies
    (7, 8, 3, 90.5),   # Sara        - Web Technologies
    (8, 11, 3, 55.0),  # Vivaan      - Web Technologies
    (9, 4, 4, 95.0),   # Ananya      - Image Processing
    (10, 7, 4, 83.0),  # Vihaan      - Image Processing
    (11, 12, 4, 87.5), # Tanvi       - Image Processing
    (12, 3, 2, 77.0),  # Rohan       - Microprocessors
    (13, 1, 2, 85.0),  # Aarav       - Microprocessors
]


def main():
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript(DDL)
    con.executemany(
        "INSERT INTO students (student_id,name,department,average_marks,"
        "attendance_percentage,assignment_score,pending_fees) "
        "VALUES (?,?,?,?,?,?,?)", STUDENTS)
    con.executemany(
        "INSERT INTO courses (course_id,course_name,department,credits,"
        "difficulty_score) VALUES (?,?,?,?,?)", COURSES)
    con.executemany(
        "INSERT INTO enrollments (enrollment_id,student_id,course_id,marks) "
        "VALUES (?,?,?,?)", ENROLLMENTS)
    con.commit()
    counts = {
        t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("students", "courses", "enrollments")}
    con.close()
    print(f"built {DB} with {counts}")


if __name__ == "__main__":
    main()
