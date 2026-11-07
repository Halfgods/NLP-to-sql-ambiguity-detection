#!/usr/bin/env python3
"""Build the two interconnected demo SQLite databases.

data/people.db    students(student_id, name, department, year_of_study)
data/academics.db courses(course_id, course_name, credits, difficulty_score)
                  enrollments(enrollment_id, student_id -> people.students,
                              course_id -> courses.course_id, grade, fees_due)

The bridge is `enrollments`: its student_id matches the OTHER file's
students.student_id. SQLite cannot enforce a foreign key across database
files, so the relationship is modeled as a shared key and joined with ATTACH.
`students.db` is intentionally untouched — the held-out SQL eval suite runs
against it verbatim; this only replaces it as the UI demo default.
"""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PEOPLE = ROOT / "data" / "people.db"
ACADEMICS = ROOT / "data" / "academics.db"

STUDENTS = [
    (1, "Aarav Shah", "Computer Engineering", 2),
    (2, "Meera Patil", "Information Technology", 3),
    (3, "Rohan Desai", "Computer Engineering", 1),
    (4, "Ananya Rao", "Electronics Engineering", 4),
    (5, "Kabir Mehta", "Information Technology", 2),
    (6, "Isha Kulkarni", "Computer Engineering", 3),
    (7, "Vihaan Joshi", "Electronics Engineering", 1),
    (8, "Sara Fernandes", "Information Technology", 3),
    (9, "Arjun Nair", "Computer Engineering", 4),
    (10, "Diya Kapoor", "Electronics Engineering", 2),
    (11, "Vivaan Sharma", "Information Technology", 1),
    (12, "Tanvi Iyer", "Electronics Engineering", 3),
    (13, "Aditya Bose", "Computer Engineering", 2),
    (14, "Nisha Verma", "Information Technology", 4),
]

COURSES = [
    (1, "Algorithms", 4, 9.2),
    (2, "Data Structures", 3, 8.5),
    (3, "Machine Learning", 4, 9.0),
    (4, "Databases", 3, 7.4),
    (5, "Compilers", 4, 8.7),
]

# enrollment_id, student_id, course_id, grade (0-100), fees_due (rupees)
ENROLLMENTS = [
    (1, 1, 1, 93, 0), (2, 1, 4, 88, 0), (3, 1, 3, 90, 0),
    (4, 2, 1, 85, 2500), (5, 2, 5, 83, 2500), (6, 2, 3, 91, 2500),
    (7, 3, 2, 89, 0), (8, 3, 5, 86, 0), (9, 3, 4, 92, 0),
    (10, 4, 2, 87, 5000), (11, 4, 3, 95, 5000), (12, 4, 5, 90, 5000),
    (13, 5, 1, 80, 1500), (14, 5, 3, 78, 1500), (15, 6, 2, 96, 0),
    (16, 6, 5, 94, 0), (17, 6, 3, 97, 0), (18, 7, 1, 84, 3000),
    (19, 7, 4, 82, 3000), (20, 8, 2, 90, 0), (21, 8, 5, 88, 0),
    (22, 9, 1, 75, 7500), (23, 9, 4, 73, 7500), (24, 10, 3, 91, 0),
    (25, 10, 2, 89, 0), (26, 10, 5, 93, 0), (27, 11, 1, 66, 10000),
    (28, 11, 4, 62, 10000), (29, 12, 2, 92, 1200), (30, 12, 3, 90, 1200),
    (31, 13, 4, 85, 4000), (32, 13, 1, 82, 4000), (33, 13, 5, 87, 4000),
    (34, 14, 3, 97, 0), (35, 14, 2, 95, 0), (36, 14, 5, 93, 0),
]


def build_people():
    conn = sqlite3.connect(PEOPLE)
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS students;
            DROP TABLE IF EXISTS departments;
            CREATE TABLE students (
                student_id    INTEGER PRIMARY KEY,
                name          TEXT NOT NULL,
                department    TEXT NOT NULL,
                year_of_study INTEGER NOT NULL
            );
            CREATE TABLE departments (
                department   TEXT PRIMARY KEY,
                head_name    TEXT NOT NULL,
                total_seats  INTEGER NOT NULL
            );
            INSERT INTO departments VALUES
                ('Computer Engineering', 'Prof. R. Menon', 120),
                ('Information Technology', 'Prof. S. Rao', 90),
                ('Electronics Engineering', 'Prof. A. Bose', 105);
            """)
        conn.executemany("INSERT INTO students VALUES (?, ?, ?, ?)", STUDENTS)
        conn.commit()
    finally:
        conn.close()


def build_academics():
    conn = sqlite3.connect(ACADEMICS)
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS enrollments;
            DROP TABLE IF EXISTS courses;
            CREATE TABLE courses (
                course_id        INTEGER PRIMARY KEY,
                course_name      TEXT NOT NULL,
                credits          INTEGER NOT NULL,
                difficulty_score REAL NOT NULL
            );
            CREATE TABLE enrollments (
                enrollment_id INTEGER PRIMARY KEY,
                student_id    INTEGER NOT NULL,
                course_id     INTEGER NOT NULL,
                grade         REAL NOT NULL,
                fees_due      REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (course_id)  REFERENCES courses(course_id)
            );
            """)
        conn.executemany("INSERT INTO courses VALUES (?, ?, ?, ?)", COURSES)
        conn.executemany(
            "INSERT INTO enrollments VALUES (?, ?, ?, ?, ?)", ENROLLMENTS)
        conn.commit()
    finally:
        conn.close()


def main():
    build_people()
    build_academics()
    # interconnection smoke-test: bridge join across the two files,
    # verified here once with ATTACH so the shipped claim is measured.
    conn = sqlite3.connect("file:" + str(ACADEMICS) + "?mode=ro", uri=True)
    try:
        conn.execute("ATTACH DATABASE 'file:" + str(PEOPLE) +
                     "?mode=ro' AS people")
        rows = conn.execute(
            "SELECT c.course_name, COUNT(*) FROM enrollments e "
            "JOIN people.students s USING (student_id) "
            "JOIN courses c USING (course_id) GROUP BY 1 ORDER BY 1").fetchall()
        print("cross-file join OK —", rows)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
