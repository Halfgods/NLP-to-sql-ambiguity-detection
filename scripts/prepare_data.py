"""Build the 5k SQL teacher pairs + held-out evals for QLoRA fine-tuning.

Sources
-------
1. b-mc2/sql-create-context   (78,577 pairs: CREATE TABLE context + question -> SELECT)
   -> 4,200 train + 40 held-out eval            (source: "sql_create_context")
2. gretelai/synthetic_text_to_sql (teacher/GPT-4 pairs)
   -> 300 train after filtering out PostgreSQL-only syntax  (source: "gretel")
3. Domain pairs generated over the project's real schema (students, courses)
   -> 500 train                                 (source: "clarion_domain")
4. Hand-written domain eval: 30 gold question/SQL pairs -> eval_domain.jsonl

Output
------
data/train_pairs.jsonl    5,000 rows  {"schema", "question", "sql", "source"}
data/eval_public.jsonl      40 rows   held-out sql-create-context
data/eval_domain.jsonl      30 rows   hand-written gold for students.db
"""

import json
import random
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
DB = ROOT / "students.db"

SEED = 42
N_CC_TRAIN, N_CC_EVAL = 4200, 40
N_GRETEL = 300
N_DOMAIN = 500

rng = random.Random(SEED)

# PostgreSQL/dialect-only constructs we never want to teach (runtime is SQLite)
PG_ONLY = re.compile(
    r"(::|\bILIKE\b|\bSERIAL\b|AT TIME ZONE|WITHIN GROUP|\bGENERATED\b|\bRETURNING\b|"
    r"\bFETCH FIRST\b|\bOVER\(\)|\|\|)", re.IGNORECASE)


def ok_context(ctx: str) -> bool:
    if not ctx or len(ctx) > 1600:
        return False
    # keep schemas small: few tables, few lines
    return ctx.count("CREATE TABLE") <= 4 and len(ctx.splitlines()) <= 45


def ok_question(q: str) -> bool:
    return 8 <= len(q) <= 300


def ok_sql(sql: str) -> bool:
    if not sql or len(sql) > 700:
        return False
    if PG_ONLY.search(sql):
        return False
    # single statement, read-only
    return sql.strip().rstrip(";").count(";") == 0 and sql.strip().upper().startswith(
        ("SELECT", "WITH"))


def load_public():
    cc = pd.read_parquet(
        next(RAW.glob("sql-create-context/**/*.parquet")))[["question", "context", "answer"]]
    cc = [(str(q).strip(), str(c).strip(), str(a).strip())
          for q, c, a in zip(cc.question, cc.context, cc.answer)]
    cc = [(q, c, a) for q, c, a in cc if ok_question(q) and ok_context(c) and ok_sql(a)]
    rng.shuffle(cc)
    # dedupe identical questions
    seen, uniq = set(), []
    for q, c, a in cc:
        if q.lower() not in seen:
            seen.add(q.lower())
            uniq.append((q, c, a))
    rng.shuffle(uniq)
    eval_pub = [{"schema": c, "question": q, "sql": a, "source": "sql_create_context"}
                for q, c, a in uniq[:N_CC_EVAL]]
    train_cc = [{"schema": c, "question": q, "sql": a, "source": "sql_create_context"}
                for q, c, a in uniq[N_CC_EVAL:N_CC_EVAL + N_CC_TRAIN]]
    gretel = pd.read_parquet(RAW / "gretel" / "synthetic_text_to_sql_train.snappy.parquet")
    rows = [(str(r.sql_prompt).strip(), str(r.sql_context).strip(), str(r.sql).strip())
            for r in gretel.itertuples()
            if str(r.sql_task_type) in ("analytics and reporting", "data retrieval")
            and "INSERT" not in str(r.sql_context).upper()]
    rows = [(q, c, a) for q, c, a in rows if ok_question(q) and ok_context(c) and ok_sql(a)]
    rng.shuffle(rows)
    train_g = [{"schema": c, "question": q, "sql": a, "source": "gretel"}
               for q, c, a in rows[:N_GRETEL]]
    return train_cc, train_g, eval_pub


# ---------------------------------------------------------------- domain pairs

DEPTS = ["Computer Engineering", "Information Technology", "Electronics Engineering"]
NUM_COL = {
    "average_marks": "average marks", "attendance_percentage": "attendance percentage",
    "assignment_score": "assignment score", "pending_fees": "pending fees"}
PREFIX = ["", "Please ", "Can you ", "For the student database, "]
SUFFIX = ["", "?", " please.", ". Thank you."]


def domain_pairs(n=N_DOMAIN):
    """Templates over the real students/courses schema, with value jitter."""
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    names = [r[0] for r in conn.execute("SELECT name FROM students")]
    surnames = sorted({nm.split()[-1] for nm in names})
    conn.close()
    out, reuse = [], set()

    def add(questions, sql):
        for probe in questions:
            q = rng.choice(PREFIX) + probe + rng.choice(SUFFIX)
            key = (q.lower(), sql)
            if key not in reuse:
                reuse.add(key)
                out.append({"schema": DOMAIN_SCHEMA, "question": q.strip(), "sql": sql,
                            "source": "clarion_domain"})

    while len(out) < n:
        pick = rng.choice
        col, phrase = pick(list(NUM_COL.items()))
        d1, d2 = pick(DEPTS), pick(DEPTS)
        k = pick(range(1, 6))
        att = rng.choice([60, 65, 70, 72, 75, 80, 85, 88, 90])
        marks = rng.choice([70, 75, 80, 85, 88, 90, 92])
        fees = rng.choice([0, 0, 1000, 2000, 2500, 5000, 7500, 10000])
        assign = rng.choice([70, 75, 80, 85, 90, 95])
        intent = rng.randrange(18)
        if intent == 0:
            add([f"How many students are there in total",
                 f"Count the total number of students in the database"], 
                "SELECT COUNT(*) AS total_students FROM students;")
        elif intent == 1:
            add([f"List all students in the {d1} department",
                 f"Show me every student from {d1}"],
                f"SELECT student_id, name, department, average_marks, pending_fees FROM students WHERE department = '{d1}';")
        elif intent == 2:
            add([f"What is the average attendance percentage in each department"],
                "SELECT department, ROUND(AVG(attendance_percentage), 2) AS avg_attendance FROM students GROUP BY department;")
        elif intent == 3:
            add([f"Show the top {k} students by average marks",
                 f"Which {k} students have the highest average marks"],
                f"SELECT student_id, name, average_marks FROM students ORDER BY average_marks DESC LIMIT {k};")
        elif intent == 4:
            add([f"Which students still have pending fees"], 
                "SELECT student_id, name, department, pending_fees FROM students WHERE pending_fees > 0 ORDER BY pending_fees DESC;")
        elif intent == 5:
            add([f"How many students owe more than {fees} in pending fees",
                 f"Count the students whose pending fees exceed {fees}"],
                f"SELECT COUNT(*) AS students_with_fees FROM students WHERE pending_fees > {fees};")
        elif intent == 6:
            add([f"Which students have attendance below {att} percent"], 
                f"SELECT student_id, name, department, attendance_percentage FROM students WHERE attendance_percentage < {att};")
        elif intent == 7:
            add([f"What is the overall average of {phrase}",
                 f"Compute the average {phrase} across all students"],
                f"SELECT ROUND(AVG({col}), 2) AS avg_value FROM students;")
        elif intent == 8:
            add([f"Which student has the highest {phrase}"], 
                f"SELECT name, {col} FROM students ORDER BY {col} DESC LIMIT 1;")
        elif intent == 9:
            add([f"Which student has the lowest {phrase}"], 
                f"SELECT name, {col} FROM students ORDER BY {col} ASC LIMIT 1;")
        elif intent == 10:
            add([f"How many students are there in the {d1} department"], 
                f"SELECT COUNT(*) AS student_count FROM students WHERE department = '{d1}';")
        elif intent == 11:
            add([f"How many students does each department have"], 
                "SELECT department, COUNT(*) AS student_count FROM students GROUP BY department;")
        elif intent == 12:
            sub = pick(surnames)
            add([f"Which students' names end with {sub}",
                 f"Find students whose name contains the surname {sub}"], 
                f"SELECT student_id, name, department FROM students WHERE name LIKE '%{sub}';")
        elif intent == 13:
            add([f"Which departments have more than {k} students"], 
                f"SELECT department, COUNT(*) AS count FROM students GROUP BY department HAVING COUNT(*) > {k};")
        elif intent == 14:
            add([f"List all students ordered from the best to the worst assignment score"], 
                "SELECT student_id, name, department, assignment_score FROM students ORDER BY assignment_score DESC;")
        elif intent == 15:
            add([f"Which students have pending fees of {fees} or less and {phrase} at least {marks}",
                 f"Students with pending fees no more than {fees} and {phrase} at or above {marks}"], 
                f"SELECT student_id, name, {col}, pending_fees FROM students WHERE pending_fees <= {fees} AND {col} >= {marks};")
        elif intent == 16:
            add([f"Which courses are ordered from the most to the least difficult"], 
                "SELECT course_id, course_name, credits, difficulty_score FROM courses ORDER BY difficulty_score DESC;")
        elif intent == 17:
            add([f"How many credits are offered in total across all courses",
                 f"What is the total number of credits available in the courses table"], 
                "SELECT SUM(credits) AS total_credits FROM courses;")
    rng.shuffle(out)
    return out[:n]


DOMAIN_SCHEMA = (
    "CREATE TABLE students (student_id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
    "department TEXT NOT NULL, average_marks REAL NOT NULL, attendance_percentage REAL NOT NULL, "
    "assignment_score REAL NOT NULL, pending_fees REAL NOT NULL);\n"
    "CREATE TABLE courses (course_id INTEGER, course_name TEXT, credits INTEGER, difficulty_score REAL);")

# ------------------------------------------------------------- hand-made evals
# 30 phrasings with gold SQL; deliberately different from the training templates.
DOMAIN_EVAL = [
    ("Who owes the most in pending fees?", "SELECT name, pending_fees FROM students ORDER BY pending_fees DESC LIMIT 1;"),
    ("Average pending fees per department", "SELECT department, ROUND(AVG(pending_fees), 2) AS avg_pending_fees FROM students GROUP BY department;"),
    ("Give me the full record for the student named Isha Kulkarni.", "SELECT * FROM students WHERE name = 'Isha Kulkarni';"),
    ("difference between highest and lowest average marks", "SELECT MAX(average_marks) - MIN(average_marks) AS spread FROM students;"),
    ("List students attending less than 70% of classes.", "SELECT student_id, name, attendance_percentage FROM students WHERE attendance_percentage < 70;"),
    ("Which department's students have the best attendance on average?", "SELECT department, ROUND(AVG(attendance_percentage), 2) AS avg_attendance FROM students GROUP BY department ORDER BY avg_attendance DESC LIMIT 1;"),
    ("Total pending fees across the cohort", "SELECT ROUND(SUM(pending_fees), 2) AS total_pending_fees FROM students;"),
    ("Which three students score lowest on assignments?", "SELECT name, assignment_score FROM students ORDER BY assignment_score ASC LIMIT 3;"),
    ("Are there any students with zero pending fees? Count them.", "SELECT COUNT(*) AS no_fees FROM students WHERE pending_fees = 0;"),
    ("number of engineering departments represented", "SELECT COUNT(DISTINCT department) AS dept_count FROM students;"),
    ("Students in Electronics Engineering with average marks above 85", "SELECT name, average_marks FROM students WHERE department = 'Electronics Engineering' AND average_marks > 85;"),
    ("Sort students by pending fees, highest first, then name alphabetically", "SELECT name, department, pending_fees FROM students ORDER BY pending_fees DESC, name ASC;"),
    ("What's the mean difficulty score of our course catalogue?", "SELECT ROUND(AVG(difficulty_score), 2) AS avg_difficulty FROM courses;"),
    ("Course with the most credits; if tied, show all of them", "SELECT course_name, credits FROM courses WHERE credits = (SELECT MAX(credits) FROM courses);"),
    ("Everything about the Databases course", "SELECT * FROM courses WHERE course_name = 'Databases';"),
    ("Which courses are 4-credit courses?", "SELECT course_name, difficulty_score FROM courses WHERE credits = 4;"),
    ("How strong is the correlation between attendance and marks? — no, simpler: which students have both attendance below 80% and marks below 80?",
     "SELECT name, attendance_percentage, average_marks FROM students WHERE attendance_percentage < 80 AND average_marks < 80;"),
    ("top half of students by marks", "SELECT name, average_marks FROM students ORDER BY average_marks DESC LIMIT 7;"),
    ("Students whose fees equal exactly 2500", "SELECT name, department, pending_fees FROM students WHERE pending_fees = 2500;"),
    ("Second best average marks (the runner-up student)", "SELECT name, average_marks FROM students ORDER BY average_marks DESC LIMIT 1 OFFSET 1;"),
    ("Which IT students have pending fees under 3000?", "SELECT name, pending_fees FROM students WHERE department = 'Information Technology' AND pending_fees < 3000;"),
    ("Average assignment score, grouped from high to low", "SELECT department, ROUND(AVG(assignment_score), 2) AS avg_assignment FROM students GROUP BY department ORDER BY avg_assignment DESC;"),
    ("Does anyone hit perfect attendance? Name them.", "SELECT name, attendance_percentage FROM students WHERE attendance_percentage >= 99;"),
    ("Names starting with V", "SELECT name FROM students WHERE name LIKE 'V%';"),
    ("rough fees summary: min, max, and average", "SELECT MIN(pending_fees) AS min_fees, MAX(pending_fees) AS max_fees, ROUND(AVG(pending_fees), 2) AS avg_fees FROM students;"),
    ("hardest course", "SELECT course_name FROM courses ORDER BY difficulty_score DESC LIMIT 1;"),
    ("sum of attendance... meaning total across students", "SELECT ROUND(SUM(attendance_percentage), 2) AS total_attendance FROM students;"),
    ("Students with pending fees above 4000 but average marks below 90", "SELECT name, average_marks, pending_fees FROM students WHERE pending_fees > 4000 AND average_marks < 90;"),
    ("every unique department name", "SELECT DISTINCT department FROM students;"),
    ("How many courses have a difficulty under 9?", "SELECT COUNT(*) AS easier_courses FROM courses WHERE difficulty_score < 9;"),
]


def main():
    DATA.mkdir(exist_ok=True)
    cc_train, g_train, eval_pub = load_public()
    dom_train = domain_pairs()
    train = cc_train + g_train + dom_train
    rng.shuffle(train)

    with open(DATA / "train_pairs.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA / "eval_public.jsonl", "w") as f:
        for r in eval_pub:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    schema = DOMAIN_SCHEMA
    with open(DATA / "eval_domain.jsonl", "w") as f:
        for q, sql in DOMAIN_EVAL:
            f.write(json.dumps({"schema": schema, "question": q, "sql": sql,
                                "source": "clarion_domain_eval"}, ensure_ascii=False) + "\n")

    print(f"train      : {len(train):5d} "
          f"(sql_create_context={len(cc_train)}, gretel={len(g_train)}, domain={len(dom_train)})")
    print(f"eval_public: {len(eval_pub):5d}")
    print(f"eval_domain: {len(DOMAIN_EVAL):5d}")


if __name__ == "__main__":
    main()
