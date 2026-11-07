#!/usr/bin/env python3
"""build_sql_eval_v2.py — builds two new held-out_eval databases and extends
evals/heldout/sql_eval.jsonl to ≥100 pairs across ≥4 schemas.

Honesty rules:
* the original 30 pairs are preserved byte-for-byte;
* `expected_result_row_count` is NOT hand-claimed — it is filled in by running
  each `gold_sql` against the real database file; gold SQL that errors is a
  generator bug and FAILS the build;
* the generator + DB builds are checked in, so the eval set is reproducible.

Run:  .venv/bin/python evals/tmp/build_sql_eval_v2.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HELDOUT = ROOT / "evals" / "heldout"
SQL_SET = HELDOUT / "sql_eval.jsonl"
TMP = ROOT / "evals" / "tmp"

# ---------------------------------------------------------------- new schemas

LIBRARY_DDL = """
DROP TABLE IF EXISTS loans;
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS members;
CREATE TABLE members (
  member_id INTEGER PRIMARY KEY,
  member_name TEXT NOT NULL,
  joined_on TEXT NOT NULL,
  active INTEGER NOT NULL
);
CREATE TABLE books (
  book_id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  author TEXT NOT NULL,
  genre TEXT,
  publication_year INTEGER,
  copies_total INTEGER NOT NULL,
  copies_available INTEGER NOT NULL
);
CREATE TABLE loans (
  loan_id INTEGER PRIMARY KEY,
  book_id INTEGER NOT NULL REFERENCES books(book_id),
  member_id INTEGER NOT NULL REFERENCES members(member_id),
  loaned_on TEXT NOT NULL,
  due_on TEXT NOT NULL,
  returned_on TEXT
);
"""

ORDERS_DDL = """
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY,
  customer_name TEXT NOT NULL,
  city TEXT NOT NULL,
  signup_date TEXT NOT NULL
);
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY,
  product_name TEXT NOT NULL,
  category TEXT NOT NULL,
  unit_price REAL NOT NULL,
  units_in_stock INTEGER NOT NULL
);
CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
  order_date TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE TABLE order_items (
  item_id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL REFERENCES orders(order_id),
  product_id INTEGER NOT NULL REFERENCES products(product_id),
  quantity INTEGER NOT NULL,
  unit_price REAL NOT NULL
);
"""

LIBRARY_MEMBERS = [
    (1, "Asha Iyer", "2023-01-15", 1),
    (2, "Rohit Verma", "2023-03-02", 1),
    (3, "Neha Nair", "2023-05-21", 0),
    (4, "Dev Menon", "2023-07-11", 1),
    (5, "Ira Bhatt", "2023-09-30", 1),
]
LIBRARY_BOOKS = [
    (1, "The Turing Game", "R. Srinivasan", "Science", 2019, 4, 2),
    (2, "Monsoon Metrics", "A. Kapoor", "Poetry", 2021, 2, 1),
    (3, "Ganges of Data", "R. Srinivasan", "Fiction", 2022, 3, 3),
    (4, "SQLite Dhamaka", "P. Iyengar", "Science", 2020, 5, 0),
    (5, "Monsoon Rains", "A. Kapoor", "Fiction", 2018, 2, 2),
    (6, "The Turing Book", "P. Iyengar", "History", 2017, 1, 1),
    (7, "Late Night Calculus", "M. Deshpande", "Science", 2023, 3, 1),
]
LIBRARY_LOANS = [
    (1, 1, 1, "2024-01-05", "2024-01-19", "2024-01-12"),
    (2, 2, 2, "2024-01-20", "2024-02-03", None),
    (3, 4, 3, "2024-02-01", "2024-02-15", None),
    (4, 1, 4, "2024-02-10", "2024-02-24", "2024-02-22"),
    (5, 3, 5, "2024-02-12", "2024-02-26", None),
    (6, 7, 1, "2024-03-01", "2024-03-15", None),
    (7, 5, 2, "2024-03-04", "2024-03-18", "2024-03-20"),
    (8, 4, 5, "2024-03-06", "2024-03-20", None),
    (9, 2, 4, "2024-03-11", "2024-03-25", None),
    (10, 6, 3, "2024-03-15", "2024-03-29", None),
]

ORDERS_CUSTOMERS = [
    (1, "Asha Iyer", "Mumbai", "2023-01-10"),
    (2, "Rohit Verma", "Delhi", "2023-02-14"),
    (3, "Neha Nair", "Mumbai", "2023-03-30"),
    (4, "Dev Menon", "Pune", "2023-05-08"),
    (5, "Ira Bhatt", "Delhi", "2023-08-19"),
    (6, "Kabir Seth", "Pune", "2023-11-11"),
]
ORDERS_PRODUCTS = [
    (1, "USB-C Cable", "Electronics", 350.0, 40),
    (2, "Mechanical Keyboard", "Electronics", 2400.0, 12),
    (3, "Standing Desk", "Furniture", 12500.0, 5),
    (4, "Notebook A5", "Stationery", 60.0, 120),
    (5, "Gel Pens Pack", "Stationery", 180.0, 75),
    (6, "Monitor Riser", "Furniture", 900.0, 20),
    (7, "Webcam HD", "Electronics", 1800.0, 0),
]
ORDERS_ORDER = [
    (1, 1, "2024-01-08", "delivered"),
    (2, 2, "2024-01-15", "delivered"),
    (3, 1, "2024-02-02", "delivered"),
    (4, 3, "2024-02-18", "shipped"),
    (5, 4, "2024-03-01", "delivered"),
    (6, 5, "2024-03-09", "pending"),
    (7, 6, "2024-03-20", "shipped"),
    (8, 2, "2024-03-28", "pending"),
]
ORDERS_ITEMS = [
    (1, 1, 1, 2, 350.0),
    (2, 1, 4, 5, 60.0),
    (3, 2, 2, 1, 2400.0),
    (4, 2, 5, 3, 180.0),
    (5, 3, 6, 1, 900.0),
    (6, 3, 4, 2, 60.0),
    (7, 4, 1, 4, 350.0),
    (8, 5, 3, 1, 12500.0),
    (9, 5, 5, 2, 180.0),
    (10, 6, 4, 10, 60.0),
    (11, 7, 2, 2, 2400.0),
    (12, 7, 6, 1, 900.0),
    (13, 8, 1, 1, 350.0),
    (14, 8, 7, 1, 1800.0),
]


def build_library():
    db = TMP / "library_demo.db"
    con = sqlite3.connect(db)
    con.executescript(LIBRARY_DDL)
    con.executemany("INSERT INTO members VALUES (?,?,?,?)", LIBRARY_MEMBERS)
    con.executemany("INSERT INTO books VALUES (?,?,?,?,?,?,?)", LIBRARY_BOOKS)
    con.executemany("INSERT INTO loans VALUES (?,?,?,?,?,?)", LIBRARY_LOANS)
    con.commit()
    con.close()
    return db


def build_orders():
    db = TMP / "orders_demo.db"
    con = sqlite3.connect(db)
    con.executescript(ORDERS_DDL)
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)", ORDERS_CUSTOMERS)
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?)", ORDERS_PRODUCTS)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS_ORDER)
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", ORDERS_ITEMS)
    con.commit()
    con.close()
    return db


# ------------------------------------------------------------------ new pairs
# Written by hand, then verified below (gold must run + row count is recorded,
# not asserted by hand). Question wording deliberately mirrors how real users
# ask ("most", "top", "how many", "which") without copying gold phrasing.

STUDENTS_EXTRA = [
    ("Which student has the highest attendance percentage?",
     "SELECT name FROM students ORDER BY attendance_percentage DESC LIMIT 1;", "order_limit"),
    ("How much pending fees is outstanding in total across all students?",
     "SELECT SUM(pending_fees) FROM students;", "aggregate"),
]

LIBRARY_PAIRS = [
    ("How many books are written by R. Srinivasan?",
     "SELECT COUNT(*) FROM books WHERE author = 'R. Srinivasan';", "aggregate"),
    ("How many loans are still open (never returned)?",
     "SELECT COUNT(*) FROM loans WHERE returned_on IS NULL;", "aggregate"),
    ("Which members joined in 2023 after March? List their names.",
     "SELECT member_name FROM members WHERE joined_on > '2023-03-31';", "filter"),
    ("List all books whose available copies are less than total copies.",
     "SELECT title FROM books WHERE copies_available < copies_total;", "filter"),
    ("Which books have no copies available right now?",
     "SELECT title FROM books WHERE copies_available = 0;", "filter"),
    ("How many loans were returned after their due date?",
     "SELECT COUNT(*) FROM loans WHERE returned_on > due_on;", "date"),
    ("Who returned a book late? Give me the member names.",
     "SELECT DISTINCT m.member_name FROM loans l JOIN members m "
     "ON l.member_id = m.member_id WHERE l.returned_on > l.due_on;", "join"),
    ("For each genre, how many books does the library own?",
     "SELECT genre, COUNT(*) FROM books GROUP BY genre;", "group_having"),
    ("List genres that own at least two distinct titles.",
     "SELECT genre, COUNT(*) AS n FROM books GROUP BY genre HAVING n >= 2;", "group_having"),
    ("Show the number of loans per member, sorted from most to least.",
     "SELECT m.member_name, COUNT(*) AS n FROM loans l JOIN members m "
     "ON l.member_id = m.member_id GROUP BY m.member_id ORDER BY n DESC;", "join"),
    ("Which author has the most copies owned by the library?",
     "SELECT author, SUM(copies_total) AS t FROM books GROUP BY author "
     "ORDER BY t DESC LIMIT 1;", "order_limit"),
    ("Top 2 oldest books by publication year.",
     "SELECT title, publication_year FROM books ORDER BY publication_year ASC LIMIT 2;", "order_limit"),
    ("How many members are currently active?",
     "SELECT COUNT(*) FROM members WHERE active = 1;", "aggregate"),
    ("List the titles of every book borrowed by members named Dev Menon.",
     "SELECT b.title FROM loans l JOIN members m ON l.member_id = m.member_id "
     "JOIN books b ON l.book_id = b.book_id WHERE m.member_name = 'Dev Menon';", "join"),
    ("Average publication year of Science books.",
     "SELECT AVG(publication_year) FROM books WHERE genre = 'Science';", "aggregate"),
    ("Which member borrowed the greatest number of distinct books?",
     "SELECT member_id FROM loans GROUP BY member_id ORDER BY COUNT(DISTINCT book_id) "
     "DESC LIMIT 1;", "nested"),
    ("Loans that were due in February 2024 — list loan ids.",
     "SELECT loan_id FROM loans WHERE due_on LIKE '2024-02-%';", "date"),
    ("For each book, how many times has it been loaned out?",
     "SELECT book_id, COUNT(*) FROM loans GROUP BY book_id;", "group_having"),
    ("Books that have never been loaned (list titles).",
     "SELECT title FROM books WHERE book_id NOT IN (SELECT book_id FROM loans);", "nested"),
    ("Total copies available across the whole library.",
     "SELECT SUM(copies_available) FROM books;", "aggregate"),
    ("List members who joined in the first half of 2023 (names, join dates).",
     "SELECT member_name, joined_on FROM members WHERE joined_on <= '2023-06-30' "
     "ORDER BY joined_on;", "date"),
    ("Which authors have written more than one book in the library?",
     "SELECT author FROM books GROUP BY author HAVING COUNT(*) > 1;", "group_having"),
    ("Members who have never borrowed anything — list their names.",
     "SELECT member_name FROM members WHERE member_id NOT IN "
     "(SELECT member_id FROM loans);", "nested"),
    ("Which books were borrowed by more than one different member?",
     "SELECT book_id FROM loans GROUP BY book_id HAVING COUNT(DISTINCT member_id) > 1;", "group_having"),
    ("List every loan that was returned late, with loan ids.",
     "SELECT loan_id FROM loans WHERE returned_on > due_on;", "date"),
    ("Sort the books by copies available, most available first.",
     "SELECT title FROM books ORDER BY copies_available DESC;", "order_limit"),
    ("Which member is the most recent one to join (name and date)?",
     "SELECT member_name, joined_on FROM members ORDER BY joined_on DESC LIMIT 1;", "order_limit"),
    ("How many distinct authors does the library currently hold?",
     "SELECT COUNT(DISTINCT author) FROM books;", "aggregate"),
    ("Books whose title contains the word 'Turing'.",
     "SELECT title FROM books WHERE title LIKE '%Turing%';", "filter"),
    ("Total number of loans per month of the loan date (year-month).",
     "SELECT SUBSTR(loaned_on, 1, 7) AS ym, COUNT(*) FROM loans GROUP BY ym ORDER BY ym;", "date"),
]

ORDERS_PAIRS = [
    ("How many customers are from Delhi?",
     "SELECT COUNT(*) FROM customers WHERE city = 'Delhi';", "aggregate"),
    ("List product names with zero stock.",
     "SELECT product_name FROM products WHERE units_in_stock = 0;", "filter"),
    ("Which orders were placed in March 2024? Give order ids.",
     "SELECT order_id FROM orders WHERE order_date LIKE '2024-03-%';", "date"),
    ("Total quantity of items ordered across all orders.",
     "SELECT SUM(quantity) FROM order_items;", "aggregate"),
    ("What is the average unit price of Electronics products?",
     "SELECT AVG(unit_price) FROM products WHERE category = 'Electronics';", "aggregate"),
    ("List each order id with its total amount (sum of quantity times price).",
     "SELECT order_id, SUM(quantity * unit_price) AS total FROM order_items "
     "GROUP BY order_id ORDER BY order_id;", "arithmetic"),
    ("Which customer spent the most across all their orders (total value)?",
     "SELECT o.customer_id FROM orders o JOIN order_items oi ON o.order_id = oi.order_id "
     "GROUP BY o.customer_id ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1;", "join"),
    ("Top 3 most expensive products by unit price.",
     "SELECT product_name, unit_price FROM products ORDER BY unit_price DESC LIMIT 3;", "order_limit"),
    ("How many orders are still pending?",
     "SELECT COUNT(*) FROM orders WHERE status = 'pending';", "filter"),
    ("For each status, how many orders?",
     "SELECT status, COUNT(*) FROM orders GROUP BY status;", "group_having"),
    ("Which cities have more than one customer? Show city and the count.",
     "SELECT city, COUNT(*) AS n FROM customers GROUP BY city HAVING n > 1;", "group_having"),
    ("List Electronics product names along with how many units remain.",
     "SELECT product_name, units_in_stock FROM products "
     "WHERE category = 'Electronics' ORDER BY product_name;", "filter"),
    ("Which customer from Pune has never placed an order (name)?",
     "SELECT customer_name FROM customers WHERE city = 'Pune' AND customer_id "
     "NOT IN (SELECT customer_id FROM orders);", "nested"),
    ("Product names ordered on 2024-02-02.",
     "SELECT DISTINCT p.product_name FROM order_items oi JOIN orders o "
     "ON oi.order_id = o.order_id JOIN products p ON oi.product_id = p.product_id "
     "WHERE o.order_date = '2024-02-02';", "join"),
    ("How many distinct products have been ordered at least once?",
     "SELECT COUNT(DISTINCT product_id) FROM order_items;", "aggregate"),
    ("For every product, how many units were sold in total?",
     "SELECT product_id, SUM(quantity) FROM order_items GROUP BY product_id;", "group_having"),
    ("Customers who signed up in 2023 — names and dates, newest first.",
     "SELECT customer_name, signup_date FROM customers ORDER BY signup_date DESC;", "order_limit"),
    ("What's the total inventory value (stock times price) per category?",
     "SELECT category, SUM(units_in_stock * unit_price) FROM products GROUP BY category;", "arithmetic"),
    ("Which order contains the highest single item-quantity line?",
     "SELECT order_id FROM order_items ORDER BY quantity DESC LIMIT 1;", "order_limit"),
    ("Average order value (sum over items) across all orders.",
     "SELECT AVG(total) FROM (SELECT SUM(quantity * unit_price) AS total "
     "FROM order_items GROUP BY order_id);", "nested"),
    ("List the product names bought by Asha Iyer.",
     "SELECT DISTINCT p.product_name FROM order_items oi JOIN orders o "
     "ON oi.order_id = o.order_id JOIN products p ON oi.product_id = p.product_id "
     "JOIN customers c ON o.customer_id = c.customer_id "
     "WHERE c.customer_name = 'Asha Iyer';", "join"),
    ("How many items does each order have (count of lines)?",
     "SELECT order_id, COUNT(*) FROM order_items GROUP BY order_id;", "group_having"),
    ("Which pending orders belong to Delhi customers (order ids)?",
     "SELECT o.order_id FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
     "WHERE o.status = 'pending' AND c.city = 'Delhi';", "join"),
    ("Coolest product name alphabetically last.",
     "SELECT product_name FROM products ORDER BY product_name DESC LIMIT 1;", "order_limit"),
    ("Which products are low on stock (fewer than 15 units)? List names.",
     "SELECT product_name FROM products WHERE units_in_stock < 15;", "filter"),
    ("For each city, what is the total value of goods its customers ordered?",
     "SELECT c.city, SUM(oi.quantity * oi.unit_price) FROM orders o "
     "JOIN customers c ON o.customer_id = c.customer_id "
     "JOIN order_items oi ON oi.order_id = o.order_id GROUP BY c.city;", "join"),
    ("Customers with more than one order — names and their order counts.",
     "SELECT c.customer_name, COUNT(*) AS n FROM customers c JOIN orders o "
     "ON o.customer_id = c.customer_id GROUP BY c.customer_id HAVING n > 1;", "join"),
    ("What is the total value of items in pending orders?",
     "SELECT SUM(oi.quantity * oi.unit_price) FROM orders o JOIN order_items oi "
     "ON oi.order_id = o.order_id WHERE o.status = 'pending';", "join"),
    ("Which product was ordered by the greatest number of distinct orders?",
     "SELECT product_id FROM order_items GROUP BY product_id "
     "ORDER BY COUNT(DISTINCT order_id) DESC, product_id LIMIT 1;", "nested"),
]

# join_demo.db (built by evals/tmp/make_join_demo_db.py) is the honest JOIN
# sandbox: real students rows + courses + enrollments. students.db is a single
# table, so join questions live here.
JOIN_PAIRS = [
    ("For every course, how many students are enrolled? Show course name and count.",
     "SELECT c.course_name, COUNT(*) FROM courses c JOIN enrollments e "
     "ON e.course_id = c.course_id GROUP BY c.course_id ORDER BY c.course_id;", "join"),
    ("List each student's name with the number of courses they take.",
     "SELECT s.name, COUNT(*) FROM students s JOIN enrollments e "
     "ON e.student_id = s.student_id GROUP BY s.student_id ORDER BY s.name;", "join"),
    ("Which students take no course at all? List their names.",
     "SELECT name FROM students WHERE student_id NOT IN "
     "(SELECT student_id FROM enrollments);", "nested"),
    ("What is the average marks students scored in each department's courses "
     "(students joined to enrollments)?",
     "SELECT s.department, AVG(e.marks) FROM students s JOIN enrollments e "
     "ON e.student_id = s.student_id GROUP BY s.department;", "join"),
    ("Which course is the most enrolled, by enrollment count?",
     "SELECT course_id FROM enrollments GROUP BY course_id "
     "ORDER BY COUNT(*) DESC LIMIT 1;", "order_limit"),
    ("For each department, how many credits do its courses offer in total?",
     "SELECT department, SUM(credits) FROM courses GROUP BY department;", "group_having"),
    ("Show students who scored above 90 in any enrollment (name and that mark).",
     "SELECT s.name, e.marks FROM students s JOIN enrollments e "
     "ON e.student_id = s.student_id WHERE e.marks > 90;", "join"),
    ("Which departments offer the most courses (name plus count, top 2)?",
     "SELECT department, COUNT(*) AS n FROM courses GROUP BY department "
     "ORDER BY n DESC LIMIT 2;", "order_limit"),
    ("Enrollment counts for courses of the Computer Engineering department.",
     "SELECT c.course_name, COUNT(*) FROM courses c JOIN enrollments e "
     "ON e.course_id = c.course_id WHERE c.department = 'Computer Engineering' "
     "GROUP BY c.course_id ORDER BY c.course_name;", "join"),
    ("Which course carries the highest total marks across its enrollments?",
     "SELECT e.course_id FROM enrollments e GROUP BY e.course_id "
     "ORDER BY SUM(e.marks) DESC LIMIT 1;", "join"),
]


def main():
    lib_db = build_library()
    top_db = build_orders()

    def verify(pairs, db_path, db_key):
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        out = []
        for q, sql, cat in pairs:
            try:
                cur = con.execute(sql)
                n = len(cur.fetchall())
            except sqlite3.Error as exc:
                raise SystemExit(f"GOLD SQL FAILED for {q!r}:\n  {sql}\n  {exc}")
            out.append({"question": q, "gold_sql": sql, "category": cat,
                        "db": db_key, "expected_result_row_count": n})
        con.close()
        return out

    old = [json.loads(l) for l in SQL_SET.read_text().splitlines() if l.strip()]
    new_pairs = (verify(STUDENTS_EXTRA, ROOT / "students.db", "students")
                 + verify(LIBRARY_PAIRS, lib_db, "library_demo")
                 + verify(ORDERS_PAIRS, top_db, "orders_demo")
                 + verify(JOIN_PAIRS, TMP / "join_demo.db", "join_demo"))

    # Idempotent merge: dedupe by question keep-first, so re-running this
    # generator never duplicates or reorders the originals at the top.
    seen, merged = set(), []
    for p in old + new_pairs:
        key = (p["question"].strip().lower(), p["db"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)
    SQL_SET.write_text("\n".join(json.dumps(p) for p in merged) + "\n")

    dropped = len(old) + len(new_pairs) - len(merged)
    if dropped:
        print(f"(!) dropped {dropped} duplicate (question, db) rows already present")

    from collections import Counter
    by_db = Counter(p["db"] for p in merged)
    by_cat = Counter(p["category"] for p in merged)
    print(f"total pairs: {len(merged)} ({len(old)} original preserved)")
    print("by db:", dict(by_db))
    print("by category:", dict(by_cat))


if __name__ == "__main__":
    main()
