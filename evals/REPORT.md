# Clarion — held-out evaluation report (2026-10-31T10:05:08)

## suite status

| suite | status | detail |
|---|---|---|
| classifier | run | artifact: AmbiguityClassifier (module.AmbiguityClassifier.load()) |
| sql | run | model: Qwen2.5-Coder-1.5B-Instruct (SQL + explicit-write QLoRA, GGUF Q4_K_M) |

**Classifier** — artifact `AmbiguityClassifier (module.AmbiguityClassifier.load())` (scored 101/101, 0 prediction errors)

- accuracy: **0.604** — macro-F1: **0.606**
- ambiguity precision **0.8421** / recall **0.4706** / F1 **0.6038**

per-class prec/recall/F1/support:

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| CLEAR | 0.4783 | 0.6667 | 0.557 | 33 |
| AMBIGUOUS | 0.8421 | 0.4706 | 0.6038 | 34 |
| UNDERSPECIFIED | 0.6389 | 0.6765 | 0.6571 | 34 |

confusion matrix (rows = gold, cols = ['CLEAR', 'AMBIGUOUS', 'UNDERSPECIFIED']):

CLEAR| gold \ pred |  | AMBIGUOUS| gold \ pred |  | UNDERSPECIFIED |
|---|---|---|---|
| CLEAR | 22 | 3 | 8 |
| AMBIGUOUS | 13 | 16 | 5 |
| UNDERSPECIFIED | 11 | 0 | 23 |

per-schema accuracy:

| schema | n | accuracy |
|---|---|---|
| students_college | 27 | 0.5556 |
| employees_single_metric | 11 | 0.7273 |
| employees_multi_metric | 13 | 0.6154 |
| vet_clinic_visits | 16 | 0.5625 |
| bike_share_rides | 16 | 0.6875 |
| cancer_screening_registry | 18 | 0.5556 |

**SQL suite** — server model `Qwen2.5-Coder-1.5B-Instruct (SQL + explicit-write QLoRA, GGUF Q4_K_M)` (gguf: `qwen25-coder-1.5b-sql-write-q4_k_m.gguf`, sha256 `6a801b7b27c23a27…`, 986048160 bytes)

prompt parity: production (READ_ONLY mode, SQL_SYSTEM_PROMPT, pipeline.build_schema, max_tokens=220, temp=0)

schema sources: {'pipeline.build_schema (production)': 101} · server warmup 0.0 s

- pairs evaluated: **101**/101
- sql produced: 1.0 • valid read-only: 1.0 • invalid-SQL rate: 0.0
- executed: 0.9604
- execution accuracy (normalized: projection+2dp): **0.7822** · strict (4dp exact): **0.6733** (passed only the fair comparator: 11)
- latency s: mean 1.55 / median 1.34 / p95 2.81 (total 156.1)

| category | n | exec_ok | accuracy | mean latency |
|---|---|---|---|---|
| filter | 15 | 13 | 0.8667 | 1.31 |
| aggregate | 17 | 17 | 1.0 | 0.88 |
| group_having | 16 | 12 | 0.75 | 1.38 |
| order_limit | 16 | 11 | 0.6875 | 1.35 |
| join | 22 | 15 | 0.6818 | 2.37 |
| date | 6 | 5 | 0.8333 | 1.78 |
| nested | 7 | 4 | 0.5714 | 1.7 |
| arithmetic | 2 | 2 | 1.0 | 1.52 |

failed pairs:

- **EXEC-FAIL** Which departments have an average attendance below 87? — result mismatch · got: `SELECT department FROM students WHERE attendance_percentage < 87;`
- **EXEC-FAIL** Give the names and departments of students whose pending fees are abov — result mismatch · got: `SELECT name, department FROM students WHERE pending_fees > (SELECT AVG(pending_fees) FROM `
- **EXEC-FAIL** Which student owes the most in pending fees? Give the name. — result mismatch · got: `SELECT name FROM students WHERE pending_fees > (SELECT MAX(pending_fees) FROM students);`
- **EXEC-FAIL** Give the name of the student who has the fifth highest attendance perc — result mismatch · got: `SELECT name FROM students ORDER BY attendance_percentage DESC LIMIT 1;`
- **EXEC-FAIL** Which members joined in 2023 after March? List their names. — result mismatch · got: `SELECT member_name FROM members WHERE joined_on > '2023-03-01' AND active = 1;`
- **EXEC-FAIL** Who returned a book late? Give me the member names. — result mismatch · got: `SELECT DISTINCT m.member_name FROM members m INNER JOIN loans l ON m.member_id = l.member_`
- **EXEC-FAIL** Show the number of loans per member, sorted from most to least. — result mismatch · got: `SELECT member_id, COUNT(*) as num_loans FROM loans GROUP BY member_id ORDER BY num_loans D`
- **EXEC-FAIL** Top 2 oldest books by publication year. — result mismatch · got: `SELECT title, author, publication_year FROM books ORDER BY publication_year DESC LIMIT 2;`
- **EXEC-FAIL** Which member borrowed the greatest number of distinct books? — result mismatch · got: `SELECT m.member_name, COUNT(DISTINCT l.book_id) AS total_books_borrowed FROM members m JOI`
- **EXEC-FAIL** Which customer spent the most across all their orders (total value)? — result mismatch · got: `SELECT customers.customer_name, SUM(order_items.quantity * order_items.unit_price) as tota`
- **EXEC-FAIL** For every product, how many units were sold in total? — result mismatch · got: `SELECT products.product_name, SUM(order_items.quantity) as total_units_sold FROM products `
- **INVALID** Which order contains the highest single item-quantity line? — OperationalError: no such column: orders.product_id · got: `SELECT orders.order_id, orders.customer_id, orders.order_date, orders.status FROM orders I`
- **EXEC-FAIL** Average order value (sum over items) across all orders. — result mismatch · got: `SELECT AVG(quantity * unit_price) FROM order_items INNER JOIN orders ON order_items.order_`
- **EXEC-FAIL** List the product names bought by Asha Iyer. — result mismatch · got: `SELECT DISTINCT t1.product_name FROM products AS t1 JOIN order_items AS t2 ON t1.product_i`
- **INVALID** Which course is the most enrolled, by enrollment count? — OperationalError: ambiguous column name: course_id · got: `SELECT course_name FROM courses JOIN enrollments ON courses.course_id = enrollments.course`
- **INVALID** Show students who scored above 90 in any enrollment (name and that mar — OperationalError: no such column: name · got: `SELECT name, MAX(marks) FROM enrollments GROUP BY student_id HAVING MAX(marks) > 90;`
- **EXEC-FAIL** Which books were borrowed by more than one different member? — result mismatch · got: `SELECT title FROM books WHERE book_id IN (SELECT DISTINCT book_id FROM loans GROUP BY book`
- **EXEC-FAIL** Books whose title contains the word 'Turing'. — result mismatch · got: `SELECT * FROM books WHERE title LIKE '%Turing';`
- **INVALID** Total number of loans per month of the loan date (year-month). — OperationalError: near "FROM": syntax error · got: `SELECT EXTRACT(MONTH FROM loaned_on) AS month, COUNT(*) AS total_loans FROM loans GROUP BY`
- **EXEC-FAIL** Which product was ordered by the greatest number of distinct orders? — result mismatch · got: `SELECT p.product_name FROM products p INNER JOIN order_items oi ON p.product_id = oi.produ`
- **EXEC-FAIL** Enrollment counts for courses of the Computer Engineering department. — result mismatch · got: `SELECT course_id, COUNT(*) AS enrollment_count FROM enrollments WHERE student_id IN (SELEC`
- **EXEC-FAIL** Which course carries the highest total marks across its enrollments? — result mismatch · got: `SELECT courses.course_name, SUM(enrollments.marks) as total_marks FROM courses INNER JOIN `
