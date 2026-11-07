"""Schema DDL blocks used by the held-out classifier eval set.

Students DDL is the EXACT DDL of the real students.db. The others are invented
concrete schemas (two of them deliberately unlike anything in train_pairs.jsonl:
sql_create_context table_name_* wiki tables plus students/courses).
"""

STD = """CREATE TABLE students (student_id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL, average_marks REAL NOT NULL, attendance_percentage REAL NOT NULL, assignment_score REAL NOT NULL, pending_fees REAL NOT NULL);"""

EMP_S = """CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL, salary REAL NOT NULL);"""

EMP_M = """CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL, base_salary REAL NOT NULL, bonus_percentage REAL NOT NULL, stock_value_usd REAL NOT NULL, travel_reimbursement REAL NOT NULL);"""

VET = """CREATE TABLE vet_visits (visit_id INTEGER PRIMARY KEY, pet_name TEXT NOT NULL, owner_surname TEXT NOT NULL, species TEXT NOT NULL, month TEXT NOT NULL, weight_kg REAL NOT NULL, diagnosis TEXT NOT NULL, followup_required INTEGER NOT NULL, visit_cost REAL NOT NULL);"""

BIKE = """CREATE TABLE rides (ride_id INTEGER PRIMARY KEY, bike_id INTEGER NOT NULL, rider_name TEXT NOT NULL, start_station TEXT NOT NULL, end_station TEXT NOT NULL, month TEXT NOT NULL, duration_minutes INTEGER NOT NULL, fare_cents INTEGER NOT NULL);"""

SCR = """CREATE TABLE screenings (record_id INTEGER PRIMARY KEY, patient_code TEXT NOT NULL, age INTEGER NOT NULL, county TEXT NOT NULL, screening_type TEXT NOT NULL, result TEXT NOT NULL, month TEXT NOT NULL, referral_status TEXT NOT NULL, patient_share_of_cost REAL NOT NULL);"""

SCHEMAS = {
    "students_college": STD,
    "employees_single_metric": EMP_S,
    "employees_multi_metric": EMP_M,
    "vet_clinic_visits": VET,
    "bike_share_rides": BIKE,
    "cancer_screening_registry": SCR,
}
