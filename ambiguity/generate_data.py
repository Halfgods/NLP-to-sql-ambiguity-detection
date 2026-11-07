"""Generate labeled question--schema pairs for the Clarion ambiguity classifier.

Creates data/ambiguity_train/{train,val,test,test_unseen}.jsonl with rows
    {"schema_id", "schema_text", "question", "label"}

Design contract (why labels depend on question AND schema):
* every ambiguity mechanism is driven by which columns/tables exist in the
  rendered schema, and the *same question string* is emitted with different
  labels on different schemas (see the *-twin mechanisms),
* column/table names and phrasings are randomized inside each schema family so
  a single token can never decide the label by itself,
* two schema families (flights, streaming) are held out entirely: they never
  appear in train/val/test and form the unseen-schema slice.

Run:  python -m ambiguity.generate_data
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

SEED = 20260430
LABELS = ("CLEAR", "AMBIGUOUS", "UNDERSPECIFIED")
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "ambiguity_train"

VERBS = ("Show", "List", "Get", "Retrieve", "Pull up")
MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
YEARS = (2023, 2024, 2025)

# ----------------------------------------------------------------------------
# schema blueprints.
# table  = {"key", "names": [...], optional "whole_p": prob the table exists,
#           "cols": [(col_key, [name variants], TYPE, opts)]}
# opts   = {"pk": 1} | {"nn": 1} | {"p": float}   (p=0.0 -> decoy/rival column:
#          present only when a mechanism explicitly requests it)
# ----------------------------------------------------------------------------

FAMILIES: dict = {}

FAMILIES["university"] = {
    "tables": {
        "students": {
            "names": ["students", "student_records", "enrolled_students"],
            "cols": [
                ("id", ["student_id", "sid", "student_no"], "INTEGER", {"pk": 1}),
                ("name", ["full_name", "student_name", "name"], "TEXT", {"nn": 1}),
                ("major", ["major", "program", "field_of_study"], "TEXT", {"p": 0.55}),
                ("email", ["email", "student_email", "email_address"], "TEXT", {"p": 0.35}),
                ("phone", ["phone", "phone_number", "contact_number"], "TEXT", {"p": 0.2}),
                ("gpa", ["gpa", "grade_point_average", "cumulative_gpa"], "REAL", {"p": 0.65}),
                ("status", ["status", "enrollment_status"], "TEXT", {"p": 0.4}),
            ],
        },
        "instructors": {
            "names": ["instructors", "faculty", "teaching_staff"],
            "cols": [
                ("id", ["instructor_id", "faculty_id", "staff_id"], "INTEGER", {"pk": 1}),
                ("name", ["full_name", "instructor_name", "name"], "TEXT", {"nn": 1}),
                ("dept", ["department", "dept", "academic_department"], "TEXT", {"p": 0.6}),
                ("salary", ["salary", "annual_salary", "base_pay"], "REAL", {"p": 0.8}),
                ("d_bonus", ["bonus", "yearly_bonus", "performance_bonus"], "REAL", {"p": 0.0}),
                ("d_stipend", ["stipend", "research_stipend"], "REAL", {"p": 0.0}),
                ("d_grant", ["grant_income", "grant_money", "grant_funding"], "REAL", {"p": 0.0}),
                ("d_summer", ["summer_teaching_pay", "extra_teaching_pay"], "REAL", {"p": 0.0}),
                ("email", ["email", "work_email", "instructor_email"], "TEXT", {"p": 0.25}),
            ],
        },
        "courses": {
            "names": ["courses", "course_catalog", "classes"],
            "cols": [
                ("id", ["course_id", "cid"], "INTEGER", {"pk": 1}),
                ("name", ["title", "course_name", "course_title"], "TEXT", {"nn": 1}),
                ("dept", ["department", "dept"], "TEXT", {"p": 0.55}),
                ("credits", ["credits", "credit_hours"], "INTEGER", {"p": 0.4}),
                ("room", ["room", "building_room"], "TEXT", {"p": 0.2}),
            ],
        },
        "enrollments": {
            "names": ["enrollments", "registrations", "course_registrations"],
            "cols": [
                ("id", ["enrollment_id", "reg_id"], "INTEGER", {"pk": 1}),
                ("student_id", ["student_id", "sid"], "INTEGER", {"p": 0.95}),
                ("course_id", ["course_id", "cid"], "INTEGER", {"p": 0.95}),
                ("grade", ["grade", "final_grade"], "TEXT", {"p": 0.35}),
                ("enrolled_on", ["enrolled_on", "enrollment_date", "registered_on"], "TEXT", {"p": 0.8}),
                ("conf_on", ["payment_received_on", "confirmed_on", "confirmation_date"], "TEXT", {"p": 0.0}),
            ],
        },
    },
    "mech": {
        "super": {"table": "instructors", "metric": "salary",
                  "metric_words": ["salary", "annual salary", "base pay"],
                  "decoys": ["d_bonus", "d_stipend", "d_grant", "d_summer"],
                  "ents": ["instructors", "faculty members", "teaching staff"],
                  "phrases": ["the highest-paid {e}", "the best-paid {e}",
                              "the top earners among the {e}"],
                  "named_deg": "highest"},
        "attr": [
            {"table": "students", "col": "email", "words": ["email addresses", "email address"],
             "ents": ["students", "enrolled students"]},
            {"table": "students", "col": "phone", "words": ["phone numbers"],
             "ents": ["students"]},
            {"table": "instructors", "col": "email", "words": ["work email addresses"],
             "ents": ["instructors", "faculty members"]},
        ],
        "date": [{"table": "enrollments", "col": "enrolled_on",
                  "pp": ["recorded", "registered", "made"],
                  "ents": ["enrollments", "course registrations"]}],
        "recency": [{"table": "enrollments", "base": "enrolled_on",
                     "rivals": ["conf_on"],
                     "ents": ["enrollments", "registrations", "course registrations"]}],
        "vague": [{"table": "students", "col": "gpa",
                   "adjs": ["strong", "high-performing"],
                   "ents": ["students", "enrolled students"]}],
        "cat": [{"table": "students", "col": "status",
                 "vals": ["active", "graduated", "on leave"], "ents": ["students"]}],
        "noattr": [{"a": "home addresses", "e": "students"},
                   {"a": "home addresses", "e": "instructors"},
                   {"a": "tuition fees", "e": "courses"},
                   {"a": "advisor names", "e": "students"}],
        "counts": [{"table": "students", "ent": "students"},
                   {"table": "courses", "ent": "courses"},
                   {"table": "enrollments", "ent": "enrollments"}],
        "ref_ents": ["student", "course", "instructor", "enrollment"],
        "decoy_lists": [{"table": "students", "word": "names", "ents": "students"},
                        {"table": "courses", "word": "course titles", "ents": "courses"}],
    },
}

FAMILIES["retail"] = {
    "tables": {
        "employees": {
            "names": ["employees", "staff", "store_employees"],
            "cols": [
                ("id", ["employee_id", "staff_id", "emp_id"], "INTEGER", {"pk": 1}),
                ("name", ["full_name", "employee_name", "name"], "TEXT", {"nn": 1}),
                ("role", ["job_title", "role", "position"], "TEXT", {"p": 0.5}),
                ("salary", ["salary", "monthly_wage", "base_pay"], "REAL", {"p": 0.8}),
                ("d_bonus", ["bonus", "annual_bonus", "yearly_bonus"], "REAL", {"p": 0.0}),
                ("d_commission", ["commission", "sales_commission"], "REAL", {"p": 0.0}),
                ("d_overtime", ["overtime_pay", "extra_hours_pay"], "REAL", {"p": 0.0}),
                ("d_tips", ["tips", "tip_income"], "REAL", {"p": 0.0}),
                ("hire_on", ["hire_date", "hired_on", "start_date"], "TEXT", {"p": 0.25}),
            ],
        },
        "customers": {
            "names": ["customers", "client_records", "shoppers"],
            "cols": [
                ("id", ["customer_id", "client_id", "shopper_id"], "INTEGER", {"pk": 1}),
                ("name", ["full_name", "customer_name", "name"], "TEXT", {"nn": 1}),
                ("email", ["email", "customer_email", "email_address"], "TEXT", {"p": 0.35}),
                ("phone", ["phone", "phone_number"], "TEXT", {"p": 0.2}),
                ("city", ["city", "home_city"], "TEXT", {"p": 0.3}),
            ],
        },
        "products": {
            "names": ["products", "catalog_items", "store_items"],
            "cols": [
                ("id", ["product_id", "item_id", "sku_id"], "INTEGER", {"pk": 1}),
                ("name", ["product_name", "title", "item_name"], "TEXT", {"nn": 1}),
                ("category", ["category", "product_category", "department"], "TEXT", {"p": 0.55}),
                ("price", ["price", "retail_price", "list_price"], "REAL", {"p": 0.8}),
                ("d_sale", ["sale_price", "discounted_price"], "REAL", {"p": 0.0}),
                ("d_clearance", ["clearance_price", "markdown_price"], "REAL", {"p": 0.0}),
                ("d_cost", ["wholesale_cost", "supplier_cost", "unit_cost"], "REAL", {"p": 0.0}),
                ("d_msrp", ["msrp", "sticker_price"], "REAL", {"p": 0.0}),
                ("description", ["description", "product_description"], "TEXT", {"p": 0.4}),
                ("stock", ["stock_quantity", "stock_on_hand", "inventory_count"], "INTEGER", {"p": 0.3}),
            ],
        },
        "orders": {
            "names": ["orders", "customer_orders", "sales_orders"],
            "cols": [
                ("id", ["order_id", "sale_id"], "INTEGER", {"pk": 1}),
                ("cust_id", ["customer_id", "client_id"], "INTEGER", {"p": 0.95}),
                ("total", ["order_total", "total_amount", "amount_due"], "REAL", {"p": 0.75}),
                ("order_date", ["order_date", "placed_on", "created_at"], "TEXT", {"p": 0.8}),
                ("ship_date", ["shipped_on", "ship_date", "dispatched_at"], "TEXT", {"p": 0.0}),
                ("status", ["order_status", "status"], "TEXT", {"p": 0.45}),
            ],
        },
        "order_items": {
            "names": ["order_items", "sale_lines", "order_lines"],
            "cols": [
                ("id", ["order_item_id", "sale_line_id"], "INTEGER", {"pk": 1}),
                ("order_id", ["order_id", "sale_id"], "INTEGER", {"p": 0.95}),
                ("product_id", ["product_id", "item_id"], "INTEGER", {"p": 0.95}),
                ("qty", ["quantity", "units"], "INTEGER", {"p": 0.75}),
            ],
        },
    },
    "mech": {
        "super": [
            {"table": "employees", "metric": "salary",
             "metric_words": ["salary", "monthly wage", "base pay"],
             "decoys": ["d_bonus", "d_commission", "d_overtime", "d_tips"],
             "ents": ["employees", "staff", "staff members"],
             "phrases": ["the highest-paid {e}", "the best-paid {e}",
                         "the top earners among the {e}"],
             "named_deg": "highest"},
            {"table": "products", "metric": "price",
             "metric_words": ["price", "retail price", "list price"],
             "decoys": ["d_sale", "d_clearance", "d_cost", "d_msrp"],
             "ents": ["products", "items", "catalog items"],
             "phrases": ["the most expensive {e}", "the priciest {e}"],
             "named_deg": "highest"},
        ],
        "attr": [
            {"table": "products", "col": "description",
             "words": ["product descriptions", "descriptions"],
             "ents": ["products", "items"]},
            {"table": "customers", "col": "email", "words": ["email addresses", "email address"],
             "ents": ["customers", "shoppers"]},
            {"table": "customers", "col": "phone", "words": ["phone numbers"],
             "ents": ["customers"]},
        ],
        "date": [{"table": "orders", "col": "order_date",
                  "pp": ["placed", "recorded", "made"],
                  "ents": ["orders", "customer orders"]}],
        "recency": [{"table": "orders", "base": "order_date",
                     "rivals": ["ship_date"],
                     "ents": ["orders", "customer orders"]}],
        "vague": [{"table": "products", "col": "price",
                   "adjs": ["cheap", "budget"], "ents": ["products", "items"]}],
        "cat": [{"table": "orders", "col": "status",
                 "vals": ["pending", "shipped", "delivered", "cancelled"],
                 "ents": ["orders"]}],
        "noattr": [{"a": "home addresses", "e": "employees"},
                   {"a": "office locations", "e": "employees"},
                   {"a": "tax amounts", "e": "orders"},
                   {"a": "supplier contact info", "e": "products"}],
        "counts": [{"table": "orders", "ent": "orders"},
                   {"table": "customers", "ent": "customers"},
                   {"table": "products", "ent": "products"}],
        "ref_ents": ["product", "order", "customer", "employee"],
        "decoy_lists": [{"table": "employees", "word": "names", "ents": "employees"},
                        {"table": "products", "word": "product names", "ents": "products"}],
    },
}

FAMILIES["library"] = {
    "tables": {
        "members": {
            "names": ["members", "library_members", "readers"],
            "cols": [
                ("id", ["member_id", "reader_id", "card_id"], "INTEGER", {"pk": 1}),
                ("name", ["full_name", "member_name", "name"], "TEXT", {"nn": 1}),
                ("email", ["email", "member_email", "email_address"], "TEXT", {"p": 0.35}),
                ("phone", ["phone", "phone_number"], "TEXT", {"p": 0.2}),
                ("joined", ["joined_on", "membership_date", "registered_on"], "TEXT", {"p": 0.25}),
            ],
        },
        "books": {
            "names": ["books", "library_titles", "collection_items"],
            "cols": [
                ("id", ["book_id", "title_id"], "INTEGER", {"pk": 1}),
                ("name", ["title", "book_title", "title_name"], "TEXT", {"nn": 1}),
                ("author", ["author", "author_name"], "TEXT", {"p": 0.6}),
                ("times", ["times_borrowed", "borrow_count", "total_checkouts"], "INTEGER", {"p": 0.85}),
                ("d_requests", ["hold_requests", "request_count", "reservation_count"], "INTEGER", {"p": 0.0}),
                ("d_wishlist", ["wishlist_count", "saved_count"], "INTEGER", {"p": 0.0}),
                ("d_rating", ["rating_score", "average_rating"], "REAL", {"p": 0.0}),
                ("isbn", ["isbn", "isbn_number", "isbn_code"], "TEXT", {"p": 0.3}),
                ("pub_year", ["publication_year", "published_year"], "INTEGER", {"p": 0.45}),
                ("shelf", ["shelf_location", "shelf_code"], "TEXT", {"p": 0.2}),
            ],
        },
        "loans": {
            "names": ["loans", "checkouts", "borrowing_records"],
            "cols": [
                ("id", ["loan_id", "checkout_id"], "INTEGER", {"pk": 1}),
                ("book_id", ["book_id", "title_id"], "INTEGER", {"p": 0.95}),
                ("member_id", ["member_id", "reader_id"], "INTEGER", {"p": 0.95}),
                ("loan_date", ["loan_date", "checked_out_on", "borrowed_on"], "TEXT", {"p": 0.8}),
                ("due_date", ["due_date", "return_by_date"], "TEXT", {"p": 0.55}),
                ("returned_on", ["returned_on", "return_date"], "TEXT", {"p": 0.0}),
            ],
        },
    },
    "mech": {
        "super": {"table": "books", "metric": "times",
                  "metric_words": ["times borrowed", "borrow count", "total checkouts"],
                  "decoys": ["d_requests", "d_wishlist", "d_rating"],
                  "ents": ["books", "titles"],
                  "phrases": ["the most borrowed {e}", "the most checked-out {e}"],
                  "named_deg": "highest"},
        "attr": [
            {"table": "books", "col": "isbn", "words": ["ISBN numbers", "ISBN codes"],
             "ents": ["books", "titles"]},
            {"table": "members", "col": "email", "words": ["email addresses", "email address"],
             "ents": ["members", "library members"]},
        ],
        "date": [{"table": "loans", "col": "loan_date",
                  "pp": ["issued", "recorded", "made"],
                  "ents": ["loans", "checkouts"]}],
        "recency": [{"table": "loans", "base": "loan_date", "rivals": ["returned_on"],
                     "ents": ["loans", "checkouts"]}],
        "vague": [
            {"table": "books", "col": "times", "adjs": ["popular"],
             "ents": ["books", "titles"]},
            {"table": "loans", "col": "loan_date", "adjs": ["recent"],
             "ents": ["loans", "checkouts"]},
        ],
        "noattr": [{"a": "home addresses", "e": "members"},
                   {"a": "cover images", "e": "books"},
                   {"a": "late fees", "e": "loans"}],
        "counts": [{"table": "books", "ent": "books"},
                   {"table": "members", "ent": "members"},
                   {"table": "loans", "ent": "loans"}],
        "ref_ents": ["book", "member", "loan"],
        "decoy_lists": [{"table": "books", "word": "book titles", "ents": "books"},
                        {"table": "members", "word": "names", "ents": "members"}],
    },
}

FAMILIES["hospital"] = {
    "tables": {
        "patients": {
            "names": ["patients", "patient_records", "admitted_patients"],
            "cols": [
                ("id", ["patient_id", "medical_record_number", "mrn_id"], "INTEGER", {"pk": 1}),
                ("name", ["full_name", "patient_name", "name"], "TEXT", {"nn": 1}),
                ("age", ["age", "patient_age"], "INTEGER", {"p": 0.4}),
                ("blood", ["blood_type", "blood_group"], "TEXT", {"p": 0.35}),
                ("dob", ["date_of_birth", "birth_date"], "TEXT", {"p": 0.25}),
                ("email", ["email", "patient_email"], "TEXT", {"p": 0.3}),
                ("phone", ["phone", "contact_number"], "TEXT", {"p": 0.2}),
            ],
        },
        "visits": {
            "names": ["visits", "encounters", "appointments"],
            "cols": [
                ("id", ["visit_id", "encounter_id", "appointment_id"], "INTEGER", {"pk": 1}),
                ("patient_id", ["patient_id"], "INTEGER", {"p": 0.95}),
                ("visit_date", ["visit_date", "visit_on", "scheduled_for"], "TEXT", {"p": 0.8}),
                ("checked_in", ["checked_in_on", "check_in_time"], "TEXT", {"p": 0.0}),
                ("discharge", ["discharge_date", "released_on"], "TEXT", {"p": 0.0}),
                ("duration", ["duration_days", "length_of_stay_days"], "INTEGER", {"p": 0.5}),
                ("cost", ["cost", "total_cost", "treatment_cost"], "REAL", {"p": 0.8}),
                ("d_bill", ["billed_amount", "amount_billed", "bill_total"], "REAL", {"p": 0.0}),
                ("d_copay", ["copay_amount", "copayment", "out_of_pocket"], "REAL", {"p": 0.0}),
                ("d_ins", ["insurance_charge", "insured_amount"], "REAL", {"p": 0.0}),
                ("diagnosis", ["diagnosis", "primary_diagnosis"], "TEXT", {"p": 0.5}),
                ("disp", ["disposition", "visit_type"], "TEXT", {"p": 0.35}),
            ],
        },
        "treatments": {
            "names": ["treatments", "treatment_records", "procedures"],
            "cols": [
                ("id", ["treatment_id", "procedure_id"], "INTEGER", {"pk": 1}),
                ("patient_id", ["patient_id"], "INTEGER", {"p": 0.95}),
                ("tname", ["treatment_name", "procedure_name", "treatment"], "TEXT", {"p": 0.85}),
                ("tdate", ["treatment_date", "performed_on"], "TEXT", {"p": 0.5}),
            ],
        },
    },
    "mech": {
        "super": {"table": "visits", "metric": "cost",
                  "metric_words": ["cost", "total cost", "treatment cost"],
                  "decoys": ["d_bill", "d_copay", "d_ins"],
                  "ents": ["visits", "patient visits", "encounters"],
                  "phrases": ["the most expensive {e}", "the costliest {e}"],
                  "named_deg": "highest"},
        "attr": [
            {"table": "patients", "col": "blood", "words": ["blood types", "blood type"],
             "ents": ["patients"]},
            {"table": "patients", "col": "email", "words": ["email addresses"],
             "ents": ["patients"]},
        ],
        "date": [{"table": "visits", "col": "visit_date",
                  "pp": ["recorded", "happened", "made"],
                  "ents": ["visits", "patient visits"]}],
        "recency": [{"table": "visits", "base": "visit_date",
                     "rivals": ["checked_in", "discharge"],
                     "ents": ["visits", "patient visits", "encounters"]}],
        "vague": [{"table": "visits", "col": "duration",
                   "adjs": ["long", "lengthy"], "ents": ["visits", "patient visits"]}],
        "cat": [{"table": "visits", "col": "disp",
                 "vals": ["outpatient", "inpatient", "emergency"],
                 "ents": ["visits"]}],
        "noattr": [{"a": "insurance policy numbers", "e": "patients"},
                   {"a": "insurance providers", "e": "patients"},
                   {"a": "home addresses", "e": "patients"}],
        "counts": [{"table": "patients", "ent": "patients"},
                   {"table": "visits", "ent": "visits"},
                   {"table": "treatments", "ent": "treatments"}],
        "ref_ents": ["patient", "visit", "treatment"],
        "decoy_lists": [{"table": "patients", "word": "names", "ents": "patients"},
                        {"table": "treatments", "word": "treatment names", "ents": "treatments"}],
    },
}

FAMILIES["logistics"] = {
    "tables": {
        "shipments": {
            "names": ["shipments", "freight_records", "consignments"],
            "cols": [
                ("id", ["shipment_id", "consignment_id", "freight_id"], "INTEGER", {"pk": 1}),
                ("dest", ["destination", "destination_city", "delivery_city"], "TEXT", {"p": 0.55}),
                ("weight", ["weight_kg", "gross_weight_kg", "shipment_weight_kg"], "REAL", {"p": 0.55}),
                ("declared", ["declared_value", "stated_value", "declared_worth"], "REAL", {"p": 0.85}),
                ("d_ins", ["insured_value", "insurance_value"], "REAL", {"p": 0.0}),
                ("d_customs", ["customs_value", "duty_value"], "REAL", {"p": 0.0}),
                ("d_replace", ["replacement_value", "documented_value"], "REAL", {"p": 0.0}),
                ("dispatch", ["dispatch_date", "shipped_on", "sent_on"], "TEXT", {"p": 0.8}),
                ("delivery", ["delivery_date", "delivered_on"], "TEXT", {"p": 0.0}),
                ("priority", ["priority", "shipping_priority"], "TEXT", {"p": 0.45}),
                ("volume", ["volume_m3", "cargo_volume_m3"], "REAL", {"p": 0.3}),
            ],
        },
        "warehouses": {
            "names": ["warehouses", "depots", "distribution_centers"],
            "cols": [
                ("id", ["warehouse_id", "depot_id"], "INTEGER", {"pk": 1}),
                ("name", ["warehouse_name", "name", "depot_name"], "TEXT", {"nn": 1}),
                ("city", ["city", "location_city"], "TEXT", {"p": 0.55}),
                ("phone", ["phone", "contact_number", "warehouse_phone"], "TEXT", {"p": 0.35}),
                ("pallets", ["capacity_pallets", "pallet_capacity"], "INTEGER", {"p": 0.25}),
            ],
        },
        "vehicles": {
            "names": ["vehicles", "trucks", "delivery_vehicles"],
            "cols": [
                ("id", ["vehicle_id", "truck_id"], "INTEGER", {"pk": 1}),
                ("name", ["vehicle_name", "nickname", "name"], "TEXT", {"nn": 1}),
                ("plate", ["plate_number", "license_plate", "registration_plate"], "TEXT", {"p": 0.4}),
                ("capacity", ["capacity_kg", "max_load_kg", "load_capacity"], "REAL", {"p": 0.45}),
                ("vtype", ["vehicle_type", "asset_type"], "TEXT", {"p": 0.3}),
            ],
        },
    },
    "mech": {
        "super": {"table": "shipments", "metric": "declared",
                  "metric_words": ["declared value", "stated value", "declared worth"],
                  "decoys": ["d_ins", "d_customs", "d_replace"],
                  "ents": ["shipments", "consignments", "freight records"],
                  "phrases": ["the most valuable {e}", "the highest-value {e}"],
                  "named_deg": "highest"},
        "attr": [
            {"table": "warehouses", "col": "phone", "words": ["phone numbers"],
             "ents": ["warehouses", "depots"]},
            {"table": "vehicles", "col": "plate",
             "words": ["license plate numbers", "plate numbers"],
             "ents": ["vehicles", "trucks"]},
        ],
        "date": [{"table": "shipments", "col": "dispatch",
                  "pp": ["dispatched", "shipped", "sent"],
                  "ents": ["shipments", "consignments"]}],
        "recency": [{"table": "shipments", "base": "dispatch", "rivals": ["delivery"],
                     "ents": ["shipments", "consignments"]}],
        "vague": [{"table": "shipments", "col": "weight",
                   "adjs": ["heavy", "bulky"], "ents": ["shipments", "consignments"]}],
        "cat": [{"table": "shipments", "col": "priority",
                 "vals": ["express", "standard"], "ents": ["shipments"]}],
        "noattr": [{"a": "warehouse addresses", "e": "warehouses"},
                   {"a": "driver names", "e": "shipments"},
                   {"a": "fuel costs", "e": "vehicles"}],
        "counts": [{"table": "shipments", "ent": "shipments"},
                   {"table": "warehouses", "ent": "warehouses"},
                   {"table": "vehicles", "ent": "vehicles"}],
        "ref_ents": ["shipment", "warehouse", "vehicle"],
        "decoy_lists": [{"table": "warehouses", "word": "names", "ents": "warehouses"},
                        {"table": "vehicles", "word": "names", "ents": "vehicles"}],
    },
}

FAMILIES["flights"] = {
    "tables": {
        "flights": {
            "names": ["flights", "scheduled_flights"],
            "cols": [
                ("id", ["flight_id", "flight_record_id"], "INTEGER", {"pk": 1}),
                ("fno", ["flight_number", "flight_code"], "TEXT", {"p": 0.7}),
                ("origin", ["origin", "origin_code", "from_airport"], "TEXT", {"p": 0.65}),
                ("dest", ["destination", "destination_code", "to_airport"], "TEXT", {"p": 0.65}),
                ("depart", ["departure_time", "departs_on", "scheduled_departure"], "TEXT", {"p": 0.8}),
                ("arrive", ["arrival_time", "lands_on", "scheduled_arrival"], "TEXT", {"p": 0.0}),
                ("duration", ["duration_minutes", "flight_duration_minutes", "duration"], "INTEGER", {"p": 0.8}),
                ("d_km", ["distance_km", "kilometers_flown"], "INTEGER", {"p": 0.0}),
                ("d_mi", ["distance_miles", "miles_flown"], "INTEGER", {"p": 0.0}),
                ("gate", ["gate", "boarding_gate"], "TEXT", {"p": 0.35}),
                ("terminal", ["terminal", "departure_terminal"], "TEXT", {"p": 0.25}),
                ("status", ["status", "flight_status"], "TEXT", {"p": 0.45}),
            ],
        },
        "bookings": {
            "names": ["bookings", "reservations", "flight_bookings"],
            "cols": [
                ("id", ["booking_id", "reservation_id"], "INTEGER", {"pk": 1}),
                ("flight_id", ["flight_id"], "INTEGER", {"p": 0.95}),
                ("passenger", ["passenger_name", "customer_name"], "TEXT", {"p": 0.8}),
                ("seat", ["seat_number", "seat_code"], "TEXT", {"p": 0.35}),
                ("booked", ["booked_on", "booking_date"], "TEXT", {"p": 0.75}),
                ("total", ["total_price", "amount_paid"], "REAL", {"p": 0.2}),
            ],
        },
        "aircraft": {
            "names": ["aircraft", "planes", "fleet_records"],
            "cols": [
                ("id", ["aircraft_id", "tail_id"], "INTEGER", {"pk": 1}),
                ("name", ["model", "aircraft_model", "plane_model"], "TEXT", {"nn": 1}),
                ("seats", ["seat_count", "economy_seats"], "INTEGER", {"p": 0.4}),
            ],
        },
    },
    "mech": {
        "super": {"table": "flights", "metric": "duration",
                  "metric_words": ["flight duration", "duration", "flight time"],
                  "decoys": ["d_km", "d_mi"],
                  "ents": ["flights"],
                  "phrases": ["the longest {e}"],
                  "named_deg": "longest"},
        "attr": [
            {"table": "flights", "col": "gate", "words": ["gate assignments", "gates"],
             "ents": ["flights"]},
            {"table": "flights", "col": "terminal", "words": ["terminals"],
             "ents": ["flights"]},
        ],
        "date": [{"table": "bookings", "col": "booked",
                  "pp": ["made", "recorded"],
                  "ents": ["bookings", "reservations"]}],
        "recency": [{"table": "flights", "base": "depart", "rivals": ["arrive"],
                     "ents": ["flights"]}],
        "vague": [{"table": "flights", "col": "duration", "adjs": ["long"],
                   "ents": ["flights"]}],
        "cat": [{"table": "flights", "col": "status",
                 "vals": ["cancelled", "delayed", "on time"], "ents": ["flights"]}],
        "noattr": [{"a": "pilot names", "e": "flights"},
                   {"a": "meals served", "e": "flights"},
                   {"a": "weather forecasts", "e": "flights"}],
        "counts": [{"table": "flights", "ent": "flights"},
                   {"table": "bookings", "ent": "bookings"}],
        "ref_ents": ["flight", "booking", "aircraft"],
        "decoy_lists": [{"table": "bookings", "word": "passenger names", "ents": "bookings"},
                        {"table": "aircraft", "word": "model names", "ents": "aircraft"}],
    },
}

FAMILIES["streaming"] = {
    "tables": {
        "users": {
            "names": ["users", "listeners", "accounts"],
            "cols": [
                ("id", ["user_id", "account_id"], "INTEGER", {"pk": 1}),
                ("name", ["username", "handle", "display_name"], "TEXT", {"nn": 1}),
                ("email", ["email", "user_email", "email_address"], "TEXT", {"p": 0.45}),
                ("phone", ["phone", "phone_number"], "TEXT", {"p": 0.2}),
                ("plan", ["plan", "subscription_plan"], "TEXT", {"p": 0.4}),
                ("joined", ["joined_on", "signup_date"], "TEXT", {"p": 0.25}),
            ],
        },
        "playlists": {
            "names": ["playlists", "user_playlists", "collections"],
            "cols": [
                ("id", ["playlist_id", "collection_id"], "INTEGER", {"pk": 1}),
                ("user_id", ["user_id", "owner_id"], "INTEGER", {"p": 0.95}),
                ("name", ["playlist_name", "name", "title"], "TEXT", {"nn": 1}),
                ("is_public", ["is_public", "public_flag"], "INTEGER", {"p": 0.25}),
                ("created", ["created_on", "created_at"], "TEXT", {"p": 0.2}),
            ],
        },
        "tracks": {
            "names": ["tracks", "songs", "song_catalog"],
            "cols": [
                ("id", ["track_id", "song_id"], "INTEGER", {"pk": 1}),
                ("name", ["title", "track_name", "song_name"], "TEXT", {"nn": 1}),
                ("artist", ["artist", "artist_name", "performer"], "TEXT", {"p": 0.6}),
                ("genre", ["genre", "music_genre"], "TEXT", {"p": 0.5}),
                ("plays", ["play_count", "stream_count", "total_plays"], "INTEGER", {"p": 0.85}),
                ("d_download", ["download_count", "downloads_total"], "INTEGER", {"p": 0.0}),
                ("d_like", ["like_count", "favorites_count"], "INTEGER", {"p": 0.0}),
                ("d_share", ["share_count", "shares_total"], "INTEGER", {"p": 0.0}),
                ("d_skip", ["skip_count", "skips_total"], "INTEGER", {"p": 0.0}),
                ("record_label", ["record_label", "label_name"], "TEXT", {"p": 0.3}),
                ("released", ["release_date", "released_on"], "TEXT", {"p": 0.45}),
                ("seconds", ["duration_seconds", "length_seconds"], "INTEGER", {"p": 0.3}),
            ],
        },
        "plays": {
            "names": ["plays", "play_events", "listen_events"],
            "cols": [
                ("id", ["play_id", "event_id"], "INTEGER", {"pk": 1}),
                ("track_id", ["track_id", "song_id"], "INTEGER", {"p": 0.95}),
                ("user_id", ["user_id", "listener_id"], "INTEGER", {"p": 0.9}),
                ("played", ["played_at", "played_on", "occurred_at"], "TEXT", {"p": 0.8}),
                ("device", ["device", "source_device"], "TEXT", {"p": 0.2}),
            ],
        },
    },
    "mech": {
        "super": {"table": "tracks", "metric": "plays",
                  "metric_words": ["play count", "stream count", "total plays"],
                  "decoys": ["d_download", "d_like", "d_share", "d_skip"],
                  "ents": ["tracks", "songs"],
                  "phrases": ["the most played {e}", "the most streamed {e}"],
                  "named_deg": "highest"},
        "attr": [
            {"table": "users", "col": "email", "words": ["email addresses", "email address"],
             "ents": ["users", "listeners"]},
            {"table": "tracks", "col": "record_label", "words": ["record labels"],
             "ents": ["tracks", "songs"]},
        ],
        "date": [{"table": "plays", "col": "played",
                  "pp": ["recorded", "logged", "made"],
                  "ents": ["plays", "play events"]}],
        "vague": [{"table": "tracks", "col": "plays",
                   "adjs": ["popular", "well-known"], "ents": ["tracks", "songs"]}],
        "cat": [{"table": "users", "col": "plan",
                 "vals": ["premium", "free", "family"], "ents": ["users"]}],
        "noattr": [{"a": "billing invoices", "e": "users"},
                   {"a": "album covers", "e": "tracks"},
                   {"a": "concert dates", "e": "songs"}],
        "counts": [{"table": "tracks", "ent": "tracks"},
                   {"table": "users", "ent": "users"},
                   {"table": "plays", "ent": "plays"}],
        "ref_ents": ["track", "playlist", "user"],
        "decoy_lists": [{"table": "users", "word": "usernames", "ents": "users"},
                        {"table": "playlists", "word": "playlist names", "ents": "playlists"}],
    },
}

# ----------------------------------------------------------------------------
# generic question/string helpers
# ----------------------------------------------------------------------------

def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _variants_phrase(name: str) -> str:
    return name.replace("_", " ")


def _q(verb: str, body: str) -> str:
    return _cap(f"{verb} {body}")


def _pick(rng, seq):
    return rng.choice(list(seq))


# ----------------------------------------------------------------------------
# schema instantiation + rendering
# ----------------------------------------------------------------------------

def _word_variants(variants, rng):
    """Return [pretty words] for a column name-variant list."""
    return [_variants_phrase(v) for v in variants]


def spawn_family(fam, rng, requests=None):
    """Instantiate the whole family schema as DDL text.

    ``requests``: {(table_key, col_key): "present"|"absent"}; every other column
    uses its blueprint base probability (pk/nn columns always present).  All
    random name draws happen BEFORE any presence filtering, so two spawn calls
    with the same seed and different requests differ only in column presence
    (that is exactly the twin-pair requirement).
    """
    requests = requests or {}
    out = []
    for tkey, tdef in fam["tables"].items():
        tname = _pick(rng, tdef["names"])
        decisions = []          # (name, ctype, opts, mode-derived keep)
        for ckey, variants, ctype, opts in tdef["cols"]:
            cname = _pick(rng, variants)          # deterministic draw first
            mode = requests.get((tkey, ckey))
            if mode == "present":
                keep = True
            elif mode == "absent":
                keep = False
            elif opts.get("pk") or opts.get("nn"):
                keep = True
            else:
                keep = rng.random() < opts.get("p", 0.5)
            decisions.append((cname, ctype, opts, keep))
        out.append((tname, decisions))
    rendered = []
    for tname, decisions in out:
        pk_cols = [d for d in decisions if d[3] and d[2].get("pk")]
        rest = [d for d in decisions if d[3] and not d[2].get("pk")]
        rng.shuffle(rest)
        lines = []
        for cname, ctype, opts, _keep in pk_cols + rest:
            decl = f"{cname} {ctype}"
            if opts.get("pk"):
                decl += " PRIMARY KEY"
            elif opts.get("nn"):
                decl += " NOT NULL"
            lines.append(decl)
        rendered.append(
            f"CREATE TABLE {tname} (\n  " + ",\n  ".join(lines) + "\n);")
    return "\n\n".join(rendered)


def variants_for(fam, table_key, col_key):
    for tkey, tdef in fam["tables"].items():
        if tkey == table_key:
            for ckey, variants, _t, _o in tdef["cols"]:
                if ckey == col_key:
                    return variants
    raise KeyError((table_key, col_key))


# ----------------------------------------------------------------------------
# mechanism builders: (fam, spec, rng) -> (requests, question)
# Each builder encapsulates one label semantics; twin builders produce the two
# members of a same-string/different-schema pair.
# ----------------------------------------------------------------------------

def _b_super_requests(spec, rng, decoys_present: bool):
    req = {(spec["table"], spec["metric"]): "present"}
    if decoys_present:
        for dkey in rng.sample(list(spec["decoys"]),
                               min(2, len(spec["decoys"])) if len(spec["decoys"]) < 2 else 2):
            req[(spec["table"], dkey)] = "present"
    return req


def _b_super_question(spec, rng):
    ent = _pick(rng, spec["ents"])
    body = _pick(rng, spec["phrases"]).format(e=ent)
    return f"{_pick(rng, VERBS)} {body}."


def b_super_pair(fam, spec, rng):
    """Return [(CLEAR, requests), (AMBIGUOUS, requests)] sharing one question."""
    q = _b_super_question(spec, rng)
    r_clear = _b_super_requests(spec, rng, decoys_present=False)
    r_amb = _b_super_requests(spec, rng, decoys_present=True)
    return [(("CLEAR", q), r_clear), (("AMBIGUOUS", q), r_amb)]


def b_super_extra_amb(fam, spec, rng):
    q = _b_super_question(spec, rng)
    return ("AMBIGUOUS", q), _b_super_requests(spec, rng, decoys_present=True)


def _named_word_pools(spec, fam):
    """[(col_key, pretty word)] for metric + decoy words of a super spec."""
    pools = []
    for ck in [spec["metric"], *spec["decoys"]]:
        for v in variants_for(fam, spec["table"], ck):
            pools.append((ck, _variants_phrase(v)))
    return pools


def _spec_cols(fam, table_key):
    for tkey, tdef in fam["tables"].items():
        if tkey == table_key:
            return tdef["cols"]
    raise KeyError(table_key)


def b_named_pair(fam, spec, rng):
    """Same question naming one specific metric/decoy word: CLEAR if the column
    exists in the schema, UNDERSPECIFIED if it does not."""
    ck, w = _pick(rng, _named_word_pools(spec, fam))
    ent = _pick(rng, spec["ents"])
    q = f"{_pick(rng, VERBS)} the {ent} with the {spec['named_deg']} {w}."
    # decoys (other than the named one) add realistic pressure in both twins
    r = {(spec["table"], spec["metric"]): "present"}
    for dkey in spec["decoys"]:
        if dkey != ck and rng.random() < 0.4:
            r[(spec["table"], dkey)] = "present"
    r_present = {**r, (spec["table"], ck): "present"}
    r_absent = {**r, (spec["table"], ck): "absent"}
    return [(("CLEAR", q), r_present), (("UNDERSPECIFIED", q), r_absent)]


def _attr_body(spec, rng):
    return _pick(rng, spec["words"]), _pick(rng, spec["ents"])


def b_attr_pair(fam, spec, rng):
    word, ent = _attr_body(spec, rng)
    if rng.random() < 0.5:
        q = f"{_pick(rng, VERBS)} the {word} of all the {ent}."
    else:
        q = f"List all the {ent} with their {word}."
    return [(("CLEAR", q), {(spec["table"], spec["col"]): "present"}),
            (("UNDERSPECIFIED", q), {(spec["table"], spec["col"]): "absent"})]


def b_date_pair(fam, spec, rng):
    ent = _pick(rng, spec["ents"])
    pp = _pick(rng, spec["pp"])
    month = _pick(rng, MONTHS)
    year = _pick(rng, YEARS)
    if rng.random() < 0.5:
        q = f"How many {ent} were {pp} in {month} {year}?"
    else:
        q = f"{_pick(rng, VERBS)} the {ent} {pp} in {month} {year}."
    return [(("CLEAR", q), {(spec["table"], spec["col"]): "present"}),
            (("UNDERSPECIFIED", q), {(spec["table"], spec["col"]): "absent"})]


def b_recency_pair(fam, spec, rng):
    ent = _pick(rng, spec["ents"])
    word = _pick(rng, ("most recent", "latest", "newest"))
    q = f"{_pick(rng, VERBS)} the {word} {ent}."
    r_c = {(spec["table"], spec["base"]): "present"}
    r_a = {**r_c, (spec["table"], _pick(rng, spec["rivals"])): "present"}
    return [(("CLEAR", q), r_c), (("AMBIGUOUS", q), r_a)]


def b_recency_extra_amb(fam, spec, rng):
    ent = _pick(rng, spec["ents"])
    word = _pick(rng, ("most recent", "latest", "newest"))
    q = f"{_pick(rng, VERBS)} the {word} {ent}."
    r = {(spec["table"], _pick(rng, spec["rivals"])): "present"}
    return ("AMBIGUOUS", q), r


def b_vague_u(fam, spec, rng):
    adj = _pick(rng, spec["adjs"])
    ent = _pick(rng, spec["ents"])
    if rng.random() < 0.75:
        q = f"{_pick(rng, VERBS)} the {adj} {ent}."
    else:
        q = f"Which {ent} are {adj}?"
    return ("UNDERSPECIFIED", q), {}


def b_cat_pair(fam, spec, rng):
    ent = _pick(rng, spec["ents"])
    val = _pick(rng, spec["vals"])
    q = f"{_pick(rng, VERBS)} the {val} {ent}."
    return [(("CLEAR", q), {(spec["table"], spec["col"]): "present"}),
            (("UNDERSPECIFIED", q), {(spec["table"], spec["col"]): "absent"})]


def b_noattr_u(fam, spec, rng):
    if rng.random() < 0.5:
        q = f"{_pick(rng, VERBS)} the {spec['a']} of all the {spec['e']}."
    else:
        q = f"List all the {spec['e']} with their {spec['a']}."
    return ("UNDERSPECIFIED", q), {}


REF_BODY = ["that {e}", "the one you mentioned", "it", "the {e} we talked about",
            "everything about that {e}", "the same one as last time",
            "the top stuff", "the details of that {e}", "the interesting one"]


def b_ref_u(fam, _spec, rng):
    ent = _pick(rng, fam["mech"]["ref_ents"])
    q = f"{_pick(rng, VERBS)} {_pick(rng, REF_BODY).format(e=ent)}."
    return ("UNDERSPECIFIED", q), {}


def b_count_c(fam, spec, rng):
    spec = _pick(rng, fam["mech"]["counts"])
    if rng.random() < 0.4:
        q = f"How many {spec['ent']} do we have in total?"
    else:
        q = f"How many {spec['ent']} are there?"
    return ("CLEAR", q), {}


def b_decoy_c(fam, spec, rng):
    """CLEAR question about an unrelated listing + metric-based decoy columns
    present elsewhere (decoys must not imply AMBIGUOUS on their own)."""
    sup = _pick(rng, _mech_specs(fam, "super"))
    req = _b_super_requests(sup, rng, decoys_present=True)
    if rng.random() < 0.5:
        lst = _pick(rng, fam["mech"]["decoy_lists"])
        q = f"{_pick(rng, VERBS)} the {lst['word']} of all the {lst['ents']}."
    else:
        cnt = _pick(rng, fam["mech"]["counts"])
        q = f"How many {cnt['ent']} are there?"
    return ("CLEAR", q), req


def _mech_specs(fam, mech):
    sp = fam["mech"][mech]
    return sp if isinstance(sp, list) else [sp]

# ----------------------------------------------------------------------------
# quotas: (mech, n) per family; pair mechs emit n of EACH side of the twin.
# Seen families aim 50/50/50; unseen families 40/40/40. Totals = 990 >= 900.
# ----------------------------------------------------------------------------

FULL = [("super_pair", 12), ("super_extra_amb", 18), ("named_pair", 3),
        ("decoy_c", 8), ("attr_pair", 8), ("date_pair", 5),
        ("recency_pair", 5), ("recency_extra_amb", 15), ("cat_pair", 4),
        ("count_c", 5), ("vague_u", 8), ("noattr_u", 12), ("ref_u", 10)]

QUOTA = {
    "university": FULL, "retail": FULL, "hospital": FULL, "logistics": FULL,
    "library": [("super_pair", 12), ("super_extra_amb", 18), ("named_pair", 3),
                ("decoy_c", 11), ("attr_pair", 8), ("date_pair", 5),
                ("recency_pair", 5), ("recency_extra_amb", 15),
                ("count_c", 6), ("vague_u", 12), ("noattr_u", 13), ("ref_u", 9)],
    "flights": [("super_pair", 12), ("super_extra_amb", 10), ("named_pair", 2),
                ("decoy_c", 7), ("attr_pair", 6), ("date_pair", 5),
                ("recency_pair", 3), ("recency_extra_amb", 15), ("cat_pair", 3),
                ("count_c", 2), ("vague_u", 6), ("noattr_u", 10), ("ref_u", 8)],
    "streaming": [("super_pair", 12), ("super_extra_amb", 28), ("named_pair", 2),
                  ("decoy_c", 7), ("attr_pair", 6), ("date_pair", 5),
                  ("cat_pair", 3), ("count_c", 5), ("vague_u", 6),
                  ("noattr_u", 10), ("ref_u", 8)],
}

SEEN_FAMILIES = ["university", "retail", "library", "hospital", "logistics"]
UNSEEN_FAMILIES = ["flights", "streaming"]

# mech -> (spec list key or None, builder)
_TWIN_MECHS = {
    "super_pair": ("super", b_super_pair), "named_pair": ("super", b_named_pair),
    "attr_pair": ("attr", b_attr_pair), "date_pair": ("date", b_date_pair),
    "recency_pair": ("recency", b_recency_pair), "cat_pair": ("cat", b_cat_pair),
}
_SPEC_MECHS = {
    "super_extra_amb": ("super", b_super_extra_amb),
    "recency_extra_amb": ("recency", b_recency_extra_amb),
    "vague_u": ("vague", b_vague_u), "noattr_u": ("noattr", b_noattr_u),
}
_PLAIN_MECHS = {"decoy_c": b_decoy_c, "count_c": b_count_c, "ref_u": b_ref_u}


def generate_examples(rng):
    examples = []
    for fam_name, fam in FAMILIES.items():
        seq = 0
        for mech, n in QUOTA[fam_name]:
            for _ in range(n):
                if mech in _TWIN_MECHS:
                    key, builder = _TWIN_MECHS[mech]
                    spec = _pick(rng, _mech_specs(fam, key))
                    items = builder(fam, spec, rng)
                elif mech in _SPEC_MECHS:
                    key, builder = _SPEC_MECHS[mech]
                    spec = _pick(rng, _mech_specs(fam, key))
                    items = [builder(fam, spec, rng)]
                else:
                    builder = _PLAIN_MECHS[mech]
                    items = [builder(fam, None, rng)]
                pair_seed = rng.getrandbits(64)
                for (label, question), requests in items:
                    seed = pair_seed if len(items) > 1 else rng.getrandbits(64)
                    schema_text = spawn_family(fam, random.Random(seed), requests)
                    seq += 1
                    examples.append({
                        "schema_id": f"{fam_name}-{seq:04d}",
                        "schema_text": schema_text,
                        "question": question,
                        "label": label,
                    })
    return examples


def _stratified_split(items, rng):
    """Per (family, label) 70/15/15 split."""
    groups = {}
    for i, ex in enumerate(items):
        groups.setdefault((ex["schema_id"].rsplit("-", 1)[0], ex["label"]), []).append(i)
    train, val, test = [], [], []
    for key in sorted(groups):
        idxs = groups[key]
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = round(0.70 * n)
        n_val = round(0.15 * n)
        train += idxs[:n_tr]
        val += idxs[n_tr:n_tr + n_val]
        test += idxs[n_tr + n_val:]
    return train, val, test


def _write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as fh:
        for ex in items:
            fh.write(json.dumps(
                {"schema_id": ex["schema_id"], "schema_text": ex["schema_text"],
                 "question": ex["question"], "label": ex["label"]}) + "\n")


def main():
    rng = random.Random(SEED)
    examples = generate_examples(rng)

    # exact-duplicate (schema_text, question) removal
    seen_keys, unique = set(), []
    for ex in examples:
        key = (ex["schema_text"], ex["question"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(ex)
    dropped = len(examples) - len(unique)
    examples = unique

    seen = [e for e in examples if e["schema_id"].rsplit("-", 1)[0] in SEEN_FAMILIES]
    unseen = [e for e in examples if e["schema_id"].rsplit("-", 1)[0] in UNSEEN_FAMILIES]
    tr_idx, va_idx, te_idx = _stratified_split(seen, rng)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUT_DIR / "train.jsonl", [seen[i] for i in tr_idx])
    _write_jsonl(OUT_DIR / "val.jsonl", [seen[i] for i in va_idx])
    _write_jsonl(OUT_DIR / "test.jsonl", [seen[i] for i in te_idx])
    _write_jsonl(OUT_DIR / "test_unseen.jsonl", unseen)

    # ---- guards ------------------------------------------------------------
    assert len(examples) >= 900, f"dataset too small: {len(examples)}"
    seen_names = {e["schema_id"].rsplit("-", 1)[0] for e in seen}
    unseen_names = {e["schema_id"].rsplit("-", 1)[0] for e in unseen}
    assert seen_names == set(SEEN_FAMILIES), seen_names
    assert unseen_names == set(UNSEEN_FAMILIES), unseen_names

    # same question string must appear under more than one label (question+schema)
    q_labels = {}
    for e in examples:
        q_labels.setdefault(e["question"], set()).add(e["label"])
    multi = [q for q, ls in q_labels.items() if len(ls) > 1]
    assert len(multi) >= 60, f"too few label-diverse questions: {len(multi)}"

    counts = Counter(e["label"] for e in examples)
    for lab in LABELS:
        assert counts[lab] > 0
    print(f"schema families: seen={sorted(seen_names)} unseen={sorted(unseen_names)}")
    print(f"examples: {len(examples)} (dropped exact dups: {dropped})")
    print(f"label counts: {dict(counts)}")
    for path, _n in (("train.jsonl", len(tr_idx)), ("val.jsonl", len(va_idx)),
                     ("test.jsonl", len(te_idx)), ("test_unseen.jsonl", len(unseen))):
        with open(OUT_DIR / path, encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
        c = Counter()
        with open(OUT_DIR / path, encoding="utf-8") as fh:
            for line in fh:
                c[json.loads(line)["label"]] += 1
        print(f"  {path}: {n} rows {dict(c)}")
    print(f"label-diverse question strings: {len(multi)}"
          f" (example: {multi[0] if multi else '-'})")


if __name__ == "__main__":
    main()




