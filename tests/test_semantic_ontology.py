"""Tests for the lightweight semantic ontology (Auto-Ontology-inspired)."""

import json
import os
import sqlite3

import pytest

import semantic_ontology as so
from ambiguity.analysis import analyze_ambiguity
from pipeline import build_schema


def _make_db(path, statements, rows=()):
    conn = sqlite3.connect(path)
    for statement in statements:
        conn.execute(statement)
    for table, records in rows:
        conn.executemany(
            f"INSERT INTO {table} VALUES ({','.join('?' * len(records[0]))})",
            records)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def student_db(tmp_path):
    """The schema from improvements.md, plus an unrelated course metric."""
    return _make_db(
        tmp_path / "student.db",
        ["CREATE TABLE demostudents ("
         "student_id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
         "department TEXT NOT NULL, average_marks REAL, "
         "attendance_percentage REAL, assignment_score REAL, "
         "pending_fees REAL)",
         "CREATE TABLE courses (course_id INTEGER PRIMARY KEY, "
         "course_name TEXT, difficulty_score REAL)"],
        [("demostudents", [(1, "Asha Iyer", "Computer", 91.5, 88.0, 94.0, 0.0),
                           (2, "Rohit Verma", "Civil", 72.0, 61.0, 68.0, 2500.0)]),
         ("courses", [(1, "Databases", 3.5)])])


def test_ontology_maps_doc_example_concepts(student_db):
    ontology = so.build_ontology(student_db)
    assert ontology["database"] == "student.db"
    entities = {entity["table"]: entity for entity in ontology["entities"]}
    student = entities["demostudents"]
    assert student["name"] == "Student"
    assert student["attributes"] == {
        "academic_performance": ["average_marks", "assignment_score"],
        "attendance": ["attendance_percentage"],
        "outstanding_fees": ["pending_fees"],
    }
    # "score" in courses must not become student academic performance.
    assert entities["courses"]["attributes"] == {}
    assert ontology["schema_metadata"]["table_count"] == 2
    assert ontology["schema_metadata"]["concept_catalog_version"] == 1


def test_ontology_is_stored_next_to_database_and_invalidates(student_db):
    ontology = so.build_ontology(student_db)
    path = so.ontology_path(student_db)
    assert path.exists()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["fingerprint"] == ontology["fingerprint"]

    conn = sqlite3.connect(student_db)
    conn.execute("ALTER TABLE demostudents ADD COLUMN exam_score REAL")
    conn.close()
    stat = student_db.stat()
    os.utime(student_db, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    refreshed = so.build_ontology(student_db)
    assert refreshed["fingerprint"] != ontology["fingerprint"]
    entity = {e["table"]: e for e in refreshed["entities"]}["demostudents"]
    assert "exam_score" in entity["attributes"]["academic_performance"]


def test_retrieval_maps_synonyms_and_respects_table_scope(student_db):
    ontology = so.build_ontology(student_db)

    owes = so.retrieve_concepts(ontology, "Show students who owe money.",
                                tables=["demostudents"])
    assert [concept["concept"] for concept in owes] == ["Outstanding Fees"]
    assert owes[0]["mapped_columns"] == ["demostudents.pending_fees"]
    assert owes[0]["verified"] is False

    performance = so.retrieve_concepts(
        ontology, "students with poor academic performance",
        tables=["demostudents"])
    assert [concept["concept"] for concept in performance] == [
        "Academic Performance"]
    assert performance[0]["mapped_columns"] == [
        "demostudents.average_marks", "demostudents.assignment_score"]

    attendance = so.retrieve_concepts(ontology, "attendance percentage",
                                      tables=["demostudents"])
    assert [concept["concept"] for concept in attendance] == ["Attendance"]

    assert so.retrieve_concepts(ontology, "happiest employees") == []
    assert so.retrieve_concepts(ontology, "difficulty score",
                                tables=["courses"]) == []


def test_render_labels_mappings_as_suggestions(student_db):
    concepts = so.retrieve_concepts(so.build_ontology(student_db),
                                    "students who owe money")
    rendered = so.render_concepts(concepts)
    assert "Outstanding Fees" in rendered
    assert "students.pending_fees" in rendered
    assert "not verified" in rendered and "not interchangeable" in rendered


def test_analysis_uses_ontology_for_answerability_and_ambiguity(student_db):
    clear = analyze_ambiguity("students who owe money", student_db)
    assert clear.label == "CLEAR"
    assert clear.decision_rule == "fallback-heuristic"
    assert clear.evidence["metric_named"] is True
    assert [c["concept"] for c in clear.evidence["ontology"]] == [
        "Outstanding Fees"]

    ambiguous = analyze_ambiguity(
        "students with poor academic performance", student_db)
    assert ambiguous.label == "AMBIGUOUS"
    assert ambiguous.decision_rule == "fallback-heuristic"

    missing = analyze_ambiguity("happiest employees", student_db)
    assert missing.label == "UNDERSPECIFIED"


def test_analysis_and_sql_prompt_receive_semantic_context(student_db):
    class FakeClassifier:
        def __init__(self):
            self.seen = None

        def predict(self, question, schema_text):
            self.seen = schema_text
            return {"label": "CLEAR", "probs": {"CLEAR": 0.9}}

    classifier = FakeClassifier()
    analyze_ambiguity("students who owe money", student_db,
                      classifier=classifier)
    assert "semantic hints" in classifier.seen
    assert "Outstanding Fees" in classifier.seen

    prompt = build_schema(student_db, [], question="students who owe money")
    assert "-- Outstanding Fees: demostudents.pending_fees" in prompt
