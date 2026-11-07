"""Tests for db_introspect.py and ambiguity/analysis.py (Clarion).

Isolation rules honoured here:
* the peer's ambiguity/classifier.py is NEVER imported — a stub classifier
  class is defined inline and injected into analyze_ambiguity;
* no model_runtime/network/llama-server usage — the SLM is a FakeSLM that
  returns canned JSON strings, including one wrapped in ```json fences and one
  invalid payload;
* databases are tiny, throwaway SQLite files built in pytest's tmp_path.
"""

import json
import sqlite3

import pytest

import db_introspect as di
from ambiguity.analysis import (AmbiguityDecision, analyze_ambiguity,
                                build_slm_messages, parse_slm_json,
                                split_questions)


# --------------------------------------------------------------------------
# fake clients (no peer imports, no real models)
# --------------------------------------------------------------------------

class FakeClassifier:
    """Stub with the peer contract: predict(question, schema_text) -> {"label","probs"}."""

    def __init__(self, label, probs=None):
        self.label = label
        self.probs = probs
        self.seen_calls = []

    def predict(self, question, schema_text):
        self.seen_calls.append((question, schema_text))
        return {"label": self.label, "probs": self.probs}


class FakeSLM:
    """Stub with the LlamaServerRuntime.generate_chat signature shape."""

    def __init__(self, response):
        self.response = response
        self.blocks = []  # "ok" -> 1, "err" -> 0

    def generate_chat(self, messages, **kwargs):
        self.messages = messages
        self.blocks.append(1)
        return self.response


class BrokenSLM:
    def generate_chat(self, messages, **kwargs):
        raise RuntimeError("server is down")


# canned SLM payloads ------------------------------------------------------------

META_CLEAR = {"label": "CLEAR", "reason": "single salary reading",
              "interpretations": [], "clarification_question": "",
              "note_on_missing_data": ""}

META_AMBIG = {
    "label": "AMBIGUOUS",
    "reason": "two numeric pay metrics",
    "interpretations": [
        {"meaning": "highest base salary",
         "relevant_columns": ["employees.salary"]},
        {"meaning": "highest bonus",
         "relevant_columns": ["employees.bonus"]},
    ],
    "clarification_question": "Did you mean base salary or a bonus?",
    "note_on_missing_data": ""}

META_UNDER = {
    "label": "UNDERSPECIFIED",
    "reason": "no happiness column in the schema",
    "interpretations": [],
    "clarification_question": "Which stored metric should I use instead?",
    "note_on_missing_data": "schema has no happiness-related column"}
SLM_CLEAR = json.dumps(META_CLEAR)
SLM_AMBIG = json.dumps(META_AMBIG)
SLM_UNDER = json.dumps(META_UNDER)

# --------------------------------------------------------------------------
# temp databases
# --------------------------------------------------------------------------

def _make_db(path, stmts, rows):
    conn = sqlite3.connect(path)
    for s in stmts:
        conn.execute(s)
    for table, records in rows:
        conn.executemany(
            f"INSERT INTO {table} VALUES ({','.join('?' * len(records[0]))})",
            records)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def salary_only_db(tmp_path):
    """Pay metric 1 only: an unambiguous schema for 'highest salary'."""
    return _make_db(
        tmp_path / "salary_only.db",
        ["CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, "
         "employee_name TEXT NOT NULL, salary REAL)"],
        [("employees", [(1, "Asha Iyer", 52000.0), (2, "Rohit Verma", 61000.5),
                        (3, "Neha Nair", 47500.0)])])


@pytest.fixture
def pay_mix_db(tmp_path):
    """Pay metric 2 variety (salary/bonus/commission) + FK to departments."""
    return _make_db(
        tmp_path / "pay_mix.db",
        ["CREATE TABLE departments (department_id INTEGER PRIMARY KEY, "
         "department_name TEXT NOT NULL)",
         "CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, "
         "employee_name TEXT NOT NULL, salary REAL NOT NULL, bonus REAL, "
         "commission REAL, department_id INTEGER, "
         "FOREIGN KEY (department_id) REFERENCES departments(department_id))"],
        [("departments", [(1, "Sales"), (2, "Engineering"), (3, "Support")]),
         ("employees", [(1, "Asha Iyer", 52000.0, 4000.0, 900.0, 2),
                        (2, "Rohit Verma", 61000.5, 8000.0, 0.0, 1),
                        (3, "Neha Nair", 47500.0, 1500.0, 300.0, 1),
                        (4, "Dev Menon", 54500.0, 2500.0, 640.0, 3)])])


@pytest.fixture
def named_pair_db(tmp_path):
    """No FK, same column name across tables -> name_match heuristic."""
    return _make_db(
        tmp_path / "named_pair.db",
        ["CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, "
         "employee_name TEXT NOT NULL)",
         "CREATE TABLE payroll (payroll_id INTEGER PRIMARY KEY, "
         "employee_id INTEGER, amount REAL)"],
        [("employees", [(1, "Asha Iyer"), (2, "Rohit Verma")]),
         ("payroll", [(10, 1, 1500.0), (11, 2, 2200.0)])])


@pytest.fixture
def quoted_db(tmp_path):
    """Identifiers with spaces and embedded double quotes must survive."""
    db = tmp_path / "quoted.db"
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE "order details" ('
                 '"row id" INTEGER PRIMARY KEY, '
                 '"weird ""col""" TEXT)')
    conn.execute('INSERT INTO "order details" VALUES (1, \'a quote: "z"\')')
    conn.execute('CREATE TABLE plain_notes ("note id" INTEGER, body TEXT)')
    conn.execute("INSERT INTO plain_notes VALUES (9, 'hello')")
    conn.commit()
    conn.close()
    return db


# --------------------------------------------------------------------------
# db_introspect: schema extraction, samples, quoting
# --------------------------------------------------------------------------

def test_extract_schema_counts_and_samples(salary_only_db):
    schema = di.extract_schema(salary_only_db)
    assert list(schema["tables"]) == ["employees"]
    cols = schema["tables"]["employees"]["columns"]
    assert [c["name"] for c in cols] == ["employee_id", "employee_name", "salary"]
    by_name = {c["name"]: c for c in cols}
    assert by_name["employee_id"]["primary_key"] is True
    assert by_name["employee_name"]["notnull"] is True
    assert by_name["salary"]["primary_key"] is False
    samples = schema["tables"]["employees"]["sample_values"]
    assert len(samples["salary"]) == 3 and all(v in {"52000.0", "61000.5", "47500.0"}
                                               for v in samples["salary"])
    assert samples["employee_name"] == ["'Asha Iyer'", "'Rohit Verma'", "'Neha Nair'"]


def test_extract_schema_fk_and_relationship(pay_mix_db):
    schema = di.extract_schema(pay_mix_db)
    assert sorted(schema["tables"]) == ["departments", "employees"]
    employees = schema["tables"]["employees"]
    assert employees["foreign_keys"] == [
        {"from": "department_id", "table": "departments", "to": "department_id"}]
    fk_rels = [r for r in schema["relationships"] if r["kind"] == "foreign_key"]
    assert len(fk_rels) == 1
    rel = fk_rels[0]
    assert (rel["left_table"], rel["left_column"],
            rel["right_table"], rel["right_column"]) == (
        "employees", "department_id", "departments", "department_id")
    assert rel["join_sql"] == '"employees"."department_id" = "departments"."department_id"'


def test_extract_schema_name_match_relationship(named_pair_db):
    schema = di.extract_schema(named_pair_db)
    matches = [r for r in schema["relationships"] if r["kind"] == "name_match"]
    assert len(matches) == 1
    rel = matches[0]
    assert rel["join_sql"] == '"employees"."employee_id" = "payroll"."employee_id"'
    assert isinstance(tuple(sorted([rel["left_table"], rel["right_table"]])), tuple)


def test_quote_ident_escaping():
    assert di.quote_ident('a"b') == '"a""b"'
    assert di.quote_ident("plain") == '"plain"'


def test_quoted_identifiers_end_to_end(quoted_db):
    schema = di.extract_schema(quoted_db)
    assert sorted(schema["tables"]) == ["order details", "plain_notes"]
    cols = schema["tables"]["order details"]["columns"]
    assert [c["name"] for c in cols] == ["row id", 'weird "col"']
    assert cols[0]["primary_key"] is True
    rendered = di.render_compact(schema, tables=["order details", "plain_notes"])
    assert "order details" in rendered and "weird \"col\"" in rendered
    assert repr('a quote: "z"') in rendered or "a quote:" in rendered
    assert "hello" in rendered


# --------------------------------------------------------------------------
# db_introspect: relevance + metric helper
# --------------------------------------------------------------------------

def test_select_relevant_salary_only(salary_only_db, pay_mix_db):
    s1 = di.extract_schema(salary_only_db)
    assert di.select_relevant(s1, "who earns the highest salary") == ["employees"]
    s2 = di.extract_schema(pay_mix_db)
    # 'highest salary' must not drag departments in, but "by department" must
    assert di.select_relevant(s2, "who earns the highest salary") == ["employees"]
    assert di.select_relevant(s2, "average salary by department") == [
        "departments", "employees"]
    assert di.select_relevant(s2, "attendance percentage of students") == []


def test_select_relevant_sample_value_match(tmp_path):
    db = _make_db(
        tmp_path / "seg.db",
        ["CREATE TABLE staff (staff_id INTEGER PRIMARY KEY, team TEXT, "
         "payout REAL)"],
        [("staff", [(1, "Sales", 5000.0), (2, "Engineering", 7000.0)])])
    schema = di.extract_schema(db)
    # only the sample value 'Sales' connects the question to the table
    assert di.select_relevant(schema, "list the sales people payouts") == ["staff"]
    assert di.select_relevant(schema, "list the marketing people") == []


def test_select_relevant_cap_max_tables(tmp_path):
    db = _make_db(
        tmp_path / "three.db",
        ["CREATE TABLE alpha (alpha_score REAL)",
         "CREATE TABLE bravo (bravo_count INTEGER)",
         "CREATE TABLE charlie (charlie_total REAL)"],
        [("alpha", [(1.0,)]), ("bravo", [(2,)]), ("charlie", [(3.0,)])])
    schema = di.extract_schema(db)
    got = di.select_relevant(schema, "alpha bravo charlie", max_tables=2)
    assert got == ["alpha", "bravo"]  # equal scores break alphabetically
    got4 = di.select_relevant(schema, "alpha bravo charlie", max_tables=4)
    assert got4 == ["alpha", "bravo", "charlie"]


def test_select_relevant_empty_question(salary_only_db):
    schema = di.extract_schema(salary_only_db)
    assert di.select_relevant(schema, "") == []
    assert di.select_relevant(schema, "the and or !!") == []


def test_has_named_metric(pay_mix_db, salary_only_db):
    s = di.extract_schema(pay_mix_db)
    assert di.has_named_metric(s, "who has the highest bonus") == [
        {"table": "employees", "column": "bonus", "match": "bonus"}]
    assert di.has_named_metric(s, "happiest employees") == []
    assert di.has_named_metric(s, "highest") == []   # superlative alone names nothing
    asserts = di.has_named_metric(s, ["salary"])    # token input is accepted
    assert asserts and asserts[0]["column"] == "salary"
    s1 = di.extract_schema(salary_only_db)
    assert di.has_named_metric(s1, "highest bonus") == []  # bonus absent here
    assert di.has_named_metric(s1, "highest salary")[0]["column"] == "salary"


# --------------------------------------------------------------------------
# render_compact
# --------------------------------------------------------------------------

def test_render_compact_content(pay_mix_db):
    schema = di.extract_schema(pay_mix_db)
    text = di.render_compact(schema)
    for expected in ("table employees: employee_id INTEGER PRIMARY KEY",
                     "commission REAL", "samples:", "'Rohit Verma'",
                     "relation: employees.department_id -> "
                     "departments.department_id (foreign_key)"):
        assert expected in text


def test_render_compact_table_filter(quoted_db):
    schema = di.extract_schema(quoted_db)
    text = di.render_compact(schema, tables=["plain_notes"])
    assert "plain_notes" in text and "order details" not in text
    assert di.render_compact(schema, tables=[]) == "(schema: no tables available)"


# --------------------------------------------------------------------------
# robust JSON parsing
# --------------------------------------------------------------------------

def test_parse_slm_json_fences_and_prose():
    verdict = parse_slm_json("```json\n" + SLM_AMBIG + "\n```")
    assert verdict["label"] == "AMBIGUOUS" and len(verdict["interpretations"]) == 2
    assert verdict["interpretations"][1]["relevant_columns"] == ["employees.bonus"]
    prose = parse_slm_json("Here is my answer: " + SLM_CLEAR + " hope that helps")
    assert prose["label"] == "CLEAR"


def test_parse_slm_json_braces_inside_strings_first_object_wins():
    tricky = ("Answer: {\"label\": \"CLEAR\", "
              "\"reason\": \"use { and \\\"} \\\" carefully\"} extra {\"junk\": 1}")
    verdict = parse_slm_json(tricky)
    assert verdict["label"] == "CLEAR"
    assert '"junk"' not in (verdict["reason"] or "") and "carefully" in verdict["reason"]


def test_parse_slm_json_invalid_and_edge_cases():
    assert parse_slm_json("I really cannot produce JSON today.") is None
    assert parse_slm_json("{\"label\": \"MAYBE\"}") is None          # invalid label
    assert parse_slm_json("{\"label\": \"clear\"}")["label"] == "CLEAR"  # case normalised
    assert parse_slm_json('{"label": "CLEAR", "confidence": 0.9}'
                          )["label"] == "CLEAR"
    verdict = parse_slm_json(SLM_UNDER)
    assert verdict["note_on_missing_data"].startswith("schema has no happiness")


# --------------------------------------------------------------------------
# analyze_ambiguity end to end with fake clients
# --------------------------------------------------------------------------

def test_analyze_clear_agreement(salary_only_db):
    cls = FakeClassifier("CLEAR", {"CLEAR": 0.9, "AMBIGUOUS": 0.05,
                                   "UNDERSPECIFIED": 0.05})
    slm = FakeSLM(SLM_CLEAR)
    d = analyze_ambiguity("who earns the highest salary", salary_only_db,
                          classifier=cls, slm_client=slm)
    assert d.label == "CLEAR" and d.decision_rule == "agreement"
    assert d.options == [] and d.clarify_question == ""
    assert d.evidence["classifier"]["probs"]["CLEAR"] == 0.9
    assert d.evidence["slm"]["label"] == "CLEAR"
    assert isinstance(d, AmbiguityDecision)
    # the SLM saw the rendered schema, not a literal hardcode
    user_msg = slm.messages[-1]["content"]
    assert "employees" in user_msg and "salary" in user_msg
    assert 'salary' in cls.seen_calls[0][1]


def test_analyze_ambiguous_pay_mix_agreement(pay_mix_db):
    cls = FakeClassifier("AMBIGUOUS", {"CLEAR": 0.2, "AMBIGUOUS": 0.6,
                                       "UNDERSPECIFIED": 0.2})
    slm = FakeSLM("```json\n" + SLM_AMBIG + "\n```")
    d = analyze_ambiguity("who has the highest pay", pay_mix_db,
                          classifier=cls, slm_client=slm)
    assert d.label == "AMBIGUOUS" and d.decision_rule == "agreement"
    assert [o["option"] for o in d.options] == ["highest base salary",
                                                "highest bonus"]
    assert d.options[0]["note"] == "uses employees.salary"
    assert d.clarify_question == "Did you mean base salary or a bonus?"


def test_analyze_conflict_conservative_when_slm_unclear(salary_only_db):
    d = analyze_ambiguity("who earns the highest salary", salary_only_db,
                          classifier=FakeClassifier("CLEAR", {"CLEAR": 0.8,
                                                              "AMBIGUOUS": 0.2}),
                          slm_client=FakeSLM("```json\n" + SLM_AMBIG + "\n```"))
    assert d.label == "AMBIGUOUS"          # UNCLEAR wins
    assert d.decision_rule == "conflict->clarify"
    assert len(d.options) == 2


def test_analyze_conflict_conservative_when_classifier_unclear(pay_mix_db):
    # Deliberate semantics change (2026-10-30, live-bug fix): a lone AMBIGUOUS
    # classifier can no longer force a clarify when the SLM saw the schema and
    # returned CLEAR — that path carried no chips and dead-ended the UI
    # ("highest base salary" looped forever). The SLM's supported reading wins.
    d = analyze_ambiguity("who has the highest base salary", pay_mix_db,
                          classifier=FakeClassifier("AMBIGUOUS",
                                                    {"CLEAR": 0.3, "AMBIGUOUS": 0.62}),
                          slm_client=FakeSLM(SLM_CLEAR))
    assert d.label == "CLEAR"
    assert d.decision_rule == "slm-clear-overrides"


def test_analyze_classifier_clear_still_yields_to_slm_care(pay_mix_db):
    # The conservative direction is preserved: when the classifier's CLEAR is
    # outvoted by an SLM non-CLEAR label, clarify still fires (with the SLM's
    # chips).
    d = analyze_ambiguity("who has the highest pay", pay_mix_db,
                          classifier=FakeClassifier("CLEAR", {"CLEAR": 0.8,
                                                              "AMBIGUOUS": 0.2}),
                          slm_client=FakeSLM(SLM_AMBIG))
    assert d.label == "AMBIGUOUS"
    assert d.decision_rule == "conflict->clarify"
    assert len(d.options) == 2


def test_analyze_no_named_metric_keeps_clarify(pay_mix_db):
    # "best performers" names no stored metric: the lone classifier-AMBIGUOUS
    # does NOT get overridden — this is the demo's flagship clarify path.
    d = analyze_ambiguity("show me the best folks around", pay_mix_db,
                          classifier=FakeClassifier("AMBIGUOUS",
                                                    {"CLEAR": 0.2, "AMBIGUOUS": 0.7}),
                          slm_client=FakeSLM(SLM_CLEAR))
    assert d.label == "AMBIGUOUS"
    assert d.decision_rule == "conflict->clarify"


def test_analyze_chitchat_short_circuits_models(salary_only_db):
    # "hi" / "?" must never reach the classifier or the SLM (they misread it as
    # an ambiguous SQL ask and burned ~7 s returning an unanswerable card).
    cls = FakeClassifier("AMBIGUOUS", {"AMBIGUOUS": 0.9})
    slm = FakeSLM(SLM_AMBIG)
    d = analyze_ambiguity("hi", salary_only_db, classifier=cls, slm_client=slm)
    assert d.label == "UNDERSPECIFIED" and d.decision_rule == "chitchat-rule"
    assert cls.seen_calls == [] and slm.blocks == []   # zero model calls
    assert len(d.options) == 2
    assert "employees" in d.options[0]["option"]
    assert d.clarify_question.startswith("Hi!")
    d2 = analyze_ambiguity("?", salary_only_db, classifier=cls, slm_client=slm)
    assert d2.label == "UNDERSPECIFIED" and d2.decision_rule == "chitchat-rule"
    d3 = analyze_ambiguity("thanks a lot", salary_only_db, classifier=cls,
                           slm_client=slm)
    assert d3.decision_rule == "chitchat-rule"
    # a real question is never mistaken for chit-chat
    d4 = analyze_ambiguity("ok now show the highest salary", salary_only_db,
                           classifier=cls, slm_client=slm)
    assert d4.decision_rule != "chitchat-rule"


def test_split_questions_multi_line_compound():
    # the phrasing from the live bug report: compound ask spread over newlines
    text = ("For each product category in my uploaded data,\n"
            "what is the total quantity in stock,\n"
            "and which category has the largest quantity? "
            "Use whatever tables you need.")
    questions = split_questions(text)
    assert len(questions) == 2
    assert questions[0] == "what is the total quantity in stock"
    assert questions[1] == "and which category has the largest quantity?"
    assert split_questions("Show me the best players") == []
    assert split_questions("How many orders are pending?") == []


def test_analyze_conflicting_unclear_labels_go_underspecified(pay_mix_db):
    d = analyze_ambiguity("totally unclear request", pay_mix_db,
                          classifier=FakeClassifier("AMBIGUOUS",
                                                    {"AMBIGUOUS": 0.7}),
                          slm_client=FakeSLM(SLM_UNDER))
    assert d.label == "UNDERSPECIFIED"
    assert d.decision_rule == "conflict->clarify"
    assert d.clarify_question == "Which stored metric should I use instead?"


def test_analyze_weak_classifier_defers_to_slm_clear(pay_mix_db):
    d = analyze_ambiguity("who earns the highest salary", pay_mix_db,
                          classifier=FakeClassifier("AMBIGUOUS",
                                                    {"CLEAR": 0.5, "AMBIGUOUS": 0.36}),
                          slm_client=FakeSLM(SLM_CLEAR))
    assert d.label == "CLEAR"
    assert d.decision_rule == "confidence-threshold"


def test_analyze_weak_underspecified_classifier_defers_to_slm(salary_only_db):
    d = analyze_ambiguity("who earns the highest salary", salary_only_db,
                          classifier=FakeClassifier("UNDERSPECIFIED",
                                                    {"CLEAR": 0.55,
                                                     "UNDERSPECIFIED": 0.41}),
                          slm_client=FakeSLM(SLM_CLEAR))
    assert d.label == "CLEAR" and d.decision_rule == "confidence-threshold"


def test_analyze_invalid_slm_json_falls_back_to_classifier(salary_only_db):
    slm = FakeSLM("```\nnot json at all\n```")
    d = analyze_ambiguity("who earns the highest salary", salary_only_db,
                          classifier=FakeClassifier("CLEAR", {"CLEAR": 0.9}),
                          slm_client=slm)
    assert d.evidence["slm"] is None           # parse failure -> unusable SLM
    assert d.label == "CLEAR" and d.decision_rule == "classifier-only"
    assert "slm unavailable" in d.reason


def test_analyze_slm_error_keeps_classifier(salary_only_db):
    d = analyze_ambiguity("who earns the highest salary", salary_only_db,
                          classifier=FakeClassifier("UNDERSPECIFIED",
                                                    {"UNDERSPECIFIED": 0.7}),
                          slm_client=BrokenSLM())
    assert d.evidence["slm"] is None and "slm_error" in d.evidence
    assert d.label == "UNDERSPECIFIED" and d.decision_rule == "classifier-only"


def test_analyze_slm_only(pay_mix_db):
    d = analyze_ambiguity("who are the happiest employees", pay_mix_db,
                          classifier=None, slm_client=FakeSLM(SLM_UNDER))
    assert d.label == "UNDERSPECIFIED" and d.decision_rule == "slm-only"
    assert d.options == []
    assert d.clarify_question == "Which stored metric should I use instead?"
    assert "happiness" in d.reason


def test_analyze_no_signals_fallback_heuristic(pay_mix_db):
    d = analyze_ambiguity("who earns the highest salary", pay_mix_db)
    assert d.label == "CLEAR" and d.decision_rule == "fallback-heuristic"
    d2 = analyze_ambiguity("who are the happiest employees", pay_mix_db)
    assert d2.label == "UNDERSPECIFIED" and d2.decision_rule == "fallback-heuristic"
    d3 = analyze_ambiguity("attendance percentage of students", pay_mix_db)
    assert d3.label == "UNDERSPECIFIED"   # no table matches at all


def test_analyze_no_database_and_empty_question(tmp_path):
    d = analyze_ambiguity("who earns the most", tmp_path / "missing.db")
    assert d.label == "UNDERSPECIFIED" and d.decision_rule == "no-database"
    assert d.evidence == {"classifier": None, "slm": None}
    empty = analyze_ambiguity("   ", tmp_path / "whatever.db")
    assert empty.label == "UNDERSPECIFIED" and empty.decision_rule == "empty-question"


def test_slm_prompt_encodes_label_policy():
    messages = build_slm_messages("who earns the highest salary", "table employees: salary REAL")
    assert "AMBIGUOUS" in messages[1]["content"]
    assert "happiest" in messages[1]["content"]          # the canonical example
    assert "Never invent" in messages[1]["content"]
    assert "json" in messages[0]["content"].lower()
