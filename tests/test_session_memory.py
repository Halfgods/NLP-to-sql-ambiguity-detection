"""Boundary tests for the worker's in-process conversation memory."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


SERVER_DIR = Path(__file__).resolve().parents[1] / "UI" / "server"
sys.path.insert(0, str(SERVER_DIR))
_stdout = sys.stdout
try:
    model_worker = importlib.import_module("model_worker")
finally:
    # The JSON-lines worker redirects stdout to stderr when run as a process;
    # importing it for unit tests must not change pytest's output stream.
    sys.stdout = _stdout


def test_session_cap_is_eight_and_existing_session_is_retained(monkeypatch):
    model_worker.SESSION_MEMORY.clear()
    ticks = iter(float(n) for n in range(100))
    monkeypatch.setattr(model_worker.time, "monotonic", lambda: next(ticks))

    for i in range(8):
        model_worker._session(f"tab-{i}")
    retained = model_worker.SESSION_MEMORY["tab-0"]
    assert len(model_worker.SESSION_MEMORY) == 8

    assert model_worker._session("tab-0") is retained
    assert len(model_worker.SESSION_MEMORY) == 8
    model_worker._session("tab-8")
    assert len(model_worker.SESSION_MEMORY) == 8
    assert "tab-0" not in model_worker.SESSION_MEMORY


def test_expired_session_is_replaced(monkeypatch):
    model_worker.SESSION_MEMORY.clear()
    now = [10.0]
    monkeypatch.setattr(model_worker.time, "monotonic", lambda: now[0])
    first = model_worker._session("tab")
    first["question"] = "old question"
    first["updated"] = now[0]

    now[0] += model_worker.SESSION_TTL_S + 0.1
    fresh = model_worker._session("tab")
    assert fresh is not first
    assert "question" not in fresh
    assert len(model_worker.SESSION_MEMORY) == 1


def test_typed_clarification_matches_worker_chip_labels():
    analysis = SimpleNamespace(options=[
        {"label": "Average marks", "note": "uses students.average_marks"},
        {"label": "Assignment score", "note": "uses grades.assignment_score"},
    ])
    session = {"pending_clarify": True, "analysis": analysis}

    assert model_worker._typed_clarify_answer(session, "average marks") == "Average marks"
    assert model_worker._typed_clarify_answer(session, "Assignment score please") == "Assignment score"
    assert model_worker._typed_clarify_answer(session, "show all students") is None
    session["pending_clarify"] = False
    assert model_worker._typed_clarify_answer(session, "Average marks") is None


def test_answer_payload_flags_model_give_up_sentinel():
    result = SimpleNamespace(
        columns=["1"], rows=[(1,)], error=None,
        sql="SELECT 1 WHERE 0", tool_calls=[])
    payload = model_worker._answer_payload(
        result, SimpleNamespace(reason="r", source="rule_baseline",
                                to_dict=lambda: {}),
        None, 0.0)
    assert "couldn't turn that into a meaningful SQL query" in payload["reply"]


def test_multi_question_message_answers_every_sub_question(monkeypatch, tmp_path):
    """Compound messages ("1. …? 2. …?") run each sub-question through the
    standard pipeline — no clarify card round-trip between the two asks."""
    model_worker.SESSION_MEMORY.clear()

    def fake_analyze(question, db_path=None):
        ambiguous = "ordered the most" in question
        return SimpleNamespace(
            is_ambiguous=ambiguous,
            reason="relevant tables: orders",
            source="ambiguity:test",
            options=[],
            to_dict=lambda: {"decision": "ambiguous" if ambiguous else "clear"},
        )

    def fake_answer(question, db_path=None, on_progress=None):
        return SimpleNamespace(
            columns=["n"], rows=[(len(question),)], error=None,
            sql=f"SELECT 1 -- {question}", tool_calls=[])

    monkeypatch.setattr(model_worker, "analyze_question", fake_analyze)
    monkeypatch.setattr(model_worker, "answer_question", fake_answer)

    payload = model_worker._query({
        "message": "1. How many orders are pending?\n"
                   "2. Which customer has ordered the most by amount?",
        "dbPath": str(tmp_path / "workspace.db"),
    })
    assert payload["status"] == "multi"
    answers = payload["answers"]
    assert answers[0]["status"] == "ok" and answers[0]["sql"].endswith(
        "-- How many orders are pending?")
    # an ambiguous sub-question can't strand a card mid-batch
    assert answers[1]["status"] == "clarify_needed"
    assert "ordered the most by amount?" in answers[1]["reply"]

    session = model_worker._session("default")
    assert session["question"].startswith("Which customer")  # last sub kept


def test_referential_follow_up_requires_reference_and_prior_question():
    session = {"question": "Show students in Computer", "clarification": "active students"}

    context = model_worker._referential_context(session, "and their pending fees too")
    assert "Show students in Computer" in context
    assert "active students" in context
    assert model_worker._referential_context(session, "Show pending fees") is None
    assert model_worker._referential_context({}, "show those students") is None


def test_ambiguous_reply_explains_the_competing_readings():
    analysis = SimpleNamespace(
        options=[
            {"label": "Average marks", "note": "uses average_marks"},
            {"label": "Assignment score", "note": "uses assignment_score"},
        ],
        to_dict=lambda: {"decision": "ambiguous",
                         "clarify_question": "Which metric should I use?"},
        reason="internal classifier diagnostics",
    )

    payload = model_worker._clarify_payload(analysis)
    assert "Average marks" in payload["reply"]
    assert "Assignment score" in payload["reply"]
    assert "Which metric should I use?" in payload["reply"]
    assert "internal classifier diagnostics" not in payload["reply"]
    assert payload["ambiguities"][0]["options"][0]["label"] == "Average marks"


def test_query_typed_answer_replays_original_question(monkeypatch, tmp_path):
    class Analysis:
        source = "ambiguity:agreement"
        reason = "two metrics are supported"
        options = [{"label": "Average marks", "note": ""}]
        is_ambiguous = True

        def to_dict(self):
            return {"decision": "ambiguous", "clarify_question": "Which metric?"}

    analysis = Analysis()
    calls = []
    result = SimpleNamespace(error=None, columns=["name"], rows=[["Asha"]],
                             sql="SELECT name FROM students", tool_calls=[])
    model_worker.SESSION_MEMORY.clear()
    model_worker.ANALYSIS_CACHE.clear()
    monkeypatch.setattr(model_worker, "analyze_question", lambda *a, **k: analysis)
    monkeypatch.setattr(model_worker, "answer_question",
                        lambda *a, **k: calls.append((a, k)) or result)
    db = tmp_path / "query.db"
    db.touch()

    opened = model_worker._query({"sessionId": "typed", "dbPath": str(db),
                                  "message": "Show the best students"})
    answered = model_worker._query({"sessionId": "typed", "dbPath": str(db),
                                    "message": "Average marks"})

    assert opened["status"] == "clarify"
    assert answered["status"] == "ok"
    assert calls[0][0][0] == "Show the best students"
    assert calls[0][1]["clarification"] == "Average marks"


def test_query_referential_follow_up_carries_prior_turn(monkeypatch, tmp_path):
    class Analysis:
        source = "ambiguity:agreement"
        reason = "clear"
        options = []
        is_ambiguous = False

        def to_dict(self):
            return {"decision": "clear", "reason": self.reason}

    calls = []
    result = SimpleNamespace(error=None, columns=["name"], rows=[["Asha"]],
                             sql="SELECT name FROM students", tool_calls=[])
    model_worker.SESSION_MEMORY.clear()
    model_worker.ANALYSIS_CACHE.clear()
    monkeypatch.setattr(model_worker, "analyze_question", lambda *a, **k: Analysis())
    monkeypatch.setattr(model_worker, "answer_question",
                        lambda *a, **k: calls.append((a, k)) or result)
    db = tmp_path / "followup.db"
    db.touch()

    model_worker._query({"sessionId": "followup", "dbPath": str(db),
                         "message": "Show students in Computer"})
    response = model_worker._query({"sessionId": "followup", "dbPath": str(db),
                                    "message": "and their pending fees too"})

    assert response["referential"] is True
    assert "Show students in Computer" in calls[1][1]["clarification"]
