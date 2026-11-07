"""JSON-lines worker that exposes the real NL2SQL pipeline to the React UI.

The Vite dev server spawns this file once (see ``model_bridge.js``) and keeps it
alive, so Qwen is loaded once instead of once per request. stdin/stdout carry
one JSON object per line:

    {"id": 1, "action": "warmup"}
    {"id": 2, "action": "query", "message": "...", "answers": {"interpretation": 0},
     "dbPath": "/abs/path/to/clarion.db"}

``sessionId`` (optional, string) scopes conversational intent memory: resolv-
ing typed replies to an open clarify card and keyword-gated referential
follow-ups ("and their bonus too"). One id per UI tab; the memory stores turn
metadata only (question text, last clarification label, card state), is
TTL'd (30 min) and capped at 8 concurrent sessions. Omitting it uses a
shared "default" scope.

Responses are ``{"id": ..., "ok": true, "data": ...}`` or
``{"id": ..., "ok": false, "error": "..."}``.

Only the protocol may use stdout; everything else is redirected to stderr so
library banners can never corrupt a response.
"""

import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr  # transformers/torch banners must not hit the protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import (  # noqa: E402
    InputError,
    analyze_question,
    answer_question,
    confirm_write,
    explicitly_requests_write,
    is_schema_question,
    propose_write,
    split_write_followup,
    schema_summary,
    validate_question,
)
from ambiguity.analysis import split_questions  # noqa: E402
import pipeline as _pipeline  # noqa: E402
parse_select_tree = _pipeline.parse_select_tree

INTERPRETATION_ID = "interpretation"
PROCEED_CHIP = "Proceed with the most literal reading"
ANALYSIS_CACHE: dict = {}
ANALYSIS_CACHE_LIMIT = 32

# ---- session-scoped conversational intent memory (P1-4) --------------------
# Keyed by the UI-provided sessionId; only ever holds turn metadata (question
# text, last clarification label, pending clarify card) — never data rows.
SESSION_MEMORY: dict = {}
SESSION_LIMIT = 8
SESSION_TTL_S = 30 * 60
_REFERENTIAL_RE = re.compile(
    r"\b(this|that|those|these|it|its|they|them|same|again|also|too|previous|"
    r"above|earlier|last one|last query|last result|those students|the same)\b",
    re.IGNORECASE)
MODEL_INFO: dict = {}
PENDING_WRITES: dict = {}
WRITE_CONFIRM_TTL_S = 300


def _send(payload: dict) -> None:
    PROTOCOL_OUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL_OUT.flush()


def _ok(request_id, data: dict) -> None:
    _send({"id": request_id, "ok": True, "data": data})


def _error(request_id, message: str, code: str = "error") -> None:
    _send({"id": request_id, "ok": False, "error": message, "code": code})


def _db_path(raw) -> Path:
    if raw:
        return Path(raw).resolve()
    default = PROJECT_ROOT / "UI" / "uploads" / "clarion.db"
    if default.exists():
        return default
    return PROJECT_ROOT / "data" / "people.db"


def _db_stamp(path: Path) -> tuple:
    """Invalidate cached analyses when the workspace database changes."""
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return 0, 0


def _cache_key(question: str, path: Path) -> tuple:
    return question, str(path), _db_stamp(path)


def _session(session_id: str) -> dict:
    """Resolve or create the session-memory entry, pruning stale sessions."""
    now = time.monotonic()
    for key, entry in list(SESSION_MEMORY.items()):
        if entry["updated"] + SESSION_TTL_S < now:
            SESSION_MEMORY.pop(key, None)
    entry = SESSION_MEMORY.get(session_id)
    if entry is not None and entry["updated"] + SESSION_TTL_S >= now:
        return entry
    while len(SESSION_MEMORY) >= SESSION_LIMIT:
        oldest = min(SESSION_MEMORY, key=lambda k: SESSION_MEMORY[k]["updated"])
        SESSION_MEMORY.pop(oldest, None)
    entry = {"updated": now, "turn": 0}
    SESSION_MEMORY[session_id] = entry
    return entry


def _remember_turn(session: dict, *, question: str, analysis=None,
                   clarification: str | None = None, pending_clarify: bool,
                   status: str):
    session.update({
        "question": question,
        "analysis": getattr(analysis, "options", None) and analysis or None,
        "clarification": clarification,
        "pending_clarify": pending_clarify,
        "status": status,
        "updated": time.monotonic(),
        "turn": session.get("turn", 0) + 1,
    })


def _typed_clarify_answer(session: dict, text: str) -> str | None:
    """When the last card of this session asked for a clarification and the
    user simply *typed* one of the offered readings, resolve it to that
    option's label."""
    if not session.get("pending_clarify"):
        return None
    analysis = session.get("analysis")
    if analysis is None:
        return None
    options = getattr(analysis, "options", None) or []
    norm = text.strip().casefold()
    if not norm:
        return None
    for opt in options:
        label = (opt.get("label") if isinstance(opt, dict) else str(opt)
                 or "").strip().casefold()
        if not label:
            continue
        if norm == label or (norm in label) or (label in norm and len(label) >= 8):
            return (opt.get("label") if isinstance(opt, dict) else str(opt))
    return None


def _referential_context(session: dict, question: str) -> str | None:
    """Compose a carry-over hint when a follow-up clearly references the
    previous turn ("and their bonus too"). Strictly keyword-gated — no
    speculative rewriting of the user's words."""
    if not _REFERENTIAL_RE.search(question):
        return None
    last_q = session.get("question")
    if not last_q or last_q.casefold() == question.strip().casefold():
        return None
    parts = [f"The user's message refers to their previous question: "
             f"'{last_q}'."]
    if session.get("clarification"):
        parts.append(f"That question was clarified as: "
                     f"'{session['clarification']}'.")
    return " ".join(parts)


def _remember_clear_pending(session: dict):
    if session.get("pending_clarify"):
        session["pending_clarify"] = False


def _answer_payload(result, analysis, clarification, started) -> dict:
    """Build the normal (non-clarify, non-write) reply payload."""
    detector = _detector_payload(analysis) if analysis is not None else None
    if clarification:
        summary = f"Interpretation: {clarification}"
    elif analysis is not None:
        summary = analysis.reason
    else:
        summary = None

    if result.error:
        return {
            "status": "ok",
            "reply": "I couldn't run that safely, so nothing was executed. "
                     + result.error,
            "summary": "Generated SQL was rejected before execution",
            "sql": None,
            "explanation": None,
            "results": None,
            "clarification": clarification,
            "error": result.error,
            "tools": result.tool_calls,
            "ambiguities": [],
            "detector": detector,
            "latency_s": round(time.perf_counter() - started, 3),
        }

    columns = result.columns or []
    rows = [
        {column: value for column, value in zip(columns, row)}
        for row in (result.rows or [])
    ]
    if analysis is not None and analysis.source == "schema_meta":
        reply = "Here are the tables and columns in the connected database."
    elif result.sql and re.match(r"^\s*SELECT\s+1\s+WHERE\s+0\b",
                                 result.sql, re.IGNORECASE):
        # The system prompt tells the model to answer unknown requests with
        # this never-matching sentinel — surface it as "no answer" honestly.
        reply = ("The model couldn't turn that into a meaningful SQL query "
                 "(safe fallback executed — 0 rows means *no answer*, not "
                 "*no data*). Try naming the table and column.")
    else:
        reply = (f"I ran a read-only query and got {len(rows)} row"
                 f"{'' if len(rows) == 1 else 's'}.")
    explanation = _explain_read_query(result.sql) if result.sql else None
    return {
        "status": "ok",
        "reply": reply,
        "summary": summary,
        "sql": result.sql,
        "explanation": explanation,
        "results": {"columns": columns, "rows": rows},
        "clarification": clarification,
        "error": None,
        "tools": result.tool_calls,
        "ambiguities": [],
        "detector": detector,
        "latency_s": round(time.perf_counter() - started, 3),
    }


def _explain_read_query(sql: str) -> str | None:
    """Describe the validated query's main SELECT without another model call."""
    try:
        sqlx = getattr(_pipeline, "sqlx", None)
        if sqlx is None:
            return None
        expression, error = parse_select_tree(sql)
        if error or expression is None or not isinstance(expression, sqlx.Select):
            return None
        parts = []
        selected = [item.sql(dialect="sqlite") for item in expression.expressions]
        if selected and selected != ["*"]:
            parts.append("returns " + ", ".join(f"`{item}`" for item in selected))
        else:
            parts.append("returns matching rows")

        from_clause = expression.args.get("from_")
        tables = []
        if from_clause is not None:
            tables.extend(table.name for table in from_clause.find_all(sqlx.Table))
        for join in expression.args.get("joins") or []:
            tables.extend(table.name for table in join.find_all(sqlx.Table))
        tables = list(dict.fromkeys(name for name in tables if name))
        if tables:
            parts.append("from " + ", ".join(f"`{name}`" for name in tables))

        where = expression.args.get("where")
        if where is not None:
            parts.append("filters rows with `" + where.this.sql(dialect="sqlite") + "`")
        group = expression.args.get("group")
        if group is not None:
            parts.append("groups by " + ", ".join(
                f"`{item.sql(dialect='sqlite')}`" for item in group.expressions))
        order = expression.args.get("order")
        if order is not None:
            parts.append("sorts by " + ", ".join(
                f"`{item.sql(dialect='sqlite')}`" for item in order.expressions))
        limit = expression.args.get("limit")
        if limit is not None:
            parts.append("limits the output to " + limit.expression.sql(dialect="sqlite") + " rows")
        return "This query " + "; ".join(parts) + "."
    except Exception:
        # Explanations are a usability aid; they must never affect safe execution.
        return None


def _cache_put(key: tuple, analysis) -> None:
    if len(ANALYSIS_CACHE) >= ANALYSIS_CACHE_LIMIT:
        ANALYSIS_CACHE.pop(next(iter(ANALYSIS_CACHE)))
    ANALYSIS_CACHE[key] = analysis


def _detector_payload(analysis) -> dict:
    return analysis.to_dict()


def _clarify_payload(analysis) -> dict:
    chips = [o if isinstance(o, dict) else {"label": o, "note": ""}
             for o in analysis.options]
    extra = getattr(analysis, "to_dict", lambda: {})()
    clarify_q = extra.get("clarify_question", "") or ""
    # Never dead-end: an AMBIGUOUS card with zero chips (SLM parse failure,
    # classifier-only dissent) leaves the user nothing to click, so synthesize
    # a proceed chip that can always complete the turn. The chips are written
    # back onto the analysis so `_selected_clarification` can resolve the
    # click later (they did not exist in the original options list).
    if not chips and extra.get("decision") == "ambiguous":
        chips = [{"label": PROCEED_CHIP,
                  "note": "SQL will be written exactly as asked"}]
        analysis._options = chips
    question = clarify_q or "Which reading did you mean?"
    labels = [str(chip.get("label", "")).strip().replace("\n", " ")[:160]
              for chip in chips if str(chip.get("label", "")).strip()]
    if extra.get("decision") == "ambiguous" and len(labels) >= 2:
        choices = "; ".join(labels[:2])
        if len(labels) > 2:
            choices += "; and other readings"
        reply = (f"I found multiple schema-supported readings: {choices}. "
                 f"{question}")
    else:
        reply = clarify_q or (
            "Before I write SQL, I need to pin down the meaning: "
            + analysis.reason)
    return {
        "status": "clarify",
        "reply": reply,
        "summary": None,
        "sql": None,
        "results": None,
        "clarification": None,
        "error": None,
        "tools": [],
        "ambiguities": [{
            "id": INTERPRETATION_ID,
            "kind": "interpretation",
            "question": question,
            "options": chips,
        }],
        "detector": _detector_payload(analysis),
    }


def _selected_clarification(analysis, answers: dict):
    """Resolve the clicked option (or a custom string) from the UI answers."""
    value = answers.get(INTERPRETATION_ID)
    if value is None and len(answers) == 1:
        value = next(iter(answers.values()))
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if not value.isdigit():
            return value
        value = int(value)

    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= index < len(analysis.options):
        option = analysis.options[index]
        if isinstance(option, dict):
            return option.get("label", "")
        return option
    return None


def _schema_reply(path: Path, started: float) -> dict:
    """Schema-shaped questions are answered by introspection, not SQL."""
    try:
        text = schema_summary(path)
    except Exception as exc:  # noqa: BLE001 - introspection failure is reply text
        text = (f"I could not read the schema from {path.name}: "
                f"{type(exc).__name__}. Try re-importing the dataset.")
    return {
        "status": "ok",
        "reply": text,
        "summary": "Read-only schema introspection; no SQL was generated",
        "sql": None,
        "results": None,
        "clarification": None,
        "error": None,
        "tools": [],
        "ambiguities": [],
        "detector": {"decision": "clear", "source": "schema_meta",
                     "reason": "schema request answered by read-only introspection"},
        "latency_s": round(time.perf_counter() - started, 3),
    }


def _multi_answer(sub_questions, session, path, report_progress, started):
    """Run every sub-question of a compound message as its own query.

    Returns {"status": "multi", "answers": [payload...]}; the UI renders one
    chat bubble per sub-answer (SQL + rows). Sub-questions that would elicit a
    clarify card are reported as clarify_needed instead — a card halfway
    through a batch would strand the user's remaining questions.
    """
    payloads = []
    total = len(sub_questions)
    for index, sub in enumerate(sub_questions, start=1):
        if report_progress is not None:
            def sub_progress(event, prefix=f"{index}/{total}"):
                if event.get("type") == "stage" and event.get("label"):
                    event = {**event, "label": f"{prefix}: {event['label']}"}
                report_progress(event)
        else:
            sub_progress = None
        payloads.append(_run_single_question(
            sub, path, sub_progress, time.perf_counter()))
    # Session memory references the last sub-question; follow-up pronouns
    # ("and their fees too") attach to the most recent ask.
    _remember_turn(session, question=sub_questions[-1], analysis=None,
                   clarification=None, pending_clarify=False, status="ok")
    return {"status": "multi", "answers": payloads,
            "latency_s": round(time.perf_counter() - started, 3)}


def _run_single_question(question, path, on_progress, started) -> dict:
    """The standard read pipeline for one sub-question of a compound message."""
    key = _cache_key(question, path)
    analysis = ANALYSIS_CACHE.get(key)
    if analysis is None:
        analysis = analyze_question(question, db_path=path)
        _cache_put(key, analysis)
    if analysis.is_ambiguous:
        return {
            "status": "clarify_needed",
            "reply": (f"I can't answer \u201c{question}\u201d as part of a batch: "
                      "it needs clarification first — send it on its own."),
            "summary": None, "sql": None, "results": None, "error": None,
            "tools": [], "detector": _detector_payload(analysis),
            "latency_s": round(time.perf_counter() - started, 3),
        }
    result = answer_question(question, db_path=path, on_progress=on_progress)
    return _answer_payload(result, analysis, None, started)


def _query(request: dict, request_id=None) -> dict:
    started = time.perf_counter()
    question = validate_question(request.get("message"))
    path = _db_path(request.get("dbPath"))
    progress_enabled = bool(request.get("stream")) and request_id is not None
    session_id = (request.get("sessionId")
                  if isinstance(request.get("sessionId"), str)
                  else "default")
    session = _session(session_id)

    now = time.monotonic()
    for key, staged in list(PENDING_WRITES.items()):
        if staged["expires"] < now:
            PENDING_WRITES.pop(key, None)
    outstanding = next((entry for entry in PENDING_WRITES.values()
                        if entry["db_path"] == str(path)), None)
    if outstanding:
        return {"status": "write_pending",
                "reply": "There is an unconfirmed database change. Confirm or cancel its preview before starting another request.",
                "sql": None, "results": None}
    def report_progress(event):
        if progress_enabled:
            _send({"id": request_id, "progress": event})

    report_progress({"type": "stage", "label": "Checking schema and interpreting the question"})
    if is_schema_question(question):
        _remember_turn(session, question=question, analysis=None,
                       clarification=None, pending_clarify=False,
                       status="schema")
        return _schema_reply(path, started)
    # P1-4 case B: the user typed a reading instead of clicking a chip on the
    # previous clarify card → answer the ORIGINAL question with that term.
    # A chip click / card-input reply re-sends the original question (with the
    # `answers` payload), so Case B only fires for plain new messages that
    # match an offered reading.
    if not explicitly_requests_write(question):
        answers = request.get("answers") or {}
        if not isinstance(answers, dict):
            answers = {}
        typed = None
        if (not answers and session.get("pending_clarify")
                and session.get("question")
                and question.casefold() != session["question"].casefold()):
            typed = _typed_clarify_answer(session, question)
        if typed:
            if typed == PROCEED_CHIP:
                typed = ("the user chose to proceed with the question "
                         "exactly as written")
            original = session["question"]
            analysis = session["analysis"]
            session["pending_clarify"] = False
            report_progress({"type": "stage", "label":
                             "Using your answer to the previous question"})
            result = answer_question(
                original, clarification=typed,
                db_path=path, on_progress=report_progress
                if progress_enabled else None)
            _remember_turn(session, question=original, analysis=analysis,
                           clarification=typed, pending_clarify=False,
                           status="ok_typed_answer")
            return _answer_payload(result, analysis, typed, started)

    if explicitly_requests_write(question):
        report_progress({"type": "stage", "label": "Preparing a rollback-only write preview"})
        write_question, followup = split_write_followup(question)
        try:
            proposal = propose_write(write_question, db_path=path)
        except InputError as exc:
            return {"status": "ok",
                    "reply": "I couldn't prepare that change safely. " + str(exc),
                    "summary": "No data was changed", "sql": None,
                    "results": None, "error": str(exc)}
        token = secrets.token_urlsafe(24)
        PENDING_WRITES[token] = {
            "sql": proposal["sql"], "question": write_question,
            "followup": followup, "db_path": str(path),
            "expires": time.monotonic() + WRITE_CONFIRM_TTL_S,
            "db_fingerprint": proposal.get("db_fingerprint"),
        }
        # Bound memory and discard expired confirmations.
        now = time.monotonic()
        for key, staged in list(PENDING_WRITES.items()):
            if staged["expires"] < now or len(PENDING_WRITES) > 32:
                PENDING_WRITES.pop(key, None)
        preview = proposal.get("preview") or {}
        reply = (f"This explicit write request would affect "
                 f"{proposal['affected_rows']} row"
                 f"{'s' if proposal['affected_rows'] != 1 else ''}. "
                 "The preview was rolled back; no data has changed. "
                 "Confirm below to apply it.")
        if preview.get("duplicate_existing"):
            reply += (f" Warning: identical row(s) already exist in "
                      f"`{preview.get('table_name', 'the table')}` — "
                      "the statement may duplicate data.")
        elif preview.get("net_zero_change"):
            reply += " Note: the values would end up identical to what is already stored (no net change)."
        if re.search(r"\band\s+(?:a\s+new\b|another\b|an\s+additional\b|"
                     r"also\b|add\b|insert\b|update\b|change\b|modify\b)",
                     question, re.IGNORECASE):
            reply += (" Note: your message asked for more than one change — "
                      "only one statement per turn is staged. Confirm this, "
                      "then send the remaining change separately.")
        id_note = " ".join(_pipeline.insert_id_warnings(proposal["sql"], path))
        if id_note:
            reply += f" Warning: {id_note}"
        _remember_turn(session, question=question, analysis=None,
                       clarification=None, pending_clarify=False,
                       status="write_preview")
        return {
            "status": "confirm_write",
            "reply": reply,
            "summary": "Rollback-only preview",
            "sql": proposal["sql"],
            "affected_rows": proposal["affected_rows"],
            "preview": {
                "columns": preview.get("columns") or [],
                "before": preview.get("before") or [],
                "after": preview.get("after") or [],
                "rows_omitted": bool(preview.get("rows_omitted")),
                "duplicate_existing": bool(preview.get("duplicate_existing")),
                "net_zero_change": bool(preview.get("net_zero_change")),
            },
            "write_token": token,
            "latency_s": round(time.perf_counter() - started, 3),
        }
    # Compound messages ("1. …? 2. …?") are split and answered in one batch —
    # the user gets every answer, no clarification-card round trip per part.
    if not explicitly_requests_write(question):
        sub_questions = split_questions(question)
        if len(sub_questions) >= 2:
            return _multi_answer(
                sub_questions, session, path,
                report_progress if progress_enabled else None, started)

    answers = request.get("answers") or {}
    if not isinstance(answers, dict):
        answers = {}

    key = _cache_key(question, path)
    analysis = ANALYSIS_CACHE.get(key)
    if analysis is None:
        analysis = analyze_question(question, db_path=path)
        _cache_put(key, analysis)
    report_progress({"type": "stage", "label": "Question analysis complete"})

    if analysis.is_ambiguous:
        clarification = _selected_clarification(analysis, answers)
        if clarification is None:
            # P1-4: remember the open question so a *typed* reply to this card
            # (instead of a chip click) can resolve to the original question.
            _remember_turn(session, question=question, analysis=analysis,
                           clarification=None, pending_clarify=True,
                           status="clarify")
            payload = _clarify_payload(analysis)
            payload["latency_s"] = round(time.perf_counter() - started, 3)
            return payload
        if clarification == PROCEED_CHIP:
            clarification = (
                "the user chose to proceed with the question exactly as written")
    else:
        clarification = None

    # P1-4 case C: referential follow-up ("and their pending fees too") —
    # keyword-gated, only when this turn has no fresh clarification of its own.
    if clarification is None and session.get("question"):
        referential = _referential_context(session, question)
    else:
        referential = None

    result = answer_question(
        question,
        clarification=clarification or referential,
        db_path=path,
        on_progress=report_progress if progress_enabled else None,
    )

    # P1-4 case A (chip click resolved) / C / unambiguous turn recording.
    _remember_turn(session, question=question, analysis=analysis,
                   clarification=clarification or
                   (referential if referential else None),
                   pending_clarify=False,
                   status="ok" if not result.error else "rejected")
    payload = _answer_payload(result, analysis, clarification, started)
    if referential and not clarification:
        payload["referential"] = True
    return payload


def _confirm_write(request: dict) -> dict:
    token = request.get("token")
    staged = PENDING_WRITES.pop(token, None) if isinstance(token, str) else None
    if not staged or staged["expires"] < time.monotonic():
        raise InputError("This write confirmation expired. Please submit the request again.")
    affected = confirm_write(staged["sql"], staged["question"],
                             db_path=staged["db_path"],
                             expect_fingerprint=staged.get("db_fingerprint"))
    outcome = {"status": "write_complete", "affected_rows": affected,
              "reply": (f"Applied the confirmed change to {affected} row"
                        f"{'s' if affected != 1 else ''}.")}
    if staged.get("followup"):
        result = answer_question(staged["followup"], db_path=staged["db_path"])
        if result.error:
            outcome["reply"] += " The write succeeded, but the follow-up query failed: " + result.error
        else:
            outcome["followup_sql"] = result.sql
            columns = result.columns or []
            outcome["results"] = {
                "columns": columns,
                "rows": [{column: value for column, value in zip(columns, row)}
                         for row in (result.rows or [])],
            }
            outcome["reply"] += f" Here are the requested results ({len(result.rows or [])} rows)."
    return outcome


def _cancel_write(request: dict) -> dict:
    token = request.get("token")
    staged = PENDING_WRITES.pop(token, None) if isinstance(token, str) else None
    if not staged:
        raise InputError("This write preview expired or was already resolved.")
    return {"status": "write_cancelled", "reply": "Cancelled the write preview. No data was changed."}


def _warmup() -> dict:
    import model_runtime

    tokenizer, model = model_runtime.load_model()
    MODEL_INFO.clear()
    MODEL_INFO.update({
        "model": model_runtime.MODEL_NAME,
        "gguf": str(getattr(model, "model", "")),
        "device": str(getattr(model, "device", "unknown")),
        "dtype": str(getattr(model, "dtype", "")),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "adapter": "LoRA" if getattr(model, "peft_config", None) else None,
    })
    return dict(MODEL_INFO)


def main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            action = request.get("action")

            if action == "warmup":
                _ok(request_id, _warmup())
            elif action == "query":
                _ok(request_id, _query(request, request_id=request_id))
            elif action == "confirm_write":
                _ok(request_id, _confirm_write(request))
            elif action == "cancel_write":
                _ok(request_id, _cancel_write(request))
            elif action == "ping":
                _ok(request_id, {"pong": True, "loaded": bool(MODEL_INFO),
                                 "pid": os.getpid()})
            else:
                _error(request_id, f"Unknown action: {action!r}", code="bad_request")
        except InputError as exc:
            _error(request_id, str(exc), code="input")
        except FileNotFoundError as exc:
            _error(request_id, str(exc), code="model_missing")
        except Exception as exc:  # keep the worker alive for the next request
            _error(request_id, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
