"""Clarion ambiguity decision layer: classifier + SLM verdicts combined.

``analyze_ambiguity(question, db_path, classifier=None, slm_client=None)``
integrates the schema-aware 3-class classifier (a peer module is injected via
the ``classifier`` argument — this module never imports it) with a generic
small-LLM client (``slm_client.generate_chat(messages, ...)``) and returns one
``AmbiguityDecision``:

* labels: ``CLEAR`` / ``AMBIGUOUS`` / ``UNDERSPECIFIED``
* evidence: raw signals kept for the UI/debugging (never trusted on its own —
  the SLM's *self-confidence* is ignored by design, only its structural verdict
  is used)
* options: clarify chips ``{"option": meaning, "note": short}``
``decision_rule`` records which signal won:
``agreement`` | ``conflict->clarify`` | ``no-metric-guard`` |
``confidence-threshold`` | ``classifier-only`` |
``classifier-below-threshold`` | ``slm-only`` | ``slm-clear-overrides`` |
``slm-resolves-conflict`` |
``chitchat-rule`` | ``fallback-heuristic`` |
``no-database`` | ``empty-question``.

Pure stdlib: sqlite3 access only through ``db_introspect`` and the rule-based
``semantic_ontology`` module (read-only), no network, no server processes
started here (the caller passes a client in). Retrieved semantic concepts are
appended to the schema text and are always labelled suggestions.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:  # normal case: project root already on sys.path
    import db_introspect as di
    import semantic_ontology as so
except ImportError:  # tolerate imports from other working directories
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import db_introspect as di
    import semantic_ontology as so

LABELS = ("CLEAR", "AMBIGUOUS", "UNDERSPECIFIED")
MAX_OPTIONS = 4
# Classifier probabilities below this count as "weak" when a *named* metric
# exists and the schema-aware SLM returned CLEAR: the historical live-bug was
# a weak AMBIGUOUS on "highest base salary" dead-ending the UI, so a weak
# classifier defers to a structurally valid SLM verdict.
CLASSIFIER_WEAK_CONFIDENCE = 0.5
# Prior per-class block thresholds for the *classifier-only* path (no SLM
# verdict available). Used when the injected classifier does not carry tuned
# ``block_thresholds`` (stubs, artifacts from before precision tuning); a
# trained artifact's tuned values always take precedence.
DEFAULT_BLOCK_THRESHOLDS = {"AMBIGUOUS": 0.70, "UNDERSPECIFIED": 0.70}
SLM_MAX_TOKENS = 260

# Greetings / acknowledgements ("hi", "thanks", "?") are not data questions:
# they are routed before any model call, so they cost no classifier/SLM
# latency and can never be misread as an ambiguous query (reproduced live:
# "hi" burned 6.8 s and returned an unanswerable clarify card).
_CHITCHAT_RE = re.compile(
    r"^(?:h[i!]+|hello+|hey+|yo|sup|thanks(?: a lot| again)?|thank you|thx"
    r"|ok(?:ay)?|alright|cool|nice|great|good (?:morning|afternoon|evening|day)"
    r"|by+e|goodbye|see (?:you|ya)|how are you|what'?s up|help|\?)+"
    r"[\s!.?！？]*$",
    re.IGNORECASE,
)
_MAX_CHITCHAT_WORDS = 4

# A message that asks several things at once ("1. How many orders are pending?
# 2. Which customer has ordered the most?" ) cannot be answered honestly by a
# single query — the model blends them and one sub-question comes back wrong.
# Detected deterministically before any model call; the clarify card offers the
# detected sub-questions as chips the UI re-fires one at a time.
_QUESTION_START_RE = re.compile(
    r"^(?:and\s+|or\s+|also\s+|then\s+|plus\s+)?"
    r"(?:what|which|who|whose|whom|when|where|why|how|show|list|tell|give|"
    r"find|count|name|display|return|fetch|summarize|summarise|total)\b",
    re.IGNORECASE)


def split_questions(text: str):
    """Sub-questions within one user message; [] when there is only one ask.

    Splits on newlines and after '?'/'.!' endings, strips numbered/bullet
    prefixes, and keeps segments that read as a question (trailing '?' or a
    question/answer-request opener). Falls back to one for prose that merely
    mentions an interrogative mid-sentence.
    """
    stripped = str(text or "").strip()
    if not stripped:
        return []
    parts = re.split(r"\n+|(?<=[?!.])\s+", stripped)
    questions = []
    for part in parts:
        part = re.sub(r"^\s*(?:\d+\s*[.):]|\d+\)|[-*•])[ \t]*", "", part.strip())
        part = part.strip(" \t;:,")
        if len(part.split()) < 2:
            continue
        if part.endswith("?") or _QUESTION_START_RE.match(part):
            questions.append(part[:200])
    return questions if len(questions) >= 2 else []


def is_chitchat(text: str) -> bool:
    tokens = str(text).strip().split()
    if len(tokens) > _MAX_CHITCHAT_WORDS:
        return False
    return bool(_CHITCHAT_RE.match(str(text).strip()))


def _department_performance_options(schema: dict, question: str):
    """Use a real department→student→enrollment path for vague performance asks."""
    q = str(question).lower()
    if not re.search(r"\bdepartments?\b", q) or not re.search(
            r"\b(best|top|perform(?:ance|ing)|doing best)\b", q):
        return None

    tables = schema.get("tables", {})
    student_tables = [
        name for name, info in tables.items()
        if {c["name"].lower() for c in info.get("columns", [])}
        .issuperset({"department", "student_id"})
    ]
    grade_metrics = [
        (name, c["name"])
        for name, info in tables.items()
        if "student_id" in {c["name"].lower() for c in info.get("columns", [])}
        for c in info.get("columns", [])
        if c["name"].lower() in {"grade", "marks", "mark", "score", "gpa"}
        and di._is_numeric_type(c["type"])
    ]
    for student_table in student_tables:
        for grade_table, metric in grade_metrics:
            if student_table == grade_table:
                continue
            metric_label = metric.replace("_", " ")
            return [
                {"option": f"Rank departments by average enrollment {metric_label}",
                 "note": f"joins {student_table}.student_id to "
                         f"{grade_table}.student_id; groups by department"},
                {"option": f"Rank departments by their highest individual {metric_label}",
                 "note": f"uses MAX({metric}) across {grade_table} rows per department"},
            ]
    return None

SLM_SYSTEM_PROMPT = (
    "You are the disambiguation analyst for a natural-language-to-SQL assistant. "
    "Decide whether a question about the database has one clear, schema-supported "
    "reading, several different readings, or cannot be answered. "
    "Return ONLY one JSON object — no prose, no code fences.")

SLM_USER_TEMPLATE = """SCHEMA:
{schema_text}

QUESTION: {question}

Pick exactly one label:
- CLEAR: one reasonable, schema-supported reading. Superlative wording that maps
  onto exactly one numeric column in the schema (e.g. "highest" onto `salary`) is
  a single reading, so it is CLEAR.
- AMBIGUOUS: multiple materially different readings that the schema does support
  (different metric, different table, different grouping — e.g. "highest pay"
  with both salary and bonus columns).
- UNDERSPECIFIED: missing information ("Show that student", "the first one") or
  the question asks for data the schema does not provide (e.g. "happiest
  employees" when there is no happiness/satisfaction column). Never invent
  tables or columns; judge only against the schema above.

Answer with JSON exactly in this shape:
{{"label": "CLEAR|AMBIGUOUS|UNDERSPECIFIED",
  "reason": "one short sentence why",
  "interpretations": [{{"meaning": "...", "relevant_columns": ["table.column"]}}],
  "clarification_question": "...",
  "note_on_missing_data": "..."}}
Interpretations must be empty for CLEAR, and carry 2-3 different readings for
AMBIGUOUS. For UNDERSPECIFIED fill "note_on_missing_data"."""


@dataclass
class AmbiguityDecision:
    """One decision about how (or whether) a question can be answered."""

    label: str                     # CLEAR | AMBIGUOUS | UNDERSPECIFIED
    reason: str                    # human-readable justification
    options: list = field(default_factory=list)  # {"option": ..., "note": ...}
    clarify_question: str = ""
    evidence: dict = field(default_factory=dict)  # classifier/slm signals
    decision_rule: str = ""
    question: str = ""
    relevant_tables: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# robust SLM parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _first_balanced_object(text: str):
    """First well-formed ``{...}`` span, string- and escape-aware."""
    depth = start = 0
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:  # stray closer: keep scanning
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return text[start: i + 1]
    return None


def _strip_fences(text: str) -> str:
    fence = _FENCE.search(text)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def parse_slm_json(raw):
    """Best-effort first-JSON-object extraction + validation.

    Tries the fenced body, then the first balanced object, then the full string;
    returns a normalised verdict dict, or ``None`` on any parse/validation
    failure (the caller then treats the SLM as unavailable). The SLM's
    ``confidence`` field, if any, is deliberately dropped.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    stripped = _strip_fences(raw)
    candidates = []
    if stripped and stripped != raw:
        candidates.append(stripped)
    for span in (_first_balanced_object(stripped), _first_balanced_object(raw)):
        if span and span not in candidates:
            candidates.append(span)
    candidates.append(raw)
    for object_text in candidates:
        if not object_text:
            continue
        try:
            verdict = json.loads(object_text)
        except (json.JSONDecodeError, ValueError):
            continue
        return validate_slm_verdict(verdict)  # invalid structure -> None
    return None


def validate_slm_verdict(verdict):
    """pydantic-free structural validation + normalisation of an SLM verdict."""
    if not isinstance(verdict, dict):
        return None
    label = str(verdict.get("label", "")).strip().upper()
    if label not in LABELS:
        return None

    interpretations = []
    raw_interps = verdict.get("interpretations")
    if isinstance(raw_interps, list):
        for item in raw_interps[:MAX_OPTIONS]:
            if not isinstance(item, dict):
                continue
            meaning = str(item.get("meaning", "")).strip()
            if not meaning:
                continue
            raw_cols = item.get("relevant_columns", [])
            columns = []
            if isinstance(raw_cols, (list, tuple)):
                columns = [str(c).strip() for c in raw_cols if str(c).strip()]
            interpretations.append(
                {"meaning": meaning, "relevant_columns": columns[:MAX_OPTIONS]})

    def _text(key):
        val = verdict.get(key)
        return str(val).strip() if val else ""

    return {
        "label": label,
        "reason": _text("reason") or _text("justification"),
        "interpretations": interpretations,
        "clarification_question": _text("clarification_question"),
        "note_on_missing_data": _text("note_on_missing_data"),
    }


def build_slm_messages(question: str, schema_text: str) -> list:
    """OpenAI-style chat messages for the ambiguity analyst."""
    return [
        {"role": "system", "content": SLM_SYSTEM_PROMPT},
        {"role": "user", "content": SLM_USER_TEMPLATE.format(
            schema_text=schema_text, question=question)},
    ]


# --------------------------------------------------------------------------
# signal acquisition (each wrapped: one bad signal never breaks the decision)
# --------------------------------------------------------------------------

def _norm_probs(probs):
    """Normalise {label: p}; tolerate 0-100 scales and lowercase keys."""
    if not isinstance(probs, (dict, tuple)):
        return {}
    items = []
    if isinstance(probs, dict):
        items = list(probs.items())
    if not items:
        return {}
    norm = {str(k).strip().upper(): float(v) for k, v in items
            if str(v).strip() not in {"", "nan"}}
    if norm and max(norm.values()) > 1.5:  # probabilities were stored as %
        norm = {k: v / 100.0 for k, v in norm.items()}
    return {k: v for k, v in norm.items() if k in LABELS}


def _run_classifier(classifier, question: str, schema_text: str):
    """Call classifier.predict(question, schema_text); None on any failure."""
    if classifier is None:
        return None
    try:
        raw = classifier.predict(question, schema_text)
        if not isinstance(raw, dict):
            return None
        label = str(raw.get("label", "")).strip().upper()
        if label not in LABELS:
            return None
        probs = _norm_probs(raw.get("probs"))
        return {"label": label, "probs": probs}
    except Exception:  # a broken classifier must never kill the decision
        return None


def _run_slm(slm_client, question: str, schema_text: str):
    """Ask the SLM; returns (verdict|None, error_note|"")."""
    if slm_client is None:
        return None, ""
    try:
        raw = slm_client.generate_chat(
            build_slm_messages(question, schema_text),
            max_tokens=SLM_MAX_TOKENS, temperature=0.0)
    except Exception as exc:
        return None, f"slm call failed: {exc}"
    verdict = parse_slm_json(raw)
    if verdict is None:
        return None, "slm response was not valid JSON"
    return verdict, ""


# --------------------------------------------------------------------------
# combination logic
# --------------------------------------------------------------------------

def _combine(cls, slm, metric_named=False, block_thresholds=None):
    """Return (label, decision_rule) from classifier + SLM verdicts.

    Division of labour: the SLM is schema-aware but answers one question at a
    time with fresh eyes; the classifier is statistically calibrated-ish but
    shallow. The rules, in order:

    1. ``agreement`` — same label, no arbitration needed.
    2. SLM sees a problem, classifier says CLEAR → follow the SLM
       (``conflict->clarify``). Clarification is the conservative direction;
       the historical live-bug "highest base salary" loop proves the cost of
       doing the opposite.
    3. SLM says CLEAR, classifier disagrees:
       * the question names **no stored metric** → keep the classifier's
         label (``conflict->clarify``). "best folks" with zero metric-named
         columns is a genuine structural gap the SLM demonstrably misses;
       * a metric IS named and the classifier is weak (p <
         ``CLASSIFIER_WEAK_CONFIDENCE``) → CLEAR (``confidence-threshold``), a
         debatable call the SLM gets right;
       * a metric IS named and the classifier is confident → CLEAR
         (``slm-clear-overrides``).
    4. Classifier-only (SLM unavailable): CLEAR passes; non-CLEAR blocks only
       at its per-class block threshold (precision-tuned on val, else the
       ``DEFAULT_BLOCK_THRESHOLDS`` prior).
    """
    blocks = dict(DEFAULT_BLOCK_THRESHOLDS)
    for k, v in (block_thresholds or {}).items():
        try:
            blocks[k] = float(v)
        except (TypeError, ValueError):
            pass

    if cls and slm:
        c_lbl, s_lbl = cls["label"], slm["label"]
        if c_lbl == s_lbl:
            return c_lbl, "agreement"
        if s_lbl != "CLEAR":
            return s_lbl, "conflict->clarify"
        # s_lbl == CLEAR, c_lbl != CLEAR: classifier wants a clarify the SLM
        # was happy to answer. Metric naming arbitrates.
        probability = (cls.get("probs") or {}).get(c_lbl, 0.0)
        if not metric_named:
            return c_lbl, "conflict->clarify"
        if probability < CLASSIFIER_WEAK_CONFIDENCE:
            return "CLEAR", "confidence-threshold"
        return "CLEAR", "slm-clear-overrides"
    if cls:
        c_lbl = cls["label"]
        probability = (cls.get("probs") or {}).get(c_lbl, 0.0)
        if c_lbl == "CLEAR":
            return "CLEAR", "classifier-only"
        blocks_label = blocks.get(c_lbl, 1.01)
        if probability >= blocks_label:
            return c_lbl, "classifier-only"
        return "CLEAR", "classifier-below-threshold"
    if slm:
        return slm["label"], "slm-only"
    return None, "fallback-heuristic"


def _fallback_label(schema: dict, relevant_tables, metric_count: int):
    """Deterministic last resort when neither analyser produced a signal.

    Ontology columns count here: one mapped metric is a supported reading,
    several are genuinely different readings, zero is missing information.
    """
    if not schema.get("tables") or not relevant_tables:
        return "UNDERSPECIFIED"
    if metric_count > 1:
        return "AMBIGUOUS"
    return "CLEAR" if metric_count == 1 else "UNDERSPECIFIED"


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def analyze_ambiguity(question, db_path, classifier=None, slm_client=None) -> AmbiguityDecision:
    """Combined 3-class decision for ``question`` over the database.

    ``classifier``: object exposing ``.predict(question, schema_text) ->
    {"label", "probs"}`` (the peer's ambiguity classifier is injected).
    ``slm_client``: object exposing ``.generate_chat(messages, ...)`` (the
    llama-server runtime or anything duck-type compatible).
    """
    question = question.strip() if isinstance(question, str) else ""
    if not question:
        return AmbiguityDecision(
            label="UNDERSPECIFIED",
            reason="The question is empty, so there is nothing to interpret.",
            evidence={"classifier": None, "slm": None},
            decision_rule="empty-question", question="")

    evidence = {"classifier": None, "slm": None}
    schema, tables, schema_text, schema_error = {}, [], "", ""
    try:
        schema = di.extract_schema(db_path)
        tables = di.select_relevant(schema, question, max_tables=4) \
            or sorted(schema["tables"])
        schema_text = di.render_compact(schema, tables=tables)
    except Exception as exc:  # missing/corrupt DB: conservative, no signals
        schema_error = str(exc)

    if schema_error:
        return AmbiguityDecision(
            label="UNDERSPECIFIED",
            reason=f"Introspection failed for {db_path!r}: {schema_error}. "
                   "The question cannot be judged against any schema.",
            evidence=evidence,
            decision_rule="no-database", question=question)

    if is_chitchat(question):
        example = tables[0] if tables else "the uploaded data"
        return AmbiguityDecision(
            label="UNDERSPECIFIED",
            reason="Reads as a greeting/acknowledgement rather than a data "
                   "question, so no SQL was attempted.",
            options=[
                {"option": f"Count the rows in {example}",
                 "note": f"simple summary of {example}"},
                {"option": f"List the rows in {example}",
                 "note": f"browse {example}"},
            ],
            clarify_question="Hi! Ask me something about your data, for example:",
            evidence={"classifier": None, "slm": None},
            decision_rule="chitchat-rule", question=question,
            relevant_tables=tables)

    performance_options = _department_performance_options(schema, question)
    if performance_options:
        return AmbiguityDecision(
            label="AMBIGUOUS",
            reason=("Department performance can mean the average grade across "
                    "enrollments or the highest individual grade; both are "
                    "supported by the department-to-student-to-enrollment join."),
            options=performance_options,
            clarify_question="How should department performance be measured?",
            evidence={"classifier": None, "slm": None,
                      "schema_rule": "department_performance_join"},
            decision_rule="schema-join-performance-rule", question=question,
            relevant_tables=tables)

    ontology_concepts = []
    try:  # additive layer: retrieval failure must not block the classifier
        ontology = so.build_ontology(db_path, schema=schema)
        ontology_concepts = so.retrieve_concepts(ontology, question,
                                                 tables=tables)
        rendered = so.render_concepts(ontology_concepts)
        if rendered:
            schema_text = f"{schema_text}\n\n{rendered}"
    except Exception:  # noqa: BLE001
        ontology_concepts = []
    evidence["ontology"] = ontology_concepts

    cls = _run_classifier(classifier, question, schema_text)
    evidence["classifier"] = cls

    slm, slm_error = _run_slm(slm_client, question, schema_text)
    evidence["slm"] = slm
    if slm_error:
        evidence["slm_error"] = slm_error

    semantic_metrics = so.metric_columns(ontology_concepts)
    direct_metrics = [f"{m['table']}.{m['column']}"
                      for m in di.has_named_metric(schema, question)]
    named_metrics = list(dict.fromkeys(direct_metrics + semantic_metrics))
    metric_named = bool(direct_metrics) or len(semantic_metrics) == 1
    evidence["metric_named"] = metric_named
    evidence["named_metrics"] = named_metrics
    block_thresholds = dict(getattr(classifier, "block_thresholds", {}) or {})
    label, decision_rule = _combine(cls, slm, metric_named=metric_named,
                                    block_thresholds=block_thresholds)

    reason_parts = [f"relevant tables: {', '.join(tables) or 'none'}",
                    f"rule: {decision_rule}"]

    if decision_rule == "fallback-heuristic":
        fallback_metrics = direct_metrics or semantic_metrics
        label = _fallback_label(schema, tables, len(set(fallback_metrics)))
    if cls:
        prob = (cls.get("probs") or {}).get(cls["label"])
        prob_str = f", p={prob:.2f}" if prob is not None else ""
        reason_parts.append(f"classifier={cls['label']}{prob_str}")
    if slm and slm.get("reason"):
        reason_parts.append(f"slm: {slm['reason']}")
    elif slm:
        reason_parts.append(f"slm={slm['label']}")
    if slm_error:
        reason_parts.append(f"slm unavailable ({slm_error})")
    if slm and slm.get("note_on_missing_data"):
        reason_parts.append(f"missing data: {slm['note_on_missing_data']}")

    decision = AmbiguityDecision(
        label=label, reason="; ".join(reason_parts),
        clarify_question="", evidence=evidence, decision_rule=decision_rule,
        question=question, relevant_tables=tables)

    if label == "AMBIGUOUS":
        decision.options = _options_from_slm(slm)
        if decision.options:
            decision.clarify_question = (
                (slm.get("clarification_question") or "").strip()
                or "Which of these readings should the SQL answer use?")
        else:
            decision.clarify_question = (
                "The request has multiple readings; please restate it naming "
                "the exact column (or table) to use.")
    elif label == "UNDERSPECIFIED":
        decision.options = []
        decision.clarify_question = _clarify_underspecified(slm, question)
    else:
        decision.options = []
        decision.clarify_question = ""
    return decision


def _options_from_slm(slm) -> list:
    """Clarify chips: slm interpretations -> {"option", "note"}."""
    options = []
    if not slm:
        return options
    for interp in slm.get("interpretations", []):
        cols = interp.get("relevant_columns") or []
        note = f"uses {', '.join(cols[:MAX_OPTIONS])}" if cols else ""
        options.append({"option": interp["meaning"], "note": note})
    return options


def _clarify_underspecified(slm, question: str) -> str:
    if slm:
        cs = slm.get("clarification_question") or ""
        note = slm.get("note_on_missing_data")
        if cs:
            return cs
        if note:
            return f"The request can't be answered as asked ({note}). " \
                   "Please rephrase to an attribute listed in the schema."
    return ("The request names information this database does not provide; "
            "please rephrase using the available columns.")


__all__ = [
    "AmbiguityDecision", "analyze_ambiguity", "build_slm_messages",
    "parse_slm_json", "validate_slm_verdict", "SLM_SYSTEM_PROMPT",
    "SLM_USER_TEMPLATE",
]
