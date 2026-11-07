"""Clarion pipeline, rebuilt around the fine-tuned GGUF SQL model (simplified).

The original project's detector-adapter / tool-agent files were not in this
workspace, so this module provides the same interface used by
``UI/server/model_worker.py`` with an honest, leaner design:

* detector      — rule-based "clear" decision (no invented confidence scores)
* ontology      — rule-based semantic concepts retrieved as prompt hints
* SQL generator — the Q4_K_M GGUF (one query, schema + question prompt)
* validation    — read-only query checks plus confirmed INSERT/UPDATE execution
* repair        — a single grounded retry that feeds the SQLite error back to
                  the model if the first SQL fails to validate or execute

Normal questions execute in read-only mode. Explicit writes are previewed in a
rollback-only transaction and committed only after a separate confirmation.
"""

import re
import sqlite3
import sys
import time
from pathlib import Path

from model_runtime import SQL_SYSTEM_PROMPT, get_runtime
import db_introspect as di
import semantic_ontology as so

ROW_CAP = 100
MAX_ROWS_RETURNED = 100

ROOT = Path(__file__).resolve().parent


def _default_db():
    """Use the combined demo workspace; students.db is evaluation-only."""
    return ROOT / "UI" / "uploads" / "clarion.db"


class InputError(ValueError):
    pass


# --------------------------------------------------------------- question I/O

def validate_question(raw) -> str:
    if raw is None or not isinstance(raw, str):
        raise InputError("Question must be a non-empty string.")
    q = raw.strip()
    if not q:
        raise InputError("Question must be a non-empty string.")
    if len(q) > 1000:
        raise InputError("Question too long (max 1000 characters).")
    return q


# ------------------------------------------------------------------- schema

def _tables(conn):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")
    return [r[0] for r in cur.fetchall()]


def _table_schema(conn, table, question=""):
    quoted_table = di.quote_ident(table)
    cols = conn.execute(f"PRAGMA table_info({quoted_table})").fetchall()
    if len(cols) > 48:
        tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        relevant = [c for c in cols if c[5] or
                    any(part in tokens for part in re.findall(
                        r"[a-z0-9]+", str(c[1]).lower()))]
        chosen = {c[1] for c in relevant}
        for c in cols:
            if len(chosen) >= 48:
                break
            chosen.add(c[1])
        cols = [c for c in cols if c[1] in chosen]
    names, decl, samples = [], [], []
    for _, name, ctype, notnull, _df, pk in cols:
        names.append(name)
        parts = [f"{di.quote_ident(name)} {ctype or 'TEXT'}"]
        if pk:
            parts.append("PRIMARY KEY")
        elif notnull:
            parts.append("NOT NULL")
        decl.append(" ".join(parts))
        sample = conn.execute(
            f"SELECT DISTINCT {di.quote_ident(name)} FROM {quoted_table} "
            f"WHERE {di.quote_ident(name)} IS NOT NULL LIMIT 5"
        ).fetchall()
        example = ", ".join(str(v[0])[:40] for v in sample[:4])
        samples.append(f"-- {name}: e.g. {example}" if example else f"-- {name}")
    body = ",\n  ".join(decl)
    ddl = f"CREATE TABLE {quoted_table} (\n  {body}\n);"
    return ddl, "\n".join(samples)


SCHEMA_CHAR_BUDGET = 7000


def _relationship_hints(schema: dict, selected: list) -> list:
    """Ready-to-use JOIN lines for the tables actually shown to the model.

    Only pairs where BOTH sides are in the selected subset are emitted —
    partial join keys are exactly how wrong-result queries get written. FK
    relationships come first (authoritative), then name-match heuristics.
    """
    chosen = set(selected)
    lines = []
    for rel in schema.get("relationships", []):
        if rel["left_table"] in chosen and rel["right_table"] in chosen:
            lines.append(f"-- join: {rel['kind']}: {rel['join_sql']}")
    return lines


def build_schema(db_path: Path, tool_calls: list, on_progress=None, question="") -> str:
    try:
        schema = di.extract_schema(db_path)   # cached by (path, mtime_ns, size)
    except Exception:
        schema = {"tables": {}, "relationships": []}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        blocks = []
        all_tables = _tables(conn)
        detected = []
        if question:
            try:
                detected = di.select_relevant(schema, question, max_tables=6)
            except Exception:
                detected = []
        selected = detected or all_tables[:6]
        used_chars = 0
        for table in selected:   # relevance order: best tables render first
            ddl, notes = _table_schema(conn, table, question)
            block = ddl + "\n" + notes
            if used_chars and used_chars + len(block) > SCHEMA_CHAR_BUDGET:
                continue
            blocks.append(block)
            used_chars += len(block)
            call = {"arguments": f"describe_table({table})",
                    "observation": notes, "ok": True}
            tool_calls.append(call)
            if on_progress:
                on_progress({"type": "tool", "tool": call})
        hints = _relationship_hints(schema, selected)
        if hints:
            hint_block = "-- join helpers (verified relationships):\n" \
                         + "\n".join(hints)
            if used_chars + len(hint_block) <= SCHEMA_CHAR_BUDGET:
                blocks.append(hint_block)
                used_chars += len(hint_block)
        semantic = so.prompt_for(db_path, question, schema=schema,
                                 tables=selected)
        if semantic and used_chars + len(semantic) <= SCHEMA_CHAR_BUDGET:
            blocks.append(semantic)
            used_chars += len(semantic)
        return "\n\n".join(blocks)
    finally:
        conn.close()


# ----------------------------------------------------------------- detector

class QuestionAnalysis:
    """Detector result adapter matching UI/server/model_worker.py expectations.

    Wraps the hybrid decision (Edwards classifier + SLM analysis) when that
    layer is available; falls back to the rule-baseline "clear" reading when
    artifacts are missing or the layer errors — degraded, never crashed.
    """

    def __init__(self, question, tables, decision=None):
        self.question = question
        self.tables = tables
        self._decision = decision
        if decision is None:
            self.source = "rule_baseline"
            self._label = "CLEAR"
            self._reason = ("Read as a data question over table"
                            f"{'' if len(tables) == 1 else 's'} "
                            f"{', '.join(tables) or '?'}. No blocking ambiguity detected.")
            self._options = []
        else:
            self.source = f"ambiguity:{decision.decision_rule}"
            self._label = decision.label
            self._reason = decision.reason
            chips = [{"label": o.get("option", ""), "note": o.get("note", "")}
                     for o in decision.options if o.get("option")]
            if decision.label == "UNDERSPECIFIED" and not chips:
                # runnable example asks instead of a dead-end chip the flow
                # can never honour; the UI fires these as their own questions
                if self.tables:
                    t = self.tables[0]
                    chips = [{"label": f"Count the rows in {t}",
                              "note": f"simple summary of {t}"},
                             {"label": f"List the rows in {t}",
                              "note": f"browse {t}"}]
            self._options = chips

    @property
    def is_ambiguous(self):
        return self._label != "CLEAR"

    @property
    def reason(self):
        return self._reason

    @property
    def options(self):
        return self._options

    def to_dict(self):
        base = {"decision": self._label.lower(), "reason": self.reason,
                "rule_fired": None, "source": self.source,
                "is_fallback": self.source == "rule_baseline",
                "ambiguous": self.is_ambiguous, "options": self.options}
        if self._decision is not None:
            base["evidence"] = self._decision.evidence
            # Expose the classifier's ambiguity probability when available.
            # This is a model estimate, not a calibrated measure from the
            # combined classifier/SLM decision.
            classifier = (self._decision.evidence or {}).get("classifier") or {}
            probs = classifier.get("probs") or {}
            ambiguity_probability = probs.get("AMBIGUOUS")
            if isinstance(ambiguity_probability, (int, float)):
                base["ambiguity_probability"] = max(0.0, min(1.0, float(ambiguity_probability)))
            base["clarify_question"] = self._decision.clarify_question
            base["decision_rule"] = self._decision.decision_rule
        return base


def analyze_question(question: str, db_path=None):
    db_path = Path(db_path) if db_path else _default_db()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = _tables(conn)
    finally:
        conn.close()
    decision = combined_ambiguity_decision(question, db_path)
    return QuestionAnalysis(question, tables, decision=decision)


def combined_ambiguity_decision(question, db_path):
    """Hybrid detector: MiniLM classifier + GGUF SLM structured analysis.

    Returns an AmbiguityDecision or None (fallback path, never raises).
    """
    try:
        from ambiguity.analysis import analyze_ambiguity
        from model_runtime import get_runtime
        return analyze_ambiguity(question, db_path,
                                 classifier=load_classifier(),
                                 slm_client=get_runtime())
    except Exception as exc:  # noqa: BLE001 - detector must never kill the flow
        print(f"[pipeline] ambiguity layer degraded: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return None


_CLASSIFIER_CACHE = {}

def load_classifier():
    """Lazily load ambiguity/artifacts; returns None when unavailable."""
    for key, value in list(_CLASSIFIER_CACHE.items()):
        return value
    try:
        from ambiguity.classifier import AmbiguityClassifier
        clf = AmbiguityClassifier.load(ROOT / "ambiguity" / "artifacts")
        _CLASSIFIER_CACHE["clf"] = clf
        return clf
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] classifier unavailable: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        _CLASSIFIER_CACHE["clf"] = None
        return None


# --------------------------------------------------------- schema meta replies

_SCHEMA_Q_RE = re.compile(
    r"\b(?:schema|structure|table definitions|describe\b|what tables|which tables|"
    r"list (?:all |the )?tables|show (?:me )?(?:the )?tables?|"
    r"what columns|which columns|columns (?:does|are|have|of))\b",
    re.IGNORECASE)

# A deterministic gate, separate from model output. Writes are considered only
# for direct imperative/request forms; the user must still confirm the preview.
_WRITE_VERB = r"(?:insert|add|update|change|modify|set)"
_EXPLICIT_WRITE_RE = re.compile(
    rf"^(?:\s*(?:please\s+)?{_WRITE_VERB}\b|"
    rf"\s*(?:can|could|would)\s+(?:you|u)\s+{_WRITE_VERB}\b|"
    rf"\s*i\s+(?:want|would\s+like)\s+you\s+to\s+{_WRITE_VERB}\b|"
    rf"\s*please\s+{_WRITE_VERB}\b)", re.IGNORECASE)
_WRITE_TARGET_RE = re.compile(
    r"^\s*(?:insert|add|update|change|modify|set)\s+"
    r"(?:a\s+|an\s+|the\s+|new\s+|all\s+|each\s+|every\s+)?"
    r"(?:row|record|entry|student|department|course|enrollment|"
    r"students|departments|courses|enrollments|[a-z_]+\.[a-z_]+|"
    r"[a-z_]+\s+(?:to|in|with|where|whose|having|named|called)\b)",
    re.IGNORECASE)


def explicitly_requests_write(question: str) -> bool:
    """True only for direct INSERT/UPDATE style requests, never model guesses."""
    q = str(question or "").strip()
    if re.search(r"\b(?:add|set)\s+up\b", q, re.IGNORECASE):
        return False
    return bool(_EXPLICIT_WRITE_RE.search(q) or _WRITE_TARGET_RE.search(q))


_WRITE_READ_FOLLOWUP_RE = re.compile(
    r"\s+(?:and\s+then|then)\s+(?=(?:show|list|display|fetch|query|tell)\b)",
    re.IGNORECASE)


def split_write_followup(question: str):
    """Split a direct write followed by a read request into two operations."""
    match = _WRITE_READ_FOLLOWUP_RE.search(str(question or ""))
    if not match:
        return str(question or "").strip(), None
    write_part = str(question)[:match.start()].strip()
    followup = str(question)[match.end():].strip()
    if not explicitly_requests_write(write_part) or not followup:
        return str(question or "").strip(), None
    return write_part, followup


def is_schema_question(question: str) -> bool:
    """Meta-questions about the DB itself: answered by introspection, no SQL."""
    return bool(_SCHEMA_Q_RE.search(question or ""))


def schema_summary(db_path=None) -> str:
    """Human-readable table/column listing (read-only introspection)."""
    db_path = Path(db_path) if db_path else _default_db()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = _tables(conn)
    finally:
        conn.close()
    if not tables:
        return ("This database has no tables yet. Drop a CSV on the left to "
                "import one.")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    parts = [f"This database has {len(tables)} table"
             f"{'' if len(tables) == 1 else 's'}:"]
    try:
        for name in tables:
            safe = str(name).replace('"', '""')
            rows = list(conn.execute(f'PRAGMA table_info("{safe}")'))
            cols = []
            for row in rows:
                col_name, col_type, pk = row[1], row[2], row[5]
                label = f"{col_name} {col_type}".strip()
                if pk:
                    label += " PK"
                cols.append(label)
            parts.append(f"• {name}: " + ", ".join(cols))
    finally:
        conn.close()
    return "\n".join(parts)


# ------------------------------------------------------------ SQL generation

_SELECT_WORD = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|"
    r"vacuum|reindex|grant|revoke|merge)\b|\breplace\s+into\b",
    re.IGNORECASE)
_LIMIT_RE = re.compile(r"\blimit\b(\s*\d+)?\s*(;)?\s*$", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the first SQL statement out of whatever the model returned."""
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    text = text.strip("`[]").strip()
    if not text:
        return ""
    # if the model added prose, forward from the first SELECT/WITH keyword
    m = re.search(r"(select|with)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start():]
    # single statement only
    cut = text.find(";")
    if cut != -1:
        text = text[:cut + 1]
    return text.strip()


try:  # optional: the project works without sqlglot, it just validates less
    import sqlglot
    import sqlglot.expressions as sqlx
    import sqlglot.errors as sql_errors
    _HAS_SQLGLOT = sqlglot.__version__ >= "20"
except Exception:                     # ImportError or a broken install
    _HAS_SQLGLOT = False

_WRITE_EXPR_TYPES = None


def _write_expr_types():
    global _WRITE_EXPR_TYPES
    if _WRITE_EXPR_TYPES is None and _HAS_SQLGLOT:
        _WRITE_EXPR_TYPES = tuple(
            cls for name in (
                "Insert", "Update", "Delete", "Drop", "Create", "Alter",
                "AlterTable", "TruncateTable", "Attach", "Detach", "Pragma",
                "Analyze", "Merge", "Grant", "Revoke", "Transaction", "Commit",
                "Rollback", "Vacuum",
            ) if hasattr(sqlx, name)
            for cls in [getattr(sqlx, name)])
    return _WRITE_EXPR_TYPES or ()


def parse_select_tree(sql: str):
    """(expr, "") if ``sql`` parses as a single read-only SELECT/WITH tree.

    (None, reason) otherwise: parse failure, multiple statements, or any
    write/DDL/transaction expression anywhere in the tree.
    """
    if not _HAS_SQLGLOT:
        return None, "sqlglot unavailable"
    if not sql or not sql.strip():
        return None, "no SQL produced"
    try:
        statements = sqlglot.parse(sql, dialect="sqlite")
    except (sql_errors.ParseError, ValueError) as exc:
        return None, f"SQL failed to parse: {str(exc)[:160]}"
    statements = [s for s in statements if s is not None]
    if not statements:
        return None, "no SQL produced"
    if len(statements) > 1:
        return None, "contains multiple statements"
    expr = statements[0]
    forbidden = _write_expr_types()
    if forbidden:
        for node in expr.walk():
            if isinstance(node, forbidden):
                return None, (f"contains a {node.key.upper()} statement — "
                              "read-only mode")
    return expr, ""


def _expr_is_read_query(expr) -> bool:
    """True for SELECT / UNION / INTERSECT / EXCEPT / parenthesized subquery.

    ``WITH ... SELECT`` parses as a Select with a ``with`` arg, so it is
    covered. Anything odder (VALUES, PRAGMA remnants, garbage) is refused —
    the 1.5B model is only ever supposed to answer with one query.
    """
    if not _HAS_SQLGLOT:
        return False
    return isinstance(expr, (sqlx.Select, sqlx.Union, sqlx.Intersect,
                             sqlx.Except, sqlx.Subquery))


def _has_limit_anywhere(expr) -> bool:
    return _HAS_SQLGLOT and expr.find(sqlx.Limit) is not None


def validate_read_only(sql: str):
    """Return (clean SQL, "") or (None, human reason).

    Primary gate: a full sqlglot parse (dialect sqlite), which correctly
    handles WITH RECURSIVE, aliases, CTE names, quoted identifiers and
    comments inside SQL — the cases the old regex misjudged. A statement that
    fails *to parse* falls back to the legacy regex gate so a formatting
    quirk doesn't hard-block the UI; that fallback is strictly weaker (a
    hostile payload that dodges sqlglot-dialect parsing may dodge regexes
    too), but execution stays read-only behind `mode=ro` connections, so the
    worst case is a sqlite error reaching the retry loop, not data loss.
    Multiple statements and any write/DDL/transaction node are rejected.
    """
    cleaned = sql.strip()
    if not cleaned:
        return None, "no SQL produced"

    expr, why = parse_select_tree(cleaned)
    if _HAS_SQLGLOT:
        if expr is None:
            if "SQL failed to parse" in why:
                return _validate_read_only_regex(cleaned)
            return None, why or "SQL rejected"
        if not _expr_is_read_query(expr):
            return None, "only a single SELECT/WITH query is allowed"

        if not _has_limit_anywhere(expr):
            body = cleaned.rstrip().rstrip(";").rstrip()
            return f"{body} LIMIT {ROW_CAP};", ""
        return cleaned, ""
    return _validate_read_only_regex(cleaned)


def _validate_read_only_regex(sql: str):
    """Legacy regex gate, kept only as the sqlglot-unavailable fallback."""
    if not _SELECT_WORD.match(sql):
        return None, "does not start with SELECT or WITH"
    if _FORBIDDEN.search(sql):
        return None, "contains a write/DDL keyword"
    body = sql.rstrip(";").strip()
    if ";" in body:
        return None, "contains multiple statements"
    # row cap
    if not re.search(r"\blimit\b", sql, re.IGNORECASE):
        sql = f"{body} LIMIT {ROW_CAP};"
    return sql, ""


def run_query(sql: str, db_path: Path):
    """Read-only execution. SQLite failures come back through the note
    channel (``columns is None`` + "sqlite error: ...") instead of crashing
    the caller — a hallucinated table/column is a *model* error to retry,
    not a crash."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        try:
            cur = conn.execute(sql)
        except sqlite3.Error as exc:
            return None, [], f"sqlite error: {exc}"
        if cur.description is None:
            return [], ["no result set"], "returned no result set"
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS_RETURNED + 1)
        over = len(rows) > MAX_ROWS_RETURNED
        return columns, [tuple(r) for r in rows[:MAX_ROWS_RETURNED]], \
            (f"capped at {MAX_ROWS_RETURNED} rows" if over else "")
    finally:
        conn.close()


# ------------------------------------------------- schema-reference validation

_TABLE_REF_RE = re.compile(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)",
                           re.IGNORECASE)
_TABLE_DOT_COL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_*]*)",
                               re.IGNORECASE)
_CTE_NAME_RE = re.compile(r"\b(?:with|and)\s+([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(",
                          re.IGNORECASE)


def _load_tables(db_path) -> dict:
    """{lower_table: {lower_col, ...}} straight from the live database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {}
        for name in _tables(conn):
            safe = str(name).replace('"', '""')
            cols = {r[1].lower() for r in conn.execute(f'PRAGMA table_info("{safe}")')}
            tables[str(name).lower()] = cols
        return tables
    finally:
        conn.close()


def _subquery_outputs(sq) -> set:
    """Output column names of a derived table / subquery (best effort)."""
    inner = getattr(sq, "this", None)
    if isinstance(inner, sqlx.Select):
        names = set()
        for e in inner.expressions:
            if isinstance(e, sqlx.Star):
                return set()   # '*' outputs unknowable statically → opaque
            names.add(e.alias_or_name.lower())
            if isinstance(e, sqlx.Column):
                names.add(e.name.lower())
        return names
    return set()


def schema_mismatch_reason(sql: str, db_path: Path) -> str:
    """Human reason when SQL names tables/columns absent from this DB.

    AST walk via sqlglot (primary): FROM/JOIN targets must be real tables or
    CTE/alias names; ``table.column`` qualifiers must resolve to the aliased
    (or real) table and the column must exist in its columns list. Derived
    tables (``FROM (SELECT ...) AS x``) are resolved through their SELECT
    aliases; unknown inner shapes are skipped rather than false-positived.
    Falls back to the legacy regex checks (weaker) when sqlglot can't parse
    the statement or is unavailable. Returns "" when nothing is provably
    wrong — the classic 1.5B hallucination (``FROM employees`` on a database
    that only has ``students``) is what this is for.
    """
    tables = _load_tables(db_path)
    if not tables:
        return ""

    expr, why = parse_select_tree(sql)
    use_ast = _HAS_SQLGLOT and expr is not None
    if _HAS_SQLGLOT and expr is None and "SQL failed to parse" not in why:
        return ""          # rejected elsewhere (multi/write); not a mismatch

    if use_ast:
        # ---- aliases in scope ------------------------------------------------
        aliases = {}                       # alias -> real table name (lower)
        opaque_aliases = set()             # derived-table aliases (no clear cols)
        derived_cols = {}                  # derived alias -> allowed columns
        for cte in expr.find_all(sqlx.CTE):
            aliases[cte.alias_or_name.lower()] = None   # CTE names table-like
        for sub in expr.find_all(sqlx.Subquery):
            alias = (sub.alias or "").lower()
            if alias:
                derived_cols[alias] = _subquery_outputs(sub)
                if not derived_cols[alias]:
                    opaque_aliases.add(alias)
        for tbl in expr.find_all(sqlx.Table):
            real = (tbl.name or "").lower()
            alias = (tbl.alias or "").lower()
            if alias:
                aliases[alias] = real        # alias of a real table OR of a CTE

        # ---- FROM / JOIN targets --------------------------------------------
        for tbl in expr.find_all(sqlx.Table):
            real = (tbl.name or "").lower()
            if not real:
                continue
            if real in tables or real in aliases or real in derived_cols \
                    or real in opaque_aliases:
                continue
            return (f"references table '{tbl.name}' which does not exist in "
                    f"this database (available: {', '.join(sorted(tables))})")

        # ---- qualified columns -----------------------------------------------
        for col in expr.find_all(sqlx.Column):
            qual = (col.table or "").lower()
            if not qual:
                continue                    # unqualified: ambiguous by design
            cname = col.name.lower()
            if cname == "*":
                continue
            if qual not in aliases and qual not in derived_cols \
                    and qual not in opaque_aliases and qual not in tables:
                return (f"references '{qual}.{col.name}' but '{qual}' is not a "
                        f"table or alias in this query")
            real = aliases.get(qual, qual)  # alias -> table, else the word itself
            if qual in derived_cols:
                allowed = derived_cols[qual]
                if allowed and cname not in allowed:
                    return (f"references column '{col.name}' which is not "
                            f"selected by subquery alias '{qual}'")
                continue
            if qual not in opaque_aliases and real in tables \
                    and cname not in tables[real]:
                return (f"references column '{col.name}' which does not exist "
                        f"in table '{real}'")
        return ""

    return _schema_mismatch_reason_regex(sql, tables)


class QueryResult:
    def __init__(self):
        self.sql = None
        self.columns = []
        self.rows = []
        self.error = None
        self.tool_calls = []


def answer_question(question, clarification=None, db_path=None,
                    on_progress=None):
    result = QueryResult()
    started = time.perf_counter()
    db_path = Path(db_path) if db_path else _default_db()

    if on_progress:
        on_progress({"type": "stage", "label": "Checking database schema"})
    schema = build_schema(db_path, result.tool_calls, on_progress=on_progress,
                          question=question)
    runtime = get_runtime()

    last_error = None
    for attempt in range(2):
        if on_progress:
            on_progress({"type": "stage", "label": "Generating SQL"})
        user = (f"Operation mode: READ_ONLY\n\n{schema}\n\n"
                f"Question: {question}\n\nWrite one SQLite query.")
        if clarification:
            user = (f"Operation mode: READ_ONLY\n\n{schema}\n\nQuestion: {question}\n\n"
                    f"The user clarified the intended meaning: {clarification}\n\n"
                    "Write one SQLite query for that interpretation.")
        if attempt and last_error:
            user = (f"Operation mode: READ_ONLY\n\n{schema}\n\nQuestion: {question}\n\n"
                    f"A previous attempt produced SQL that failed:\n"
                    f"{result.sql}\nSQLite error (or validator message): {last_error}\n\n"
                    "Write one corrected SQLite query.")
        raw = runtime.generate_chat(
            [{"role": "system", "content": SQL_SYSTEM_PROMPT},
             {"role": "user", "content": user}], max_tokens=220)
        result.sql = extract_sql(raw)
        call = {
            "arguments": f"generate_sql (attempt {attempt + 1})",
            "observation": result.sql[:200], "ok": bool(result.sql)}
        result.tool_calls.append(call)
        if on_progress:
            on_progress({"type": "tool", "tool": call})
        if not result.sql:
            last_error = "model returned no SQL"
            continue
        sql, why = validate_read_only(result.sql)
        call = {"arguments": f"validate_sql: {result.sql[:120]}",
                "observation": why or "accepted (read-only, row-capped)",
                "ok": sql is not None}
        result.tool_calls.append(call)
        if on_progress:
            on_progress({"type": "tool", "tool": call})
        if sql is None:
            last_error = f"SQL rejected: {why}"
            continue
        why2 = schema_mismatch_reason(sql, db_path)
        if why2:
            call = {
                "arguments": f"validate_schema_refs: {sql[:120]}",
                "observation": why2, "ok": False}
            result.tool_calls.append(call)
            if on_progress:
                on_progress({"type": "tool", "tool": call})
            last_error = f"SQL rejected: {why2}"
            continue
        result.columns, result.rows, capped = run_query(sql, db_path)
        if result.columns is None:            # sqlite failure surfaces as text
            call = {
                "arguments": f"run_select: {sql[:120]}",
                "observation": capped, "ok": False}
            result.tool_calls.append(call)
            if on_progress:
                on_progress({"type": "tool", "tool": call})
            last_error = capped
            continue
        if capped:
            call = {"arguments": f"run_select (row cap)",
                    "observation": capped, "ok": True}
            result.tool_calls.append(call)
            if on_progress:
                on_progress({"type": "tool", "tool": call})
        if on_progress:
            on_progress({"type": "stage", "label": "Query finished"})
        result.sql = sql
        result.error = None
        return result

    result.error = last_error or "could not generate a valid read-only query"
    result.sql = None
    result.columns, result.rows = [], []
    return result


_WRITE_START = re.compile(r"^\s*(INSERT\s+INTO|UPDATE)\b", re.IGNORECASE)
_WRITE_TARGET = re.compile(
    r"^\s*(?:INSERT\s+INTO|UPDATE)\s+(?:OR\s+\w+\s+)?"
    r"(?:\"((?:[^\"]|\"\")+)\"|`([^`]+)`|\[([^]]+)\]|([A-Za-z_][A-Za-z0-9_]*))",
    re.IGNORECASE)


def _write_target(sql: str):
    match = _WRITE_TARGET.match(sql)
    if not match:
        return None
    return next((part.replace('""', '"') for part in match.groups() if part), None)


def validate_write_sql(sql: str, question: str, db_path: Path):
    """Restrict generated writes to one INSERT/UPDATE on an existing table."""
    sql = (sql or "").strip()
    if not _WRITE_START.match(sql):
        return None, "Only one INSERT or UPDATE statement is allowed."
    target = _write_target(sql)
    if not target:
        return None, "Could not identify the target table."
    if re.search(r"\b(DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|"
                 r"REPLACE|TRUNCATE)\b", sql, re.IGNORECASE):
        return None, "The statement contains a prohibited operation."
    if re.match(r"^\s*UPDATE\b", sql, re.IGNORECASE) and not re.search(
            r"\bWHERE\b", sql, re.IGNORECASE):
        if not re.search(r"\b(?:all|every|each)\b", question, re.IGNORECASE):
            return None, "UPDATE needs a WHERE clause unless you explicitly asked to change every row."
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = {name.lower() for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'")}
        if target.lower() not in names:
            return None, f"Target table '{target}' does not exist."
        if re.match(r"^\s*UPDATE\b", sql, re.IGNORECASE):
            if not re.search(r"\b(?:all|every|each)\b", question, re.IGNORECASE):
                where = re.search(r"\bWHERE\b(.*?)(?:\bRETURNING\b|;|$)",
                                  sql, re.IGNORECASE | re.DOTALL)
                set_clause = re.search(r"\bSET\b(.*?)\bWHERE\b", sql,
                                       re.IGNORECASE | re.DOTALL)
                set_columns = set()
                if set_clause:
                    set_columns = {m.group(1).strip('"`[]').lower()
                                   for m in re.finditer(
                                       r"(?:^|,)\s*[\"`\[]?([A-Za-z_][A-Za-z0-9_]*)[\"`\]]?\s*=",
                                       set_clause.group(1))}
                cols = conn.execute(
                    f'PRAGMA table_info("{target.replace(chr(34), chr(34) * 2)}")'
                ).fetchall()
                key_cols = {str(c[1]).lower() for c in cols
                            if c[5] or str(c[1]).lower() == "id"
                            or str(c[1]).lower().endswith("_id")
                            or str(c[1]).lower().endswith("_name")}
                grounded = False
                if where:
                    condition = re.compile(
                        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:'((?:[^']|'')*)'|"
                        r"(-?\d+(?:\.\d+)?))", re.IGNORECASE)
                    for match in condition.finditer(where.group(1)):
                        col = match.group(1).lower()
                        value = (match.group(2).replace("''", "'")
                                 if match.group(2) is not None else match.group(3))
                        if col in set_columns:
                            continue
                        if match.group(2) is not None:
                            if value.casefold() in question.casefold():
                                grounded = True
                                break
                        elif col in key_cols and re.search(
                                rf"\b(?:id|student|course|enrollment|row)\D{{0,12}}{re.escape(value)}\b",
                                question, re.IGNORECASE):
                            grounded = True
                            break
                if not grounded:
                    return None, ("The UPDATE target could not be grounded in your request. "
                                  "Name the specific row, such as a course name or ID.")
    finally:
        conn.close()
    return sql, ""


_PREVIEW_ROW_CAP = 20        # rows rendered per side of the diff
_PREVIEW_SAMPLE_CAP = 1000   # rows of the snapshot probed for duplicates


def _db_fingerprint(db_path) -> str:
    """Statistical fingerprint of the database file (mtime_ns + size).

    Attached to a write preview and re-checked at confirm time so a preview
    generated before another process's write cannot be confirmed blind
    against changed state (TOCTOU).
    """
    st = Path(db_path).stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def _jsonable_cell(v):
    return v.hex() if isinstance(v, bytes) else v


def _diff_write(conn, target: str, is_insert: bool) -> dict:
    """Column-level before/after diff of ``target`` inside the open txn.

    Snapshot semantics: ``_preview_before`` was created *before* the write
    executed, so
      * before-diff = rows present pre-write that now differ (UPDATE) or
        nothing (INSERT),
      * after-diff = the row versions produced by the write.
    A ``SELECT *`` subquery inside a derived table is opaque — likewise here
    the snapshot is opaque to column-level WHERE probing, so duplicate checks
    use row-wise EXISTS with NULL-safe ``IS``.
    """
    qi = di.quote_ident(target)
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({qi})")]
    info = {"columns": cols, "before": [], "after": [],
            "rows_omitted": False, "net_zero_change": False,
            "duplicate_existing": False}

    before_sql = (f"SELECT * FROM _preview_before EXCEPT SELECT * FROM {qi} "
                  f"LIMIT {_PREVIEW_ROW_CAP + 1}")
    before = conn.execute(before_sql).fetchall()
    if len(before) > _PREVIEW_ROW_CAP:
        info["rows_omitted"] = True
        before = before[:_PREVIEW_ROW_CAP]
    info["before"] = [[_jsonable_cell(v) for v in row] for row in before]

    after_sql = (f"SELECT * FROM {qi} EXCEPT SELECT * FROM _preview_before "
                 f"LIMIT {_PREVIEW_ROW_CAP + 1}")
    after = conn.execute(after_sql).fetchall()
    if len(after) > _PREVIEW_ROW_CAP:
        info["rows_omitted"] = True
        after = after[:_PREVIEW_ROW_CAP]
    info["after"] = [[_jsonable_cell(v) for v in row] for row in after]

    if not before and not after:
        info["net_zero_change"] = True   # wrote values equal to current state

    if is_insert and after:
        cols_pragma = conn.execute(f"PRAGMA table_info({qi})").fetchall()
        pk_cols = {name for (_cid, name, *_rest, pk) in
                   [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in cols_pragma]
                   if pk}
        probe_cols = [c for c in cols if c not in pk_cols] or cols
        probe = [[_jsonable_cell(r[cols.index(c)]) for c in probe_cols]
                 for r in after]
        conds = " AND ".join(f"{di.quote_ident(c)} IS ?" for c in probe_cols)
        for row_vals in probe:
            hit = conn.execute(
                f"SELECT 1 FROM _preview_before WHERE {conds} LIMIT 1",
                row_vals).fetchone()
            if not hit:
                info["duplicate_existing"] = False
                break
        else:
            info["duplicate_existing"] = True   # every new row already exists
    return info


def _apply_write_transaction(sql: str, db_path: Path, *, preview: bool):
    """Run a validated write under an SQLite authorizer and transaction.

    Returns ``(affected_rows, preview_info)``; ``preview_info`` is a dict
    describing what the write WOULD do (before/after row diffs capped at
    ``_PREVIEW_ROW_CAP``, duplicate-row warning, net-zero-change flag) and is
    ``None`` for apply mode. The temp snapshot name ``_preview_before`` is
    created inside the (rolled back) transaction, so preview mode leaves the
    database byte-identical; the authorizer additionally refuses any write to
    anything but ``target`` plus the preview temp table itself.
    """
    target = _write_target(sql)
    temp = "_preview_before"
    qi_temp = di.quote_ident(temp)
    SYSTEM_CATALOGS = {"sqlite_temp_master", "sqlite_master"}
    # only these PRAGMA invocations may run inside the snapshot txn — they are
    # pure schema readers (table_info/column_list/index_list/foreign_keys),
    # required for CREATE TEMP TABLE ... AS SELECT compilation; every
    # configuration pragma (journal_mode, secure_delete, ...) stays denied.
    _READONLY_PRAGMAS = {"table_info", "table_xinfo", "column_list",
                         "index_list", "index_info", "index_xinfo",
                         "foreign_key_list", "table_list"}
    allowed_actions = {
        sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_DELETE,
        # temp-snapshot machinery, names restricted to the _preview_ prefix
        sqlite3.SQLITE_CREATE_TEMP_TABLE, sqlite3.SQLITE_DROP_TEMP_TABLE,
    }

    def authorize(action, arg1, arg2, _database, _trigger):
        name = "" if arg1 is None else str(arg1)
        lname = name.lower()
        # CREATE/DROP TEMP TABLE internally writes the sqlite_temp_master
        # catalog; temp DDL is validated by the _preview_ prefix on its
        # own object name and every catalog row it generates may pass.
        if lname in SYSTEM_CATALOGS:
            if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE):
                return sqlite3.SQLITE_OK
            # SQLite's own DDL machinery maintains the catalog row via an
            # internal UPDATE (column 'type'); that is only permitted on the
            # session-local temp catalog — UPDATE sqlite_master through user
            # SQL stays denied (it is DDL by another name).
            if action == sqlite3.SQLITE_UPDATE and lname == "sqlite_temp_master":
                if arg2 in (None, "type", "name", "tbl_name", "rootpage", "sql"):
                    return sqlite3.SQLITE_OK
            return (sqlite3.SQLITE_OK if action in
                    (sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT)
                    else sqlite3.SQLITE_DENY)
        if action == sqlite3.SQLITE_PRAGMA:
            return (sqlite3.SQLITE_OK if lname in _READONLY_PRAGMAS
                    else sqlite3.SQLITE_DENY)
        if action in (sqlite3.SQLITE_CREATE_TEMP_TABLE,
                      sqlite3.SQLITE_DROP_TEMP_TABLE):
            return (sqlite3.SQLITE_OK if lname.startswith("_preview_")
                    else sqlite3.SQLITE_DENY)
        if action == sqlite3.SQLITE_DELETE:
            return (sqlite3.SQLITE_OK if lname.startswith("_preview_")
                    else sqlite3.SQLITE_DENY)
        if action not in allowed_actions:
            return sqlite3.SQLITE_DENY
        if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE):
            if lname.startswith("_preview_"):
                return sqlite3.SQLITE_OK
            return (sqlite3.SQLITE_OK if lname == target.lower()
                    else sqlite3.SQLITE_DENY)
        return sqlite3.SQLITE_OK

    conn = sqlite3.connect(str(db_path), timeout=5)
    preview_info = None
    try:
        conn.set_authorizer(authorize)
        conn.execute("BEGIN IMMEDIATE")
        if preview:
            conn.execute(f"CREATE TEMP TABLE {qi_temp} AS "
                         f"SELECT * FROM {di.quote_ident(target)}")
        cursor = conn.execute(sql)
        affected = max(0, cursor.rowcount)
        if preview:
            preview_info = _diff_write(
                conn, target, re.match(r"^\s*INSERT\b", sql, re.IGNORECASE)
                is not None)
            preview_info["table_name"] = target
            conn.rollback()
        else:
            conn.commit()
        return affected, preview_info
    except Exception:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


def propose_write(question: str, db_path=None):
    """Generate and rollback-preview one write; never commit here.

    The reply dict carries a column-level ``preview`` (before/after row
    diffs, duplicate-row warning, net-zero flag) and a ``db_fingerprint``
    snapshot for the confirm step to re-check.
    """
    db_path = Path(db_path) if db_path else _default_db()
    calls = []
    schema = build_schema(db_path, calls, question=question)
    user = (f"Operation mode: WRITE\n\n{schema}\n\nQuestion: {question}\n\n"
            "The user explicitly requested this change. Write one INSERT or UPDATE "
            "statement only. Use a WHERE clause for UPDATE unless all rows were "
            "explicitly requested. Never delete data.")
    raw = get_runtime().generate_chat(
        [{"role": "system", "content": SQL_SYSTEM_PROMPT},
         {"role": "user", "content": user}], max_tokens=220)
    sql = extract_sql(raw)
    sql, why = validate_write_sql(sql, question, db_path)
    if sql is None:
        raise InputError(f"Generated write was rejected: {why}")
    fingerprint = _db_fingerprint(db_path)
    try:
        affected, preview = _apply_write_transaction(sql, db_path, preview=True)
    except sqlite3.Error as exc:
        raise InputError(f"Generated write failed its rollback-only preview: {exc}") from exc
    return {"sql": sql, "affected_rows": affected, "preview": preview,
            "db_fingerprint": fingerprint}


def _insert_literal(node):
    """(value, is_null) for the literal classes the write model may emit."""
    if isinstance(node, sqlx.Null):
        return None, True
    if isinstance(node, sqlx.Literal):
        return str(node.this), False
    if isinstance(node, sqlx.Neg) and isinstance(node.this, sqlx.Literal):
        return "-" + str(node.this.this), False
    return None, False


def insert_id_warnings(sql: str, db_path=None):
    """Warnings about the integer *_id columns of a generated INSERT.

    Imported tables have no PRIMARY KEY/autoincrement, so an id that is
    omitted, NULL, or copied from an existing row silently becomes data you
    cannot reference later. Returns [] when sqlglot is unavailable or the
    statement is not a single plain INSERT-VALUES — the audit is additive;
    validation already guarantees the statement itself is safe to preview.
    """
    if not _HAS_SQLGLOT:
        return []
    try:
        statements = sqlglot.parse(sql, dialect="sqlite")
        statements = [s for s in statements if s is not None]
        if len(statements) != 1:
            return []
        expr = statements[0]
    except Exception:
        return []
    if not isinstance(expr, sqlx.Insert) or not isinstance(
            expr.expression, sqlx.Values):
        return []
    target = expr.this
    # sqlite dialect wraps `INSERT INTO t (a, b) …` in a Schema node; a bare
    # `INSERT INTO t …` (no column list) points at the Table directly.
    if isinstance(target, sqlx.Schema):
        columns = [col.name for col in target.expressions]
        target = target.this
    else:
        columns = [col.name for col in expr.expressions]
    if not isinstance(target, sqlx.Table):
        return []
    table = target.name
    tuples = expr.expression.expressions
    try:
        conn = sqlite3.connect(f"file:{Path(db_path) if db_path else _default_db()}"
                               f"?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    warnings = []
    try:
        info = list(conn.execute(f'PRAGMA table_info("{table}")'))
        id_cols = {row[1].lower(): row[1] for row in info
                   if str(row[1]).lower().endswith("_id")
                   and "int" in str(row[2]).lower()}
        for name in id_cols.values():
            if name.lower() not in columns:
                warnings.append(f"{name} was not provided — it will be "
                                "stored as NULL (the table has no "
                                "auto-increment), so the row can't be "
                                "referenced by id. Include an id if you "
                                "need one.")
        quoted = table.replace('"', '""')
        for row_number, row in enumerate(tuples, start=1):
            for index, column in enumerate(columns):
                id_name = id_cols.get(column)
                if id_name is None or index >= len(row.expressions):
                    continue
                value, is_null = _insert_literal(row.expressions[index])
                if is_null:
                    warnings.append(f"{column} is set to NULL — the row can't "
                                    "be referenced by id later; include an "
                                    "id if you need one.")
                    continue
                if value is None:
                    continue
                compare = int(value) if value.lstrip("-").isdigit() else value
                try:
                    (row_count,) = conn.execute(
                        f'SELECT COUNT(*) FROM "{quoted}" WHERE '
                        f'"{id_name}" = ?', (compare,)).fetchone()
                except sqlite3.Error:
                    continue
                if row_count:
                    shown = f"row {row_number}: " if len(tuples) > 1 else ""
                    warnings.append(
                        f"{shown}{column} {value} already exists in "
                        f"`{table}` — this row will share the id with an "
                        "existing row. Pick a fresh id if the rows are "
                        "meant to differ.")
    finally:
        conn.close()
    return warnings


def confirm_write(sql: str, question: str, db_path=None,
                  expect_fingerprint=None):
    """Revalidate and commit a previously staged, explicitly requested write.

    With ``expect_fingerprint`` set (the fingerprint captured at preview
    time), a database that changed underneath the preview is refused — the
    user is asked to regenerate the preview rather than confirm blind.
    """
    db_path = Path(db_path) if db_path else _default_db()
    if expect_fingerprint is not None:
        live = _db_fingerprint(db_path)
        if live != expect_fingerprint:
            raise InputError(
                "The database has changed since this preview was generated "
                "(another write may have landed). Start a new request to "
                "review a fresh preview before applying.")
    clean, why = validate_write_sql(sql, question, db_path)
    if clean is None:
        raise InputError(f"Write was rejected: {why}")
    try:
        affected, _ = _apply_write_transaction(clean, db_path, preview=False)
        return affected
    except sqlite3.Error as exc:
        raise InputError(f"Write failed; transaction rolled back: {exc}") from exc


__all__ = ["InputError", "validate_question", "analyze_question", "answer_question",
           "explicitly_requests_write", "split_write_followup", "propose_write",
           "confirm_write"]
