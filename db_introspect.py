"""Dynamic SQLite introspection for Clarion.

Everything the ambiguity layer and the prompts need to know about a database is
derived at runtime from the file itself — no hardcoded table or column names
anywhere. Functions here never write (read-only connections) and never spawn
anything; they are pure stdlib ``sqlite3``.

Contract (kept stable, other modules may rely on it):

* ``extract_schema(db_path)`` -> ``{"tables": {...}, "relationships": [...]}``
  where each table entry has ``columns`` (list of {name, type, primary_key,
  notnull}), ``foreign_keys`` and ``sample_values`` (<= 4 distinct non-null
  reprs per column).
* ``render_compact(schema, tables=None)`` -> deterministic text block for
  prompts: table + columns + types + samples (+ relationships when available).
* ``select_relevant(schema, question, max_tables=4)`` -> table-name subset via
  token-overlap scoring (deterministic, cheap, schema-driven).
* ``has_named_metric(schema, question_or_tokens)`` -> candidate numeric columns
  whose names overlap the question wording (empty list == falsy, so it doubles
  as a predicate: "no listed metric supports this question -> UNDERSPECIFIED").
* ``quote_ident(name)`` -> safely double-quoted SQL identifier.

Rendered prompt text intentionally shows identifiers without quotes (a prompt is
not executed); every fragment that *could* be pasted into SQL (join hints,
PRAGMA calls, sample queries) uses ``quote_ident``.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from urllib.parse import quote

# Words that are question furniture, never a concept/metric name.
_STOPWORDS = {
    "a", "also", "all", "am", "an", "and", "any", "are", "as", "at", "be",
    "both", "but", "by", "can", "did", "do", "does", "each", "every", "find",
    "for", "from", "give", "give", "has", "have", "how", "i", "in", "is", "it",
    "its", "list", "many", "may", "me", "much", "my", "no", "of", "on", "or",
    "our", "per", "please", "show", "so", "than", "that", "the", "their",
    "there", "these", "this", "those", "to", "too", "top", "us", "use", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "whose", "why", "will", "with", "would", "you", "your",
}

# Column names too generic, on their own, to imply a relationship between two
# tables (e.g. both ``invoices`` and ``users`` may have a ``name``).
_GENERIC_COLUMN_NAMES = {
    "id", "name", "type", "value", "values", "code", "notes", "comment",
    "comments", "date", "datetime", "created_at", "updated_at", "deleted_at",
    "description", "state", "status", "active", "enabled", "version", "rowid",
    "seq", "seqno", "position", "index",
}

_NUMERIC_HINT = re.compile(r"INT|REAL|FLOA|DOUB|NUM|DEC|MONEY|SERIAL|BIT", re.I)
_WORD_TOKEN = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def quote_ident(name: str) -> str:
    """Return ``name`` as a safely double-quoted SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def _connect_ro(db_path) -> sqlite3.Connection:
    """Open a SQLite file strictly read-only; raise if it does not exist."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    uri = quote(str(path.resolve()), safe="/")
    return sqlite3.connect(f"file:{uri}?mode=ro", uri=True)


def _tokenize(text: str):
    return _WORD_TOKEN.findall(str(text).lower()) if text else []


def _plural_stem(token: str) -> str:
    """Crude stem: salaries->salary, marks->mark (len guard keeps 3-4 char ids)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _token_match(a: str, b: str) -> bool:
    """Equal, or a shared prefix of >= 4 chars (marks~mark, salaries~salary)."""
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
        return True
    return False


def _matched_question_tokens(name, q_raw: set) -> set:
    """Which question tokens are lexically related to a given identifier/text?

    Both sides are compared raw and 3-plural-stemmed, plus a >= 4-char shared
    prefix rule (marks~mark, salaries~salary). Report the *original* question
    tokens so callers get human-readable matches.
    """
    cvar = set()
    for tok in _tokenize(str(name)):
        cvar.add(tok)
        cvar.add(_plural_stem(tok))
    if not cvar:
        return set()
    matched = set()
    for qt in q_raw:
        qvar = {qt, _plural_stem(qt)}
        if qvar & cvar or any(_token_match(a, b) for a in qvar for b in cvar):
            matched.add(qt)
    return matched


def _is_identifier_column(col: dict) -> bool:
    """PK/_id columns are not 'metrics' for the answerability check."""
    lname = col["name"].lower()
    return bool(col["primary_key"]) or lname.endswith("_id") or lname == "id"


def _is_numeric_type(decl: str) -> bool:
    return bool(_NUMERIC_HINT.search(decl or ""))


def _q_tokens(question) -> list:
    """Filtered raw question tokens (works for question text or token lists)."""
    if isinstance(question, str):
        source = [question]
    else:  # caller supplies "question-ish tokens"
        source = [str(tok) for tok in question if isinstance(tok, str)]
    raw = []
    seen = set()
    for text in source:
        for tok in _tokenize(text):
            if len(tok) < 3 or tok in _STOPWORDS or tok in seen:
                continue
            seen.add(tok)
            raw.append(tok)
    return raw


def _shorten(reprv: str, limit: int = 32) -> str:
    if len(reprv) <= limit:
        return reprv
    return reprv[: limit - 3] + "..."


# --------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------

def _sample_values(conn, table: str, column: str, limit: int = 4):
    """<= ``limit`` distinct non-null values of a column, as short reprs."""
    qt, qc = quote_ident(table), quote_ident(column)
    rows = conn.execute(
        f"SELECT DISTINCT {qc} FROM {qt} WHERE {qc} IS NOT NULL LIMIT {int(limit)}"
    ).fetchall()
    return [_shorten(repr(v[0])) for v in rows]


def _pk_column(conn, table: str):
    """First PRIMARY KEY column of ``table`` (for FKs declared without TO)."""
    try:
        for _cid, name, _ctype, _nn, _dflt, pk in conn.execute(
            f"PRAGMA table_info({quote_ident(table)})"
        ).fetchall():
            if pk:
                return name
    except sqlite3.Error:
        pass
    return None


def _infer_relationships(tables) -> list:
    """Candidate joins: FK constraints first, then same-column-name heuristics."""
    rels, seen = [], set()

    def add(kind, t1, c1, t2, c2):
        # FKs are added first; a name-match duplicate of the same pair is skipped
        key = tuple(sorted([(t1, c1), (t2, c2)]))
        if key in seen:
            return
        seen.add(key)
        rels.append({
            "left_table": t1, "left_column": c1,
            "right_table": t2, "right_column": c2,
            "kind": kind,
            "join_sql": f"{quote_ident(t1)}.{quote_ident(c1)} = "
                        f"{quote_ident(t2)}.{quote_ident(c2)}",
        })

    # 1) declared FOREIGN KEY constraints
    for table, info in tables.items():
        for fk in info["foreign_keys"]:
            ref_table = fk["table"]
            if ref_table in tables and fk["to"] in {c["name"] for c in tables[ref_table]["columns"]}:
                add("foreign_key", table, fk["from"], ref_table, fk["to"])

    # 2) column-name heuristics among columns that survive across tables
    for t1, info1 in tables.items():
        for t2, info2 in tables.items():
            if t1.lower() >= t2.lower():
                continue  # each unordered pair once, deterministic order
            cols2 = {c["name"].lower(): c for c in info2["columns"]}
            for c1 in info1["columns"]:
                c2 = cols2.get(c1["name"].lower())
                if c2 is None:
                    continue
                norm = c1["name"].lower()
                if norm in _GENERIC_COLUMN_NAMES:
                    continue
                same_kind = (
                    _is_numeric_type(c1["type"]) == _is_numeric_type(c2["type"])
                )
                if same_kind:
                    add("name_match", t1, c1["name"], t2, c2["name"])

    rels.sort(key=lambda r: (
        0 if r["kind"] == "foreign_key" else 1,
        r["left_table"].lower(), r["left_column"].lower(),
        r["right_table"].lower(), r["right_column"].lower(),
    ))
    return rels


# --------------------------------------------------------------------------
# extraction (+ mtime/size-keyed cache)
# --------------------------------------------------------------------------

_SCHEMA_CACHE: dict = {}
_SCHEMA_CACHE_MAX = 8
_CACHE_VERSION = 1  # bump when the introspection output shape/logic changes


def cache_clear():
    """Drop all memoized schema introspections (tests, file re-registration)."""
    _SCHEMA_CACHE.clear()


def cache_stats():
    return {"entries": len(_SCHEMA_CACHE),
            "max": _SCHEMA_CACHE_MAX,
            "version": _CACHE_VERSION}


def _extract_schema_uncached(db_path) -> dict:
    conn = _connect_ro(db_path)
    try:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY lower(name)"
        ).fetchall()

        tables = {}
        for (tname,) in table_rows:
            columns, samples, fks = [], {}, []
            pk_names = []
            try:
                pragma_rows = conn.execute(
                    f"PRAGMA table_info({quote_ident(tname)})"
                ).fetchall()
                fk_rows = conn.execute(
                    f"PRAGMA foreign_key_list({quote_ident(tname)})"
                ).fetchall()
            except sqlite3.Error:  # shadow/damaged table: note and move on
                tables[tname] = {"columns": [], "foreign_keys": [],
                                 "sample_values": {}, "error": "introspection failed"}
                continue
            for _cid, name, ctype, notnull, _dflt, pk in pragma_rows:
                name = str(name)
                decl = (ctype or "").strip().upper() or "ANY"
                columns.append({
                    "name": name,
                    "type": decl,
                    "primary_key": bool(pk),
                    "notnull": bool(notnull),
                })
                samples[name] = _sample_values(conn, tname, name)
                if pk:
                    pk_names.append(name)
            for row in fk_rows:
                # (id, seq, table, from, to, on_update, on_delete, match)
                ref, frm, to = row[2], row[3], row[4]
                if not to:
                    to = _pk_column(conn, ref) or ""
                fks.append({"from": str(frm), "table": str(ref), "to": str(to)})
            tables[tname] = {"columns": columns, "foreign_keys": fks,
                             "sample_values": samples}
        return {"tables": tables, "relationships": _infer_relationships(tables)}
    finally:
        conn.close()


def extract_schema(db_path) -> dict:
    """Introspect a SQLite database: tables, columns, FKs, samples, joins.

    Memoized by (resolved path, st_mtime_ns, st_size, cache_version): repeated
    questions in one session pay the introspection+sampling cost exactly once
    per file version. Returns a defensive deep copy so callers can't corrupt
    the cache; the file signature is re-stat()'d on every call (cheap) so a
    write/CSV import invalidates naturally and cache staleness can never
    outlive the mtime.
    """
    import copy
    from pathlib import Path as _Path

    path = _Path(db_path)
    try:
        st = path.stat()
        sig = (str(path.resolve()), st.st_mtime_ns, st.st_size, _CACHE_VERSION)
    except OSError:          # vanished/unopenable path: bypass the cache
        return _extract_schema_uncached(db_path)

    hit = _SCHEMA_CACHE.get(sig)
    if hit is None:
        hit = _extract_schema_uncached(db_path)
        _SCHEMA_CACHE[sig] = hit
        # tiny LRU: drop the oldest signature when over capacity
        while len(_SCHEMA_CACHE) > _SCHEMA_CACHE_MAX:
            oldest = next(iter(_SCHEMA_CACHE))
            _SCHEMA_CACHE.pop(oldest, None)
    return copy.deepcopy(hit)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_compact(schema: dict, tables=None) -> str:
    """Deterministic prompt block: tables + columns + types + samples."""
    table_names = [t for t in schema.get("tables", {})]
    if tables is not None:
        chosen = list(tables)
    elif table_names:
        chosen = sorted(table_names)
    else:
        return "(schema: no tables available)"

    lines = []
    for table in chosen:
        info = schema.get("tables", {}).get(table)
        if not info:
            continue  # unknown table requested: just skip it
        col_parts = []
        for col in info["columns"]:
            part = f"{col['name']} {col['type']}"
            if col["primary_key"]:
                part += " PRIMARY KEY"
            elif col["notnull"]:
                part += " NOT NULL"
            col_parts.append(part)
        lines.append(f"table {table}: " + ", ".join(col_parts))
        sample_bits = [f"{col}: {', '.join(vals)}"
                       for col, vals in info.get("sample_values", {}).items() if vals]
        if sample_bits:
            lines.append("  samples: " + "; ".join(sample_bits))

    for rel in schema.get("relationships", []):
        pair = {rel["left_table"], rel["right_table"]}
        if pair.issubset(set(chosen)):
            lines.append(
                f"relation: {rel['left_table']}.{rel['left_column']} -> "
                f"{rel['right_table']}.{rel['right_column']} ({rel['kind']})")

    if not lines:
        return "(schema: no tables available)"
    return "\n".join(lines)


# --------------------------------------------------------------------------
# lexical relevance + metric helper
# --------------------------------------------------------------------------

def select_relevant(schema: dict, question: str, max_tables: int = 4) -> list:
    """Table subset lexically related to ``question`` (deterministic).

    Scoring per table: +3 per matching table-name token, +2 per matching column
    name, +1 per matching non-null sample value of a text-ish column. Tables
    with score 0 are dropped; ties break alphabetically. Returns [] when nothing
    matches (caller decides how to fall back).
    """
    q_raw = set(_q_tokens(question))
    if not q_raw:
        return []
    scores = {}
    for table, info in schema.get("tables", {}).items():
        score = 0.0
        score += 3.0 * len(_matched_question_tokens(str(table), q_raw))
        for col in info["columns"]:
            if _matched_question_tokens(col["name"], q_raw):
                score += 2.0
            # sample values participate only if the column looks textual
            if not _is_numeric_type(col["type"]):
                for val in info["sample_values"].get(col["name"], []):
                    if _matched_question_tokens(val, q_raw):
                        score += 1.0
                        break
        if score > 0:
            scores[table] = score
    ranked = sorted(scores, key=lambda t: (-scores[t], t.lower()))
    return ranked[: max(0, int(max_tables))]


def has_named_metric(schema: dict, question) -> list:
    """Candidate numeric columns whose name overlaps the question wording.

    ``question`` may be the question text or an iterable of tokens. Returns a
    list of ``{"table", "column", "match"}`` best-match first — an *empty list*
    means the wording names no stored metric (typical root of
    UNDERSPECIFIED: "happiest employees" with no happiness column). Identifier
    columns (primary keys, *_id) are never candidates: mentioning "employee id"
    does not make a metric available.
    """
    q_raw = set(_q_tokens(question))
    if not q_raw or not schema.get("tables"):
        return []
    out = []
    for table, info in schema["tables"].items():
        for col in info["columns"]:
            if not _is_numeric_type(col["type"]) or _is_identifier_column(col):
                continue
            matched = _matched_question_tokens(col["name"], q_raw)
            if matched:
                out.append({"table": table, "column": col["name"],
                            "match": "/".join(sorted(matched)),
                            "hits": len(matched)})
    out.sort(key=lambda r: (-r["hits"], r["table"].lower(), r["column"].lower()))
    return [{"table": r["table"], "column": r["column"], "match": r["match"]}
            for r in out]


__all__ = [
    "extract_schema", "render_compact", "select_relevant", "has_named_metric",
    "quote_ident", "cache_clear", "cache_stats",
]
