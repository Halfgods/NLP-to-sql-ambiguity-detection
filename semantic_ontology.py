"""Lightweight semantic ontology for Clarion (Auto-Ontology-inspired).

For each SQLite file this module derives a small ontology from the live schema:

* entities  -- table -> suggested semantic attributes -> real columns
* relationships -- the joins ``db_introspect`` already inferred
* schema_metadata -- counts and the concept-catalog version

The mappings are deterministic suggestions, never facts: the module contains no
model, network, Docker, PostgreSQL or vector store. They are stored next to the
database as ``<database>.ontology.json`` and regenerated when the file's
mtime/size changes. Retrieval is lexical (the same token matching used by schema
selection), so it adds semantic recall (``owe money`` -> ``pending_fees``)
without pretending that two mapped columns are interchangeable.

Public API:
    build_ontology(db_path, schema=None) -> dict
    retrieve_concepts(ontology, question, max_concepts=5, tables=None) -> list[dict]
    render_concepts(concepts) -> str
    prompt_for(db_path, question, schema=None, tables=None) -> str
    metric_columns(concepts) -> list[str]
    ontology_path(db_path) -> Path
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import db_introspect as di

ONTOLOGY_VERSION = 1  # bump when CONCEPT_CATALOG or the stored shape changes
ONTOLOGY_SUFFIX = ".ontology.json"
MAX_CONCEPTS = 5
_CACHE: dict = {}
_CACHE_MAX = 8

# Domain vocabulary used to map user wording onto column-name evidence. Keep it
# small: every entry must be defensible for a generic SQLite database, and a
# concept only exists for a table when its columns actually match.
CONCEPT_CATALOG = (
    {
        "id": "academic_performance",
        "name": "Academic Performance",
        "description": "Represents a student's academic achievement.",
        "aliases": ("academic", "performance", "grade", "grades", "marks",
                    "scores", "gpa", "results", "exam", "assignment"),
        "column_tokens": ("academic", "performance", "grade", "grades", "mark",
                          "marks", "score", "scores", "gpa", "result",
                          "results", "exam", "assignment", "assignments"),
        "entity_hints": ("student", "pupil", "learner", "enrollment",
                         "enrolment"),
        "kind": "metric",
    },
    {
        "id": "attendance",
        "name": "Attendance",
        "description": "Represents how often a student attended class.",
        "aliases": ("attendance", "present", "absent", "absence",
                    "absenteeism"),
        "column_tokens": ("attendance", "present", "absent", "absence",
                          "absenteeism"),
        "entity_hints": ("student", "pupil", "learner", "enrollment",
                         "enrolment"),
        "kind": "metric",
    },
    {
        "id": "outstanding_fees",
        "name": "Outstanding Fees",
        "description": "Represents money a student still owes.",
        "aliases": ("fee", "fees", "due", "dues", "owe", "owed", "outstanding",
                    "unpaid", "arrears", "debt", "money", "balance"),
        "column_tokens": ("fee", "fees", "due", "dues", "pending",
                          "outstanding", "balance", "arrears", "unpaid",
                          "payment", "payments"),
        "entity_hints": (),
        "kind": "metric",
    },
    {
        "id": "compensation",
        "name": "Compensation",
        "description": "Represents what an employee is paid.",
        "aliases": ("pay", "salary", "wage", "wages", "income", "earnings",
                    "bonus", "commission", "compensation"),
        "column_tokens": ("pay", "payout", "salary", "salaries", "wage",
                          "wages", "income", "earnings", "bonus", "commission",
                          "compensation"),
        "entity_hints": (),
        "kind": "metric",
    },
)

# Entity nouns used only for readable names (``demostudents`` -> Student).
_ENTITY_NOUNS = (
    "student", "teacher", "instructor", "employee", "staff", "customer",
    "user", "account", "order", "product", "course", "department",
    "enrollment", "enrolment", "invoice", "payment", "patient", "supplier",
    "vendor", "project", "task", "sale", "flight", "hotel", "book", "class",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text) -> set:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _entity_name(table: str, info: dict) -> str:
    """Readable entity name; deterministic and only cosmetic."""
    table_name = str(table).lower()
    for noun in _ENTITY_NOUNS:
        if noun in table_name:
            return noun.title()
    for column in info.get("columns", []):
        name = str(column.get("name", "")).lower()
        if name.endswith("_id"):
            for noun in _ENTITY_NOUNS:
                if noun in name[:-3]:
                    return noun.title()
    words = [w for w in _TOKEN_RE.findall(table_name) if w]
    if not words:
        return "Entity"
    words[-1] = _singular(words[-1])
    return " ".join(word.title() for word in words)


def _column_concept_match(concept: dict, column: dict) -> bool:
    """True when a real numeric, non-identifier column expresses the concept."""
    if di._is_identifier_column(column) or not di._is_numeric_type(column["type"]):
        return False
    return bool(_tokens(column["name"]) & set(concept["column_tokens"]))


def _entity_in_scope(table: str, info: dict, concept: dict) -> bool:
    """Student-only concepts must not map ``courses.difficulty_score``."""
    hints = concept.get("entity_hints") or ()
    if not hints:
        return True
    words = _tokens(table)
    for column in info.get("columns", []):
        words |= _tokens(column.get("name"))
    for hint in hints:
        if any(hint == word or (len(hint) >= 4 and hint in word) for word in words):
            return True
    return False


def _generate_ontology(db_path: Path, schema: dict, fingerprint: str) -> dict:
    entities = []
    tables = schema.get("tables", {}) if isinstance(schema, dict) else {}
    for table in sorted(tables):
        info = tables[table] or {}
        attributes = {}
        for concept in CONCEPT_CATALOG:
            if not _entity_in_scope(table, info, concept):
                continue
            columns = [column["name"] for column in info.get("columns", [])
                       if _column_concept_match(concept, column)]
            if columns:
                attributes[concept["id"]] = columns
        entities.append({
            "name": _entity_name(table, info),
            "table": table,
            "attributes": attributes,
        })

    return {
        "database": db_path.name,
        "version": ONTOLOGY_VERSION,
        "fingerprint": fingerprint,
        "entities": entities,
        "relationships": list(schema.get("relationships", [])
                              if isinstance(schema, dict) else []),
        "schema_metadata": {
            "table_count": len(entities),
            "column_count": sum(len((tables.get(entity["table"]) or {})
                                    .get("columns", [])) for entity in entities),
            "concept_catalog_version": ONTOLOGY_VERSION,
            "mapping_policy": "suggested, not verified; columns are not interchangeable",
            "source": "dynamic_sqlite_introspection",
        },
    }


def ontology_path(db_path) -> Path:
    """The JSON file stored alongside ``db_path``."""
    return Path(f"{Path(db_path)}{ONTOLOGY_SUFFIX}")


def _fingerprint(db_path: Path) -> str:
    try:
        stat = db_path.stat()
    except OSError:
        return ""
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _read_stored(path: Path, fingerprint: str):
    try:
        ontology = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(ontology, dict):
        return None
    if ontology.get("version") != ONTOLOGY_VERSION:
        return None
    if ontology.get("fingerprint") != fingerprint:
        return None
    if not isinstance(ontology.get("entities"), list):
        return None
    return ontology


def _write_stored(db_path: Path, ontology: dict) -> None:
    target = ontology_path(db_path)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(ontology, indent=2), encoding="utf-8")
        tmp.replace(target)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def build_ontology(db_path, schema: dict = None) -> dict:
    """Generate (or load) the ontology for ``db_path``.

    The file is reused only while its recorded fingerprint matches the current
    database mtime/size; a changed database always regenerates the ontology.
    Failures to persist (read-only directory) degrade to an in-memory result.
    """
    path = Path(db_path)
    if schema is None:
        schema = di.extract_schema(path)
    fingerprint = _fingerprint(path)
    key = (str(path.resolve()), fingerprint, ONTOLOGY_VERSION)

    cached = _CACHE.get(key)
    if cached is not None:
        return copy.deepcopy(cached)

    ontology = None
    if fingerprint:
        ontology = _read_stored(ontology_path(path), fingerprint)
    if ontology is None:
        ontology = _generate_ontology(path, schema, fingerprint)
        if fingerprint:
            _write_stored(path, ontology)

    if fingerprint:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)), None)
        _CACHE[key] = ontology
    return copy.deepcopy(ontology)


def _concepts_from_entities(entities: list) -> list:
    """Flatten stored entity attributes into retrieval-ready concept records."""
    concepts = []
    for definition in CONCEPT_CATALOG:
        mapped_columns, related_entities = [], []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            columns = (entity.get("attributes") or {}).get(definition["id"]) or []
            if not columns:
                continue
            table = entity.get("table")
            if not table:
                continue
            related_entities.append(entity.get("name") or _entity_name(table, {}))
            for column in columns:
                mapping = f"{table}.{column}"
                if mapping not in mapped_columns:
                    mapped_columns.append(mapping)
        if mapped_columns:
            concepts.append({
                "concept": definition["name"],
                "description": definition["description"],
                "kind": definition["kind"],
                "mapped_columns": mapped_columns,
                "related_entities": list(dict.fromkeys(related_entities)),
                "verified": False,
            })
    return concepts


def retrieve_concepts(ontology: dict, question: str,
                      max_concepts: int = MAX_CONCEPTS, tables=None) -> list:
    """Concepts whose name, aliases or mapped columns the question mentions.

    ``tables`` optionally restricts retrieval to the schema subset already
    selected for the question, so a concept is not imported from a table the
    query will not touch.
    """
    entities = ontology.get("entities") if isinstance(ontology, dict) else None
    if not isinstance(entities, list) or not entities:
        return []
    if tables is not None:
        allowed = {str(table) for table in tables}
        entities = [entity for entity in entities
                    if isinstance(entity, dict) and entity.get("table") in allowed]
        if not entities:
            return []
    question_tokens = set(di._q_tokens(question))
    if not question_tokens:
        return []

    scored = []
    for concept in _concepts_from_entities(entities):
        column_names = [mapping.rsplit(".", 1)[-1]
                        for mapping in concept["mapped_columns"]]
        aliases = [alias for definition in CONCEPT_CATALOG
                   if definition["name"] == concept["concept"]
                   for alias in definition["aliases"]]
        haystack = " ".join([concept["concept"], *aliases, *column_names])
        matched = di._matched_question_tokens(haystack, question_tokens)
        if matched:
            scored.append((len(matched), concept["concept"].lower(), concept))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [concept for _score, _name, concept in scored[:max(0, int(max_concepts))]]


def render_concepts(concepts: list) -> str:
    """Compact prompt block; explicitly labels every mapping as unverified."""
    if not concepts:
        return ""
    lines = ["-- semantic hints (suggested, not verified; mapped columns are "
             "not interchangeable):"]
    for concept in concepts:
        columns = ", ".join(concept.get("mapped_columns") or [])
        lines.append(f"-- {concept['concept']}: {columns}")
    return "\n".join(lines)


def prompt_for(db_path, question: str, schema: dict = None, tables=None) -> str:
    """Ready-to-append prompt text; empty on any failure (additive layer)."""
    try:
        ontology = build_ontology(db_path, schema=schema)
        return render_concepts(
            retrieve_concepts(ontology, question, tables=tables))
    except Exception:  # noqa: BLE001 - semantic layer must never kill a query
        return ""


def metric_columns(concepts: list) -> list:
    """Unique qualified metric columns behind the retrieved concepts."""
    columns = []
    for concept in concepts:
        if concept.get("kind") != "metric":
            continue
        for column in concept.get("mapped_columns") or []:
            if column not in columns:
                columns.append(column)
    return columns


__all__ = [
    "CONCEPT_CATALOG", "ONTOLOGY_VERSION", "build_ontology",
    "retrieve_concepts", "render_concepts", "prompt_for", "metric_columns",
    "ontology_path",
]
