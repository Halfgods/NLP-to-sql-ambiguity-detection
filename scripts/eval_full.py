"""eval_full.py — Clarion held-out evaluation orchestrator (brutal-tester build).

Measures EVERYTHING the project claims, with graceful skips: any missing or
unloadable artifact yields a printed "NOT RUN (<reason>)" line instead of a
crash, so this script never lies by omission and never dies by dependency.

Suites
  classifier — ambiguity 3-class detector vs evals/heldout/classifier_eval.jsonl
               (loads ambiguity.classifier if importable; duck-typed adapter)
  sql        — the SERVED gguf (model_runtime.get_runtime(), identity hashed)
               vs evals/heldout/sql_eval.jsonl (execution accuracy against the
               four real eval databases; production-parity prompt = READ_ONLY
               system prompt + pipeline.build_schema schema + max_tokens 220;
               graded on a projection-aware 2dp comparator with the strict
               4dp rate reported alongside)

Outputs: evals/report-latest.json + evals/REPORT.md
Then run evals/generate_brutal_report.py for the honest memo.

CPU ONLY.  Read-only DB access everywhere (mode=ro URIs).
"""

import argparse
import importlib
import json
import math
import re
import sqlite3
import statistics
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from inspect import signature
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVALS = ROOT / "evals"
HELDOUT = EVALS / "heldout"

CLASSIFIER_SET = HELDOUT / "classifier_eval.jsonl"
SQL_SET = HELDOUT / "sql_eval.jsonl"
STUDENTS_DB = ROOT / "students.db"
JOIN_DB = EVALS / "tmp" / "join_demo.db"

LABELS = ("CLEAR", "AMBIGUOUS", "UNDERSPECIFIED")

# ---------------------------------------------------------------- utilities


def not_run(reason, extra=None):
    return {"status": "not_run", "reason": reason, **(extra or {})}


def load_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_label(value):
    """Map whatever a predictor returns to one of the three contract labels.
    Returns label or None."""
    if isinstance(value, (dict,)):
        if "label" in value:
            value = value["label"]
        elif "decision" in value:
            value = value["decision"]
        elif "prediction" in value:
            value = value["prediction"]
        elif "ambiguity" in value:
            value = value["ambiguity"]
        else:
            probs = value.get("probs") or value.get("probabilities") or \
                value.get("labels")
            if isinstance(probs, dict) and probs:
                value = max(probs.items(), key=lambda kv: kv[1])[0]
            else:
                return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None
    s = value.strip().upper()
    if s.startswith("CLEAR"):
        return "CLEAR"
    if s.startswith("AMB") or s == "AMBIG":
        return "AMBIGUOUS"
    if s.startswith("UNDER") or s.startswith("UNSPEC"):
        return "UNDERSPECIFIED"
    return None


# ------------------------------------------------------------- macro-F1 etc.

def prf(n_pred_correct, n_pred, n_gold):
    p = n_pred_correct / n_pred if n_pred else 0.0
    r = n_pred_correct / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def classification_metrics(y_true, y_pred, labels=LABELS):
    """Hand-rolled: confusion matrix, accuracy, per-class + macro P/R/F1, and
    binary ambiguity-detection precision/recall/F1 (AMBIGUOUS = positive)."""
    n = len(y_true)
    assert n == len(y_pred)
    idx = {lab: i for i, lab in enumerate(labels)}
    conf = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        conf[idx[t]][idx[p]] += 1
    acc = round(sum(conf[i][i] for i in range(len(labels))) / n, 4) if n else 0.0

    per_class, f1s = {}, []
    for lab in labels:
        i = idx[lab]
        tp = conf[i][i]
        fp = sum(conf[j][i] for j in range(len(labels))) - tp
        fn = sum(conf[i]) - tp
        p, r, f1 = prf(tp, tp + fp, tp + fn)
        per_class[lab] = {"precision": p, "recall": r, "f1": f1,
                          "support": sum(conf[i])}
        f1s.append(f1)
    macro_f1 = round(sum(f1s) / len(labels), 4)

    amb_idx = idx["AMBIGUOUS"]
    amb_p, amb_r, amb_f1 = prf(
        conf[amb_idx][amb_idx],
        sum(conf[j][amb_idx] for j in range(len(labels))),
        sum(conf[amb_idx]))
    return {
        "n": n,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "ambiguity": {"precision": amb_p, "recall": amb_r, "f1": amb_f1},
        "confusion_matrix": {
            "columns_are": list(labels),
            "rows_are": list(labels),
            "matrix": conf,
        },
    }


# ------------------------------------------- classifier predictor discovery

_CLASSIFY_HINT_WORDS = ("predict", "classify", "infer", "label", "analyze",
                        "detect", "categorize", "judge", "resolve")
_FACTORY_HINT_WORDS = ("get_classifier", "load_classifier", "build_classifier",
                       "make_classifier", "create_classifier")


def _safe_call(fn, question, schema_text):
    """Call a candidate predict function with whatever argument shape it takes."""
    mistakes = []
    try:
        args = signature(fn).parameters
        n_req = sum(1 for p in args.values()
                    if p.default is p.empty
                    and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                                   p.KEYWORD_ONLY))
    except (TypeError, ValueError):
        n_req = None
    call_attempts = [
        ("(q, schema)", lambda: fn(question, schema_text)),
        ("(q)", lambda: fn(question)),
        ("(question=..., schema_text=...)", lambda: fn(question=question, schema_text=schema_text)),
    ]
    for name, attempt in call_attempts:
        try:
            return attempt()
        except TypeError as exc:
            mistakes.append(f"{name}: {exc}")
    raise TypeError("no compatible signature; tried: " + " | ".join(mistakes))


def build_classifier_predictor():
    """Import ambiguity.classifier (peer's artifact) and return (predict_fn,
    describe_str).  Raises with a human reason on any failure."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        module = importlib.import_module("ambiguity.classifier")
    except ModuleNotFoundError as exc:
        raise ImportError(f"ambiguity.classifier is not importable ({exc})") from exc
    except Exception as exc:
        raise ImportError(f"ambiguity.classifier exists but fails to import: {exc}") from exc

    def describe(obj, sig_note):
        name = getattr(obj, "__name__", type(obj).__name__)
        return f"{name} ({sig_note})"

    # 1) direct module-level functions
    for attr in dir(module):
        if attr.startswith("_") or attr in ("json", "re", "time", "sys"):
            continue
        fn = getattr(module, attr)
        if not callable(fn):
            continue
        low = attr.lower()
        if any(w in low for w in _FACTORY_HINT_WORDS):
            try:
                obj = fn()  # factories: no-arg build first
            except TypeError:
                obj = _safe_call(fn, "", "")
            if callable(getattr(obj, "predict", None)):
                return (lambda q, s, o=obj: normalize_label(
                    _safe_call(o.predict, q, s))), describe(obj, f"module.{attr}()factory")
            if callable(obj):
                return (lambda q, s, o=obj: normalize_label(
                    _safe_call(o, q, s))), describe(obj, f"module.{attr}() factory")
        if any(w in low for w in _CLASSIFY_HINT_WORDS):
            return (lambda q, s, f=fn: normalize_label(
                _safe_call(f, q, s))), describe(fn, f"module.{attr}")

    # 2) module-level instance/object with predict
    for attr in ("classifier", "clf", "model", "pipeline", "detector"):
        obj = getattr(module, attr, None)
        if obj is not None and callable(getattr(obj, "predict", None)):
            return (lambda q, s, o=obj: normalize_label(
                _safe_call(o.predict, q, s))), describe(obj, f"module.{attr}.predict")

    # 3) class definition — .load() factory / no-arg init
    art_dir = ROOT / "ambiguity" / "artifacts"
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and not attr.startswith("_") \
                and any(w in attr.lower() for w in _CLASSIFY_HINT_WORDS + ("classifier",)):
            load_fn = getattr(obj, "load", None)
            if callable(load_fn):
                for label_, arg in (("load()", None), (f"load({art_dir.name}/)", art_dir)):
                    try:
                        inst = obj.load(arg) if arg else obj.load()
                    except TypeError:
                        continue
                    except Exception as exc:
                        raise ImportError(
                            f"{attr}.load({arg if arg else ''}) failed — "
                            f"trained artifacts look missing/incomplete ({exc})") from exc
                    if callable(getattr(inst, "predict", None)):
                        return (lambda q, s, o=inst: normalize_label(
                            _safe_call(o.predict, q, s))), describe(
                            inst, f"module.{attr}.load()")
            try:
                inst = obj()
                if callable(getattr(inst, "predict", None)):
                    return (lambda q, s, o=inst: normalize_label(
                        _safe_call(o.predict, q, s))), describe(inst, f"module.{attr}()")
            except Exception:
                continue
    raise ImportError("ambiguity.classifier imported but no usable predict entrypoint found")


# ------------------------------------------------------------ SQL suite bits

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|"
    r"vacuum|reindex|grant|revoke|merge)\b|\breplace\s+into\b", re.IGNORECASE)
_SELECT_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Minimal local extraction (mirrors the production contract, but owned by
    the eval so pipeline changes cannot distort the measurement)."""
    t = (text or "").strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    t = t.strip("`[] \n")
    if not t:
        return ""
    m = re.search(r"(select|with)\b", t, re.IGNORECASE)
    if m:
        t = t[m.start():]
    cut = t.find(";")
    if cut != -1:
        t = t[:cut + 1]
    return t.strip()


def is_valid_read_only(sql: str):
    """Production gate: pipeline.validate_read_only (sqlglot-backed). Any
    import failure falls back to the spec'd regex so the eval can still run
    standalone."""
    global _VALIDATE
    if _VALIDATE is None:
        try:
            _VALIDATE = importlib.import_module("pipeline").validate_read_only
        except Exception:
            _VALIDATE = False
    if _VALIDATE:
        out, why = _VALIDATE(sql)
        return out is not None, why
    return _is_valid_read_only_regex(sql)


_VALIDATE = None


def _is_valid_read_only_regex(sql: str):
    """Spec'd regex check: starts SELECT/WITH, no write keywords, single stmt."""
    if not sql:
        return False, "no SQL produced"
    if not _SELECT_START.match(sql):
        return False, "does not start with SELECT or WITH"
    if _WRITE_KEYWORDS.search(sql):
        return False, "contains a write/DDL keyword"
    body = sql.rstrip(";").strip()
    if ";" in body:
        return False, "contains multiple statements"
    return True, ""


def round_row(row, ndigits=4):
    return tuple(round(v, ndigits) if isinstance(v, float) else v for v in row)


def _normalized_col_keys(cols):
    seen, keys = set(), []
    for c in cols:
        key = re.sub(r"\s+", " ", str(c or "").strip().casefold()) or "(blank)"
        while key in seen:
            key += "#"
        seen.add(key)
        keys.append(key)
    return keys


def row_multisets_strict(rows_a, rows_b):
    """Strict comparator: exact multiset, floats rounded to 4 dp, order and
    column names ignored (the production-style grading)."""
    return Counter(map(round_row, rows_a)) == Counter(map(round_row, rows_b))


def row_multisets_normalized(cols_a, rows_a, cols_b, rows_b):
    """Fair comparator: project onto shared columns by name, floats at 2 dp.

    Two documented normalizations (both measured as grading artifacts in the
    2026-10-29 run):
    * projection — a semantically-correct answer may SELECT extra columns the
      gold query does not (6 of 11 "failures" were exactly this);
    * 2 dp rounding — the same AVG() can differ in the 3rd decimal through a
      different, mathematically-equivalent query plan.

    Returns (equal, mode); mode in {"exact", "projected", "no_shared_cols"}.
    With no usable column names it degrades to a 2 dp row-multiset check.
    """
    two_dp = lambda row: round_row(row, ndigits=2)

    if not cols_a or not cols_b:
        return (Counter(map(two_dp, rows_a)) == Counter(map(two_dp, rows_b)),
                "no_shared_cols")
    keys_a, keys_b = _normalized_col_keys(cols_a), _normalized_col_keys(cols_b)
    shared = {k for k in keys_a if k in keys_b}
    if not shared:
        return (Counter(map(two_dp, rows_a)) == Counter(map(two_dp, rows_b)),
                "no_shared_cols")
    idx_a = [i for i, k in enumerate(keys_a) if k in shared]
    idx_b = [i for i, k in enumerate(keys_b) if k in shared]
    proj_a = [tuple(r[i] for i in idx_a) for r in rows_a]
    proj_b = [tuple(r[i] for i in idx_b) for r in rows_b]
    equal = (Counter(map(two_dp, proj_a)) == Counter(map(two_dp, proj_b)))
    projected = len(set(idx_a)) != len(keys_a) or len(set(idx_b)) != len(keys_b)
    return equal, ("projected" if projected else "exact")



def run_query(sql, db_uri):
    con = sqlite3.connect(db_uri, uri=True)
    try:
        cur = con.execute(sql)
        if cur.description is None:
            raise ValueError("returned no result set")
        cols = [d[0] for d in cur.description]
        rows = [tuple(r) for r in cur.fetchall()]
        return cols, rows
    finally:
        con.close()


def render_schema_local(db_file: Path) -> str:
    """Minimal fallback renderer: PRAGMA-driven CREATE TABLE per table."""
    uri = f"file:{db_file}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    blocks = []
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND "
            "name NOT LIKE 'sqlite_%' ORDER BY name")]
        for tb in tables:
            cols = con.execute(f'PRAGMA table_info("{tb}")').fetchall()
            decl = []
            for _, name, ctype, notnull, _dflt, pk in cols:
                parts = [f"{name} {ctype}"]
                if pk:
                    parts.append("PRIMARY KEY")
                elif notnull:
                    parts.append("NOT NULL")
                decl.append(" ".join(parts))
            blocks.append(f"CREATE TABLE {tb} (\n  " + ",\n  ".join(decl) + "\n);")
    finally:
        con.close()
    return "\n\n".join(blocks)


def render_schema_via_db_introspect(db_file: Path):
    """Try the peer's db_introspect module; return (schema_text, source) or
    raise ImportError-ish reason."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        di = importlib.import_module("db_introspect")
    except Exception as exc:
        raise ImportError(f"db_introspect not importable ({exc})")
    for attr in dir(di):
        if attr.startswith("_") or attr in ("sqlite3", "re", "json", "os"):
            continue
        obj = getattr(di, attr)
        if not callable(obj):
            continue
        low = attr.lower()
        if any(w in low for w in ("render", "schema", "describe", "introspect",
                                  "context", "build")):
            try:
                out = obj(str(db_file))
            except TypeError:
                try:
                    out = obj(db_file)
                except TypeError:
                    continue
            except Exception:
                continue
            if isinstance(out, str) and out.strip():
                return out, f"db_introspect.{attr}()"
            if isinstance(out, dict) and out.get("schema_text"):
                return out["schema_text"], f"db_introspect.{attr}()"
    raise ImportError("db_introspect imported but no usable schema renderer found")


# ----------------------------------------------------------- sql suite main

def run_sql_suite(limit=None, verbose=True):
    """Returns (report_dict, status, skip_reason) with status run|not_run."""
    if not STUDENTS_DB.exists():
        return None, "not_run", "students.db not found"
    if not SQL_SET.exists():
        return None, "not_run", f"{SQL_SET.relative_to(ROOT)} missing"

    try:
        pairs = load_jsonl(SQL_SET)
    except Exception as exc:
        return None, "not_run", f"cannot parse sql_eval.jsonl: {exc}"
    if limit:
        pairs = pairs[:limit]

    try:
        from model_runtime import SQL_SYSTEM_PROMPT, get_runtime
    except Exception as exc:
        return None, "not_run", f"model_runtime not importable: {exc}"

    try:
        build_schema = importlib.import_module("pipeline").build_schema
        schema_source_default = "pipeline.build_schema (production)"
    except Exception as exc:
        build_schema = None
        schema_source_default = None

    if verbose:
        print("[eval_full] starting SQL model server (this allocates once)…")
    t_start = time.perf_counter()
    try:
        rt = get_runtime()
    except Exception as exc:
        return None, "not_run", f"model server failed to start: {exc}"
    warmup_s = round(time.perf_counter() - t_start, 1)

    # the model actually SERVED — the GGUF constant was a lie in the old eval
    served_gguf = Path(getattr(rt, "gguf", "unknown"))
    try:
        served_path = str(served_gguf.resolve().relative_to(ROOT))
    except ValueError:
        served_path = served_gguf.name
    model_identity = {"gguf": served_gguf.name, "path": served_path}
    try:
        import hashlib
        h = hashlib.sha256()
        with open(served_gguf, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        model_identity["sha256"] = h.hexdigest()
        model_identity["size_bytes"] = served_gguf.stat().st_size
        model_identity["mtime"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(served_gguf.stat().st_mtime))
    except OSError as exc:
        model_identity["identity_error"] = str(exc)

    attempts, details = [], []
    for p in pairs:
        db = p.get("db", "students")
        if db == "students":
            db_file = STUDENTS_DB
        elif db == "join_demo":
            db_file = JOIN_DB
            if not db_file.exists():
                builder = EVALS / "tmp" / "make_join_demo_db.py"
                if builder.exists():
                    import subprocess as sp
                    sp.run([sys.executable, str(builder)], check=True,
                           capture_output=True)
                else:
                    attempts.append({"question": p["question"], "status": "SKIPPED",
                                     "note": f"{JOIN_DB.name} missing and no builder"})
                    continue
        else:
            db_file = EVALS / "tmp" / f"{db}.db"
        if not db_file.exists():
            attempts.append({"question": p["question"], "status": "SKIPPED",
                             "note": f"database file missing: {db_file.name}"})
            continue

        # production parity: exact user-message shape of pipeline.answer_question
        item = {"question": p["question"], "category": p.get("category"), "db": db}
        if build_schema is not None:
            try:
                schema = build_schema(db_file, [], question=p["question"])
                schema_src = schema_source_default
            except Exception as exc:
                schema, schema_src = render_schema_local(db_file), \
                    f"local_minimal_renderer (build_schema failed: {exc})"
        else:
            try:
                schema, schema_src = render_schema_via_db_introspect(db_file)
            except ImportError:
                schema, schema_src = render_schema_local(db_file), "local_minimal_renderer"
        item["schema_source"] = schema_src
        user = (f"Operation mode: READ_ONLY\n\n{schema}\n\n"
                f"Question: {p['question']}\n\nWrite one SQLite query.")

        t0 = time.perf_counter()
        try:
            raw = rt.generate_chat(
                [{"role": "system", "content": SQL_SYSTEM_PROMPT},
                 {"role": "user", "content": user}], max_tokens=220)
            dt = time.perf_counter() - t0
        except Exception as exc:
            attempts.append({**item, "status": "ERROR",
                             "error": f"generation failed: {exc}"})
            continue

        sql = extract_sql(raw)
        item.update({"latency_s": round(dt, 2), "raw_prefix": (raw or "")[:120]})
        produced = bool(sql)
        valid, why = (is_valid_read_only(sql) if produced else (False, "no SQL"))
        item.update({"sql": sql, "valid_read_only": valid, "invalid_reason": why})
        executed = False
        if valid:
            try:
                gold_cols, gold_rows = run_query(p["gold_sql"], f"file:{db_file}?mode=ro")
                if len(gold_rows) != p.get("expected_result_row_count"):
                    item["gold_note"] = (f"gold row count changed: live {len(gold_rows)} "
                                         f"vs stored {p.get('expected_result_row_count')}")
                got_cols, got_rows = run_query(sql, f"file:{db_file}?mode=ro")
                executed = True
                item["exec_match_strict"] = row_multisets_strict(gold_rows, got_rows)
                item["exec_match_norm"], item["norm_mode"] = row_multisets_normalized(
                    gold_cols, gold_rows, got_cols, got_rows)
                item["gold_rows"] = len(gold_rows)
                item["got_rows"] = len(got_rows)
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
        item["executed"] = executed
        ok = item.get("exec_match_norm")           # graded on the fair comparator
        status = ("PASS" if ok else
                  ("PASS_STRICT_ONLY" if item.get("exec_match_strict") is True else
                   ("EXEC-FAIL" if executed else
                    ("INVALID" if produced else "EMPTY"))))
        item["status"] = status
        attempts.append(item)
        details.append(item)

    n_attempt = len(attempts)
    n_skip_bada = [a for a in attempts if a["status"] == "SKIPPED"]
    n_eval = [a for a in attempts if a["status"] in ("PASS", "PASS_STRICT_ONLY",
                                                     "EXEC-FAIL", "INVALID",
                                                     "EMPTY", "ERROR")]
    produced = [a for a in attempts if a.get("sql")]
    valids = [a for a in attempts if a.get("valid_read_only")]
    execs = [a for a in attempts if a.get("executed")]
    exec_ok_norm = [a for a in attempts if a.get("exec_match_norm")]
    exec_ok_strict = [a for a in attempts if a.get("exec_match_strict")]
    norm_only = [a for a in attempts
                 if a.get("exec_match_norm") and not a.get("exec_match_strict")]
    lats = [a["latency_s"] for a in attempts if "latency_s" in a]

    cats = {}
    for a in attempts:
        c = cats.setdefault(a.get("category"), {"n": 0, "exec_ok": 0,
                                                "mean_latency_s": None})
        c["n"] += 1
        c["exec_ok"] += 1 if a.get("exec_match_norm") else 0
    lat_by_cat = {}
    for a in attempts:
        if "latency_s" in a:
            lat_by_cat.setdefault(a.get("category"), []).append(a["latency_s"])
    for c, ls in lat_by_cat.items():
        cats[c]["mean_latency_s"] = round(statistics.mean(ls), 2)

    report = {
        "status": "run",
        "model": getattr(rt, "model_name", str(served_gguf)),
        "model_identity": model_identity,
        "prompt_parity": "production (READ_ONLY mode, SQL_SYSTEM_PROMPT, "
                         "pipeline.build_schema, max_tokens=220, temp=0)",
        "comparator": "graded on exec_match_norm (projection+2dp); strict "
                      "(4dp exact multiset) reported alongside",
        "server_warmup_s": warmup_s,
        "schema_source_counts": dict(Counter(a.get("schema_source") for a in attempts
                                             if "schema_source" in a)),
        "n_pairs": len(pairs),
        "n_attempted": n_attempt,
        "n_evaluated": len(n_eval),
        "n_skipped_bada": len(n_skip_bada),
        "sql_produced": len(produced),
        "sql_produced_rate": round(len(produced) / max(1, len(n_eval)), 4),
        "valid_read_only": len(valids),
        "valid_rate": round(len(valids) / max(1, len(produced)), 4) if produced else 0.0,
        "invalid_sql_rate": round(1 - len(valids) / max(1, len(n_eval)), 4),
        "executed": len(execs),
        "executed_rate": round(len(execs) / max(1, len(valids)), 4) if valids else 0.0,
        "exec_accuracy": round(len(exec_ok_norm) / max(1, len(n_eval)), 4),
        "exec_accuracy_strict": round(len(exec_ok_strict) / max(1, len(n_eval)), 4),
        "normalized_only_passes": len(norm_only),
        "latency_s": {
            "mean": round(statistics.mean(lats), 2) if lats else None,
            "median": round(statistics.median(lats), 2) if lats else None,
            "p95": round(sorted(lats)[int(math.ceil(len(lats) * 0.95)) - 1], 2) if lats else None,
            "total": round(sum(lats), 1) if lats else None,
        },
        "by_category": cats,
        "details": details,
    }
    return report, "run", ""


# =============================================================== classifier

def run_classifier_suite(limit=None, quiet=False):
    """Returns (report_dict, status, skip_reason)."""
    if not CLASSIFIER_SET.exists():
        return None, "not_run", f"{CLASSIFIER_SET.relative_to(ROOT)} missing"
    try:
        pairs = load_jsonl(CLASSIFIER_SET)
    except Exception as exc:
        return None, "not_run", f"cannot parse classifier_eval.jsonl: {exc}"
    if limit:
        pairs = pairs[:limit]

    try:
        predict, describe = build_classifier_predictor()
    except Exception as exc:
        return None, "not_run", str(exc)

    if not quiet:
        print(f"[eval_full] classifier artifact loaded: {describe}")

    y_true, y_pred, err_rows = [], [], []
    by_schema = {}
    for p in pairs:
        gold = p["label"]
        schema_text = p.get("schema_text") or ""
        try:
            pred = predict(p["question"], schema_text)
        except Exception as exc:
            err_rows.append({"question": p["question"], "schema_id": p["schema_id"],
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        if pred is None:
            err_rows.append({"question": p["question"], "schema_id": p["schema_id"],
                             "error": "unparseable/unknown label returned"})
            continue
        y_true.append(gold)
        y_pred.append(pred)
        s = by_schema.setdefault(p["schema_id"], {"n": 0, "ok": 0, "gold": Counter(),
                                                  "pred": Counter()})
        s["n"] += 1
        s["ok"] += 1 if pred == gold else 0
        s["gold"][gold] += 1
        s["pred"][pred] += 1

    metrics = classification_metrics(y_true, y_pred)
    schema_acc = {sid: {"accuracy": round(v["ok"] / v["n"], 4) if v["n"] else 0.0,
                        "n": v["n"], "gold": dict(v["gold"]), "pred": dict(v["pred"])}
                  for sid, v in by_schema.items()}
    report = {
        "status": "run",
        "artifact": describe,
        "n_pairs": len(pairs),
        "n_scored": len(y_true),
        "n_prediction_errors": len(err_rows),
        "prediction_errors": err_rows,
        **metrics,
        "per_schema": schema_acc,
    }
    return report, "run", ""


# ------------------------------------------------------------------- reports

def write_report(reports: dict):
    out_json = EVALS / "report-latest.json"
    out_md = EVALS / "REPORT.md"

    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "root": ".", **reports}
    out_json.write_text(json.dumps(payload, indent=2))

    md = [f"# Clarion — held-out evaluation report "
          f"({payload['generated_at']})", ""]
    cls = reports.get("classifier") or {}
    sql = reports.get("sql") or {}
    md.append("## suite status")
    md.append("")
    md.append("| suite | status | detail |")
    md.append("|---|---|---|")
    cls_status = cls.get("status") if cls.get("status") else "not_run"
    md.append(f"| classifier | {cls_status} | "
              f"{cls.get('reason', '') if isinstance(cls.get('reason'), str) else 'artifact: ' + str(cls.get('artifact',''))} |")
    md.append(f"| sql | {sql.get('status','not_run')} | "
              f"{sql.get('reason', '') if isinstance(sql.get('reason'), str) else 'model: ' + str(sql.get('model',''))} |")
    if cls.get("status") == "run":
        md += ["", f"**Classifier** — artifact `{cls['artifact']}` "
                   f"(scored {cls['n_scored']}/{cls['n_pairs']}, "
                   f"{cls['n_prediction_errors']} prediction errors)", "",
               f"- accuracy: **{cls['accuracy']}** — macro-F1: **{cls['macro_f1']}**",
               f"- ambiguity precision **{cls['ambiguity']['precision']}** / "
               f"recall **{cls['ambiguity']['recall']}** / F1 **{cls['ambiguity']['f1']}**", "",
               "per-class prec/recall/F1/support:", "",
               "| label | precision | recall | F1 | support |",
               "|---|---|---|---|---|"]
        for lab, v in cls["per_class"].items():
            md.append(f"| {lab} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |")
        cm = cls["confusion_matrix"]
        md += ["", "confusion matrix (rows = gold, cols = "
                   f"{cm['columns_are']}):", "", "| gold \\ pred | "
               " | ".join(cm["columns_are"]) + " |",
               "|---|" + "---|" * len(cm["columns_are"])]
        for rname, row in zip(cm["rows_are"], cm["matrix"]):
            md.append(f"| {rname} | " + " | ".join(str(x) for x in row) + " |")
        md += ["", "per-schema accuracy:", "",
               "| schema | n | accuracy |", "|---|---|---|"]
        for sid, v in cls["per_schema"].items():
            md.append(f"| {sid} | {v['n']} | {v['accuracy']} |")
    if sql.get("status") == "run":
        ident = sql.get("model_identity") or {}
        md += ["", f"**SQL suite** — server model `{sql['model']}` "
                   f"(gguf: `{ident.get('gguf', '?')}`, "
                   f"sha256 `{(ident.get('sha256') or '?')[:16]}…`, "
                   f"{ident.get('size_bytes', '?')} bytes)", "",
               f"prompt parity: {sql.get('prompt_parity', '?')}", "",
               f"schema sources: {sql['schema_source_counts']} · "
               f"server warmup {sql.get('server_warmup_s', '?')} s", "",
               f"- pairs evaluated: **{sql['n_evaluated']}**/{sql['n_pairs']}",
               f"- sql produced: {sql['sql_produced_rate']} • "
               f"valid read-only: {sql['valid_rate']} • "
               f"invalid-SQL rate: {sql['invalid_sql_rate']}",
               f"- executed: {sql['executed_rate']}",
               f"- execution accuracy (normalized: projection+2dp): "
               f"**{sql['exec_accuracy']}** · strict (4dp exact): "
               f"**{sql['exec_accuracy_strict']}** "
               f"(passed only the fair comparator: {sql.get('normalized_only_passes', 0)})",
               f"- latency s: mean {sql['latency_s']['mean']} / median "
               f"{sql['latency_s']['median']} / p95 {sql['latency_s']['p95']} "
               f"(total {sql['latency_s']['total']})", "",
               "| category | n | exec_ok | accuracy | mean latency |",
               "|---|---|---|---|---|"]
        for cat, c in sql["by_category"].items():
            md.append(f"| {cat} | {c['n']} | {c['exec_ok']} | "
                      f"{round(c['exec_ok'] / c['n'], 4)} | {c['mean_latency_s']} |")
        md += ["", "failed pairs:", ""]
        bad = [d for d in sql["details"] if d.get("status") != "PASS"]
        if not bad:
            md.append("(none — every generated SQL matched gold results)")
        for d in bad:
            md.append(f"- **{d['status']}** {d['question'][:70]} — "
                      f"{d.get('invalid_reason') or d.get('error') or 'result mismatch'}"
                      f" · got: `{(d.get('sql') or '')[:90]}`")
    out_md.write_text("\n".join(md) + "\n")
    return out_json, out_md


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0,
                    help="cap pairs per suite (smoke test)")
    ap.add_argument("--suites", default="all",
                    choices=["all", "classifier", "sql"])
    args = ap.parse_args()
    limit = args.limit or None

    print("=" * 68)
    print("Clarion held-out evaluation (brutal tester build)")
    print("=" * 68)

    suites = {"classifier": None, "sql": None}
    if args.suites in ("all", "classifier"):
        print("\n[1/2] classifier suite")
        rep, status, reason = run_classifier_suite(limit)
        if status != "run":
            print(f"  detector suite: NOT RUN ({reason})")
            suites["classifier"] = not_run(reason)
        else:
            suites["classifier"] = rep
            print(f"  scored {rep['n_scored']}/{rep['n_pairs']} — "
                  f"accuracy {rep['accuracy']} macro-F1 {rep['macro_f1']} "
                  f"amb-recall {rep['ambiguity']['recall']}")
    if args.suites in ("all", "sql"):
        print("\n[2/2] SQL suite")
        rep, status, reason = run_sql_suite(limit)
        if status != "run":
            print(f"  SQL suite: NOT RUN ({reason})")
            suites["sql"] = not_run(reason)
        else:
            suites["sql"] = rep
            print(f"  evaluated {rep['n_evaluated']}/{rep['n_pairs']} — "
                  f"exec accuracy {rep['exec_accuracy']} "
                  f"(strict {rep['exec_accuracy_strict']}, "
                  f"{rep['normalized_only_passes']} norm-only passes) "
                  f"invalid-SQL rate {rep['invalid_sql_rate']} "
                  f"model {rep['model_identity'].get('gguf', '?')}")

    j, m = write_report(suites)
    print(f"\nreport: {j}\n        {m}")
    if suites["classifier"] and suites["classifier"].get("status") == "not_run" \
            and suites["sql"] and suites["sql"].get("status") == "not_run":
        print("\nNOTHING was measured this run — see reasons above.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[eval_full] interrupted")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        print("\n[eval_full] crashed — fix the harness, not the ARTIFACTS"
              " (a crashed harness must be repaired before any number counts)")
        sys.exit(1)

