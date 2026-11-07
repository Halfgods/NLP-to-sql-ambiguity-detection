"""Schema-aware 3-class ambiguity classifier (CPU only).

Pipeline: SentenceTransformer("all-MiniLM-L6-v2") embedding of
    "User question:\n{question}\n\nDatabase schema:\n{schema_text}"
(QUESTION FIRST — see below) plus a small handcrafted "schema context"
feature vector, followed by a multinomial LogisticRegression
(class_weight="balanced").

WHY QUESTION-FIRST: all-MiniLM-L6-v2 truncates at 256 tokens of the
concatenated input. With the schema in front, a wide schema could push the
*question itself* past the truncation point, so the classifier literally
never saw what was asked. The question now comes first; the schema follows
and gets truncated instead (a schema-only truncation degrades feature
quality, not the basic label contract).

The schema-context features (numeric column count, numeric rivals that
share tokens with the question, named-metric flag, total column count,
question length) exist because a linear head on frozen sentence embeddings
cannot express context-dependent labels — "top X" flips CLEAR↔AMBIGUOUS
depending on how many numeric rivals the schema holds, an interaction no
frozen embedding can see.

Per-class probability thresholds are tuned on the validation split to
maximize macro-F1 (greedy coordinate ascent over additive biases) and stored
in ``thresholds.json``. Additionally, per-class *block* thresholds (used
only on the classifier-only path, when no SLM verdict is available) are
precision-targeted on the validation split and stored in ``meta.json``.

Artifacts (``ambiguity/artifacts/``):
    classifier.pkl    joblib dump {clf, label_order}
    thresholds.json   per-class additive biases + val scores
    meta.json         embedding model, labels, dataset stats, block thresholds

Constraints honored:
* ``torch.set_num_threads(4)``; device="cpu"; the GPU is never touched.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

torch.set_num_threads(4)
try:  # only warn: inter-op pool may already be set in the process
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from sentence_transformers import SentenceTransformer  # noqa: E402  (after threads)

LABELS = ["CLEAR", "AMBIGUOUS", "UNDERSPECIFIED"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"
_GRID_C = (0.5, 1.0, 2.0, 4.0, 8.0)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_relative(path) -> str:
    """Keep generated metadata free of machine-specific absolute paths."""
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return Path(path).name
CONTEXT_FEATURE_VERSION = "v1_numeric_rivals"

_NUMERIC_TYPE_RE = re.compile(
    r"\b(?:INT(?:EGER)?|REAL|FLOAT|DOUBLE|NUMERIC|DECIMAL|MONEY)\b", re.I)
_IDENT_TYPE_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(?:[A-Za-z_][A-Za-z0-9_]*)?\s*"
    r"((?:INT(?:EGER)?|REAL|FLOAT|DOUBLE|NUMERIC|DECIMAL|MONEY)\b)", re.I)
_WORD = re.compile(r"[a-z0-9]+")


def make_input_text(question: str, schema_text: str) -> str:
    """Exact representation contract for this classifier (question first).

    MiniLM truncates the concatenated input at 256 tokens; putting the
    question in front guarantees the *question* is always encoded even when
    the schema is huge (schema-only truncation degrades features, not the
    label semantics).
    """
    return f"User question:\n{question}\n\nDatabase schema:\n{schema_text}"


def _plural_stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def schema_context_features(question: str, schema_text: str) -> np.ndarray:
    """Handcrafted schema-context vector (context_feature_version).

    Returns a 5-dim float32 vector in [0, ~1.5]:
        [numeric_cols/10, rival_metrics/5, named_metric, total_cols/60,
         question_tokens/20]
    ``rival_metrics`` = numeric columns whose *name tokens* share a token
    (or a >=4-char prefix) with the question. This is the interaction a
    frozen sentence embedding fundamentally cannot see, and the one that
    makes "top X" mean CLEAR on a 1-metric schema and AMBIGUOUS on a 3-rival
    schema.
    """
    q_tokens = {t for t in _WORD.findall(str(question or "").lower()) if len(t) >= 3}
    q_variants: set[str] = set()
    for t in q_tokens:
        q_variants |= {t, _plural_stem(t)}

    numeric_cols: set[str] = set()
    all_idents: set[str] = set()
    for m in _IDENT_TYPE_RE.finditer(str(schema_text or "")):
        ident = m.group(1).lower()
        if _NUMERIC_TYPE_RE.search(m.group(2)):
            numeric_cols.add(ident)
        all_idents.add(ident)

    rivals = set()
    for ident in numeric_cols:
        ident_variants = {ident, _plural_stem(ident)} | set(_WORD.findall(ident))
        if ident_variants & q_variants or any(
                len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a))
                for a in ident_variants for b in q_variants):
            rivals.add(ident)

    n_cols = len(all_idents)
    q_len = len(q_tokens)
    vec = [
        min(len(numeric_cols) / 10.0, 1.5),
        min(len(rivals) / 5.0, 1.5),
        1.0 if rivals else 0.0,
        min(n_cols / 60.0, 1.5),
        min(q_len / 20.0, 1.5),
    ]
    return np.asarray(vec, dtype=np.float32)


def _encode(model, texts, schema_texts=None):
    """Embed `texts`; optionally append schema-context feature columns."""
    x = np.asarray(model.encode(
        list(texts), batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, device=DEVICE), dtype=np.float32)
    if schema_texts is None:
        return x
    extra = np.stack([
        schema_context_features(q, s)
        for q, s in zip(texts, schema_texts)]).astype(np.float32)
    return np.hstack([x, extra]).astype(np.float32)


def _macro_f1(y, yhat):
    return float(f1_score(y, yhat, labels=LABELS, average="macro"))


def load_jsonl(path):
    path = Path(path)
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tune_thresholds(probs_val, y_val, passes=2, span=0.3, step=0.05):
    """Greedy coordinate ascent over additive per-class biases.

    Prediction = argmax_c (p_c + b_c); ``b`` stays within [-span, span].
    Returns ({label: bias}, best val macro-F1).
    """
    grid = np.arange(-span, span + 1e-9, step)
    b = np.zeros(len(LABELS), dtype=float)

    def f1_with(bias):
        yhat = [LABELS[int(np.argmax(row + bias))] for row in probs_val]
        return _macro_f1(y_val, yhat)

    best = f1_with(b)
    for _pass in range(passes):
        improved = False
        for ci in range(len(LABELS)):
            best_b = b[ci]
            for cand in grid:
                trial = b.copy()
                trial[ci] = cand
                f1 = f1_with(trial)
                if f1 > best + 1e-6:
                    best, best_b, improved = f1, cand, True
            b[ci] = best_b
        if not improved:
            break
    return {lab: float(b[i]) for i, lab in enumerate(LABELS)}, best


class AmbiguityClassifier:
    """Loaded artifact; ``predict(question, schema_text) -> {"label","probs"}."""

    def __init__(self, model, clf, thresholds, block_thresholds=None):
        self.model = model
        self.clf = clf
        self.thresholds = {k: float(v) for k, v in thresholds.items()}
        self.block_thresholds = {
            k: float(v) for k, v in (block_thresholds or {}).items()}
        self._classes = list(getattr(clf, "classes_", LABELS))

    @classmethod
    def load(cls, out_dir):
        out_dir = Path(out_dir)
        import joblib
        blob = joblib.load(out_dir / "classifier.pkl")
        thresholds_path = out_dir / "thresholds.json"
        thresholds = {}
        if thresholds_path.exists():
            with open(thresholds_path, encoding="utf-8") as fh:
                thresholds = json.load(fh).get("biases", {})
        meta_path = out_dir / "meta.json"
        model_name = EMBED_MODEL
        block_thresholds = {}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            model_name = meta.get("embedding_model", model_name)
            block_thresholds = meta.get("block_thresholds", {})
        model = SentenceTransformer(model_name, device=DEVICE)
        return cls(model, blob["clf"], thresholds, block_thresholds)

    def predict_probs(self, question: str, schema_text: str):
        x = _encode(self.model, [make_input_text(question, schema_text)],
                    schema_texts=[schema_text])
        proba = self.clf.predict_proba(x)[0]
        return {c: float(p) for c, p in zip(self._classes, proba)}

    def predict(self, question: str, schema_text: str, use_thresholds: bool = True):
        probs = self.predict_probs(question, schema_text)
        if use_thresholds:
            bias = [self.thresholds.get(c, 0.0) for c in LABELS]
            idx = int(np.argmax([probs[c] + b for c, b in zip(LABELS, bias)]))
            label = LABELS[idx]
        else:
            label = max(probs, key=probs.get)
        return {"label": label, "probs": probs}


def _tune_block_thresholds(probs_val, y_val, target_precision=0.60,
                           step=0.05, min_threshold=0.35, max_threshold=0.95):
    """Per-class *block* thresholds, precision-targeted on the val split.

    A "block" = the classifier confidently asserts its predicted label on the
    classifier-only path (no SLM verdict available). For each non-CLEAR class
    we pick the smallest t in [min_threshold, max_threshold] such that the
    precision of {argmax == c and p_top >= t} on val reaches
    ``target_precision``. Falls back to 1.01 (never block) when no t gets
    there — being honest beats inventing a number.

    ``probs_val`` columns must be in canonical LABELS order.
    """
    probs_val = np.asarray(probs_val, dtype=float)
    out, stats = {}, {}
    for c in LABELS:
        if c == "CLEAR":
            out[c] = 0.0
            stats[c] = {"threshold": 0.0, "note": "CLEAR never blocks"}
            continue
        col = LABELS.index(c)
        denom = sum(1 for i in range(len(y_val))
                    if int(np.argmax(probs_val[i])) == col)
        best = None
        for t in np.arange(min_threshold, max_threshold + 1e-9, step):
            idx = [i for i in range(len(y_val))
                   if int(np.argmax(probs_val[i])) == col
                   and probs_val[i, col] >= t]
            if not idx:
                continue
            precision = sum(1 for i in idx if y_val[i] == c) / len(idx)
            if precision >= target_precision:
                best = {"threshold": round(float(t), 2),
                        "val_precision": round(precision, 4),
                        "val_coverage": round(len(idx) / max(1, denom), 4)}
                break
        out[c] = best["threshold"] if best else 1.01
        stats[c] = best if best else {
            "threshold": 1.01, "val_precision": None,
            "note": f"no threshold reached precision {target_precision} on val"}
    return out, stats


def train_and_save(train_path, val_path, out_dir) -> dict:
    """Embed (+context feats) -> select C by val macro-F1 -> tune thresholds."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_jsonl(train_path)
    val = load_jsonl(val_path)
    y_train = [r["label"] for r in train]
    y_val = [r["label"] for r in val]
    assert set(y_train) == set(LABELS) and set(y_val) == set(LABELS)

    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    x_train = _encode(model, [make_input_text(r["question"], r["schema_text"])
                              for r in train],
                      schema_texts=[r["schema_text"] for r in train])
    x_val = _encode(model, [make_input_text(r["question"], r["schema_text"])
                            for r in val],
                    schema_texts=[r["schema_text"] for r in val])
    encode_seconds = round(time.time() - t0, 1)

    best_clf, best_val, best_c = None, -1.0, None
    for c in _GRID_C:
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=c)
        clf.fit(x_train, y_train)
        f1 = _macro_f1(y_val, clf.predict(x_val))
        if f1 > best_val + 1e-9:
            best_clf, best_val, best_c = clf, f1, c

    yhat_val = best_clf.predict(x_val)
    if set(yhat_val) != set(LABELS):
        raise RuntimeError(
            f"class collapse: validation predictions only cover {set(yhat_val)}")

    probs_val = np.asarray(
        best_clf.predict_proba(x_val), dtype=float)
    class_order = list(best_clf.classes_)
    # store val probs ordered per canonical LABELS for threshold tuning
    probs_val = probs_val[:, [class_order.index(c) for c in LABELS]]
    thresholds, tuned_f1 = _tune_thresholds(probs_val, y_val)
    yhat_tuned = [LABELS[int(np.argmax(row + np.array(
        [thresholds[c] for c in LABELS])))] for row in probs_val]
    block_thresholds, block_stats = _tune_block_thresholds(probs_val, y_val)

    # save classifier
    import joblib
    joblib.dump({"clf": best_clf, "label_order": class_order},
                out_dir / "classifier.pkl")
    with open(out_dir / "thresholds.json", "w", encoding="utf-8") as fh:
        json.dump({
            "biases": thresholds,
            "prediction_rule": "argmax_c (p_c + bias_c)",
            "tuned_on": "val split",
            "val_macro_f1_before_thresholds": round(best_val, 4),
            "val_macro_f1_after_thresholds": round(tuned_f1, 4),
            "best_C": best_c,
        }, fh, indent=2)

    def label_counts(rows):
        from collections import Counter
        return dict(Counter(r["label"] for r in rows))

    counts = {split: label_counts(rows)
              for split, rows in (("train", train), ("val", val))}
    metrics = {
        "train_accuracy": round(float(accuracy_score(y_train, best_clf.predict(x_train))), 4),
        "train_macro_f1": round(_macro_f1(y_train, best_clf.predict(x_train)), 4),
        "val_accuracy": round(float(accuracy_score(y_val, yhat_val)), 4),
        "val_macro_f1": round(_macro_f1(y_val, yhat_val), 4),
        "val_accuracy_thresholded": round(float(accuracy_score(y_val, yhat_tuned)), 4),
        "val_macro_f1_thresholded": round(tuned_f1, 4),
        "best_C": best_c,
        "thresholds": thresholds,
        "block_thresholds": block_thresholds,
        "block_threshold_stats": block_stats,
        "n_train": len(train), "n_val": len(val),
        "embedding_model": EMBED_MODEL,
        "encode_seconds": encode_seconds,
        "labels": LABELS,
        "dataset_stats": {
            split: {**counts[split], "total": tot}
            for split, tot in (("train", len(train)), ("val", len(val)))
        },
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "embedding_model": EMBED_MODEL,
            "labels": LABELS,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": DEVICE,
            "torch_num_threads": torch.get_num_threads(),
            "representation": make_input_text("<question>", "<schema_text>"),
            "input_representation": "question_first",
            "context_feature_version": CONTEXT_FEATURE_VERSION,
            "context_feature_count": 5,
            "block_thresholds": block_thresholds,
            "block_threshold_stats": block_stats,
            "best_C": best_c,
            "thresholds": thresholds,
            "dataset_stats": metrics["dataset_stats"],
            "train_val_files": [_repo_relative(train_path), _repo_relative(val_path)],
        }, fh, indent=2)
    return metrics


__all__ = ["AmbiguityClassifier", "train_and_save", "make_input_text",
           "load_jsonl", "LABELS", "EMBED_MODEL", "schema_context_features",
           "CONTEXT_FEATURE_VERSION"]
