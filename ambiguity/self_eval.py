"""Debug self-evaluation for the Clarion ambiguity classifier.

Loads the artifacts (training first when ``--train`` is passed) and prints:
* accuracy on train / val / test / test_unseen schemas,
* 3x3 confusion matrices, macro-F1, and per-class P/R/F for AMBIGUOUS,
* the unseen-schema slice metrics (the number that matters),
* the top confusion (gold, pred) pairs with concrete example pairs,
plus a regression guard: it raises if any of the three classes is never
predicted on a slice (class collapse).

``--write-md`` regenerates ``ambiguity/SELF_EVAL.md`` from the same numbers.

Run:  python -m ambiguity.self_eval [--train] [--write-md]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, classification_report,
)

from .classifier import (
    LABELS, EMBED_MODEL, AmbiguityClassifier, load_jsonl,
    make_input_text, _encode, _macro_f1,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ambiguity_train"
DEFAULT_ART = Path(__file__).resolve().parent / "artifacts"
SLICES = ("train", "val", "test", "test_unseen")


def evaluate(clf, rows):
    texts = [make_input_text(r["question"], r["schema_text"]) for r in rows]
    x = _encode(clf.model, texts, schema_texts=[r["schema_text"] for r in rows])
    proba = np.asarray(clf.clf.predict_proba(x), dtype=float)
    order = list(clf.clf.classes_)
    proba = proba[:, [order.index(c) for c in LABELS]]
    bias = np.array([clf.thresholds.get(c, 0.0) for c in LABELS])
    thr_idx = np.argmax(proba + bias, axis=1)
    raw_idx = np.argmax(proba, axis=1)
    y = [r["label"] for r in rows]
    y_thr = [LABELS[i] for i in thr_idx]
    y_raw = [LABELS[i] for i in raw_idx]
    return {
        "n": len(rows),
        "y": y,
        "probs": proba,
        "pred_thr": y_thr,
        "pred_raw": y_raw,
        "acc_thr": float(accuracy_score(y, y_thr)),
        "f1_thr": _macro_f1(y, y_thr),
        "acc_raw": float(accuracy_score(y, y_raw)),
        "f1_raw": _macro_f1(y, y_raw),
        "cm": confusion_matrix(y, y_thr, labels=LABELS).tolist(),
        "cls_report": classification_report(
            y, y_thr, labels=LABELS, zero_division=0, output_dict=True),
    }


def _print_slice(name, res):
    print(f"\n=== {name} (n={res['n']}) ===")
    print(f"accuracy(thresholded)={res['acc_thr']:.3f}   "
          f"accuracy(raw-argmax)={res['acc_raw']:.3f}")
    print(f"macro-F1(thresholded)={res['f1_thr']:.3f}   "
          f"macro-F1(raw-argmax)={res['f1_raw']:.3f}")
    print("confusion (rows=gold, cols=pred)  order: " + ", ".join(LABELS))
    for row in res["cm"]:
        print("  " + "  ".join(str(v) for v in row))
    amb = res["cls_report"]["AMBIGUOUS"]
    und = res["cls_report"]["UNDERSPECIFIED"]
    print(f"AMBIGUOUS      P={amb['precision']:.3f} R={amb['recall']:.3f} "
          f"F1={amb['f1-score']:.3f}")
    print(f"UNDERSPECIFIED P={und['precision']:.3f} R={und['recall']:.3f} "
          f"F1={und['f1-score']:.3f}")


def top_confusions(res, errs_per_pair):
    """[(gold, pred, count, [example rows])] sorted by count."""
    pairs = Counter((r["label"], p)
                    for r, p in zip(res["_rows"], res["pred_thr"])
                    if r["label"] != p)
    bucket = defaultdict(list)
    for r, p in zip(res["_rows"], res["pred_thr"]):
        if r["label"] != p:
            bucket[(r["label"], p)].append(r)
    out = []
    for (g, p), n in pairs.most_common():
        out.append((g, p, n, bucket[(g, p)]))
    return out


def run_eval(art_dir, data_dir):
    clf = AmbiguityClassifier.load(art_dir)
    results = {}
    for slice_name in SLICES:
        rows = load_jsonl(Path(data_dir) / f"{slice_name}.jsonl")
        res = evaluate(clf, rows)
        res["_rows"] = rows
        results[slice_name] = res

    for slice_name in SLICES:
        _print_slice(slice_name, results[slice_name])

    # regression guard: no class collapse on any held-out slice
    for slice_name in ("test", "test_unseen"):
        pred_set = set(results[slice_name]["pred_thr"])
        if pred_set != set(LABELS):
            raise RuntimeError(
                f"class collapse on {slice_name}: predicted classes = {pred_set}")

    t = results["test"]
    u = results["test_unseen"]
    print("\n=== summary ===")
    print(f"seen-schema test:   acc={t['acc_thr']:.3f} macroF1={t['f1_thr']:.3f}")
    print(f"UNSEEN-schema test: acc={u['acc_thr']:.3f} macroF1={u['f1_thr']:.3f}"
          f"   (thresholds tuned on val only)")
    print(f"AMBIGUOUS unseen:   P={u['cls_report']['AMBIGUOUS']['precision']:.3f}"
          f"  R={u['cls_report']['AMBIGUOUS']['recall']:.3f}")

    return clf, results


def write_markdown(results, art_dir):
    t, v, tr, u = (results["test"], results["val"], results["train"],
                   results["test_unseen"])

    def conf_rows(res):
        return "<br>".join(
            f"| {gold} | {pred}: {n} |" for gold, pred, n, _ in
            top_confusions(res, None))

    cons_pairs = []
    for res_name in ("test", "test_unseen"):
        for gold, pred, n, exs in top_confusions(results[res_name], None):
            for ex in exs[:2]:
                cons_pairs.append((res_name, ex, gold, pred))

    amb_u = u["cls_report"]["AMBIGUOUS"]
    und_u = u["cls_report"]["UNDERSPECIFIED"]
    md = []
    md.append("# Ambiguity classifier — self-evaluation (actual numbers)\n")
    md.append("Headline number: **unseen-schema test slice** "
              f"(flights, streaming — never seen in training) "
              f"accuracy **{u['acc_thr']:.3f}**, macro-F1 "
              f"**{u['f1_thr']:.3f}**.\n")
    md.append("| slice | n | accuracy (thr) | macro-F1 (thr) | "
              "macro-F1 (raw argmax) |")
    md.append("|---|---|---|---|---|")
    for name, res in (("train", tr), ("val", v), ("test", t),
                      ("test_unseen", u)):
        md.append(f"| {name} | {res['n']} | {res['acc_thr']:.3f} | "
                  f"{res['f1_thr']:.3f} | {res['f1_raw']:.3f} |")
    md.append("")
    low = [nm for nm, res in (("seen-schema test", t),
                              ("unseen-schema test", u))
           if res["acc_thr"] < 0.80]
    if low:
        md.append(
            "**Statement: accuracy on the test slices is below 0.80** (slices: "
            f"{', '.join(low)}). Honest cause: the "
            "classifier is a linear head (LogisticRegression) on frozen "
            "all-MiniLM-L6-v2 embeddings of a single concatenated string; it "
            "must learn non-linear interactions (a ranking phrase must be "
            "read *against* which numeric columns exist in that very schema) "
            "that are only partially linear in embedding space, and the "
            "synthetic label semantics are strict (one rival column flips "
            "CLEAR to AMBIGUOUS).")
    md.append("")
    for name, res in (("test (seen schemas)", t),
                      ("test_unseen (flights, streaming)", u)):
        md.append(f"### Confusion — {name} (rows=gold, cols=pred; "
                  "order CLEAR, AMBIGUOUS, UNDERSPECIFIED)\n")
        md.append("```")
        for row in res["cm"]:
            md.append("  ".join(str(x) for x in row))
        md.append("```")
        md.append("")
        amb = res["cls_report"]["AMBIGUOUS"]
        und = res["cls_report"]["UNDERSPECIFIED"]
        md.append(f"AMBIGUOUS P={amb['precision']:.3f} R={amb['recall']:.3f} "
                  f"F1={amb['f1-score']:.3f}; UNDERSPECIFIED "
                  f"P={und['precision']:.3f} R={und['recall']:.3f} "
                  f"F1={und['f1-score']:.3f}\n")

    md.append("### Top confusions with example pairs\n")
    for slice_name, ex, gold, pred in cons_pairs:
        md.append(f"- [{slice_name}] gold **{gold}** -> predicted **{pred}** "
                  f"({ex['schema_id']}): \"{ex['question']}\" "
                  f"on schema `{ex['schema_id']}`")
    md.append("")
    md.append("### Regression guard\n")
    for name in ("test", "test_unseen"):
        pred_classes = set(results[name]["pred_thr"])
        md.append(f"- {name}: predicted classes = {sorted(pred_classes)} "
                  "→ no collapse ✔" if pred_classes == set(LABELS)
                  else f"- {name}: **COLLAPSE** predicted = {pred_classes}")
    md.append("")
    md.append("### Limitations\n")
    md.append(
        "The dataset is synthetic: template-generated schemas from 7 families "
        "with randomized naming, so distribution shift from real SQLite files "
        "is expected. Labels follow our operational semantics (multiple "
        "schema-supported readings = AMBIGUOUS; missing criterion/reference or "
        "attribute the schema cannot provide = UNDERSPECIFIED); humans can "
        "reasonably disagree at mechanism boundaries (e.g. vague adjectives "
        "could be treated as ambiguous rather than underspecified). The "
        "embedding is capped at 256 tokens, so very long user schemas get "
        "truncated at the question side. Thresholds are additive biases tuned "
        "greedily on val — not a calibration.)".replace(")", ""))
    path = Path(art_dir).parent / "SELF_EVAL.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(DEFAULT_ART))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--train", action="store_true",
                    help="(re)train first via classifier.train_and_save")
    ap.add_argument("--train-file", default="train.jsonl",
                    help="train split file under data-dir (default train.jsonl; "
                         "'train_v2.jsonl' trains on the convention-balanced "
                         "augmentation, 'train_blend.jsonl' on v1+v2 merged)")
    ap.add_argument("--val-file", default="val.jsonl",
                    help="val split file under data-dir (default val.jsonl; "
                         "'val_blend.jsonl' = v1 + convention-B val_v2)")
    ap.add_argument("--write-md", action="store_true",
                    help="write ambiguity/SELF_EVAL.md")
    args = ap.parse_args()

    if args.train:
        from .classifier import train_and_save
        metrics = train_and_save(Path(args.data_dir) / args.train_file,
                                 Path(args.data_dir) / args.val_file,
                                 args.artifacts)
        print(json.dumps(metrics, indent=2))

    _clf, results = run_eval(args.artifacts, args.data_dir)
    if args.write_md:
        write_markdown(results, args.artifacts)


if __name__ == "__main__":
    main()
