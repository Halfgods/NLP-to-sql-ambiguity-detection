"""Run fair local checkpoint and ambiguity-method comparisons for the report.

Reads project evaluation inputs and SQLite fixtures but writes only
model_comparison_results.json in this report directory; it never overwrites
the project's own evaluation artifact.
"""
import json
import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sqlite3
from ambiguity.analysis import _combine, build_slm_messages, parse_slm_json
from ambiguity.classifier import AmbiguityClassifier
import db_introspect as di
import model_runtime
import scripts.eval_full as eval_full

OUT = HERE / "model_comparison_results.json"
SQL_MODELS = [
    ("Qwen SQL-only Q4_K_M", ROOT / "models/qwen25-coder-1.5b-sql-q4_k_m.gguf"),
    ("Qwen SQL+write F16", ROOT / "models/qwen25-coder-1.5b-sql-write-f16.gguf"),
    ("Qwen SQL+write Q4_K_M (current)", ROOT / "models/qwen25-coder-1.5b-sql-write-q4_k_m.gguf"),
]


def evaluate_sql_models():
    output = []
    for name, path in SQL_MODELS:
        if not path.exists():
            output.append({"name": name, "status": "not available locally", "path": str(path)})
            continue
        print(f"Starting {name}: {path.name}", flush=True)
        rt = model_runtime.LlamaServerRuntime(gguf_path=path)
        try:
            start = time.perf_counter()
            rt.start()
            warmup = round(time.perf_counter() - start, 1)
            model_runtime.get_runtime = lambda _rt=rt: _rt
            result, status, reason = eval_full.run_sql_suite(verbose=False)
            if status != "run":
                output.append({"name": name, "status": status, "reason": reason})
                continue
            output.append({
                "name": name,
                "status": "run",
                "checkpoint": path.name,
                "size_bytes": path.stat().st_size,
                "n_evaluated": result["n_evaluated"],
                "sql_produced_rate": result["sql_produced_rate"],
                "valid_rate": result["valid_rate"],
                "executed": result["executed"],
                "exec_accuracy": result["exec_accuracy"],
                "exec_accuracy_strict": result["exec_accuracy_strict"],
                "latency_s": result["latency_s"],
                "by_category": result["by_category"],
                "server_warmup_s": warmup,
            })
            print(f"Completed {name}: normalized EX={result['exec_accuracy']}, "
                  f"strict EX={result['exec_accuracy_strict']}", flush=True)
        finally:
            rt.stop()
    return output


def evaluate_ambiguity_methods():
    pairs = [json.loads(line) for line in
             (ROOT / "evals/heldout/classifier_eval.jsonl").read_text().splitlines() if line.strip()]
    classifier = AmbiguityClassifier.load(ROOT / "ambiguity/artifacts")
    current_path = ROOT / "models/qwen25-coder-1.5b-sql-write-q4_k_m.gguf"
    rt = model_runtime.LlamaServerRuntime(gguf_path=current_path)
    gold, mini, slm_preds, hybrid = [], [], [], []
    slm_latencies = []
    slm_invalid = 0
    block_thresholds = getattr(classifier, "block_thresholds", {}) or {}
    schemas = {}
    with tempfile.TemporaryDirectory(prefix="clarion-schema-eval-") as tmp:
        try:
            rt.start()
            for row in pairs:
                key = row["schema_id"]
                if key not in schemas:
                    path = Path(tmp) / f"{len(schemas)}.db"
                    with sqlite3.connect(path) as con:
                        con.executescript(row["schema_text"])
                    schemas[key] = (path, di.extract_schema(path))
                path, schema = schemas[key]
                question, schema_text = row["question"], row["schema_text"]
                cls = classifier.predict(question, schema_text)
                mini.append(cls["label"])
                started = time.perf_counter()
                raw = rt.generate_chat(build_slm_messages(question, schema_text),
                                       max_tokens=260, temperature=0.0)
                slm_latencies.append(time.perf_counter() - started)
                verdict = parse_slm_json(raw)
                if verdict is None:
                    slm_invalid += 1
                    slm_preds.append(None)
                    verdict_for_fusion = None
                else:
                    slm_preds.append(verdict["label"])
                    verdict_for_fusion = verdict
                combined, _rule = _combine(
                    cls, verdict_for_fusion,
                    metric_named=bool(di.has_named_metric(schema, question)),
                    block_thresholds=block_thresholds)
                hybrid.append(combined or "UNDERSPECIFIED")
                gold.append(row["label"])
                if len(gold) % 10 == 0:
                    print(f"Ambiguity suite: {len(gold)}/{len(pairs)}", flush=True)
        finally:
            rt.stop()

    def report(pred):
        classes = list(eval_full.LABELS)
        conf = [[0] * (len(classes) + 1) for _ in classes]
        invalid_col = len(classes)
        for expected, guessed in zip(gold, pred):
            col = classes.index(guessed) if guessed in classes else invalid_col
            conf[classes.index(expected)][col] += 1
        per_class, f1s = {}, []
        for i, label in enumerate(classes):
            tp = conf[i][i]
            predicted = sum(conf[row][i] for row in range(len(classes)))
            support = sum(conf[i])
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                                "f1": round(f1, 4), "support": support}
            f1s.append(f1)
        amb = per_class["AMBIGUOUS"]
        return {
            "accuracy": round(sum(a == b for a, b in zip(gold, pred)) / len(gold), 4),
            "macro_f1": round(sum(f1s) / len(f1s), 4),
            "per_class": per_class,
            "ambiguity": {"precision": amb["precision"], "recall": amb["recall"], "f1": amb["f1"]},
            "confusion_matrix": {"columns_are": classes + ["INVALID"],
                                 "rows_are": classes, "matrix": conf},
        }

    return {
        "n": len(gold),
        "labels": list(eval_full.LABELS),
        "methods": [
            {"name": "MiniLM-L6-v2 + Logistic Regression", "metrics": report(mini)},
            {"name": "Qwen SLM analyst only", "metrics": report(slm_preds),
             "invalid_json": slm_invalid,
             "mean_latency_s": round(sum(slm_latencies) / max(1, len(slm_latencies)), 2)},
            {"name": "Clarion hybrid (MiniLM + Qwen)", "metrics": report(hybrid),
             "mean_slm_latency_s": round(sum(slm_latencies) / max(1, len(slm_latencies)), 2)},
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-intent", action="store_true",
                        help="score only the 101-pair ambiguity suite")
    args = parser.parse_args()
    previous = json.loads(OUT.read_text()) if args.only_intent and OUT.exists() else {}
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "sql_protocol": "evals/heldout/sql_eval.jsonl; production schema builder and SQL prompt; projection+2dp plus strict 4dp result comparison",
        "ambiguity_protocol": "evals/heldout/classifier_eval.jsonl; same 101 pairs/schema text for all three methods; Qwen Q4_K_M, JSON verdict max_tokens=260, temperature=0",
        "sql_models": previous.get("sql_models", []) if args.only_intent else evaluate_sql_models(),
        "ambiguity": evaluate_ambiguity_methods(),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
