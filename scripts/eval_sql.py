"""Evaluate the quantized SQL model.

Two suites:
  * eval_domain.jsonl (30 pairs) — executes gold and generated SQL read-only
    against students.db and compares result rows  -> execution accuracy
  * eval_public.jsonl (40 held-out public pairs) — normalized SQL equality
    (their schemas carry no data, so execution comparison is impossible)

Usage:
    .venv/bin/python scripts/eval_sql.py [--limit N] [--public-only|--domain-only]
    (starts llama-server itself; build the GGUF first via scripts/build_gguf.sh)
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_runtime import SQL_SYSTEM_PROMPT, get_runtime  # noqa: E402
import pipeline  # noqa: E402


def load(name):
    path = ROOT / "data" / name
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path)]


def norm_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").lower()
    sql = re.sub(r'"([^"]*)"', lambda m: "'" + m.group(1) + "'", sql)
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*([(),=<>])\s*", r"\1", sql)
    return sql


def norm_rows(rows):
    out = set()
    for r in rows:
        out.add(tuple(round(v, 4) if isinstance(v, float) else v for v in r))
    return out


def ask(model_runtime_rt, schema: str, question: str):
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": f"Operation mode: READ_ONLY\n\n{schema}\n\n"
                                    f"Question: {question}\n\nWrite one SQLite query."}]
    t0 = time.perf_counter()
    text = model_runtime_rt.generate_chat(messages, max_tokens=220)
    dt = time.perf_counter() - t0
    sql = pipeline.extract_sql(text)
    return sql, dt, text


def eval_domain(pairs, rt, db, limit):
    ok_exec = ok_parse = 0
    lat = []
    details = []
    for p in pairs[:limit or None]:
        sql, dt, raw = ask(rt, p["schema"], p["question"])
        lat.append(dt)
        parsed = bool(sql) and pipeline.validate_read_only(sql)[0] is not None
        executed, matched = False, False
        note = ""
        if parsed:
            ok_parse += 1
            executed = True
            try:
                _, g_rows, _ = pipeline.run_query(p["sql"].rstrip(";") + f" LIMIT {pipeline.ROW_CAP};", db)
                _, g_rows2, _ = pipeline.run_query(sql, db)
                matched = norm_rows(g_rows) == norm_rows(g_rows2)
                if matched:
                    ok_exec += 1
            except Exception as exc:
                matched = False
                note = f"exec error: {exc}"
        else:
            note = f"unparseable/unsafe: {raw[:80]!r}"
        status = "PASS" if matched else ("EXEC-OK" if parsed else "FAIL")
        details.append({"question": p["question"], "status": status,
                        "gold": p["sql"], "got": sql, "latency_s": round(dt, 2),
                        "note": note})
        print(f"[{status:8s}] {p['question'][:68]:<68} {dt:4.1f}s"
              + (f"  | {note}" if note and status != "PASS" else ""))
        if status != "PASS":
            print(f"    gold: {p['sql']}\n    got : {sql[:200]}")
    n = len(pairs[:limit or None])
    return {"suite": "domain", "n": n, "parse_ok": ok_parse, "exec_ok": ok_exec,
            "exec_accuracy": round(ok_exec / n, 3), "parse_accuracy": round(ok_parse / n, 3),
            "avg_latency_s": round(statistics.mean(lat), 2) if lat else None,
            "median_latency_s": round(statistics.median(lat), 2) if lat else None,
            "details": details}


def eval_public(pairs, rt, limit):
    ok = 0
    lat = []
    details = []
    rows = pairs[:limit or None]
    for p in rows:
        sql, dt, raw = ask(rt, p["schema"], p["question"])
        lat.append(dt)
        matched = bool(sql) and norm_sql(sql) == norm_sql(p["sql"])
        ok += matched
        details.append({"question": p["question"], "gold": p["sql"], "got": sql,
                        "match": matched, "latency_s": round(dt, 2),
                        "status": "PASS" if matched else "FAIL"})
        print(f"[{'PASS' if matched else 'FAIL':8s}] {p['question'][:68]:<68} {dt:4.1f}s")
        if not matched:
            print(f"    gold: {p['sql']}\n    got : {sql[:200]}")
    n = len(rows)
    return {"suite": "public", "n": n, "match_ok": ok,
            "normalized_match": round(ok / n, 3) if n else None,
            "avg_latency_s": round(statistics.mean(lat), 2) if lat else None,
            "details": details}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap questions per suite")
    ap.add_argument("--domain-only", action="store_true")
    ap.add_argument("--public-only", action="store_true")
    ap.add_argument("--db", default=str(ROOT / "students.db"))
    args = ap.parse_args()

    if not (ROOT / "models" / "qwen25-coder-1.5b-sql-q4_k_m.gguf").exists():
        sys.exit("Q4_K_M GGUF missing — run scripts/build_gguf.sh first (or merge+quantize).")
    rt = get_runtime()
    print("model server up:", rt.model_name)

    reports = []
    if not args.public_only:
        reports.append(eval_domain(load("eval_domain.jsonl"), rt, args.db, args.limit))
    if not args.domain_only:
        reports.append(eval_public(load("eval_public.jsonl"), rt, args.limit))

    print("\n==== SUMMARY ====")
    for r in reports:
        head = {k: v for k, v in r.items() if k != "details"}
        print(json.dumps(head, indent=2))

    ROOT.joinpath("evals").mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "evals" / f"report-{stamp}.json"
    out.write_text(json.dumps({"model": rt.model_name, "reports": reports}, indent=2))
    print("full report ->", out)


if __name__ == "__main__":
    main()
