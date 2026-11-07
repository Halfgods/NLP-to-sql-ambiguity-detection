#!/usr/bin/env python3
"""Verify each task in eecs_project/task_plan.json against its deliverables.

Ground truth over promises: a task shows DONE only when its artifacts exist on
disk. Run: .venv/bin/python scripts/taskboard.py
"""

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DONE, RUNNING, PENDING, FAILED = "DONE   ", "RUNNING", "PENDING", "FAILED "


def file_state(rel, min_lines=0):
    p = ROOT / rel
    if not p.is_file():
        return PENDING, "-"
    if min_lines:
        n = sum(1 for _ in open(p, "r", encoding="utf-8", errors="replace"))
        if n < min_lines:
            return FAILED, f"{n} lines (<{min_lines})"
        return DONE, f"{n} lines"
    return DONE, f"{p.stat().st_size // 1024} KB"


def proc_running(pattern):
    try:
        out = subprocess.run(["pgrep", "-af", pattern], capture_output=True,
                             text=True, timeout=5).stdout
        return RUNNING if out.strip() else PENDING
    except Exception:
        return PENDING


def main():
    print(f"taskboard  {time.strftime('%H:%M:%S')}")
    print("-" * 56)

    # T1 training
    adapter = ROOT / "adapters/qwen25c-1.5b-sql-lora/adapter_model.safetensors"
    if adapter.exists():
        rows1 = (DONE, f"{adapter.stat().st_size // 1_048_576} MB")
    else:
        s = proc_running("train_qlora")
        rows1 = (s, "epochs=2, GPU" if s == RUNNING else "-")
    print(f"[{'x' if rows1[0]==DONE else '~' if rows1[0]==RUNNING else ' '}] {rows1[0]}  T1 qlora-train{'':<9} {rows1[1]}")

    # T2 GGUF
    gguf = ROOT / "models/qwen25-coder-1.5b-sql-q4_k_m.gguf"
    merged = ROOT / "models/merged-sql"
    t2 = (DONE, f"{gguf.stat().st_size/1e6:.0f} MB") if gguf.exists() else \
         (RUNNING, "merged -> convert/quantize") if merged.exists() else (PENDING, "waits on T1")
    print(f"[{'x' if t2[0]==DONE else '~' if t2[0]==RUNNING else ' '}] {t2[0]}  T2 gguf-q4km{'':<12} {t2[1]}")

    # T3 classifier
    art = list((ROOT / "ambiguity/artifacts").glob("*")) if (ROOT / "ambiguity/artifacts").is_dir() else []
    selfev = (ROOT / "ambiguity/SELF_EVAL.md").exists()
    t3 = (DONE, f"{len(art)} artifacts + self-eval") if art and selfev else \
         (RUNNING, f"{len(art)} artifacts so far") if art else \
         (RUNNING, "agent working") if (ROOT / "ambiguity").is_dir() or (ROOT / "data/ambiguity_train").is_dir() else (PENDING, "-")
    print(f"[{'x' if t3[0]==DONE else '~' if t3[0]==RUNNING else ' '}] {t3[0]}  T3 classifier{'':<11} {t3[1]}")

    # T4 introspection + ambiguity
    intro = (ROOT / "db_introspect.py").exists()
    ana = (ROOT / "ambiguity/analysis.py").exists()
    tests = (ROOT / "tests").is_dir()
    t4 = (DONE, "introspect + analysis present") if intro and ana else \
         (RUNNING, "files appearing") if intro or ana or tests else (PENDING, "agent working")
    print(f"[{'x' if t4[0]==DONE else '~' if t4[0]==RUNNING else ' '}] {t4[0]}  T4 introspect{'':<12} {t4[1]}")

    # T5tester workbench
    t5_parts = [file_state("evals/heldout/classifier_eval.jsonl", 100),
                file_state("evals/heldout/sql_eval.jsonl", 25),
                file_state("scripts/eval_full.py")]
    t5 = (DONE, "heldout suites + harness") if all(st == DONE for st, _ in t5_parts) else \
         (RUNNING, "agent writing files") if any(st in (RUNNING, DONE) for st, _ in t5_parts) else (PENDING, "agent working")
    counts = ", ".join(f"{f}" for st, f in t5_parts if f != "-") or "-"
    print(f"[{'x' if t5[0]==DONE else '~' if t5[0]==RUNNING else ' '}] {t5[0]}  T5 tester-bench{'':<9} {counts}")

    # T6 integration flag
    t6 = (DONE, "ui smoke ok") if (ROOT / ".ui_smoke_ok").exists() else (PENDING, "blocked on T2-T4")
    print(f"[{'x' if t6[0]==DONE else ' '}] {t6[0]}  T6 ui-integration{'':<7} {t6[1]}")

    print("-" * 56)
    gg = " ".join(_.__name__ for _ in []) or ""
    print("ground truth = artifacts under", ROOT)
    if (ROOT / ".ui_smoke_ok").exists() is False and gg == "":
        pass


if __name__ == "__main__":
    main()
