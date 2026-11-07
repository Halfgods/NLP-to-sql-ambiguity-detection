#!/usr/bin/env python3
"""Brutal feedback generator — reads evals/report-latest.json + ambiguity/SELF_EVAL.md
and writes evals/BRUTAL_FEEDBACK.md: the honest, numbers-backed memo. No hedging."""

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"


def load(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8")


def self_eval_numbers():
    txt = load(ROOT / "ambiguity" / "SELF_EVAL.md", "")
    out = {}
    for key, pat in {
        "unseen_acc": r"(?:unseen-schema test slice.*?|test_unseen.*?)accuracy \*\*(0\.?\d*)\*\*",
        "unseen_f1": r"accuracy \*\*0?\.\d+\*\*, macro-F1 \*\*(0\.?\d*)\*\*",
    }.items():
        m = re.search(pat, txt)
        out[key] = m.group(1) if m else None
    return out


def main():
    rep_path = EVALS / "report-latest.json"
    if not rep_path.exists():
        raise SystemExit("no report-latest.json — run scripts/eval_full.py first")
    rep = json.loads(rep_path.read_text())

    lines = []
    a = lines.append
    a("# Brutal feedback — Clarion rebuild")
    a("")
    a(f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} · numbers from `evals/report-latest.json`")
    a("")

    clf = sql = None
    for item in rep if isinstance(rep, list) else rep.get("suites", rep.get("reports", [rep])):
        names = item if isinstance(item, dict) else {}

    # robust: find suites by key
    def find_suite(name):
        if isinstance(rep, dict):
            s = rep.get("suites") or rep.get("reports") or rep
            if isinstance(s, dict):
                return s.get(name)
            if isinstance(s, list):
                for r in s:
                    if r.get("suite") == name or r.get("name") == name:
                        return r
            if rep.get("suite") == name:
                return rep
        return None

    clf = find_suite("classifier")
    sql = find_suite("sql") or find_suite("domain")

    # headline metrics
    a("## What was measured")
    a("")
    if clf:
        amb = clf.get("ambiguity") or {}
        amb_rec = amb.get("recall", clf.get("ambiguity_recall",
                                            clf.get("recall_AMBIGUOUS")))
        if amb_rec is not None:
            a(f"- classifier held-out n={clf.get('n', '?')}: accuracy "
              f"**{clf.get('accuracy', '?')}**, macro-F1 "
              f"**{clf.get('macro_f1', '?')}**, ambiguity recall "
              f"**{amb_rec}**")
    if sql:
        for k in ("execution_accuracy", "exec_accuracy", "valid_rate", "invalid_rate", "avg_latency_s"):
            if k in sql:
                a(f"- SQL suite: `{k}` = **{sql[k]}**")
    a("")

    # classifier self-eval numbers
    se = self_eval_numbers()
    a("## The uncomfortable numbers (from ambiguity/SELF_EVAL.md)")
    a("")
    if se.get("unseen_acc"):
        a(f"- classifier **unseen-schema accuracy={se['unseen_acc']}**"
          + (f", macro-F1={se['unseen_f1']}" if se.get('unseen_f1') else "")
          + " — against the claimed ≥0.80 bar, this is a **miss**, not a rounding issue.")
        a("Cause 1: a linear head on frozen MiniLM embeddings must express *context-dependent* "
          "labels — 'top X' flips CLEAR↔AMBIGUOUS depending on how many numeric rivals the "
          "schema holds, an interaction a frozen sentence embedding cannot see. "
          "Cause 2: thresholds were tuned on val split of the SAME synthetic generator, "
          "so they over-fit the generator's phrasing distribution.")
        a("")

    # weaknesses picked from the report
    a("## Top weaknesses, ranked")
    a("")
    n = 0
    if se.get("unseen_acc") and float(se["unseen_acc"]) < 0.80:
        n += 1
        a(f"{n}. **Unseen-schema classifier accuracy {se['unseen_acc']} < 0.80.** The zero-overlap "
          "held-out suites prove this is not leakage — it is genuine generalization failure. "
          "Fix: two-tower features (question embed ⊗ schema-metric count embed) or an "
          "FB-logit head; and mix ≥3 human-curated schemas into training.")
    if clf:
        acc = clf.get("accuracy") or 0
        if isinstance(acc, (int, float)) and acc < 0.80:
            n += 1
            a(f"{n}. **Held-out accuracy {acc} < 0.80 overall.** Missing the acceptance bar you set.")
        amb = clf.get("ambiguity") or {}
        amb_rec = amb.get("recall", clf.get("ambiguity_recall",
                                            clf.get("recall_AMBIGUOUS")))
        if isinstance(amb_rec, (int, float)) and amb_rec < 0.7:
            n += 1
            a(f"{n}. **Ambiguity recall {amb_rec}** — the detector sleeps through real ambiguity; "
              "every miss silently guesses an interpretation and executes it.")
    if sql:
        ea = sql.get("execution_accuracy") or sql.get("exec_accuracy") or 0
        if isinstance(ea, (int, float)) and ea < 0.75:
            n += 1
            a(f"{n}. **SQL execution accuracy {ea}** below the 0.75 domain bar — inspect the failures "
              "in REPORT.md verbatim before tuning anything else.")
    a("")
    a("## Brutal suggestions (impact-ranked)")
    a("")
    a("1. **Do not trust the classifier solo.** Route every non-CLEAR union with the SLM "
      "evidence through `ambiguity/analysis.py`'s conflict rule; the measured layered result, "
      "not either component solo, is the deliverable to quote.")
    a("2. **Harden the fusion layer before demoing**: on conflict (AMBIGUOUS↔UNDERSPECIFIED) "
      "the current conservative merge right-answers rephrase-chips; a chip ending in a "
      "dead-end ('no such column') is UX malpractice.")
    a("3. **Schema-cannot-answer questions** must answer with an explicit 'this data does "
      "not exist here' card, not a fabricated query. Add a column-presence guard from "
      "`has_named_metric` to the generation prompt and reject queries referencing absent columns.")
    a("4. **Promote repair transparency**: the worker's single retry is silent; surface it "
      "in DB checks so failures are visible, not just below-fold tool names.")
    a("5. **Model card honesty**: report the GGUF serve-RSS you measured (llama-server, "
      "mmap, ctx-2048) and the eval latency distribution from report-latest.json, not "
      "chosen numbers from a spec sheet.")
    a("")
    a("## Verdict")
    a("")
    a("Would I ship this as a research demo? Yes — after the safe-execution defaults are "
      "verified publicly again. Would I call it 80% ambiguous-detection generalization? "
      "**No, and the numbers above say so louder than the README does.**")

    out = EVALS / "BRUTAL_FEEDBACK.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)
    print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()
