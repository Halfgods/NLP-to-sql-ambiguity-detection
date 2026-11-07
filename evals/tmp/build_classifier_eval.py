"""Build evals/heldout/classifier_eval.jsonl.

Pairs are hand-labeled (rationales in the parts files); the ONLY transformation
here is formatting into the 4-key JSONL contract: {"schema_id","schema_text",
"question","label"}.  Run:  .venv/bin/python evals/tmp/build_classifier_eval.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from schemas import SCHEMAS           # noqa: E402
from pairs_part1 import PART1         # noqa: E402
from pairs_part2 import PART2         # noqa: E402

OUT = ROOT / "evals" / "heldout" / "classifier_eval.jsonl"
RALE = ROOT / "evals" / "tmp" / "label_rationale.md"


def main():
    rows, rationales = [], []
    for schema_id, question, label, why in PART1 + PART2:
        rows.append({
            "schema_id": schema_id,
            "schema_text": SCHEMAS[schema_id],
            "question": question,
            "label": label,
        })
        rationales.append(("### %s | %s\n\n- **schema**: `%s`\n- **label**: %s\n- **why**: %s\n"
                           % (label, question, schema_id, label, why)))

    # guard: (schema_id, question) unique per pair, labels valid, schema ids known
    keys = [(r["schema_id"], r["question"]) for r in rows]
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    assert not dupes, f"duplicate (schema,question) pairs: {dupes}"
    assert all(r["label"] in {"CLEAR", "AMBIGUOUS", "UNDERSPECIFIED"} for r in rows)
    assert set(r["schema_id"] for r in rows) <= set(SCHEMAS)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lab = Counter(r["label"] for r in rows)
    sch = Counter(r["schema_id"] for r in rows)
    per_schema_label: dict = {}
    for r in rows:
        per_schema_label.setdefault(r["schema_id"], Counter())[r["label"]] += 1

    RALE.write_text("# Held-out classifier label rationale\n\n"
                    "One line per pair, authoring-time only (not part of the eval contract).\n\n"
                    + "\n".join(rationales) + "\n")

    print(f"wrote {OUT} ({len(rows)} pairs)")
    print("labels:", dict(lab))
    for sid in SCHEMAS:
        print(f"  {sid:28s} {dict(per_schema_label.get(sid, {}))}")


if __name__ == "__main__":
    main()
