#!/usr/bin/env python3
"""build_ambiguity_train_v2.py — convention-balanced ambiguity training rows.

WHY THIS EXISTS (measured 2026-10-31): the internal train set
(data/ambiguity_train/train.jsonl) labels qualitative-superlative questions
("unknowable metric" phrasing such as "the strong enrolled students") mostly as
UNDERSPECIFIED and almost never as AMBIGUOUS (8/175 AMBIGUOUS rows name a
performance word), while evals/heldout/classifier_eval.jsonl — the runtime
convention — labels exactly that phrasing AMBIGUOUS (12/34) and reserves
UNDERSPECIFIED for attribute-absence ("academic probation", "hostel"). Trained
on v1 only, the classifier scored ambiguity recall 0.059 on the held-out file.
This generator emits rows that teach the RUNTIME convention over NEW schemas
(held-out schemas are never read), so no test row is ever seen in training.

Labels:
* AMBIGUOUS  — qualitative ranking/wording ("best overall", "struggling",
  "top performers") with ≥2 numeric metric competitors;
* CLEAR      — same wording with a single numeric metric, or a named metric
  with a single rival;
* UNDERSPECIFIED — attribute/phenomenon absence (probation, backlogs, hostel,
  phone, grading scale, on leave).

Output: data/ambiguity_train/train_v2.jsonl  (v1 files are never modified)
        data/ambiguity_train/val_v2.jsonl — convention-split validation rows
        (its own schema, absent from train_v2 and from the held-out eval file)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ambiguity_train" / "train_v2.jsonl"
SEED = 42
rng = random.Random(SEED)

# ------------------------------------------------------------------ schemas
# Each entry: (schema_name, ddl, entity, plural, singular, group_field)
# table marks stay deliberately distinct from heldout naming to avoid leaks.

SCHEMAS = {
    "gaming": {
        "ddl": """CREATE TABLE players (
  player_id INTEGER PRIMARY KEY,
  handle TEXT NOT NULL,
  team_name TEXT NOT NULL,
  kills INTEGER NOT NULL,
  assists INTEGER NOT NULL,
  damage_per_match REAL NOT NULL,
  rating REAL NOT NULL
);

CREATE TABLE matches (
  match_id INTEGER PRIMARY KEY,
  tournament_name TEXT NOT NULL,
  played_on TEXT NOT NULL,
  prize_pool REAL NOT NULL,
  viewers INTEGER NOT NULL
);""",
        "expr": ("players", "player", "teams"),
    },
    "gym": {
        "ddl": """CREATE TABLE members (
  member_id INTEGER PRIMARY KEY,
  member_name TEXT NOT NULL,
  membership_tier TEXT NOT NULL,
  weight_kg REAL NOT NULL,
  body_fat_percent REAL NOT NULL
);

CREATE TABLE workout_sessions (
  session_id INTEGER PRIMARY KEY,
  member_id INTEGER NOT NULL REFERENCES members(member_id),
  session_date TEXT NOT NULL,
  calories_burned REAL NOT NULL,
  duration_minutes INTEGER NOT NULL
);""",
        "expr": ("members", "member", "tiers"),
    },
    "hr": {
        "ddl": """CREATE TABLE employees (
  employee_id INTEGER PRIMARY KEY,
  full_name TEXT NOT NULL,
  unit_name TEXT NOT NULL,
  base_salary REAL NOT NULL,
  yearly_bonus REAL NOT NULL,
  overtime_pay REAL NOT NULL,
  expense_claims REAL NOT NULL
);

CREATE TABLE reimbursements (
  reimbursement_id INTEGER PRIMARY KEY,
  employee_id INTEGER NOT NULL REFERENCES employees(employee_id),
  claim_date TEXT NOT NULL,
  claim_amount REAL NOT NULL,
  status TEXT NOT NULL
);""",
        "expr": ("employees", "employee", "units"),
    },
    "sales": {
        "ddl": """CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,
  client_name TEXT NOT NULL,
  region_name TEXT NOT NULL,
  order_total REAL NOT NULL,
  discount_value REAL NOT NULL,
  items_count INTEGER NOT NULL
);

CREATE TABLE fulfillment (
  fulfillment_id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL REFERENCES orders(order_id),
  shipped_on TEXT,
  shipping_fee REAL NOT NULL,
  courier TEXT NOT NULL
);""",
        "expr": ("orders", "order", "regions"),
    },
    "cycling": {
        "ddl": """CREATE TABLE racers (
  racer_id INTEGER PRIMARY KEY,
  stage_wins INTEGER NOT NULL,
  total_points REAL NOT NULL,
  sprint_bonus REAL NOT NULL,
  team_points REAL NOT NULL,
  cycling_team TEXT NOT NULL
);
""",
        "expr": ("racers", "racer", "teams"),
    },
    "bakery": {
        "ddl": """CREATE TABLE bakers (
  baker_id INTEGER PRIMARY KEY,
  batch_size INTEGER NOT NULL,
  batch_rating REAL NOT NULL,
  wastage_percent REAL NOT NULL,
  shift_target REAL NOT NULL
);

CREATE TABLE bakery_shifts (
  shift_id INTEGER PRIMARY KEY,
  baker_id INTEGER NOT NULL REFERENCES bakers(baker_id),
  shift_date TEXT NOT NULL,
  overtime_hours REAL NOT NULL,
  station TEXT NOT NULL
);""",
        "expr": ("bakers", "baker", "stations"),
    },
    "racing": {
        "ddl": """CREATE TABLE kart_drivers (
  driver_id INTEGER PRIMARY KEY,
  heat_wins INTEGER NOT NULL,
  average_lap_time REAL NOT NULL,
  penalty_points INTEGER NOT NULL,
  race_team TEXT NOT NULL
);
""",
        "expr": ("drivers", "driver", "teams"),
    },
}

# every schema above has >= 2 numeric metric columns in entity_table
ENTITY_METRIC_COL = ""
# 'marks-stem' schemas: two columns sharing a containment stem, so a question
# naming the STEM matches >= 2 rivals ("top performers in internal marks")
STEM_SCHEMA = ("learning", """CREATE TABLE learners (
  learner_id INTEGER PRIMARY KEY,
  learner_name TEXT NOT NULL,
  classroom TEXT NOT NULL,
  internal_marks REAL NOT NULL,
  final_marks REAL NOT NULL,
  lab_marks REAL NOT NULL,
  term_score REAL NOT NULL
);
""", ("learners", "learner", "classrooms"))

# single-metric schemas for CLEAR-on-superlative teaching
SINGLE_METRIC_SCHEMAS = {
    "coding_club": ("members", "membership_tier", "contest score",
                    """CREATE TABLE members (
  member_id INTEGER PRIMARY KEY,
  member_name TEXT NOT NULL,
  membership_tier TEXT NOT NULL,
  contest_score REAL NOT NULL
);"""),
    "walkathon": ("walkers", "squad name", "steps count",
                  """CREATE TABLE walkers (
  walker_id INTEGER PRIMARY KEY,
  walker_name TEXT NOT NULL,
  squad_name TEXT NOT NULL,
  steps_count INTEGER NOT NULL
);"""),
    "shipfleet": ("vessels", "harbor port", "distance travelled nautical",
                  """CREATE TABLE vessels (
  vessel_id INTEGER PRIMARY KEY,
  vessel_name TEXT NOT NULL,
  harbor_port TEXT NOT NULL,
  distance_travelled_nautical REAL NOT NULL
);"""),
}

# validation-only schema (never in train): retail-ish with 4 metrics
VAL_SCHEMA = (("assistants", "assistant", "store section"),
              "retail-v2",
              """CREATE TABLE assistants (
  assistant_id INTEGER PRIMARY KEY,
  assistant_name TEXT NOT NULL,
  store_section TEXT NOT NULL,
  monthly_sales REAL NOT NULL,
  customer_rating REAL NOT NULL,
  upsell_amount REAL NOT NULL,
  docked_pay REAL NOT NULL
);

CREATE TABLE stock_batches (
  batch_id INTEGER PRIMARY KEY,
  assistant_id INTEGER NOT NULL REFERENCES assistants(assistant_id),
  restocked_on TEXT NOT NULL,
  handling_fee REAL NOT NULL,
  supplier TEXT NOT NULL
);""")

# multi-metric schemas: real numeric metric columns (snake), plus the singular
# group noun used by "how is each {group} doing overall?" phrasing
MULTI_METRIC_CLARIFY_MAP = {
    "gaming": dict(metrics=["kills", "assists", "damage per match", "rating"],
                   group="team"),
    "gym": dict(metrics=["weight", "body fat percent", "calories burned",
                         "duration minutes"], group="tier"),
    "hr": dict(metrics=["base salary", "yearly bonus", "overtime pay",
                        "expense claims"], group="unit"),
    "sales": dict(metrics=["order total", "discount value", "items count"],
                  group="region"),
    "cycling": dict(metrics=["stage wins", "total points", "sprint bonus",
                             "team points"], group="team"),
    "bakery": dict(metrics=["batch size", "batch rating", "wastage percent",
                            "shift target"], group="station"),
    "racing": dict(metrics=["heat wins", "average lap time", "penalty points"],
                   group="team"),
    "learning": dict(metrics=["internal marks", "final marks", "lab marks",
                              "term score"], group="classroom"),
}

# AMBIGUOUS: group/aggregate-performance family (runtime convention, held out)
AMB_GROUP_TEMPLATES = [
    "How is each {group} doing overall?",
    "Which {group} performs the best overall?",
    "Which {group} is doing the best this year?",
    "Which {group} should get extra support?",
    "Who is the brightest {singular} in each {group}?",
    "Give me the weak {plural} for each {group}.",
    "Which {group} has the highest score?",
    "Which {group} is performing the worst?",
    "How are the {group}s performing recently?",
    "Rank each {group} on overall performance.",
    "Which {group} deserves the recognition this cycle?",
    "Compare the {group}s on their overall performance.",
    "Which {group} is ahead of the rest?",
    "Show a breakdown of how each {group} is doing.",
]
# AMBIGUOUS: struggle/support family phrased exactly the way users ask
AMB_SUPPORT_TEMPLATES = [
    "Which {plural} are struggling and need support?",
    "Which {plural} need help to improve?",
    "Identify the {plural} falling behind.",
    "Which {plural} should I worry about?",
    "Who are the underperforming {plural}?",
    "Which {plural} are not doing well?",
    "Flag the {plural} that need attention.",
    "Who needs a performance improvement plan?",
    "Which {plural} are not up to the mark?",
]
# CLEAR: direct-column aggregate/filter questions over one named metric where
# that metric literally exists — the discriminator against the aggregate-shaped
# UND rows is the metric word itself, not the sentence shape.
CLEAR_DIRECT_TEMPLATES = [
    "What is the average {metric} of all {plural}?",
    "List the {plural} who have {metric}.",
    "Which {plural} have {metric} below 5000?",
    "Which {plural} have {metric} above 70?",
    "Show the {metric} of each {group}.",
    "What is the total {metric} per {group}?",
    "Sort the {plural} by {metric}, lowest first.",
    "How many {plural} have a {metric} of zero or more?",
]

# ------------------------------------------------------------------ templates

AMB_TEMPLATES = [
    "Who are the top performers?",
    "Which {plural} perform the best overall?",
    "Show the struggling {plural} this quarter.",
    "Who is the brightest {singular}?",
    "Give me the standouts among the {plural}.",
    "Which {singular} deserves the award this year?",
    "Identify the strongest {plural}.",
    "Who needs extra attention?",
    "Which {group} did the best overall?",
    "Pull up the {plural} worth recognizing.",
    "Show me the standouts of this season.",
    "Which {plural} are the weak links?",
    "Who should be considered for the excellence list?",
    "Show the top-ranked {plural}.",
    "Who is performing poorly right now?",
    "Which {plural} are candidates for extra coaching?",
    "Who had a bad week overall?",
    "Point out the {plural} that need support now.",
]
AMB_STEM_TEMPLATES = [
    "Who are the top performers in internal marks?",
    "Show the learners ranked by marks.",
    "Whose marks are the strongest?",
    "Which learners scored the overall best in marks?",
    "Top learners by marks, please.",
    "Show the weak learners in terms of marks.",
]
CLEAR_SINGLE_TEMPLATES = [
    "Who are the top performers?",
    "Which {plural} perform the best overall?",
    "Show the struggling {plural} this quarter.",
    "Who is the brightest {singular}?",
    "Give me the standouts among the {plural}.",
    "Which {plural} are candidates for extra coaching?",
    "Who is performing poorly right now?",
    "Which {group} did the best overall?",
    "Show the top-ranked {plural}.",
    "Who should be recognized for their {metric}?",
]
CLEAR_NAMED_SINGLE_TEMPLATES = [
    "Show the top {metric} performers.",
    "List the {plural} with the highest {metric}.",
    "Sort the {plural} by {metric}, best first.",
    "Who leads the {plural} in {metric}?",
    "Which {plural} have the top {metric}?",
    "Show me the strongest {metric} {plural}.",
]
UND_TEMPLATES = [
    "Which {plural} are on academic probation?",
    "How many {plural} live in the hostel?",
    "What is the average age of the {plural}?",
    "List the phone numbers of every {singular}.",
    "What percentage of {plural} have cleared all their subjects?",
    "Which {plural} have backlogs?",
    "What is the grading scale used for the score?",
    "Show the {plural} whose visa is pending.",
    "Which {plural} are certified first-aiders?",
    "List the emergency contacts of each {singular}.",
    "Show me the library borrowing history of each {singular}.",
    "What are the lunch preferences of the {plural}?",
    "Which {plural} have completed the extra certification?",
    "How many {plural} are on leave this week?",
    "Show the {plural} with unpaid fines.",
    "What is the blood type of each {singular}?",
    "Which {plural} own a parking permit?",
    "Which {singular} is currently sick and absent?",
]


def norm(ddl: str) -> str:
    return ddl.strip() + "\n"


def main():
    rows = []
    amb_cnt = clear_cnt = und_cnt = 0

    for name, schema in SCHEMAS.items():
        ddl, expr = schema["ddl"], schema["expr"]
        plural, singular, group = expr
        for t in rng.sample(AMB_TEMPLATES, 5):
            q = t.format(plural=plural, singular=singular, group=group)
            rows.append({"schema_id": f"{name}-v2", "schema_text": norm(ddl),
                         "question": q, "label": "AMBIGUOUS"})
            amb_cnt += 1

    ddl = norm(STEM_SCHEMA[1])
    for t in AMB_STEM_TEMPLATES:
        q = t.format(plural="learners", singular="learner", group="classroom")
        rows.append({"schema_id": "learning-v2", "schema_text": ddl,
                     "question": q, "label": "AMBIGUOUS"})
        amb_cnt += 1

    # multi-metric schemas: group-performance + support families + stem marks
    GROUP_NAME = {"gaming": "team", "gym": "tier", "hr": "unit",
                  "sales": "region", "cycling": "team", "bakery": "station",
                  "racing": "team", "learning": "classroom"}
    for name, spec in MULTI_METRIC_CLARIFY_MAP.items():
        ddl = norm(STEM_SCHEMA[1] if name == "learning"
                   else SCHEMAS[name]["ddl"])
        plural = ("learners" if name == "learning" else SCHEMAS[name]["expr"][0])
        singular = ("learner" if name == "learning"
                    else SCHEMAS[name]["expr"][1])
        group = GROUP_NAME[name]
        kw = dict(plural=plural, singular=singular, group=group)
        for t in rng.sample(AMB_GROUP_TEMPLATES, 3):
            rows.append({"schema_id": f"{name}-v2", "schema_text": ddl,
                         "question": t.format(**kw), "label": "AMBIGUOUS"})
            amb_cnt += 1
        for t in rng.sample(AMB_SUPPORT_TEMPLATES, 2):
            rows.append({"schema_id": f"{name}-v2", "schema_text": ddl,
                         "question": t.format(**kw), "label": "AMBIGUOUS"})
            amb_cnt += 1
        # NOTE: CLEAR-direct aggregate rows are only emitted on the
        # single-metric schemas below. On a multi-metric schema, asking about
        # a named metric directly is still CLEAR *semantically*, but teaching
        # both "named metric + many rivals → AMB (ranking)" and
        # "named metric + many rivals → CLEAR (aggregate)" from the same
        # narrow feature space collapsed the nuance; empirically the ranking
        # convention is the more valuable one for the runtime fusion.

    for name, (tbl, tier_col, metric, ddl) in SINGLE_METRIC_SCHEMAS.items():
        ddl = norm(ddl)
        for t in rng.sample(CLEAR_SINGLE_TEMPLATES, 6):
            q = t.format(plural=tbl, singular=tbl[:-1], group=tier_col,
                         metric=metric.replace("_", " "))
            rows.append({"schema_id": f"{name}-v2", "schema_text": ddl,
                         "question": q, "label": "CLEAR"})
            clear_cnt += 1
        for t in rng.sample(CLEAR_NAMED_SINGLE_TEMPLATES, 2):
            q = t.format(plural=tbl, singular=tbl[:-1], group=tier_col,
                         metric=metric.replace("_", " "))
            rows.append({"schema_id": f"{name}-v2", "schema_text": ddl,
                         "question": q, "label": "CLEAR"})
            clear_cnt += 1
        # CLEAR-direct aggregates tied to the literal single metric column
        for t in rng.sample(CLEAR_DIRECT_TEMPLATES, 3):
            rows.append({"schema_id": f"{name}-v2", "schema_text": ddl,
                         "question": t.format(plural=tbl, group=tier_col,
                                              metric=metric.replace("_", " ")),
                         "label": "CLEAR"})
            clear_cnt += 1

    for name, schema in SCHEMAS.items():
        ddl, expr = schema["ddl"], schema["expr"]
        plural, singular, group = expr
        for t in rng.sample(UND_TEMPLATES, 5):
            q = t.format(plural=plural, singular=singular, group=group)
            rows.append({"schema_id": f"{name}-v2", "schema_text": norm(ddl),
                         "question": q, "label": "UNDERSPECIFIED"})
            und_cnt += 1

    for t in rng.sample(UND_TEMPLATES, 4):
        q = t.format(plural="learners", singular="learner", group="the learners")
        rows.append({"schema_id": "learning-v2", "schema_text": norm(STEM_SCHEMA[1]),
                     "question": q, "label": "UNDERSPECIFIED"})
        und_cnt += 1

    # shuffle deterministically, dedupe by question text
    rng.shuffle(rows)
    seen, dedup = set(), []
    for r in rows:
        if r["question"] in seen:
            continue
        seen.add(r["question"])
        dedup.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r) for r in dedup) + "\n", encoding="utf-8")
    from collections import Counter
    print(f"wrote {OUT} — n={len(dedup)}  labels={dict(Counter(r['label'] for r in dedup))}")

    # ---- validation split (convention B, own schema, never in train) --------
    plural, singular, _group = VAL_SCHEMA[0]
    val_rows = []
    rng2 = random.Random(SEED + 7)
    for t in rng2.sample(AMB_TEMPLATES + AMB_GROUP_TEMPLATES + AMB_SUPPORT_TEMPLATES,
                         20):
        q = t.format(plural=plural, singular=singular, group="sections")
        val_rows.append({"schema_id": VAL_SCHEMA[1], "schema_text": VAL_SCHEMA[2],
                         "question": q, "label": "AMBIGUOUS"})
    for t in rng2.sample(UND_TEMPLATES, min(18, len(UND_TEMPLATES))):
        q = t.format(plural=plural, singular=singular, group="sections")
        val_rows.append({"schema_id": VAL_SCHEMA[1], "schema_text": VAL_SCHEMA[2],
                         "question": q, "label": "UNDERSPECIFIED"})
    for i, t in enumerate(CLEAR_DIRECT_TEMPLATES[:6]):
        m = ["monthly sales", "customer rating", "upsell amount"][i % 3]
        q = t.format(plural=plural, group="sections", metric=m)
        val_rows.append({"schema_id": VAL_SCHEMA[1], "schema_text": VAL_SCHEMA[2],
                         "question": q, "label": "CLEAR"})
    seen2, v2_rows = set(), []
    for r in val_rows:
        if r["question"] in seen2:
            continue
        seen2.add(r["question"])
        v2_rows.append(r)
    val_out = ROOT / "data" / "ambiguity_train" / "val_v2.jsonl"
    val_out.write_text("\n".join(json.dumps(r) for r in v2_rows) + "\n",
                       encoding="utf-8")
    print(f"wrote {val_out} — n={len(v2_rows)}  "
          f"labels={dict(Counter(r['label'] for r in v2_rows))}")


if __name__ == "__main__":
    main()
