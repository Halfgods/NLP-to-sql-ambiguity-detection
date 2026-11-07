# Brutal feedback — Clarion rebuild

generated 2026-10-31 10:05:12 · numbers from `evals/report-latest.json`

## What was measured

- classifier held-out n=101: accuracy **0.604**, macro-F1 **0.606**, ambiguity recall **0.4706**
- SQL suite: `exec_accuracy` = **0.7822**
- SQL suite: `valid_rate` = **1.0**

## The uncomfortable numbers (from ambiguity/SELF_EVAL.md)

- classifier **unseen-schema accuracy=0.650**, macro-F1=0.645 — against the claimed ≥0.80 bar, this is a **miss**, not a rounding issue.
Cause 1: a linear head on frozen MiniLM embeddings must express *context-dependent* labels — 'top X' flips CLEAR↔AMBIGUOUS depending on how many numeric rivals the schema holds, an interaction a frozen sentence embedding cannot see. Cause 2: thresholds were tuned on val split of the SAME synthetic generator, so they over-fit the generator's phrasing distribution.

## Top weaknesses, ranked

1. **Unseen-schema classifier accuracy 0.650 < 0.80.** The zero-overlap held-out suites prove this is not leakage — it is genuine generalization failure. Fix: two-tower features (question embed ⊗ schema-metric count embed) or an FB-logit head; and mix ≥3 human-curated schemas into training.
2. **Held-out accuracy 0.604 < 0.80 overall.** Missing the acceptance bar you set.
3. **Ambiguity recall 0.4706** — the detector sleeps through real ambiguity; every miss silently guesses an interpretation and executes it.

## Brutal suggestions (impact-ranked)

1. **Do not trust the classifier solo.** Route every non-CLEAR union with the SLM evidence through `ambiguity/analysis.py`'s conflict rule; the measured layered result, not either component solo, is the deliverable to quote.
2. **Harden the fusion layer before demoing**: on conflict (AMBIGUOUS↔UNDERSPECIFIED) the current conservative merge right-answers rephrase-chips; a chip ending in a dead-end ('no such column') is UX malpractice.
3. **Schema-cannot-answer questions** must answer with an explicit 'this data does not exist here' card, not a fabricated query. Add a column-presence guard from `has_named_metric` to the generation prompt and reject queries referencing absent columns.
4. **Promote repair transparency**: the worker's single retry is silent; surface it in DB checks so failures are visible, not just below-fold tool names.
5. **Model card honesty**: report the GGUF serve-RSS you measured (llama-server, mmap, ctx-2048) and the eval latency distribution from report-latest.json, not chosen numbers from a spec sheet.

## Verdict

Would I ship this as a research demo? Yes — after the safe-execution defaults are verified publicly again. Would I call it 80% ambiguous-detection generalization? **No, and the numbers above say so louder than the README does.**
