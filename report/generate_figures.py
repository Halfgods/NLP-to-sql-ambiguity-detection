"""Regenerate empirical report plots from the saved evaluation artifact."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EVAL = ROOT.parent / "evals" / "report-latest.json"
OUT = ROOT / "figures" / "evaluation_summary.png"
COMPARISON = ROOT / "model_comparison_results.json"
COMPARISON_OUT = ROOT / "figures" / "model_comparison.png"
data = json.loads(EVAL.read_text())
clf = data["classifier"]
sql = data["sql"]

labels = ["Classifier accuracy", "Classifier macro-F1",
          "Ambiguous precision", "Ambiguous recall"]
values = [100 * clf["accuracy"], 100 * clf["macro_f1"],
          100 * clf["ambiguity"]["precision"],
          100 * clf["ambiguity"]["recall"]]
cats = list(sql["by_category"].items())
cat_labels = [name.replace("_", " ").title() for name, _ in cats]
cat_values = [100 * item["exec_ok"] / item["n"] for _, item in cats]
cat_counts = [item["n"] for _, item in cats]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [0.85, 1.45]})
fig.patch.set_facecolor("#f7f9fc")
for ax in axes:
    ax.set_facecolor("#f7f9fc")
    ax.grid(axis="x", color="#dce3ec", linewidth=0.8)
    ax.set_axisbelow(True)

ax = axes[0]
colors = ["#2676a8", "#3986b3", "#20a686", "#e08a43"]
y = list(range(len(labels)))
bars = ax.barh(y, values, color=colors, height=0.62, zorder=3)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Score (%)")
ax.set_title("Ambiguity classifier · n=101", loc="left", weight="bold", color="#173451", pad=12)
ax.axvline(80, color="#b34b4b", linestyle="--", linewidth=1.2)
for b, value in zip(bars, values):
    ax.text(value + 1.3, b.get_y() + b.get_height()/2, f"{value:.1f}%", va="center", color="#173451", weight="bold")
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#aab7c7")

ax = axes[1]
y = list(range(len(cat_labels)))
bars = ax.barh(y, cat_values, color="#3986b3", height=0.62, zorder=3)
ax.set_yticks(y, [f"{name}  (n={n})" for name, n in zip(cat_labels, cat_counts)])
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xlabel("Normalized execution accuracy (%)")
ax.set_title("Qwen2.5-Coder SQL suite · 101 pairs", loc="left", weight="bold", color="#173451", pad=12)
for b, value, count in zip(bars, cat_values, cat_counts):
    ax.text(value + 1.3, b.get_y() + b.get_height()/2, f"{value:.1f}%  ({round(value*count/100):d}/{count})", va="center", color="#173451", weight="bold", fontsize=8)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#aab7c7")

fig.suptitle("Clarion held-out evaluation", x=0.03, ha="left", fontsize=17, weight="bold", color="#173451")
fig.text(0.03, 0.925, f"Saved evaluation artifact: {data['generated_at']} · all chart values are measured, not illustrative", color="#526576", fontsize=9)
fig.text(0.03, 0.015, "Dashed line marks the 80% classifier target. SQL category rates use projection + 2-decimal result normalization.", color="#526576", fontsize=8)
fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.12, wspace=0.55)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"wrote {OUT}")

# Same-run comparison across the locally available SQL checkpoints and the
# three ambiguity approaches. The two panels intentionally use separate axes:
# MiniLM is a classifier, not a SQL generator.
cmp = json.loads(COMPARISON.read_text())
sql_models = cmp["sql_models"]
intent_methods = cmp["ambiguity"]["methods"]
short_sql = ["SQL-only\nQ4_K_M", "SQL + write\nF16", "SQL + write\nQ4_K_M"]
sql_norm = [100 * m["exec_accuracy"] for m in sql_models]
sql_strict = [100 * m["exec_accuracy_strict"] for m in sql_models]
short_intent = ["MiniLM + LR", "Qwen SLM\nonly", "Clarion\nhybrid"]
intent_acc = [100 * m["metrics"]["accuracy"] for m in intent_methods]
intent_f1 = [100 * m["metrics"]["macro_f1"] for m in intent_methods]
intent_rec = [100 * m["metrics"]["ambiguity"]["recall"] for m in intent_methods]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), gridspec_kw={"width_ratios": [1.0, 1.15]})
fig.patch.set_facecolor("#f7f9fc")
for ax in axes:
    ax.set_facecolor("#f7f9fc")
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dce3ec", linewidth=0.8)

ax = axes[0]
x = list(range(len(sql_models)))
w = 0.34
b1 = ax.bar([i-w/2 for i in x], sql_norm, w, color="#2878a8", label="Execution accuracy · projection + 2dp")
b2 = ax.bar([i+w/2 for i in x], sql_strict, w, color="#59a49b", label="Strict exact · 4dp")
ax.axhline(75, color="#b34b4b", linestyle="--", linewidth=1.1, label="75% project target (normalized)")
ax.set_xticks(x, short_sql)
ax.set_ylabel("Correct result matches (%)")
ax.set_title("Read-only SQL · n=101", loc="left", weight="bold", color="#173451", pad=12)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8, weight="bold", color="#173451")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax = axes[1]
x = list(range(len(intent_methods)))
w = 0.23
b1 = ax.bar([i-w for i in x], intent_acc, w, color="#2878a8", label="Accuracy")
b2 = ax.bar(x, intent_f1, w, color="#8457ad", label="Macro-F1")
b3 = ax.bar([i+w for i in x], intent_rec, w, color="#dd8a3c", label="AMBIGUOUS recall")
ax.set_xticks(x, short_intent)
ax.set_ylabel("Score (%)")
ax.set_title("Ambiguity detection · n=101", loc="left", weight="bold", color="#173451", pad=12)
ax.legend(frameon=False, fontsize=8, loc="upper left")
for bars in (b1, b2, b3):
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.4, weight="bold", color="#173451")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.suptitle("Clarion model and method comparison · 6 November 2026", x=0.03, ha="left", fontsize=15, weight="bold", color="#173451")
fig.text(0.03, 0.925, "Locally available checkpoints; same held-out suites within each task. MiniLM is compared on ambiguity detection, not SQL generation.", color="#526576", fontsize=8.5)
fig.text(0.03, 0.012, "SQL: blue = normalized result match; teal = strict exact; dashed red = 75% target. Intent: same 101 labels; malformed Qwen JSON counts wrong (88/101 valid).", color="#526576", fontsize=7.8)
fig.subplots_adjust(left=0.08, right=0.985, top=0.83, bottom=0.17, wspace=0.32)
fig.savefig(COMPARISON_OUT, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"wrote {COMPARISON_OUT}")
