# Goal: regenerate proposal Figure 3 (domain shift by activity) in the style of Figures 1-2
# (make_dataset_figures.py). Reads the per-activity KS CSV from ks_analysis_phase2.py; no recomputation.
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "figures" / "ks_analysis_phase2" / "3class_phase2" / "ks_test_per_activity_3class_phase2.csv"
OUT = REPO / "figures" / "dataset_overview"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["WALKING", "SITTING", "STANDING"]
BLUE, ORANGE = "#4C72B0", "#DD8452"  # same palette as Figure 1 (HAPT blue, WISDM orange)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

rows = defaultdict(list)
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows[r["activity"]].append((float(r["ks_statistic"]), int(r["significant"])))
mean_ks = [sum(k for k, _ in rows[c]) / len(rows[c]) for c in CLASSES]
sig_pct = [100 * sum(s for _, s in rows[c]) / len(rows[c]) for c in CLASSES]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, vals, fmt, title, ylab, top, color in [
    (axes[0], mean_ks, "{:.3f}", "Mean KS statistic", "Mean KS statistic (80 features)", max(mean_ks) * 1.25, BLUE),
    (axes[1], sig_pct, "{:.1f}%", "Features with significant shift", "% of features (p < 0.05)", 125, ORANGE),
]:
    bars = ax.bar(CLASSES, vals, color=color)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.set_ylim(0, top)
axes[1].set_yticks([0, 25, 50, 75, 100])
fig.suptitle("Domain shift by activity, HAPT to WISDM", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "ks_domain_shift_by_activity.png", dpi=200, bbox_inches="tight")
print(dict(zip(CLASSES, mean_ks)), dict(zip(CLASSES, sig_pct)))
