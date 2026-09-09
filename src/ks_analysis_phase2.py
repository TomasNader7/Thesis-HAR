import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# ======================
# CONFIG
# ======================
# Paths anchored to the repo root (this file lives in <repo_root>/src/).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

PHASE_TAG = "phase2"
# Output goes to figures/ks_analysis_phase2/ so it lands alongside the other
# tracked result artifacts instead of a stray dir at the repo root.
RESULTS_DIR = os.path.join(str(REPO_ROOT / "figures" / "ks_analysis_phase2"), f"3class_{PHASE_TAG}")
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS = {1: "WALKING", 2: "SITTING", 3: "STANDING"}

# ======================
# LOAD FEATURES
# ======================
# Regenerate via src/feature_extraction_HAPT.py and src/analysis_feature_dataset.py
# before running this script.
X_hapt = np.loadtxt(DATA_DIR / "interim" / "hapt_3class_output_phase2" / "X_hapt.txt")
X_wisdm = np.loadtxt(DATA_DIR / "interim" / "Filtered_datasets_and_KS_results" / "3class_wisdm_phase2" / "X_filtered.txt")

# ======================
# LOAD LABELS
# ======================
y_hapt = np.loadtxt(DATA_DIR / "interim" / "hapt_3class_output_phase2" / "y_hapt.txt").astype(int)
y_wisdm = np.loadtxt(DATA_DIR / "interim" / "Filtered_datasets_and_KS_results" / "3class_wisdm_phase2" / "y_filtered.txt").astype(int)

remap = {1: 2, 2: 3, 3: 1}
y_wisdm = np.vectorize(remap.get)(y_wisdm)

# Safety checks
assert X_hapt.shape[0] == y_hapt.shape[0], "HAPT X/y mismatch"
assert X_wisdm.shape[0] == y_wisdm.shape[0], "WISDM X/y mismatch"
assert X_hapt.shape[1] == X_wisdm.shape[1], "Feature count mismatch (alignment broken)"

n_features = X_hapt.shape[1]

# =========================
# PER-ACTIVITY KS TESTS 
# =========================
per_activity_rows = []
activities = [1, 2, 3]

for act in activities:
    Xh = X_hapt[y_hapt == act]
    Xw = X_wisdm[y_wisdm == act]

    if len(Xh) == 0 or len(Xw) == 0:
        continue

    for i in range(n_features):
        stat, p = ks_2samp(Xh[:, i], Xw[:, i])
        per_activity_rows.append([LABELS[act], i, stat, p, int(p < 0.05)])

per_activity_arr = np.array(per_activity_rows, dtype=object)

# Save per-activity CSV
per_activity_csv = os.path.join(RESULTS_DIR, "ks_test_per_activity_3class_phase2.csv")
with open(per_activity_csv, "w") as f:
    f.write("activity,feature_idx,ks_statistic,p_value,significant\n")
    for row in per_activity_rows:
        f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

print("Saved:", per_activity_csv)

# ===============================
# PER-ACTIVITY SUMMARY + PLOT 
# ===============================
activity_summary = []
for act in activities:
    act_name = LABELS[act]
    rows = [r for r in per_activity_rows if r[0] == act_name]
    if not rows:
        continue
    ks_vals = np.array([r[2] for r in rows], dtype=float)
    sig_vals = np.array([r[4] for r in rows], dtype=int)
    activity_summary.append((act_name, ks_vals.mean(), 100.0 * sig_vals.mean()))

activity_summary.sort(key=lambda x: x[1], reverse=True)

names = [x[0] for x in activity_summary]
mean_ks = [x[1] for x in activity_summary]
sig_pct = [x[2] for x in activity_summary]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(names, mean_ks)
axes[0].set_xlabel("Mean K-S Statistic")
axes[0].set_title("Domain Shift by Activity (3CLASS)")
axes[0].invert_yaxis()

axes[1].barh(names, sig_pct)
axes[1].set_xlabel("% Features with Significant Shift (p < 0.05)")
axes[1].set_title("Statistical Significance by Activity")
axes[1].invert_yaxis()

plt.tight_layout()
out_plot = os.path.join(RESULTS_DIR, "ks_per_activity_3class_phase2.png")
plt.savefig(out_plot, dpi=300)
plt.close()

print("Saved:", out_plot)

# =====================================================
# OVERALL KS BEFORE NORMALIZATION (BASELINE)  ← FIX
# =====================================================
print("\n=== Running KS Before Normalization (Baseline) ===")

ks_stats = []
p_values = []

for i in range(n_features):
    stat, p = ks_2samp(X_hapt[:, i], X_wisdm[:, i])
    ks_stats.append(stat)
    p_values.append(p)

ks_stats = np.array(ks_stats)
p_values = np.array(p_values)

np.savetxt(
    os.path.join(RESULTS_DIR, "ks_test_overall_3class_raw.csv"),
    np.column_stack((ks_stats, p_values)),
    delimiter=",",
    header="ks_statistic,p_value",
    comments=""
)

print("Median KS (Before normalization):", np.median(ks_stats))
print("Significant features (p < 0.05):",
      np.sum(p_values < 0.05), "/", len(p_values))

# =====================================================
# Z-SCORE NORMALIZATION PER DATASET
# =====================================================
from sklearn.preprocessing import StandardScaler

print("\n=== Running KS After Z-Score Normalization ===")

scaler_hapt = StandardScaler()
scaler_wisdm = StandardScaler()

X_hapt_z = scaler_hapt.fit_transform(X_hapt)
X_wisdm_z = scaler_wisdm.fit_transform(X_wisdm)

ks_stats_z = []
p_values_z = []

for i in range(X_hapt_z.shape[1]):
    stat, p = ks_2samp(X_hapt_z[:, i], X_wisdm_z[:, i])
    ks_stats_z.append(stat)
    p_values_z.append(p)

ks_stats_z = np.array(ks_stats_z)
p_values_z = np.array(p_values_z)

np.savetxt(
    os.path.join(RESULTS_DIR, "ks_test_overall_3class_zscore.csv"),
    np.column_stack((ks_stats_z, p_values_z)),
    delimiter=",",
    header="ks_statistic,p_value",
    comments=""
)

print("Median KS (After Z-score):", np.median(ks_stats_z))
print("Significant features (p < 0.05):",
      np.sum(p_values_z < 0.05), "/", len(p_values_z))

# =====================================================
# KS DISTRIBUTION PLOT (Z-SCORE)
# =====================================================
plt.figure(figsize=(7, 5))
plt.hist(ks_stats_z, bins=20)
plt.axvline(np.median(ks_stats_z), linestyle="--")
plt.xlabel("K-S Statistic")
plt.ylabel("Number of Features")
plt.title("Distribution of K-S Statistics (3CLASS - After Z-Score)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ks_distribution_3class_zscore.png"), dpi=300)
plt.close()

# ============================
# BEFORE vs AFTER COMPARISON
# ============================
print("\n=== Distribution Shift Comparison ===")
print("Median KS Before:", np.median(ks_stats))
print("Median KS After :", np.median(ks_stats_z))
print("Significant % Before:", np.mean(p_values < 0.05) * 100)
print("Significant % After :", np.mean(p_values_z < 0.05) * 100)