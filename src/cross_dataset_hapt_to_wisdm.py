# Load data
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
import os

# Re-using stacking ensemble model from Human_Activity_Recognition.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from xgboost import XGBClassifier

# ==================
#     RUN CONFIG
# ==================

# Paths are anchored to the repo root (this file lives in <repo_root>/src/),
# so the script runs the same regardless of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

PHASE_TAG = "phase2"  # this script only ever produces 80-feature output (Phase 2 methodology);
                       # Phase 1's 61-feature extractor code no longer exists in this repo.

RESULTS_DIR = os.path.join("results", PHASE_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS = ["WALKING", "SITTING", "STANDING"]

# Load HAPT (source). Regenerate via src/feature_extraction_HAPT.py
# into data/interim/hapt_3class_output_phase2/ before running this script.
X_train = np.loadtxt(DATA_DIR / "interim" / "hapt_3class_output_phase2" / "X_hapt.txt")
y_train = np.loadtxt(DATA_DIR / "interim" / "hapt_3class_output_phase2" / "y_hapt.txt").astype(int)

# Load WISDM (target). Regenerate via src/analysis_feature_dataset.py
# into data/interim/Filtered_datasets_and_KS_results/3class_wisdm_phase2/ before running this script.
X_test = np.loadtxt(DATA_DIR / "interim" / "Filtered_datasets_and_KS_results" / "3class_wisdm_phase2" / "X_filtered.txt")
y_test = np.loadtxt(DATA_DIR / "interim" / "Filtered_datasets_and_KS_results" / "3class_wisdm_phase2" / "y_filtered.txt").astype(int)

# --- FIX WISDM LABEL SEMANTICS (CRITICAL) ---
# WISDM mapping: 1=SITTING, 2=STANDING, 3=WALKING
# Desired mapping: 1=WALKING, 2=SITTING, 3=STANDING

print("HAPT labels:", np.unique(y_train))
print("WISDM labels before remap:", np.unique(y_test))
print("WISDM label counts before remap:", {label: np.sum(y_test == label) for label in np.unique(y_test)})

remap = {
    1: 2,  # SITTING  -> 2
    2: 3,  # STANDING -> 3
    3: 1   # WALKING  -> 1
}

y_test = np.vectorize(remap.get)(y_test)
print("WISDM labels after remap:", np.unique(y_test))
print("WISDM label counts after remap:", {label: np.sum(y_test == label) for label in np.unique(y_test)})
# ------------------------------------------

print("Train:", X_train.shape, y_train.shape, "labels:", sorted(set(y_train)))
print("Test :", X_test.shape, y_test.shape, "labels:", sorted(set(y_test)))

# Normalize features to [-1, 1]

scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)   




def get_advanced_stacking_model():
    level0 = [
        ('perceptron', Perceptron(max_iter=2000, tol=1e-3)),
        ('random_forest', RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42, class_weight='balanced')),
        ('xgboost', XGBClassifier(n_estimators=20, learning_rate=0.1, random_state=42)),
        ('gradient_boosting', GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42))
    ]
    level1 = LogisticRegression(solver='saga', max_iter=2000, tol=1e-4, class_weight='balanced', random_state=42)

    # Single-layer stacking model with 3-fold cross-validation
    return StackingClassifier(estimators=level0, final_estimator=level1, cv=3)

model = get_advanced_stacking_model()


# More robust stacking ensemble
"""
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
def get_stacking_ensemble(random_state: int = 42) -> StackingClassifier:
    level0 = [
        ("perceptron", Perceptron(max_iter=2000, tol=1e-3, random_state=random_state)),
        ("random_forest", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state)),
        ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=random_state)),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=random_state)),
    ]

    if HAS_XGB:
        level0.append(("xgboost", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="mlogloss",
        )))

    level1 = LogisticRegression(solver="saga", max_iter=4000, tol=1e-4, random_state=random_state)

    # Use predict_proba outputs (default) from base learners when available
    model = StackingClassifier(
        estimators=level0,
        final_estimator=level1,
        cv=3,
        n_jobs=-1,
        passthrough=False,
    )
    return model
"""

# Train on HAPT 
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"HAPT train accuracy: {train_acc:.3f}")

# Evaluate on WISDM
y_pred = model.predict(X_test)

"""
Plausible Expectation:
Likely 0.50 to 0.70
Definitely > 0.34
Definitely ≪ training accuracy
"""
test_acc = accuracy_score(y_test, y_pred)
print(f"WISDM test accuracy: {test_acc:.3f}")

# ==============================
# PERFORMANCE DEGRADATION PLOT
# ==============================
plt.figure(figsize=(6, 5))
plt.bar(["Source (HAPT)", "Target (WISDM)"], [train_acc, test_acc])
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title(f"{PHASE_TAG.upper()} – Performance Degradation")

for i, v in enumerate([train_acc, test_acc]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "performance_degradation.png"), dpi=300)
plt.close()


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Classification report
print(classification_report(
    y_test,
    y_pred,
    target_names=["WALKING", "SITTING", "STANDING"]
))

# Classification report (text)
report_txt = classification_report(y_test, y_pred, target_names=LABELS)
print(report_txt)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report_txt)

# Save confusion matrix as CSV too
np.savetxt(os.path.join(RESULTS_DIR, "confusion_matrix_raw.csv"), cm, delimiter=",", fmt="%d")

# Metrics and data records for documentation purposes in Phase 1 (61 features) and Phase 2 (70-80-90 features)

# Accuracy recording

with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
    f.write(f"Phase: {PHASE_TAG}\n")
    f.write(f"HAPT train accuracy: {train_acc:.4f}\n")
    f.write(f"WISDM test accuracy: {test_acc:.4f}\n")
    f.write(f"Train shape: {X_train.shape}\n")
    f.write(f"Test shape : {X_test.shape}\n")
    f.write(f"WISDM label counts after remap: { {label: int(np.sum(y_test == label)) for label in np.unique(y_test)} }\n")




# Raw confusion matrix plot
labels = ["WALKING", "SITTING", "STANDING"]

# Raw confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABELS,
    yticklabels=LABELS
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"{PHASE_TAG.upper()} – Confusion Matrix (HAPT → WISDM)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_raw.png"), dpi=300)
plt.close()

# Normalized confusion matrix plot (row-normalized)
row_sums = cm.sum(axis=1, keepdims=True).astype(float)
row_sums[row_sums == 0] = 1.0  # avoid division by zero
cm_norm = cm.astype(float) / row_sums

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=LABELS,
    yticklabels=LABELS
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"{PHASE_TAG.upper()} – Normalized Confusion Matrix (HAPT → WISDM)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_normalized.png"), dpi=300)
plt.close()


# Recall = how well each activity transfers.
recall = recall_score(y_test, y_pred, average=None)

plt.figure(figsize=(6, 4))
plt.bar(LABELS, recall)
plt.ylim(0, 1.05)
plt.ylabel("Recall")
plt.title(f"{PHASE_TAG.upper()} – Per-Class Recall (HAPT → WISDM)")

for i, v in enumerate(recall):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "recall_per_class.png"), dpi=300)
plt.close()
