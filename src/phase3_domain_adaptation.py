import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, List

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================
LABELS_3CLASS = ["WALKING", "SITTING", "STANDING"]
REMAPPING_WISDM = {1: 2, 2: 3, 3: 1}  # WISDM: 1=SITTING,2=STANDING,3=WALKING -> desired: 1=WALKING,2=SITTING,3=STANDING

# Paths anchored to the repo root (this file lives in <repo_root>/src/).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Regenerate via src/feature_extraction_HAPT.py before running this script.
HAPT_X_PATH = DATA_DIR / "interim" / "hapt_3class_output_phase2" / "X_hapt.txt"
HAPT_Y_PATH = DATA_DIR / "interim" / "hapt_3class_output_phase2" / "y_hapt.txt"

# Regenerate via src/analysis_feature_dataset.py before running this script.
WISDM_X_PATH = DATA_DIR / "interim" / "Filtered_datasets_and_KS_results" / "3class_wisdm_phase2" / "X_filtered.txt"
WISDM_Y_PATH = DATA_DIR / "interim" / "Filtered_datasets_and_KS_results" / "3class_wisdm_phase2" / "y_filtered.txt"

PHASE3_DIR = os.path.join("results", "phase3")
os.makedirs(PHASE3_DIR, exist_ok=True)


# =========================
# MODEL (Base learners)
# =========================
def get_base_learners(random_state: int = 42):
    return [
        ("sgd", SGDClassifier(loss='log_loss', max_iter=2000, tol=1e-3, random_state=random_state)),
        ("random_forest", RandomForestClassifier(n_estimators=20, max_depth=5, random_state=random_state)),
        ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=random_state, class_weight="balanced")),
        ("xgboost", XGBClassifier(n_estimators=20, learning_rate=0.1, random_state=random_state, eval_metric="mlogloss")),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=random_state)),
    ]


def get_meta_learner(random_state: int = 42):
    return LogisticRegression(
        solver="saga",
        max_iter=2000,
        tol=1e-4,
        class_weight="balanced",
        random_state=random_state
    )


# =========================
# DATA LOAD
# =========================
def load_hapt_wisdm_3class() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_hapt = np.loadtxt(HAPT_X_PATH)
    y_hapt = np.loadtxt(HAPT_Y_PATH).astype(int)

    X_wisdm = np.loadtxt(WISDM_X_PATH)
    y_wisdm = np.loadtxt(WISDM_Y_PATH).astype(int)

    # Remap WISDM semantics
    y_wisdm = np.vectorize(REMAPPING_WISDM.get)(y_wisdm)

    # Convert to 0-indexed for XGBoost compatibility
    y_hapt  = y_hapt  - 1   # [1,2,3] -> [0,1,2]
    y_wisdm = y_wisdm - 1   # [1,2,3] -> [0,1,2]

    return X_hapt, y_hapt, X_wisdm, y_wisdm


# =========================
# NORMALIZATION OPTIONS
# =========================
def normalize_minmax_source_only(X_src, X_tgt):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_src_n = scaler.fit_transform(X_src)
    X_tgt_n = scaler.transform(X_tgt)
    return X_src_n, X_tgt_n


def normalize_zscore_per_dataset(X_src, X_tgt):
    s_src = StandardScaler()
    s_tgt = StandardScaler()
    X_src_n = s_src.fit_transform(X_src)
    X_tgt_n = s_tgt.fit_transform(X_tgt)
    return X_src_n, X_tgt_n


# =====================================
# MASTER-TABLE METRICS HELPER  ← NEW
# =====================================
def save_experiment_metrics(out_dir: str, method_name: str,
                             y_true: np.ndarray, y_pred: np.ndarray,
                             labels=("WALKING", "SITTING", "STANDING")) -> dict:
    """
    Saves metrics.json (and classification_report.json) for one experiment.
    These files are read by build_master_table.py to produce the Word table.
    """
    os.makedirs(out_dir, exist_ok=True)

    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=list(labels),
        output_dict=True,
        zero_division=0
    )

    payload = {
        "method":          method_name,
        "accuracy":        float(acc),
        "macro_f1":        float(report["macro avg"]["f1-score"]),
        "walking_f1":      float(report["WALKING"]["f1-score"]),
        "sitting_f1":      float(report["SITTING"]["f1-score"]),
        "standing_f1":     float(report["STANDING"]["f1-score"]),
        "standing_recall": float(report["STANDING"]["recall"]),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  ── {method_name}")
    print(f"     Accuracy        : {acc:.4f}")
    print(f"     Macro F1        : {report['macro avg']['f1-score']:.4f}")
    print(f"     WALKING  F1     : {report['WALKING']['f1-score']:.4f}")
    print(f"     SITTING  F1     : {report['SITTING']['f1-score']:.4f}")
    print(f"     STANDING F1     : {report['STANDING']['f1-score']:.4f}")
    print(f"     STANDING Recall : {report['STANDING']['recall']:.4f}")

    return payload


# =====================================
# INDIVIDUAL BASE LEARNER EVALUATION
# =====================================
def exp_individual_base_learner_eval(X_hapt, y_hapt, X_wisdm, y_wisdm):
    out_dir = os.path.join(PHASE3_DIR, "0_individual_base_learners")
    os.makedirs(out_dir, exist_ok=True)

    Xs, Xt = normalize_minmax_source_only(X_hapt, X_wisdm)

    results = {}
    summary_lines = [f"{'Base Learner':<25} {'Train Acc (HAPT)':>18} {'Test Acc (WISDM)':>18}\n",
                     "-" * 63 + "\n"]

    for name, model in get_base_learners():
        m = clone(model)
        m.fit(Xs, y_hapt)

        train_acc = accuracy_score(y_hapt, m.predict(Xs))
        y_pred    = m.predict(Xt)
        test_acc  = accuracy_score(y_wisdm, y_pred)

        cm     = confusion_matrix(y_wisdm, y_pred)
        report = classification_report(y_wisdm, y_pred, target_names=LABELS_3CLASS, digits=4)

        save_metrics(out_dir, name, train_acc, test_acc, report, cm)
        save_degradation_plot(out_dir, name, train_acc, test_acc)

        results[name] = test_acc
        summary_lines.append(f"{name:<25} {train_acc:>18.4f} {test_acc:>18.4f}\n")
        print(f"  [{name}]  HAPT train: {train_acc:.4f}  |  WISDM test: {test_acc:.4f}")

    with open(os.path.join(out_dir, "summary_table.txt"), "w") as f:
        f.writelines(summary_lines)

    names = list(results.keys())
    accs  = list(results.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accs, color="steelblue", edgecolor="black")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy on WISDM (target)")
    plt.title("Individual Base Learner Accuracy: HAPT → WISDM")
    plt.xticks(rotation=15, ha="right")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{acc:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "base_learners_bar_chart.png"), dpi=300)
    plt.close()

    return results

# ==========================================
# BASE LEARNER Z-SCORE RESTORATION  
# ==========================================
def exp_base_learners_zscore(X_hapt, y_hapt, X_wisdm, y_wisdm):
    """
    Re-run every base learner after z-score normalization applied
    independently to each dataset (no label information used).

    Outputs
    -------
    results/phase3/0b_base_learners_zscore/
        zscore_vs_naive_summary.txt   — Naive | Z-score | Δ per model
        zscore_vs_naive_bar_chart.png — grouped bar chart
        {model}_zscore_metrics.txt    — per-model train/test accuracy
        {model}_zscore_confusion_matrix.png
    """
    out_dir = os.path.join(PHASE3_DIR, "0b_base_learners_zscore")
    os.makedirs(out_dir, exist_ok=True)

    # Naive (MinMax) accuracies from exp_individual_base_learner_eval
    naive_accs = {
        "sgd":               0.3865,
        "random_forest":     0.6070,
        "svm":               0.3934,
        "xgboost":           0.7427,
        "gradient_boosting": 0.5755,
    }

    Xs, Xt = normalize_zscore_per_dataset(X_hapt, X_wisdm)

    zscore_accs  = {}
    summary_lines = [
        f"{'Model':<25} {'Naive':>8} {'Z-score':>10} {'Δ':>8}\n",
        "-" * 55 + "\n"
    ]

    for name, model in get_base_learners():
        m         = clone(model)
        m.fit(Xs, y_hapt)
        train_acc = accuracy_score(y_hapt, m.predict(Xs))
        y_pred    = m.predict(Xt)
        test_acc  = accuracy_score(y_wisdm, y_pred)
        delta     = test_acc - naive_accs[name]
        cm        = confusion_matrix(y_wisdm, y_pred)
        report    = classification_report(y_wisdm, y_pred, target_names=LABELS_3CLASS, digits=4)

        save_metrics(out_dir, f"{name}_zscore", train_acc, test_acc, report, cm)

        zscore_accs[name] = test_acc
        summary_lines.append(
            f"{name:<25} {naive_accs[name]:>8.4f} {test_acc:>10.4f} {delta:>+8.4f}\n"
        )
        print(f"  [{name}]  Naive: {naive_accs[name]:.4f}  |  Z-score: {test_acc:.4f}  |  Δ: {delta:+.4f}")

    with open(os.path.join(out_dir, "zscore_vs_naive_summary.txt"), "w", encoding="utf-8") as f:
        f.writelines(summary_lines)

    print("\n  Summary →", os.path.join(out_dir, "zscore_vs_naive_summary.txt"))

    # ── Grouped bar chart ───────────────────────────────────────────────
    model_names = list(naive_accs.keys())
    naive_vals  = [naive_accs[n]  for n in model_names]
    zscore_vals = [zscore_accs[n] for n in model_names]
    deltas      = [zscore_accs[n] - naive_accs[n] for n in model_names]

    x     = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, naive_vals,  width, label="Naive (MinMax)",
                   color="steelblue",  edgecolor="black")
    bars2 = ax.bar(x + width/2, zscore_vals, width, label="Z-score",
                   color="darkorange", edgecolor="black")

    # Annotate Δ above each z-score bar
    for bar, d in zip(bars2, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{d:+.3f}", ha="center", fontsize=8,
                color="darkred", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy on WISDM (target)")
    ax.set_title("Base Learner Accuracy: Naive vs Z-score Normalization\n(HAPT → WISDM, Step 4)")
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "zscore_vs_naive_bar_chart.png"), dpi=300)
    plt.close()

    print("  Chart  →", os.path.join(out_dir, "zscore_vs_naive_bar_chart.png"))

    return zscore_accs


# ==================================================
# STACKING
# ==================================================
@dataclass
class FrozenBaseStacking:
    base_learners: List[Tuple[str, object]]
    meta_learner: object

    def fit_base(self, X, y):
        self.fitted_base_ = []
        for name, model in self.base_learners:
            m = clone(model)
            m.fit(X, y)
            self.fitted_base_.append((name, m))
        return self

    def base_proba_features(self, X) -> np.ndarray:
        feats = []
        for _, m in self.fitted_base_:
            feats.append(m.predict_proba(X))
        return np.hstack(feats)

    def fit_meta(self, X_meta, y_meta):
        self.fitted_meta_ = clone(self.meta_learner)
        self.fitted_meta_.fit(X_meta, y_meta)
        return self

    def predict(self, X):
        Z = self.base_proba_features(X)
        return self.fitted_meta_.predict(Z)

    def evaluate(self, X, y) -> Dict:
        y_pred = self.predict(X)
        acc    = accuracy_score(y, y_pred)
        cm     = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, target_names=LABELS_3CLASS, digits=4)
        return {"acc": acc, "cm": cm, "report": report, "y_pred": y_pred}


# =========================
# SAVING HELPERS
# =========================
def save_confusion_matrix(cm, out_png, title):
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS_3CLASS, yticklabels=LABELS_3CLASS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_metrics(out_dir, name, train_acc, test_acc, report, cm):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Train accuracy: {train_acc:.4f}\n")
        f.write(f"Test accuracy : {test_acc:.4f}\n")
    with open(os.path.join(out_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(report)
    np.savetxt(os.path.join(out_dir, f"{name}_confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
    save_confusion_matrix(cm, os.path.join(out_dir, f"{name}_confusion_matrix.png"),
                          title=f"{name}: Confusion Matrix (HAPT → WISDM)")


def save_degradation_plot(out_dir, name, train_acc, test_acc):
    plt.figure(figsize=(6, 5))
    plt.bar(["Source (HAPT)", "Target (WISDM)"], [train_acc, test_acc])
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(f"{name}: Performance Degradation")
    for i, v in enumerate([train_acc, test_acc]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_performance_degradation.png"), dpi=300)
    plt.close()


# =========================
# EXPERIMENTS
# =========================

def exp_feature_distribution_alignment(X_hapt, y_hapt, X_wisdm, y_wisdm):
    """
    A) MinMax source-only  → Row: Baseline (Naïve Transfer)
    B) Z-score per dataset → Row: Z-score Normalization
    """
    out_dir = os.path.join(PHASE3_DIR, "A_distribution_alignment")

    base = get_base_learners()
    meta = get_meta_learner()

    # ── A) Baseline (MinMax, source only) ──────────────────────────────
    sub_a = os.path.join(out_dir, "baseline")
    Xs, Xt = normalize_minmax_source_only(X_hapt, X_wisdm)

    modelA = FrozenBaseStacking(base, meta).fit_base(Xs, y_hapt)
    modelA.fit_meta(modelA.base_proba_features(Xs), y_hapt)

    train_res = modelA.evaluate(Xs, y_hapt)
    test_res  = modelA.evaluate(Xt, y_wisdm)

    save_metrics(sub_a, "minmax_source_only",
                 train_res["acc"], test_res["acc"],
                 test_res["report"], test_res["cm"])
    save_degradation_plot(sub_a, "minmax_source_only",
                          train_res["acc"], test_res["acc"])

    save_experiment_metrics(           # ← fills metrics.json for master table
        out_dir=sub_a,
        method_name="Baseline (Naive Transfer)",
        y_true=y_wisdm,
        y_pred=test_res["y_pred"],
    )

    # ── B) Z-score per dataset ──────────────────────────────────────────
    sub_b = os.path.join(out_dir, "zscore")
    Xs2, Xt2 = normalize_zscore_per_dataset(X_hapt, X_wisdm)

    modelB = FrozenBaseStacking(base, meta).fit_base(Xs2, y_hapt)
    modelB.fit_meta(modelB.base_proba_features(Xs2), y_hapt)

    train_res2 = modelB.evaluate(Xs2, y_hapt)
    test_res2  = modelB.evaluate(Xt2, y_wisdm)

    save_metrics(sub_b, "zscore_per_dataset",
                 train_res2["acc"], test_res2["acc"],
                 test_res2["report"], test_res2["cm"])
    save_degradation_plot(sub_b, "zscore_per_dataset",
                          train_res2["acc"], test_res2["acc"])

    save_experiment_metrics(           # ← fills metrics.json for master table
        out_dir=sub_b,
        method_name="Z-score Normalization",
        y_true=y_wisdm,
        y_pred=test_res2["y_pred"],
    )


def exp_meta_learner_adaptation(X_hapt, y_hapt, X_wisdm, y_wisdm,
                                wisdm_frac=0.10, seed=42):
    """
    Freeze base learners (trained on HAPT).
    Retrain ONLY the meta-learner on a small WISDM subset.
    Row: Meta-Only Adaptation (10%)
    """
    out_dir = os.path.join(PHASE3_DIR, "B_meta_learner_adaptation")
    sub     = os.path.join(out_dir, f"meta_only_frac{int(wisdm_frac*100)}")

    rng = np.random.default_rng(seed)
    n   = len(y_wisdm)
    idx = np.arange(n)
    rng.shuffle(idx)

    k      = int(np.floor(wisdm_frac * n))
    idx_ft = idx[:k]

    base = get_base_learners()
    meta = get_meta_learner()

    Xs, Xt = normalize_minmax_source_only(X_hapt, X_wisdm)
    X_ft_n = Xt[idx_ft]
    y_ft   = y_wisdm[idx_ft]

    model = FrozenBaseStacking(base, meta).fit_base(Xs, y_hapt)

    # --- baseline meta (HAPT-only) ---
    model.fit_meta(model.base_proba_features(Xs), y_hapt)
    base_test = model.evaluate(Xt, y_wisdm)
    save_metrics(sub, f"baseline_meta_before_adapt_frac{int(wisdm_frac*100)}",
                 model.evaluate(Xs, y_hapt)["acc"], base_test["acc"],
                 base_test["report"], base_test["cm"])

    # --- retrain meta on WISDM subset ---
    model.fit_meta(model.base_proba_features(X_ft_n), y_ft)
    after_test  = model.evaluate(Xt, y_wisdm)
    after_train = model.evaluate(Xs, y_hapt)

    save_metrics(sub, f"meta_only_adapt_frac{int(wisdm_frac*100)}",
                 after_train["acc"], after_test["acc"],
                 after_test["report"], after_test["cm"])
    save_degradation_plot(sub, f"meta_only_adapt_frac{int(wisdm_frac*100)}",
                          after_train["acc"], after_test["acc"])

    save_experiment_metrics(           # ← fills metrics.json for master table
        out_dir=sub,
        method_name=f"Meta-Only Adaptation ({int(wisdm_frac*100)}%)",
        y_true=y_wisdm,
        y_pred=after_test["y_pred"],
    )


def exp_fine_tuning_learning_curve(X_hapt, y_hapt, X_wisdm, y_wisdm,
                                   fracs=(0.05, 0.10, 0.25, 0.50), seed=42):
    """
    Retrain the full ensemble on HAPT + X% of WISDM.
    Rows: Fine-Tuning (5%), Fine-Tuning (10%), Fine-Tuning (25%), Fine-Tuning (50%)
    """
    out_dir = os.path.join(PHASE3_DIR, "C_finetune_learning_curve")
    os.makedirs(out_dir, exist_ok=True)

    Xs, Xt = normalize_minmax_source_only(X_hapt, X_wisdm)

    rng = np.random.default_rng(seed)
    n   = len(y_wisdm)
    idx = np.arange(n)
    rng.shuffle(idx)

    accs = []

    for frac in fracs:
        k      = int(np.floor(frac * n))
        idx_ft = idx[:k]
        X_ft   = Xt[idx_ft]
        y_ft   = y_wisdm[idx_ft]

        X_mix = np.vstack([Xs, X_ft])
        y_mix = np.concatenate([y_hapt, y_ft])

        model = FrozenBaseStacking(get_base_learners(), get_meta_learner())
        model.fit_base(X_mix, y_mix)
        model.fit_meta(model.base_proba_features(X_mix), y_mix)

        train_res = model.evaluate(X_mix, y_mix)
        test_res  = model.evaluate(Xt, y_wisdm)

        tag = f"finetune_frac{int(frac*100)}"
        sub = os.path.join(out_dir, tag)

        save_metrics(sub, tag,
                     train_res["acc"], test_res["acc"],
                     test_res["report"], test_res["cm"])
        save_degradation_plot(sub, tag, train_res["acc"], test_res["acc"])

        save_experiment_metrics(       # ← fills metrics.json for master table
            out_dir=sub,
            method_name=f"Fine-Tuning ({int(frac*100)}%)",
            y_true=y_wisdm,
            y_pred=test_res["y_pred"],
        )

        accs.append((frac, test_res["acc"]))

    # learning curve plot
    fr = [f * 100 for f, _ in accs]
    ac = [a        for _, a in accs]
    plt.figure(figsize=(7, 5))
    plt.plot(fr, ac, marker="o")
    plt.ylim(0, 1.0)
    plt.xlabel("WISDM labeled subset used for fine-tuning (%)")
    plt.ylabel("WISDM Accuracy")
    plt.title("Fine-tuning Learning Curve (HAPT → WISDM)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve.png"), dpi=300)
    plt.close()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    X_hapt, y_hapt, X_wisdm, y_wisdm = load_hapt_wisdm_3class()

    print("HAPT  :", X_hapt.shape,  np.unique(y_hapt,  return_counts=True))
    print("WISDM :", X_wisdm.shape, np.unique(y_wisdm, return_counts=True))

    print("\n=== 0)  Individual Base Learners: Naive (MinMax) ===")
    exp_individual_base_learner_eval(X_hapt, y_hapt, X_wisdm, y_wisdm)

    print("\n=== 0b) Individual Base Learners: Z-score  [STEP 4] ===")
    exp_base_learners_zscore(X_hapt, y_hapt, X_wisdm, y_wisdm)

    print("\n=== A)  Distribution Alignment (Baseline + Z-score) ===")
    exp_feature_distribution_alignment(X_hapt, y_hapt, X_wisdm, y_wisdm)

    print("\n=== B)  Meta-Only Adaptation (10%) ===")
    exp_meta_learner_adaptation(X_hapt, y_hapt, X_wisdm, y_wisdm, wisdm_frac=0.10)

    print("\n=== C)  Fine-Tuning Learning Curve ===")
    exp_fine_tuning_learning_curve(X_hapt, y_hapt, X_wisdm, y_wisdm,
                                   fracs=(0.05, 0.10, 0.25, 0.50))

    print("\n\nAll done.")