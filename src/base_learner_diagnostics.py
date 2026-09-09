"""
Extended base learner diagnostics 

Produces:
  results/phase3/Extend Base Learner Diagnostics/
    01_master_diagnostic_table.csv
    02_domain_collapse_recovery.csv
    03_perclass_f1_naive.csv
    03_perclass_f1_zscore.csv
    04_prediction_distribution.csv
    05_confidence_analysis.csv
    06_algorithm_sensitivity_chart.png
    07_confidence_chart.png
    08_prediction_distribution_chart.png
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# =========================
# CONFIG
# =========================
LABELS_3CLASS = ["WALKING", "SITTING", "STANDING"]
REMAPPING_WISDM = {1: 2, 2: 3, 3: 1}

# Paths anchored to the repo root (this file lives in <repo_root>/src/).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Regenerate via src/feature_extraction_HAPT.py before running this script.
HAPT_X_PATH = DATA_DIR / "interim" / "hapt_3class_output_phase2" / "X_hapt.txt"
HAPT_Y_PATH = DATA_DIR / "interim" / "hapt_3class_output_phase2" / "y_hapt.txt"
# Regenerate via src/analysis_feature_dataset.py before running this script.
WISDM_X_PATH = DATA_DIR / "interim" / "Filtered_datasets_and_KS_results" / "3class_wisdm_phase2" / "X_filtered.txt"
WISDM_Y_PATH = DATA_DIR / "interim" / "Filtered_datasets_and_KS_results" / "3class_wisdm_phase2" / "y_filtered.txt"

OUT_DIR = os.path.join("results", "phase3", "Extend Base Learner Diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# DATA LOAD
# =========================
def load_data():
    X_hapt  = np.loadtxt(HAPT_X_PATH)
    y_hapt  = np.loadtxt(HAPT_Y_PATH).astype(int) - 1
    X_wisdm = np.loadtxt(WISDM_X_PATH)
    y_wisdm = np.loadtxt(WISDM_Y_PATH).astype(int)
    y_wisdm = np.vectorize(REMAPPING_WISDM.get)(y_wisdm) - 1
    return X_hapt, y_hapt, X_wisdm, y_wisdm


# =========================
# NORMALIZATION
# =========================
def norm_minmax(X_src, X_tgt):
    sc = MinMaxScaler(feature_range=(-1, 1))
    return sc.fit_transform(X_src), sc.transform(X_tgt)

def norm_zscore(X_src, X_tgt):
    return StandardScaler().fit_transform(X_src), StandardScaler().fit_transform(X_tgt)


# =========================
# BASE LEARNERS
# =========================
def get_base_learners():
    return [
        ("SGD",               SGDClassifier(loss='log_loss', max_iter=2000, tol=1e-3, random_state=42)),
        ("SVM",               SVC(kernel="rbf", C=1.0, probability=True, random_state=42, class_weight="balanced")),
        ("Random Forest",     RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42)),
        ("XGBoost",           XGBClassifier(n_estimators=20, learning_rate=0.1, random_state=42, eval_metric="mlogloss")),
    ]


# =====================================
# STEP 1+2: MASTER DIAGNOSTIC TABLE
# =====================================
def step1_master_diagnostic_table(X_hapt, y_hapt, X_wisdm, y_wisdm, ft_frac=0.10, seed=42):
    """
    Builds full diagnostic table:
    Model | HAPT Train | WISDM Naive | WISDM Z-score | Delta Z | WISDM FineTune | Delta FT
    Also computes: Domain Collapse, Recovery, Restoration Ratio
    """
    print("\n=== STEP 1+2: Master Diagnostic Table ===")

    # Normalizations
    Xs_mm, Xt_mm     = norm_minmax(X_hapt, X_wisdm)
    Xs_zs, Xt_zs     = norm_zscore(X_hapt, X_wisdm)

    # Fine-tune subset
    rng    = np.random.default_rng(seed)
    idx    = np.arange(len(y_wisdm))
    rng.shuffle(idx)
    k      = int(np.floor(ft_frac * len(y_wisdm)))
    idx_ft = idx[:k]
    X_ft   = Xt_mm[idx_ft]
    y_ft   = y_wisdm[idx_ft]
    X_mix  = np.vstack([Xs_mm, X_ft])
    y_mix  = np.concatenate([y_hapt, y_ft])

    rows = []
    for name, model in get_base_learners():

        # --- Naive (MinMax source-only) ---
        m_naive = clone(model)
        m_naive.fit(Xs_mm, y_hapt)
        hapt_train = accuracy_score(y_hapt, m_naive.predict(Xs_mm))
        wisdm_naive = accuracy_score(y_wisdm, m_naive.predict(Xt_mm))

        # --- Z-score ---
        m_zs = clone(model)
        m_zs.fit(Xs_zs, y_hapt)
        wisdm_zscore = accuracy_score(y_wisdm, m_zs.predict(Xt_zs))

        # --- Fine-tune (retrain on HAPT + ft_frac WISDM) ---
        m_ft = clone(model)
        m_ft.fit(X_mix, y_mix)
        wisdm_ft = accuracy_score(y_wisdm, m_ft.predict(Xt_mm))

        # --- Derived metrics ---
        domain_collapse      = hapt_train - wisdm_naive
        zscore_recovery      = wisdm_zscore - wisdm_naive
        ft_recovery          = wisdm_ft - wisdm_naive
        restoration_ratio    = zscore_recovery / domain_collapse if domain_collapse != 0 else 0

        rows.append({
            "Model":              name,
            "HAPT Train":         round(hapt_train,  4),
            "WISDM Naive":        round(wisdm_naive,  4),
            "WISDM Z-score":      round(wisdm_zscore, 4),
            "Delta Z-score":      round(zscore_recovery, 4),
            "WISDM FineTune":     round(wisdm_ft,     4),
            "Delta FineTune":     round(ft_recovery,  4),
            "Domain Collapse":    round(domain_collapse, 4),
            "Restoration Ratio":  round(restoration_ratio, 4),
        })

        print(f"  [{name}]")
        print(f"    HAPT Train:        {hapt_train:.4f}")
        print(f"    WISDM Naive:       {wisdm_naive:.4f}")
        print(f"    WISDM Z-score:     {wisdm_zscore:.4f}  (Δ {zscore_recovery:+.4f})")
        print(f"    WISDM FineTune:    {wisdm_ft:.4f}  (Δ {ft_recovery:+.4f})")
        print(f"    Domain Collapse:   {domain_collapse:.4f}")
        print(f"    Restoration Ratio: {restoration_ratio:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "01_master_diagnostic_table.csv"), index=False)
    print(f"\n  Saved: 01_master_diagnostic_table.csv")
    print(df.to_string(index=False))
    return df


# =========================
# STEP 3: PER-CLASS F1
# =========================
def step3_perclass_f1(X_hapt, y_hapt, X_wisdm, y_wisdm):
    """
    Per-class F1 and STANDING recall for each base learner,
    under both Naive and Z-score normalization.
    """
    print("\n=== STEP 3: Per-Class F1 Diagnostics ===")

    Xs_mm, Xt_mm = norm_minmax(X_hapt, X_wisdm)
    Xs_zs, Xt_zs = norm_zscore(X_hapt, X_wisdm)

    rows_naive  = []
    rows_zscore = []

    for name, model in get_base_learners():
        for tag, Xs, Xt, rows in [
            ("Naive",   Xs_mm, Xt_mm, rows_naive),
            ("Z-score", Xs_zs, Xt_zs, rows_zscore),
        ]:
            m = clone(model)
            m.fit(Xs, y_hapt)
            y_pred = m.predict(Xt)
            rep = classification_report(
                y_wisdm, y_pred, target_names=LABELS_3CLASS,
                output_dict=True, zero_division=0
            )
            rows.append({
                "Model":           name,
                "WALKING F1":      round(rep["WALKING"]["f1-score"],  4),
                "SITTING F1":      round(rep["SITTING"]["f1-score"],  4),
                "STANDING F1":     round(rep["STANDING"]["f1-score"], 4),
                "STANDING Recall": round(rep["STANDING"]["recall"],   4),
                "Macro F1":        round(rep["macro avg"]["f1-score"], 4),
            })

    df_naive  = pd.DataFrame(rows_naive)
    df_zscore = pd.DataFrame(rows_zscore)

    df_naive.to_csv( os.path.join(OUT_DIR, "03_perclass_f1_naive.csv"),  index=False)
    df_zscore.to_csv(os.path.join(OUT_DIR, "03_perclass_f1_zscore.csv"), index=False)

    print("\n  Per-Class F1 — Naive:")
    print(df_naive.to_string(index=False))
    print("\n  Per-Class F1 — Z-score:")
    print(df_zscore.to_string(index=False))

    return df_naive, df_zscore


# =========================================
# STEP 4: PREDICTION DISTRIBUTION DRIFT
# =========================================
def step4_prediction_distribution(X_hapt, y_hapt, X_wisdm, y_wisdm):
    """
    For each model: % predicted as each class on
    HAPT test, WISDM Naive, WISDM Z-score.
    Reveals if model collapses to predicting one class.
    """
    print("\n=== STEP 4: Prediction Distribution Drift ===")

    Xs_mm, Xt_mm = norm_minmax(X_hapt, X_wisdm)
    Xs_zs, Xt_zs = norm_zscore(X_hapt, X_wisdm)

    rows = []
    for name, model in get_base_learners():
        m_naive = clone(model)
        m_naive.fit(Xs_mm, y_hapt)
        m_zs = clone(model)
        m_zs.fit(Xs_zs, y_hapt)

        for dataset_tag, X_eval, m in [
            ("HAPT Train",   Xs_mm, m_naive),
            ("WISDM Naive",  Xt_mm, m_naive),
            ("WISDM Z-score",Xt_zs, m_zs),
        ]:
            preds = m.predict(X_eval)
            total = len(preds)
            rows.append({
                "Model":     name,
                "Dataset":   dataset_tag,
                "% WALKING":  round(100 * np.sum(preds == 0) / total, 2),
                "% SITTING":  round(100 * np.sum(preds == 1) / total, 2),
                "% STANDING": round(100 * np.sum(preds == 2) / total, 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "04_prediction_distribution.csv"), index=False)
    print(df.to_string(index=False))

    # ── Chart: stacked bars per model per dataset ──────────────────────
    models   = df["Model"].unique()
    datasets = ["HAPT Train", "WISDM Naive", "WISDM Z-score"]
    colors   = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, len(models), figsize=(18, 5), sharey=True)
    for ax, model_name in zip(axes, models):
        sub = df[df["Model"] == model_name]
        x   = np.arange(len(datasets))
        bottom = np.zeros(len(datasets))
        for i, cls in enumerate(["% WALKING", "% SITTING", "% STANDING"]):
            vals = [float(sub[sub["Dataset"] == d][cls].values[0]) for d in datasets]
            ax.bar(x, vals, bottom=bottom, label=LABELS_3CLASS[i], color=colors[i])
            bottom += np.array(vals)
        ax.set_title(model_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("% Predicted")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Prediction Distribution Drift per Model\n(HAPT Train vs WISDM Naive vs WISDM Z-score)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "08_prediction_distribution_chart.png"), dpi=300)
    plt.close()
    print("  Saved: 08_prediction_distribution_chart.png")

    return df


# =============================
# STEP 5: CONFIDENCE ANALYSIS
# =============================
def step5_confidence_analysis(X_hapt, y_hapt, X_wisdm, y_wisdm):
    """
    Mean max predicted probability (confidence) per model per dataset.
    High confidence + wrong = boundary misalignment.
    Low confidence = model is uncertain.
    """
    print("\n=== STEP 5: Confidence Analysis ===")

    Xs_mm, Xt_mm = norm_minmax(X_hapt, X_wisdm)
    Xs_zs, Xt_zs = norm_zscore(X_hapt, X_wisdm)

    rows = []
    for name, model in get_base_learners():
        m_naive = clone(model)
        m_naive.fit(Xs_mm, y_hapt)
        m_zs = clone(model)
        m_zs.fit(Xs_zs, y_hapt)

        for dataset_tag, X_eval, m in [
            ("HAPT Train",    Xs_mm, m_naive),
            ("WISDM Naive",   Xt_mm, m_naive),
            ("WISDM Z-score", Xt_zs, m_zs),
        ]:
            proba      = m.predict_proba(X_eval)          # shape (n, 3)
            confidence = float(np.mean(np.max(proba, axis=1)))
            rows.append({
                "Model":      name,
                "Dataset":    dataset_tag,
                "Confidence": round(confidence, 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "05_confidence_analysis.csv"), index=False)
    print(df.to_string(index=False))

    # ── Grouped bar chart ──────────────────────────────────────────────
    models   = df["Model"].unique()
    datasets = ["HAPT Train", "WISDM Naive", "WISDM Z-score"]
    x        = np.arange(len(models))
    width    = 0.25
    colors   = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (ds, color) in enumerate(zip(datasets, colors)):
        vals = [float(df[(df["Model"] == m) & (df["Dataset"] == ds)]["Confidence"].values[0])
                for m in models]
        bars = ax.bar(x + i * width, vals, width, label=ds, color=color, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean Max Probability (Confidence)")
    ax.set_title("Model Confidence: HAPT Train vs WISDM Naive vs WISDM Z-score")
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_confidence_chart.png"), dpi=300)
    plt.close()
    print("  Saved: 07_confidence_chart.png")

    return df


# =====================================
# STEP 6: ALGORITHM SENSITIVITY CHART
# =====================================
def step6_sensitivity_chart(df_master):
    """
    Bar chart: Naive | Z-score | FineTune per model.
    Title: 'Algorithm Sensitivity to Cross-Dataset Shift'
    """
    print("\n=== STEP 6: Algorithm Sensitivity Chart ===")

    models   = df_master["Model"].tolist()
    naive    = df_master["WISDM Naive"].tolist()
    zscore   = df_master["WISDM Z-score"].tolist()
    finetune = df_master["WISDM FineTune"].tolist()

    x     = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, naive,    width, label="Naive (MinMax)", color="#4C72B0", edgecolor="black")
    b2 = ax.bar(x,         zscore,   width, label="Z-score",        color="#DD8452", edgecolor="black")
    b3 = ax.bar(x + width, finetune, width, label="Fine-Tune (10%)", color="#55A868", edgecolor="black")

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}",
                    ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy on WISDM (target)")
    ax.set_title("Algorithm Sensitivity to Cross-Dataset Shift\n(HAPT → WISDM)")
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_algorithm_sensitivity_chart.png"), dpi=300)
    plt.close()
    print("  Saved: 06_algorithm_sensitivity_chart.png")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Loading data...")
    X_hapt, y_hapt, X_wisdm, y_wisdm = load_data()
    print(f"HAPT : {X_hapt.shape}  WISDM: {X_wisdm.shape}")

    # STEP 1+2: Master diagnostic table + collapse/recovery metrics
    df_master = step1_master_diagnostic_table(X_hapt, y_hapt, X_wisdm, y_wisdm, ft_frac=0.10)

    # STEP 3: Per-class F1 per model
    df_naive_f1, df_zscore_f1 = step3_perclass_f1(X_hapt, y_hapt, X_wisdm, y_wisdm)

    # STEP 4: Prediction distribution drift
    df_dist = step4_prediction_distribution(X_hapt, y_hapt, X_wisdm, y_wisdm)

    # STEP 5: Confidence analysis
    df_conf = step5_confidence_analysis(X_hapt, y_hapt, X_wisdm, y_wisdm)

    # STEP 6: Algorithm sensitivity chart
    step6_sensitivity_chart(df_master)

    print("\n\n✅ All Priority 3 outputs saved to:", OUT_DIR)
    print("\nFiles produced:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")