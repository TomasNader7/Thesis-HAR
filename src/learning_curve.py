"""
priority4_learning_curve.py
----------------------------
Enhanced fine-tuning learning curve analysis.

Produces:
  results/phase3/priority4/
    01_accuracy_learning_curve.png     — main curve with reference lines + recovery annotations
    02_standing_recall_curve.png       — STANDING recall recovery curve
    03_data_efficiency_table.csv       — marginal gains + thresholds
    04_comparison_table.csv            — compact method comparison table
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ── Stacking (same as phase3) ──────────────────────────────────────────────
from sklearn.base import clone as sk_clone
from dataclasses import dataclass
from typing import List, Tuple, Dict

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

OUT_DIR = os.path.join("results", "phase3", "priority4")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# DATA + NORMALIZATION
# =========================
def load_data():
    X_hapt  = np.loadtxt(HAPT_X_PATH)
    y_hapt  = np.loadtxt(HAPT_Y_PATH).astype(int) - 1
    X_wisdm = np.loadtxt(WISDM_X_PATH)
    y_wisdm = np.loadtxt(WISDM_Y_PATH).astype(int)
    y_wisdm = np.vectorize(REMAPPING_WISDM.get)(y_wisdm) - 1
    return X_hapt, y_hapt, X_wisdm, y_wisdm

def norm_minmax(X_src, X_tgt):
    sc = MinMaxScaler(feature_range=(-1, 1))
    return sc.fit_transform(X_src), sc.transform(X_tgt)

def norm_zscore(X_src, X_tgt):
    return StandardScaler().fit_transform(X_src), StandardScaler().fit_transform(X_tgt)


# =========================
# MODEL
# =========================
def get_base_learners():
    return [
        ("sgd",               SGDClassifier(loss='log_loss', max_iter=2000, tol=1e-3, random_state=42)),
        ("random_forest",     RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)),
        ("svm",               SVC(kernel="rbf", C=1.0, probability=True, random_state=42, class_weight="balanced")),
        ("xgboost",           XGBClassifier(n_estimators=20, learning_rate=0.1, random_state=42, eval_metric="mlogloss")),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42)),
    ]

def get_meta_learner():
    return LogisticRegression(solver="saga", max_iter=2000, tol=1e-4,
                              class_weight="balanced", random_state=42)


@dataclass
class FrozenBaseStacking:
    base_learners: List[Tuple[str, object]]
    meta_learner: object

    def fit_base(self, X, y):
        self.fitted_base_ = [(n, sk_clone(m).fit(X, y)) for n, m in self.base_learners]
        return self

    def base_proba_features(self, X):
        return np.hstack([m.predict_proba(X) for _, m in self.fitted_base_])

    def fit_meta(self, X_meta, y_meta):
        self.fitted_meta_ = sk_clone(self.meta_learner).fit(X_meta, y_meta)
        return self

    def predict(self, X):
        return self.fitted_meta_.predict(self.base_proba_features(X))

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        rep = classification_report(y, y_pred, target_names=LABELS_3CLASS,
                                    output_dict=True, zero_division=0)
        return {
            "acc":             accuracy_score(y, y_pred),
            "standing_recall": rep["STANDING"]["recall"],
            "y_pred":          y_pred,
        }


# =========================
# KNOWN REFERENCE VALUES
# (from your master table)
# =========================
BASELINE_ACC      = 0.4048
ZSCORE_ACC        = 0.7279
FT50_ACC          = 0.9893

BASELINE_RECALL   = 0.638
ZSCORE_RECALL     = 0.644
FT50_RECALL       = 0.992


# =========================
# RECOVERY RATIO
# =========================
def recovery_ratio(val, baseline, upper):
    """How much of the maximum possible gap was recovered."""
    if upper == baseline:
        return 0.0
    return (val - baseline) / (upper - baseline)


# =========================
# STEP 1+2+3+4 — ACCURACY LEARNING CURVE
# =========================
def build_learning_curve(X_hapt, y_hapt, X_wisdm, y_wisdm,
                         fracs=(0.05, 0.10, 0.25, 0.50), seed=42):
    """
    Runs the fine-tuning ensemble at each fraction.
    Returns lists of (frac%, accuracy, standing_recall).
    0% point is injected as the Baseline values.
    """
    Xs, Xt = norm_minmax(X_hapt, X_wisdm)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_wisdm))
    rng.shuffle(idx)

    results = []  # (pct, acc, standing_recall)

    for frac in fracs:
        k      = int(np.floor(frac * len(y_wisdm)))
        idx_ft = idx[:k]
        X_mix  = np.vstack([Xs, Xt[idx_ft]])
        y_mix  = np.concatenate([y_hapt, y_wisdm[idx_ft]])

        model = FrozenBaseStacking(get_base_learners(), get_meta_learner())
        model.fit_base(X_mix, y_mix)
        model.fit_meta(model.base_proba_features(X_mix), y_mix)

        ev = model.evaluate(Xt, y_wisdm)
        results.append((frac * 100, ev["acc"], ev["standing_recall"]))
        print(f"  [FT {int(frac*100):>3}%]  Acc: {ev['acc']:.4f}  "
              f"STANDING Recall: {ev['standing_recall']:.4f}")

    return results


# =========================
# STEP 1–4: MAIN ACCURACY CURVE
# =========================
def plot_accuracy_curve(results):
    # Prepend 0% = Baseline
    pcts    = [0]    + [r[0] for r in results]
    accs    = [BASELINE_ACC] + [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Reference lines ────────────────────────────────────────────────
    ax.axhline(y=BASELINE_ACC, color="red",    linestyle="--", linewidth=1.5,
               label=f"Baseline — Naïve Transfer ({BASELINE_ACC*100:.1f}%)")
    ax.axhline(y=ZSCORE_ACC,   color="orange", linestyle="--", linewidth=1.5,
               label=f"Z-score Alignment ({ZSCORE_ACC*100:.1f}%)")
    ax.axhline(y=FT50_ACC,     color="green",  linestyle=":",  linewidth=1.5,
               label=f"Fine-Tune 50% — Upper Bound ({FT50_ACC*100:.1f}%)")

    # ── Learning curve ─────────────────────────────────────────────────
    ax.plot(pcts, accs, marker="o", linewidth=2.5, markersize=10,
            color="#1f77b4", zorder=5, label="Fine-Tuning (ensemble)")

    # ── Recovery ratio annotations ─────────────────────────────────────
    for pct, acc in zip(pcts, accs):
        rr = recovery_ratio(acc, BASELINE_ACC, FT50_ACC)
        label = f"{rr*100:.0f}% recovery" if pct > 0 else "Baseline"
        ax.annotate(label,
                    xy=(pct, acc),
                    xytext=(pct + 0.5, acc + 0.018),
                    fontsize=9, color="#1f77b4", fontweight="bold")

    # ── Highlight 5% point ─────────────────────────────────────────────
    ax.annotate(f"← 5% data\nachieves {results[0][1]*100:.1f}%",
                xy=(5, results[0][1]),
                xytext=(12, 0.88),
                fontsize=10,
                color="darkblue",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.8))

    # Shade the region between baseline and z-score
    ax.axhspan(BASELINE_ACC, ZSCORE_ACC, alpha=0.06, color="orange",
               label="Unsupervised gain (Z-score)")

    ax.set_xlim(-2, 55)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Target-Domain Labeled Data Used (%)", fontsize=12)
    ax.set_ylabel("Accuracy on WISDM (target domain)", fontsize=12)
    ax.set_title("Supervised Recovery from Cross-Dataset Domain Shift\n(HAPT → WISDM)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_accuracy_learning_curve.png"), dpi=300)
    plt.close()
    print("\n  Saved: 01_accuracy_learning_curve.png")
    
# =========================
# STEP 5: STANDING RECALL CURVE
# =========================
def plot_standing_recall_curve(results):
    pcts    = [0]              + [r[0] for r in results]
    recalls = [BASELINE_RECALL] + [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Reference lines
    ax.axhline(y=BASELINE_RECALL, color="red",    linestyle="--", linewidth=1.5,
               label=f"Baseline Recall ({BASELINE_RECALL:.3f})")
    ax.axhline(y=ZSCORE_RECALL,   color="orange", linestyle="--", linewidth=1.5,
               label=f"Z-score Recall ({ZSCORE_RECALL:.3f})")
    ax.axhline(y=FT50_RECALL,     color="green",  linestyle=":",  linewidth=1.5,
               label=f"Fine-Tune 50% ({FT50_RECALL:.3f})")

    # Curve
    ax.plot(pcts, recalls, marker="s", linewidth=2.5, markersize=10,
            color="#d62728", zorder=5, label="STANDING Recall (Fine-Tuning)")

    # Recovery annotations
    for pct, rec in zip(pcts, recalls):
        rr = recovery_ratio(rec, BASELINE_RECALL, FT50_RECALL)
        label = f"{rr*100:.0f}% recovery" if pct > 0 else "Baseline"
        ax.annotate(label,
                    xy=(pct, rec),
                    xytext=(pct + 0.5, rec + 0.015),
                    fontsize=9, color="#d62728", fontweight="bold")

    ax.set_xlim(-2, 55)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Target-Domain Labeled Data Used (%)", fontsize=12)
    ax.set_ylabel("STANDING Recall", fontsize=12)
    ax.set_title("STANDING Recall Recovery via Fine-Tuning\n(HAPT → WISDM)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_standing_recall_curve.png"), dpi=300)
    plt.close()
    print("  Saved: 02_standing_recall_curve.png")


# =========================
# STEP 6: DATA-EFFICIENCY METRICS
# =========================
def compute_data_efficiency(results):
    print("\n=== STEP 6: Data Efficiency Metrics ===")

    pcts    = [0]    + [r[0] for r in results]
    accs    = [BASELINE_ACC] + [r[1] for r in results]
    recalls = [BASELINE_RECALL] + [r[2] for r in results]

    rows = []
    for i, (pct, acc, rec) in enumerate(zip(pcts, accs, recalls)):
        rr_acc = recovery_ratio(acc, BASELINE_ACC, FT50_ACC)
        rr_rec = recovery_ratio(rec, BASELINE_RECALL, FT50_RECALL)

        # Marginal gain vs previous step
        if i == 0:
            marginal_acc = 0.0
            marginal_rec = 0.0
        else:
            marginal_acc = acc    - accs[i-1]
            marginal_rec = recalls[i] - recalls[i-1]

        rows.append({
            "% Data Used":         int(pct),
            "Accuracy":            round(acc, 4),
            "Recovery Ratio (Acc)":round(rr_acc, 4),
            "Marginal Acc Gain":   round(marginal_acc, 4),
            "STANDING Recall":     round(rec, 4),
            "Recovery Ratio (Rec)":round(rr_rec, 4),
            "Marginal Recall Gain":round(marginal_rec, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "03_data_efficiency_table.csv"), index=False)
    print(df.to_string(index=False))

    # Key thresholds
    print("\n  Key Thresholds:")
    for pct, acc in zip(pcts, accs):
        if acc >= ZSCORE_ACC:
            print(f"  → Exceeds Z-score accuracy ({ZSCORE_ACC:.4f}) at: {pct}% data")
            break

    for pct, acc in zip(pcts, accs):
        rr = recovery_ratio(acc, BASELINE_ACC, FT50_ACC)
        if rr >= 0.80:
            print(f"  → Reaches 80% recovery at: {pct}% data")
            break

    # Marginal gains
    if len(accs) >= 3:
        g_0_5   = accs[1] - accs[0]
        g_5_10  = accs[2] - accs[1]
        g_10_25 = accs[3] - accs[2] if len(accs) > 3 else None
        print(f"\n  Marginal accuracy gains:")
        print(f"    0% → 5%:   +{g_0_5:.4f}  ({g_0_5*100:.2f}pp)")
        print(f"    5% → 10%:  +{g_5_10:.4f}  ({g_5_10*100:.2f}pp)")
        if g_10_25 is not None:
            print(f"    10% → 25%: +{g_10_25:.4f}  ({g_10_25*100:.2f}pp)")

    return df


# =========================
# STEP 8: COMPARISON TABLE
# =========================
def save_comparison_table(results):
    pcts    = [0]    + [r[0] for r in results]
    accs    = [BASELINE_ACC] + [r[1] for r in results]
    recalls = [BASELINE_RECALL] + [r[2] for r in results]

    method_names = ["Baseline (Naïve Transfer)", "Z-score (Unsupervised)"] + \
                   [f"Fine-Tune {int(r[0])}%" for r in results]
    acc_vals     = [BASELINE_ACC, ZSCORE_ACC] + [r[1] for r in results]
    rec_vals     = [BASELINE_RECALL, ZSCORE_RECALL] + [r[2] for r in results]
    delta_vals   = [0.0, ZSCORE_ACC - BASELINE_ACC] + \
                   [r[1] - BASELINE_ACC for r in results]

    df = pd.DataFrame({
        "Method":               method_names,
        "Accuracy (%)":         [round(a * 100, 2) for a in acc_vals],
        "Δ vs Baseline (pp)":   [round(d * 100, 2) for d in delta_vals],
        "STANDING Recall":      [round(r, 3) for r in rec_vals],
    })

    df.to_csv(os.path.join(OUT_DIR, "04_comparison_table.csv"), index=False)
    print("\n  Comparison Table:")
    print(df.to_string(index=False))
    print("  Saved: 04_comparison_table.csv")
    return df


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Loading data...")
    X_hapt, y_hapt, X_wisdm, y_wisdm = load_data()
    print(f"HAPT: {X_hapt.shape}  WISDM: {X_wisdm.shape}")

    # Run fine-tuning at each fraction
    print("\n=== Running Fine-Tuning Learning Curve ===")
    results = build_learning_curve(
        X_hapt, y_hapt, X_wisdm, y_wisdm,
        fracs=(0.05, 0.10, 0.25, 0.50)
    )

    # STEP 1–4: Accuracy curve with reference lines + recovery annotations
    print("\n=== Plotting Accuracy Learning Curve ===")
    plot_accuracy_curve(results)

    # STEP 5: STANDING recall curve
    print("=== Plotting STANDING Recall Curve ===")
    plot_standing_recall_curve(results)

    # STEP 6: Data efficiency metrics
    df_eff = compute_data_efficiency(results)

    # STEP 8: Comparison table
    df_comp = save_comparison_table(results)

    print(f"\n\n All Priority 4 outputs saved to: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")